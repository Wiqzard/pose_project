from __future__ import annotations
from typing import List, Any, Optional, Tuple

from models.test import CustomGraphNet

import numpy as np
import torch
import torch.functional as F
from torch import nn


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DenoiseDiffusion:
    """
    Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        Args:
            eps_model: The εₜₕₑₜₐ(xₜ, t) model
            n_steps: The time step t
            device: The device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1.0 - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(xₜ|x₀) distribution q(xₜ|x₀) = 𝒩(xₜ; √𝐚̄ₜ x₀, (1-𝐚̄ₜ)𝐈).
        """
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        """
        Sample from q(xₜ|x₀)
            q(xₜ|x₀) = 𝒩(xₜ; √𝐚̄ₜ x₀, (1-𝐚̄ₜ)𝐈)
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from pₜₕₑₜₐ(xₜ₋₁|xₜ)
            pₜₕₑₜₐ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μₜₕₑₜₐ(xₜ, t), σₜ²𝐈)
            μₜₕₑₜₐ(xₜ, t) = (1/√𝐚ₜ) ⎛ xₜ - (√((1-𝐚̄ₜ)/𝐚ₜ))εₜₕₑₜₐ(xₜ, t) ⎞
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var**0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Simplified Loss L_simple(θ) = 𝔼_{t,x₀,ε} ⎡⎣ ∥ε - ε̂ₜₕₑₜₐ(√𝐚̄ₜ x₀ + √(1-𝐚̄ₜ)ε, t) ∥² ⎤⎦
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)


class DiffusionSampler:
    """
    ## Base class for sampling algorithms
    """

    model: CustomGraphNet

    def __init__(self, model: CustomGraphNet):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__()
        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model
        # Get number of steps the model was trained with $T$
        self.n_steps = model.n_steps

    def get_eps(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        *,
        uncond_scale: float,
        uncond_cond: Optional[torch.Tensor],
    ):
        r"""
        Get ε(xₜ, c)

        Args:
            x: Tensor of shape [batch_size, num_nodes, channels], representing xₜ
            t: Tensor of shape [batch_size], representing t
            c: Tensor of shape [batch_size, emb_size], representing the conditional embeddings c
            uncond_scale: Scalar value representing the unconditional guidance scale s. This is used for the formula:
                        εₜₕₑₜₐ(xₜ, c) = s * ε_condd(xₜ, c) + (s - 1) * ε_cond(xₜ, cᵤ)
            uncond_cond: Tensor representing the conditional embedding for an empty prompt cᵤ
        """
        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond_cond is None or uncond_scale == 1.0:
            return self.model(x, edge_index, t, c)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond_cond, c])
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        #
        return e_t

    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        """
        Sampling Loop

        Args:
            shape: The shape of the generated images in the form [batch_size, channels, height, width]
            cond: The conditional embeddings c
            temperature: The noise temperature (random noise gets multiplied by this)
            x_last: x_T. If not provided, random noise will be used.
            uncond_scale: The unconditional guidance scale s. This is used for the formula:
                        εₜₕₑₜₐ(xₜ, c) = s * ε_cₒₙd(xₜ, c) + (s - 1) * ε_cₒₙd(xₜ, cᵤ)
            uncond_cond: The conditional embedding for an empty prompt cᵤ
            skip_steps: The number of time steps to skip.

        """
        raise NotImplementedError()

    def paint(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_start: int,
        *,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
    ):
        """
        Painting Loop

        Args:
            x: Tensor of shape [batch_size, channels, height, width], representing x_T'
            cond: Tensor representing the conditional embeddings c
            t_start: The sampling step to start from, T'
            orig: The original image in the latent page which we are painting
            mask: The mask to keep the original image
            orig_noise: Fixed noise to be added to the original image
            uncond_scale: The unconditional guidance scale s. This is used for the formula:
                        εₜₕₑₜₐ(xₜ, c) = s * ε_cₒₙd(xₜ, c) + (s - 1) * ε_cₒₙd(xₜ, cᵤ)
            uncond_cond: The conditional embedding for an empty prompt cᵤ
        """
        raise NotImplementedError()

    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        """
        Sample from q(xₜ|x₀)

        Args:
            x0: Tensor of shape [batch_size, channels, height, width], representing x₀
            index: The time step index, t
            noise: The noise, ε
        """
        raise NotImplementedError()


class DDPMSampler(DiffusionSampler):
    """
    DDPM Sampler
    DDPM samples images by repeatedly removing noise step by step from pₜₕₑₜₐ(xₜ₋₁|xₜ).
    pₜₕₑₜₐ(xₜ₋₁|xₜ) = 𝒩(xₜ₋₁; μₜₕₑₜₐ(xₜ, t), 𝚻̃ₜ 𝐼)
    μₜ(xₜ, t) = (√𝐚̄ₜ₋₁ βₜ)/(1 - 𝐚̄ₜ) x₀ + (√𝐚ₜ)(1 - 𝐚̄ₜ₋₁)/(1 - 𝐚̄ₜ) xₜ
    𝚻̃ₜ = (1 - 𝐚̄ₜ₋₁)/(1 - 𝐚̄ₜ) βₜ
    x₀ = (1/√𝐚̄ₜ) xₜ - (√((1/𝐚̄ₜ) - 1))εₜₕₑₜₐ
    """

    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        """
        Args:
            model: The model used to predict noise ε_cond(xₜ, c)
        """
        super().__init__(model)

        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)))

        with torch.no_grad():
            # $\bar\alpha_t$
            alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.0]), alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = alpha_bar**0.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1.0 - alpha_bar) ** 0.5
            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = alpha_bar**-0.5
            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** 0.5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev**0.5) / (1.0 - alpha_bar)
            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (
                (1.0 - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1.0 - alpha_bar)
            )

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
    ):
        """
        Sampling Loop

        Args:
            shape: The shape of the generated images in the form [batch_size, channels, height, width]
            cond: The conditional embeddings c
            temperature: The noise temperature (random noise gets multiplied by this)
            x_last: x_T. If not provided, random noise will be used.
            uncond_scale: The unconditional guidance scale s. This is used for the formula:
                        εₜₕₑₜₐ(xₜ, c) = s * ε_cₒₙd(xₜ, c) + (s - 1) * ε_cₒₙd(xₜ, cᵤ)
            uncond_cond: The conditional embedding for an empty prompt cᵤ
            skip_steps: The number of time steps to skip, t'. We start sampling from T - t',
                        and x_last is then x_(T - t').
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        # Sampling loop
        for step in time_steps:  # monit.iterate('Sample', time_steps):
            # Time step $t$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
    ):
        """
        Sample xₜ₋₁ from pₜₕₑₜₐ(xₜ₋₁ | xₜ)

        Args:
            x: Tensor of shape [batch_size, channels, height, width], representing xₜ
            c: Tensor of shape [batch_size, emb_size], representing the conditional embeddings c
            t: Tensor of shape [batch_size], representing t
            step: The step t as an integer
            repeat_noise: Specifies whether the noise should be the same for all samples in the batch
            temperature: The noise temperature (random noise gets multiplied by this)
            uncond_scale: The unconditional guidance scale s. This is used for the formula:
                        εₜₕₑₜₐ(xₜ, c) = s * ε_cₒₙd(xₜ, c) + (s - 1) * ε_cₒₙd(xₜ, cᵤ)
            uncond_cond: The conditional embedding for an empty prompt cᵤ
        """

        # Get $\epsilon_\theta$
        e_t = self.get_eps(
            x, edge_index, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond
        )

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step]
        )
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step]
        )

        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        #
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        #
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        r"""
        Sample from q(xₜ|x₀)

        q(xₜ|x₀) = 𝒩(xₜ; √𝐚̄ₜ x₀, (1-𝐚̄ₜ)𝐈)

        Args:
            x0: Tensor of shape [batch_size, channels, height, width], representing x₀
            index: The time step index, t
            noise: The noise, ε
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise


class DiffusionWrapper(nn.Module):
    def __init__(self, diffusion_model: CustomGraphNet):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_steps: torch.Tensor,
        context: torch.Tensor,
    ):
        return self.diffusion_model(x, edge_index, time_steps, context)


class LatentDiffusion(nn.Module):
    model: DiffusionWrapper
    # first_stage_model: Autoencoder
    cond_stage_model: Any

    def __init__(
        self,
        unet_model: CustomGraphNet,
        # autoencoder: Autoencoder,
        # clip_embedder: CLIPTextEmbedder,
        backbone: Any,
        latent_scaling_factor: float,
        n_steps: int,
        linear_start: float,
        linear_end: float,
    ):
        super().__init__()
        self.model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.latent_scaling_factor = latent_scaling_factor
        self.cond_stage_model = backbone
        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    @property
    def device(self):
        """
        Get device of the model
        """
        return next(iter(self.model.parameters())).device

    def get_image_conditioning(self, images: torch.Tensor):
        """
        Get image conditioning
        Args:
            images: Image tensor (B, C, H, W)
        Returns:
            Image conditionins: tensor (B, H'*W', C')
        """
        return (
            self.cond_stage_model.forward_features(images).flatten(2).permute(0, 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ):
        """
        Predict noise
        Predict noise given the latent representation xₜ, time step t, and the conditioning context c.
        ε_cond(xₜ, c)

        """
        return self.model(x, edge_index, t, cond)
