import enum
import math
import numpy as np
import torch
import torch.nn as nn
import timm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_backbone() -> nn.Module:
    # input size is 1x3x224x224
    # output size is 1x1000
    model = timm.create_model(
        "deit3_small_patch16_224_in21ft1k", pretrained=True, in_chans=3
    )  # deit3_small_patch16_224_in21ft1k
    model.eval()
    return model


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


from models.diffusion_pose.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)

# conv_next = timm.create_model(
#    model_name="convnext_base", pretrained=True, features_only=True
# )
# result2 = conv_next(example_img)
# feats2 = conv_next.forward_features(example_img) # equivalient to features_only=True, out_indices=(3,)
# feats_pooled2 = conv_next.forward_head(feats2, pre_logits=True)
# print(result2.shape)
# print(feats2.shape)
# print(feats_pooled2.shape)
from models.test import (
    GraphResNetBlock,
    CustomGraphNet,
    MultiHeadAttentionLayer,
    GraphTransformer,
    LearnedPositionalEmbeddings,
    InfusionTransformer,
)
from data_tools.graph_tools.graph import Graph
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, TransformerConv


def main() -> int:
    bs = 1
    example_img = torch.rand(bs, 3, 224, 224)
    example_pose = torch.rand(bs, 3, 4)
    example_tranlation = example_pose[:, :, 3]
    example_rotation = example_pose[:, :, :3]
    cond = torch.randn(1, 7 * 7, 1024)

    channels = 3
    out_channels = 3
    graph = Graph.create_random_graph(20, channels)
    graph.set_edge_index()
    # graph = Graph.to_torch_geometric(graph)
    time_steps = torch.arange(0, 1, 1)

    ################################################
    graph_net = CustomGraphNet(
        in_channels=channels,
        out_channels=out_channels,
        channels=32,  # min 32
        n_res_blocks=2,
        attention_levels=[1, 2],  # 4, 6, 7, 8],
        channel_multipliers=[1, 1, 2, 2],  # 4, 4, 8, 8],
        n_heads=4,
        tf_layers=2,
        d_cond=1024,
    )
    print(
        "number of parameters: ",
        sum(p.numel() for p in graph_net.parameters() if p.requires_grad),
    )
    # out = graph_net(graph.x, graph.edge_index, time_steps, cond)
    backbone = timm.create_model(
        model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        pretrained=True,
    )
    from models.test_diffusion import LatentDiffusion, DDPMSampler, DenoiseDiffusion

    model = LatentDiffusion(
        unet_model=graph_net,
        backbone=backbone,
        latent_scaling_factor=1.0,
        n_steps=1000,
        linear_start=0,
        linear_end=1,
    )
    ddpmsampler = DDPMSampler(model)
    ddpm = DenoiseDiffusion(model, n_steps=100, device="cpu")
    cond = model.get_image_conditioning(example_img)  # 1, 1024, 7, 7
    x = torch.from_numpy(graph.feature_matrix)
    edge_index = torch.from_numpy(graph.edge_index).T

    noised_feat = ddpm.q_sample(x0=x, index=3)
    denoised_feat = ddpm.p_sample(
        x=noised_feat, edge_index=edge_index, c=cond, t=time_steps, step=3
    )
    ############################################
    graph_resnet = GraphResNetBlock(
        channels=channels, d_t_emb=256, out_channels=out_channels
    )
    gat = GATv2Conv(
        in_channels=channels, out_channels=out_channels, heads=4, concat=False
    )
    tran = TransformerConv(
        in_channels=channels, out_channels=out_channels, heads=4, concat=False
    )
    inf = InfusionTransformer(
        in_channels=channels,
        out_channels=out_channels,
        n_heads=4,
        n_layers=2,
        d_cond=256,
    )

    print(graph_resnet(graph.x, graph.edge_index).shape)
    print(inf(graph.x, graph.edge_index, cond).shape)
    print(tran(graph.x, graph.edge_index).shape)
    print(gat(graph.x, graph.edge_index).shape)

    g = torch.from_numpy(graph.adjacency_matrix).unsqueeze(0)
    h = torch.from_numpy(graph.feature_matrix).unsqueeze(0)
    t_emb = graph_net.time_step_embedding(time_steps)
    # t_emb = graph_net.time_step_embedding(time_steps)

    output = graph_net(adj_mat=g, feat_mat=h, time_steps=time_steps, cond=cond)
    print(0)

    # print(timm.models.list_models("vit*"))
    # print(timm.version.__version__)
    # model = timm.create_model(, pretrained=True)
    model = timm.create_model(
        model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        pretrained=True,
    )
    model.eval()
    result = model(example_img)
    feats = model.forward_features(example_img)  # unpooled 1x1024x7x7
    feats_pooled = model.forward_head(feats, pre_logits=True)  # 1x1024
    print(result.shape)
    print(feats.shape)
    print(feats_pooled.shape)

    betas = get_named_beta_schedule("linear", 1000)
    diffuser = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED,
        loss_type=LossType.RESCALED_KL,
    )
    pose = torch.ones(bs, 3, 3)
    timesteps = torch.tensor([1]).unsqueeze(0)
    diffused_pose = diffuser.q_sample(pose, timesteps)

    print(pose, diffused_pose)
    schedule = get_named_beta_schedule("linear", 20)

    return 0


if __name__ == "__main__":
    main()
