import itertools
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

Tensor = torch.Tensor

"""
Graph Diffusion Network
    -> Input: Graph G = (H, A), t_emb, cond where H is the node feature matrix and A is the adjacency matrix
    -> Output: H_t where H_t is the node feature matrix at time t 
    -> H_t = GDN(H, A, t_emb, cond) 
    
    -> Input graph is random noise
    -> Put trough mixture of GCN, GATv2, and other layers to produce H_t 

    
    - Since adjacency matrix is fixed, there is no glueing
"""


class CustomGraphNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        n_heads: int,
        tf_layers: int = 1,
        d_cond: int = 768,
    ) -> None:
        """
        Args:
            in_channels (int): The number of channels in the input feature map.
            out_channels (int): The number of output channels.
            channels (int): The base channel count for the model.
            n_res_blocks (int): The number of residual blocks at each level.
            attention_levels (List[int]): The levels at which attention should be performed.
            channel_multiplier (List[int]): The factors for the number of channels at each level.
            n_heads (int): The number of heads transformers.
            tf_layers (int, optional): The number of transformer layers. Defaults to 1.
            d_cond (int, optional): The dimension of the conditioning vector. Defaults to 768.
        """

        super().__init__()
        self.channels = channels
        levels = len(channel_multipliers)
        d_time_emb = channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        self.pos_emb = LearnedPositionalEmbeddings(d_model=d_cond)

        # Input block, upsamples the input channels
        self.input_blocks = nn.ModuleList([])
        self.input_blocks.append(TimestepEmbedSequential(GCN(in_channels, channels)))
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i, _ in itertools.product(range(levels), range(n_res_blocks)):
            layers = [
                GraphResNetBlock(channels, d_time_emb, out_channels=channels_list[i])
            ]
            channels = channels_list[i]

            if i in attention_levels:
                layers.append(GraphTransformer(channels, n_heads, tf_layers, d_cond))
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            GraphResNetBlock(channels, d_time_emb),
            GraphTransformer(channels, n_heads, tf_layers, d_cond),
            GraphResNetBlock(channels, d_time_emb),
        )
        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            for _ in range(n_res_blocks):  # +1
                layers = [
                    GraphResNetBlock(
                        channels + input_block_channels.pop(),
                        d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]
                if i in attention_levels:
                    layers.append(
                        GraphTransformer(channels, n_heads, tf_layers, d_cond)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_norm = normalization(channels)
        self.out_act = nn.SiLU()
        self.out = GCN(channels, out_channels)

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(
        self, adj_mat: Tensor, feat_mat: Tensor, time_steps: Tensor, cond: Tensor
    ) -> None:
        """
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # Get time step embeddings
        x_input_block = []
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)
        # pos emb
        x = feat_mat
        for module in self.input_blocks:
            x = module(feat_mat=x, adj_mat=adj_mat, t_emb=t_emb, cond=cond)
            x_input_block.append(x)

        return 0


class GraphEncoding(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        z_channels: int
    ) -> None:
        super().__init__()
        n_resulutions = len(channel_multipliers)
        self.conv_in = GCN(in_channels, channels)
        channels_list = [channels * m for m in channel_multipliers]
        self.down


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, feat_mat, adj_mat, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, GraphResNetBlock):
                x = layer(feat_mat=feat_mat, adj_mat=adj_mat, t_emb=t_emb)
            elif isinstance(layer, GraphTransformer):
                x = layer(feat_mat=feat_mat, adj_mat=adj_mat, cond=cond)
            else:
                x = layer(feat_mat=feat_mat, adj_mat=adj_mat)
        return x


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encodings[: x.shape[0]]


class GraphTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        self.proj_in = GCN(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = GCN(channels, channels)

    def forward(self, feat_mat, adj_mat, cond=None):
        h_in = feat_mat
        h = feat_mat.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        h = self.proj_in(feat_mat=h, adj_mat=adj_mat)
        # h = h.permute(0, 2, 1).reshape(bs, c_dim, n_nodes)
        for block in self.transformer_blocks:
            h = block(feat_mat=h, adj_mat=adj_mat, cond=cond)
        h = self.proj_out(h, adj_mat)
        return h + h_in


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int) -> None:
        super().__init__()
        self.attn1 = MultiHeadAttentionLayer(d_model, d_cond, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = MultiHeadAttentionLayer(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.geglu = GraphGeGLU(d_model, d_model * 4)
        self.out = GCN(d_model * 4, d_model)
        self.ff = nn.Sequential(
            nn.Sequential(
                GraphGeGLU(d_model, d_model * 4),
                nn.Dropout(0.0),
                GCN(d_model * 4, d_model),
            ),
            #            nn.Sequential(
            #                GeGLU(d_model, d_model * 4),
            #                nn.Dropout(0.0),
            #                nn.Linear(d_model * 4, d_model),
            #            ),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, feat_mat, adj_mat, cond=None):
        h = feat_mat + self.attn1(
            feat_mat=self.norm1(feat_mat), adj_mat=adj_mat, cond=cond
        )
        h = h + self.attn2(feat_mat=self.norm2(h), adj_mat=adj_mat, cond=cond)
        h = h + self.out(
            self.geglu(feat_mat=self.norm3(h), adj_mat=adj_mat), adj_mat=adj_mat
        )
        return h


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention layer. Conditional self-attention layer"""

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        self.scale = d_head**-0.5
        d_attn = d_head * n_heads
        # in projections
        #        self.q = nn.Linear(d_model, d_attn, bias=False)
        #        self.k = nn.Linear(d_cond, d_attn, bias=False)
        #        self.v = nn.Linear(d_cond, d_attn, bias=False)

        self.q = GCN(d_model, d_attn)
        self.k = GCN(d_model, d_attn)
        self.v = GCN(d_model, d_attn)

        self.k_cond = nn.Linear(d_cond, d_attn, bias=False)
        self.v_cond = nn.Linear(d_cond, d_attn, bias=False)

        # out projection
        self.to_out = GCN(d_attn, d_model)
        # self.to_out = nn.Linear(d_attn, d_model)

    def forward(self, feat_mat, adj_mat, cond=None):
        q = self.q(feat_mat, adj_mat)
        if cond is not None:
            k = self.k_cond(cond)
            v = self.v_cond(cond)
        else:
            k = self.k(feat_mat, adj_mat)
            v = self.v(feat_mat, adj_mat)
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale
        attn = torch.where(
            adj_mat.unsqueeze(1) > 0, attn, torch.tensor(-np.inf).to(attn.device)
        )
        attn_weights = torch.softmax(attn, dim=-1)
        output = torch.einsum("bhij,bjhd->bihd", attn_weights, v)
        output = output.reshape(*output.shape[:2], -1)
        output = self.to_out(output, adj_mat)
        return output


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


class GraphGeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.proj = GCN(d_in, d_out * 2)  # nn.Linear(d_in, d_out * 2)

    def forward(self, feat_mat: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(feat_mat, adj_mat).chunk(2, dim=-1)
        return x * F.gelu(gate)


class GraphUnetVarAdj(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, drop_p):
        super(GraphUnetVarAdj, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.act = nn.SiLU()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim))
            self.up_gcns.append(GCN(dim, dim))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.act(self.down_gcns[i](g, h))
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.act(self.up_gcns[i](g, h))
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z)  # .squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


class GCN(nn.Module):
    """A graph convulotional network layer (GCN)"""

    def __init__(self, in_features, out_features, bias=True):
        """
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If set to True, the layer will learn an additive bias. Defaults to True.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, feat_mat, adj_mat):
        """
        The forward function for the GCN layer.

        Args:
            x (torch.Tensor): The input features.
            adj_mat (torch.Tensor): The adjacency matrix.

        Returns:
            torch.Tensor: The output features.
        """
        ####### interchange the order of the following two lines
        # Apply the adjacency matrix
        x = torch.matmul(adj_mat, feat_mat)
        # Apply the linear transformation
        x = self.linear(x)
        return x


class GraphResNetBlock(nn.Module):
    """A graph residual network layer (GraphResNet)"""

    def __init__(self, channels: int, d_t_emb: int, *, out_channels: int = None):
        """
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If set to True, the layer will learn an additive bias. Defaults to True.
        """
        super().__init__()
        if not out_channels:
            out_channels = channels

        self.in_conv = GCN(channels, out_channels)
        self.out_conv = GCN(out_channels, out_channels)
        self.act = nn.SiLU()
        self.in_norm = normalization(channels)
        self.out_norm = normalization(out_channels)
        self.emb_layers = nn.Sequential(
            nn.Linear(d_t_emb, out_channels),
            nn.SiLU(),
        )
        if out_channels == channels:
            self.skip_connection = None  # nn.Identity()
        else:
            self.skip_connection = GCN(channels, out_channels)

    def forward(
        self, feat_mat: torch.Tensor, adj_mat: torch.Tensor, t_emb: torch.Tensor
    ):
        h = self.in_conv(
            self.act(self.in_norm(feat_mat.permute(0, 2, 1)).permute(0, 2, 1)), adj_mat
        )
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        h += t_emb  # [: ,: ,]
        h = self.out_conv(
            self.act(self.out_norm(h.permute(0, 2, 1))).permute(0, 2, 1), adj_mat
        )
        return (
            self.skip_connection(feat_mat, adj_mat) + h if self.skip_connection else h
        )


def top_k_graph(scores, g, h, k):
    batch_size, num_nodes = g.shape[:2]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)), dim=1)
    new_h = h[torch.arange(batch_size).unsqueeze(1), idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[torch.arange(batch_size).unsqueeze(1), idx, :]
    un_g = un_g[:, :, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, -1, keepdim=True)
    g = g / degrees
    return g


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)
