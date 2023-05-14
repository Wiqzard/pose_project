import itertools
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential
import numpy as np

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GATv2Conv

from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import (
    softmax,
)

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
        self.input_blocks.append(
            TimestepEmbedSequential(GCNConv(in_channels, channels))
        )
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i, _ in itertools.product(range(levels), range(n_res_blocks + 1)):
            layers = [
                GraphResNetBlock(channels, d_time_emb, out_channels=channels_list[i])
            ]
            channels = channels_list[i]

            if i in attention_levels:
                layers.append(InfusionTransformer(channels, n_heads, tf_layers, d_cond))
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            GraphResNetBlock(channels, d_time_emb),
            # GATv2Conv(
            #    channels, d_t_emb=d_time_emb, out_channels=channels, heads=n_heads
            # ),
            # GraphResNetBlock(channels, d_time_emb),
            # GATv2Conv(
            #    channels, d_t_emb=d_time_emb, out_channels=channels, heads=n_heads
            # ),
            InfusionTransformer(channels, n_heads, tf_layers, d_cond),
            GraphResNetBlock(channels, d_time_emb),
            # GATv2Conv(
            #    channels, d_t_emb=d_time_emb, out_channels=channels, heads=n_heads
            # ),
            # GraphResNetBlock(channels, d_time_emb),
            # GATv2Conv(
            #    channels, d_t_emb=d_time_emb, out_channels=channels, heads=n_heads
            # ),
        )
        self.output_blocks = nn.ModuleList([])
        for i, _ in itertools.product(reversed(range(levels)), range(n_res_blocks + 1)):
            layers = [
                GraphResNetBlock(
                    channels + input_block_channels.pop(),
                    d_time_emb,
                    out_channels=channels_list[i],
                )
            ]
            channels = channels_list[i]
            if i in attention_levels:
                layers.append(InfusionTransformer(channels, n_heads, tf_layers, d_cond))
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_norm = normalization(channels)
        self.out_act = nn.SiLU()
        self.out = GCNConv(channels, out_channels)

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
        self, x: Tensor, edge_index: Tensor, time_steps: Tensor, cond: Tensor
    ) -> None:
        """
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # Get time step embeddings
        x_input_block = []
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)
        pos_enc = self.pos_emb(cond)
        cond += pos_enc
        # maybe concat instead
        # pos emb
        for module in self.input_blocks:
            x = module(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)
            x_input_block.append(x)
            print(x.shape)
        x = self.middle_block(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=-1)
            x = module(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)
            print(x.shape)
        return self.out(x, edge_index)


class GraphEncoding(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        n_resulutions = len(channel_multipliers)
        self.conv_in = GCNConv(in_channels, channels)
        channels_list = [channels * m for m in channel_multipliers]
        self.down


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, edge_index, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, GraphResNetBlock):
                x = layer(x=x, edge_index=edge_index, t_emb=t_emb)
            elif isinstance(layer, InfusionTransformer):
                x = layer(x=x, edge_index=edge_index, cond=cond)
            else:
                x = layer(x=x, edge_index=edge_index)
        return x


class InfusionTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int,
        n_layers: int,
        d_cond: int,
        concat: bool = False,
    ) -> None:
        super().__init__()

        self.norm = nn.BatchNorm1d(channels)
        self.proj_in = GCNConv(channels, channels)
        transformer_block = Sequential(
            "x, edge_index, cond",
            [
                (
                    InfusionTransformerLayer(
                        in_channels=channels,
                        out_channels=channels,
                        n_heads=n_heads,
                        d_cond=d_cond,
                        concat=concat,
                    ),
                    "x, edge_index, cond -> x",
                ),
                (nn.LayerNorm(channels), "x -> x"),
                (
                    InfusionTransformerLayer(
                        in_channels=channels,
                        out_channels=channels,
                        n_heads=n_heads,
                        d_cond=d_cond,
                        concat=concat,
                    ),
                    "x, edge_index, cond -> x",
                ),
                (nn.LayerNorm(channels), "x -> x"),
                (GCNConv(channels, channels), "x, edge_index -> x"),
                (nn.SiLU(), "x -> x"),
                (GCNConv(channels, channels), "x, edge_index -> x"),
            ],
        )

        self.transformer_blocks = nn.ModuleList(
            [transformer_block for _ in range(n_layers)]
        )
        self.proj_out = GCNConv(channels, channels)

    def forward(self, x, edge_index, cond):
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x, edge_index)
        for block in self.transformer_blocks:
            x = block(x, edge_index, cond)
        x = self.proj_out(x, edge_index)
        return x_in + x


class InfusionTransformerLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_heads: int = 1,
        d_cond: int = 768,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = n_heads
        self.d_cond = d_cond
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], n_heads * out_channels)
        self.lin_query = Linear(in_channels[1], n_heads * out_channels)
        self.lin_value = Linear(in_channels[0], n_heads * out_channels)

        self.lin_key_cond = nn.Linear(d_cond, n_heads * out_channels)
        self.lin_value_cond = nn.Linear(d_cond, n_heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, n_heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], n_heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * n_heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        cond: OptTensor = None,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        if cond is None:
            key = self.lin_key(x[0]).view(-1, H, C)
            value = self.lin_value(x[0]).view(-1, H, C)
        else:
            key = self.lin_key_cond(cond).view(-1, H, C)
            value = self.lin_value_cond(cond).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if not isinstance(return_attention_weights, bool):
            return out
        assert alpha is not None
        if isinstance(edge_index, Tensor):
            return out, (edge_index, alpha)
        elif isinstance(edge_index, SparseTensor):
            return out, edge_index.set_value(alpha, layout="coo")

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encodings[: x.shape[0]]


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
        self.in_conv = Sequential(
            "x, edge_index",
            [
                (nn.BatchNorm1d(channels), "x -> x1"),
                (nn.Dropout(p=0.0), "x1 -> x1"),
                (GCNConv(channels, out_channels), "x1, edge_index -> x2"),
                (nn.SiLU(), "x2 -> x2"),
            ],
        )
        self.out_conv = Sequential(
            "x, edge_index",
            [
                (nn.BatchNorm1d(out_channels), "x -> x1"),
                (nn.Dropout(p=0.0), "x1 -> x1"),
                (GCNConv(out_channels, out_channels), "x1, edge_index -> x2"),
                (nn.SiLU(), "x2 -> x2"),
            ],
        )
        self.emb_layers = nn.Sequential(
            OrderedDict(
                [
                    ("emb_fc1", nn.Linear(d_t_emb, out_channels)),
                    ("act", nn.SiLU()),
                ]
            )
        )

        if out_channels == channels:
            self.skip_connection = None  # nn.Identity()
        else:
            self.skip_connection = GCNConv(channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ):
        h = self.in_conv(x, edge_index)
        if t_emb is not None:
            t_emb = self.emb_layers(t_emb).type(h.dtype)
            h += t_emb  # [: ,: ,]
        h = self.out_conv(h, edge_index)
        return (
            self.skip_connection(x, edge_index) + h
            if self.skip_connection is not None
            else x + h
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
        self.proj = GCNConv(d_in, d_out * 2)  # nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x, edge_index).chunk(2, dim=-1)
        return x * F.gelu(gate)


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


class GraphTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True
        )
        self.proj_in = GCNConv(channels, channels)
        self.transformer_blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = GCNConv(channels, channels)

    def forward(self, x, edge_index, cond=None):
        h_in = x
        h = x.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        h = self.proj_in(x=h, edge_index=edge_index)
        # h = h.permute(0, 2, 1).reshape(bs, c_dim, n_nodes)
        for block in self.transformer_blocks:
            h = block(x=h, edge_index=edge_index, cond=cond)
        h = self.proj_out(h, edge_index)
        return h + h_in


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int) -> None:
        super().__init__()
        self.attn1 = MultiHeadAttentionLayer(d_model, d_cond, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = MultiHeadAttentionLayer(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.geglu = GraphGeGLU(d_model, d_model * 4)
        self.out = GCNConv(d_model * 4, d_model)
        self.ff = nn.Sequential(
            nn.Sequential(
                GraphGeGLU(d_model, d_model * 4),
                nn.Dropout(0.0),
                GCNConv(d_model * 4, d_model),
            ),
            #            nn.Sequential(
            #                GeGLU(d_model, d_model * 4),
            #                nn.Dropout(0.0),
            #                nn.Linear(d_model * 4, d_model),
            #            ),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, cond=None):
        h = x + self.attn1(x=self.norm1(x), edge_index=edge_index, cond=cond)
        h = h + self.attn2(x=self.norm2(h), edge_index=edge_index, cond=cond)
        h = h + self.out(
            self.geglu(x=self.norm3(h), edge_index=edge_index),
            edge_index=edge_index,
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

        self.q = GCNConv(d_model, d_attn)
        self.k = GCNConv(d_model, d_attn)
        self.v = GCNConv(d_model, d_attn)

        self.k_cond = nn.Linear(d_cond, d_attn, bias=False)
        self.v_cond = nn.Linear(d_cond, d_attn, bias=False)

        # out projection
        self.to_out = GCNConv(d_attn, d_model)
        # self.to_out = nn.Linear(d_attn, d_model)

    def forward(self, x, edge_index, cond=None):
        q = self.q(x, edge_index)
        if cond is not None:
            k = self.k_cond(cond)
            v = self.v_cond(cond)
        else:
            k = self.k(x, edge_index)
            v = self.v(x, edge_index)
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale
        # attn = torch.where(
        #    edge_index.unsqueeze(1) > 0, attn, torch.tensor(-np.inf).to(attn.device)
        # )
        attn_weights = torch.softmax(attn, dim=-1)
        output = torch.einsum("bhij,bjhd->bihd", attn_weights, v)
        output = output.reshape(*output.shape[:2], -1)
        output = self.to_out(output, edge_index)
        return output
