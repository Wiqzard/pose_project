from typing import Optional, Tuple, Union
import math

import torch
import torch.nn.functional as F
from torch import NoneType, Tensor
from torch.nn import Parameter
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
from models.custom_layers import GeGLU


class GATv2Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        d_cond: int = 1024,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.d_cond = d_cond
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )
        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )

        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.inf_k = nn.Linear(self.d_cond, heads * out_channels)
        self.inf_v = nn.Linear(self.d_cond, heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        cond: OptTensor = None,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ):
        H, C = self.heads, self.out_channels
        BS = cond.shape[0]

        x_l: OptTensor = None
        x_r: OptTensor = None

        assert x.dim() == 2, "x should be [N_sum, d_in]"
        x_l = self.lin_l(x).view(-1, H, C)
        if self.share_weights:
            x_r = x_l
        else:
            x_r = self.lin_r(x).view(-1, H, C)

        if cond is not None:
            assert cond.dim() == 3, "cond should be [BS, N, d_cond]"
            cond_k = self.inf_k(cond).view(BS, -1, H, C)
            cond_v = self.inf_v(cond).view(BS, -1, H, C)
        else:
            cond_k = None
            cond_v = None

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            size=None,
        )

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        #        if cond_q_i is not None and cond_v_i is not None:
        #            alpha = (x * cond_q_i).sum(dim=-1) / math.sqrt(self.out_channels)
        #            alpha = softmax(alpha, index, ptr, size_i)
        #            out = (alpha.unsqueeze(-1) * cond_v_i) * x_j
        #        else:
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1)

    # return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class SpatTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, d_cond) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.d_cond = d_cond

        self.query_proj = nn.Linear(in_channels, n_heads * out_channels)
        self.key_proj = nn.Linear(d_cond, n_heads * out_channels)
        self.value_proj = nn.Linear(d_cond, n_heads * out_channels)

        # self.query_proj = nn.Linear(d_cond, n_heads * out_channels)
        # self.key_proj = nn.Linear(in_channels, n_heads * out_channels)
        # self.value_proj = nn.Linear(in_channels, n_heads * out_channels)

    def forward(self, x, cond=None):
        # cond shape: (batch_size, seq_len, cond_dim)
        # x shape: (sum(N_i), in_channels)

        assert cond is not None, "cond must be provided"
        if cond is None:
            cond = torch.zeros(x.shape[0], 1, 1)
        else:
            # important to work with padded graphs or fixed size graphs
            bs = cond.shape[0]
            assert x.shape[0] % bs == 0, "batch size must be a divisor of x.shape[0]"
            # transform features
            # makes use of positional invariance of transformer
            # pad graphs with zero vectors to allow for batch processing
            # split graph into batch of graphs (batch_size, num_nodes, in_channels)
            x = torch.stack(torch.chunk(x, bs, dim=0))
            query = self.query_proj(x).view(*x.shape[:2], self.n_heads, -1)
            key = self.key_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
            value = self.value_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
            attn = torch.einsum("bnhd,bmhd->bhnm", query, key) / math.sqrt(
                self.out_channels
            )
            attn_weights = F.softmax(attn, dim=-1)
            out = torch.einsum("bhnm,bmhd->bnhd", attn_weights, value)
            out = out.reshape(*out.shape[:2], -1)
            out = out.reshape(-1, out.shape[-1])  # (sum(N_i), out_channels)
        return 0


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int) -> None:
        super().__init__()
        # self.layers = nn.ModuleList([SpatTransformer(channels, channels, n_heads, d_cond) for _ in range(n_layers)])
        self.norm = nn.BatchNorm1d(channels)
        self.proj_in = GCNConv(channels, channels)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = GCNConv(channels, channels)

    def forward(self, x: Tensor, edge_index: Tensor, cond: Tensor) -> Tensor:
        assert x.ndim == 2, "x must be 2d"
        assert cond.ndim == 3, "cond must be (batch_size, seq_len, cond_dim)"
        n, d = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x, edge_index)
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = self.proj_out(x, edge_index)
        return x + x_in


class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int) -> None:
        super().__init__()

        # self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        # self.norm1 = nn.LayerNorm(d_model)
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            GeGLU(d_model, d_model * 4),
            nn.Dropout(0.0),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """For now ignore self attention"""
        # x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), cond)
        x = x + self.net(self.norm3(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.scale = d_head**-0.5

        d_attn = n_heads * d_head
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(d_cond, d_attn, bias=False)
        self.v_proj = nn.Linear(d_cond, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_model, bias=False)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        bs = cond.shape[0]
        assert x.shape[0] % bs == 0, "batch size must be a divisor of x.shape[0]"

        x = torch.stack(torch.chunk(x, bs, dim=0))
        cond = x if cond is None else cond
        q = self.q_proj(x).view(*x.shape[:2], self.n_heads, -1)
        k = self.k_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
        v = self.v_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v)
        out = out.reshape(*out.shape[:2], -1)
        return self.out_proj(out).reshape(-1, self.d_model)


import torch
from data_tools.graph_tools.graph import Graph


def main() -> int:
    gat_v2 = GATv2Conv(16, 32, heads=4, concat=False)  # concat True, returns 128
    graph = Graph.create_random_graph(100, 16)
    graph.set_edge_index()
    x = torch.from_numpy(graph.feature_matrix)
    edge_index = torch.from_numpy(graph.edge_index).T
    cond = torch.randn(10, 47, 1024)
    # out = gat_v2(x, edge_index, cond)
    cross = CrossAttention(16, 1024, 4, 32)
    spat_trans = SpatialTransformer(16, 4, 4, 1024)
    # print number of parameters
    print( sum(p.numel() for p in spat_trans.parameters()))
    out = spat_trans(x, edge_index, cond)
    #   out = cross(x, cond)
    # out2 = SpatTransformer(16, 32, 4, 1024)(x, cond)

    return 0


if __name__ == "__main__":
    main()
