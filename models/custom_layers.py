from collections import OrderedDict
from typing import Optional


import torch
import torch.nn as nn
#from torch.nn import Sequential
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential


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

    def __init__(
        self,
        channels: int,

        d_t_emb: Optional[int] = None,
        out_channels: Optional[int] = None
    ):
        """
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If set to True, the layer will learn an additive bias. Defaults to True.
        """
        super().__init__()
        if not out_channels:
            out_channels = channels
        #self.in_conv = Sequential(
        #    "x1, edge_index",
        #    [
        #        #(nn.BatchNorm1d(channels), "x -> x1"),
        #        (nn.Dropout(p=0.0), "x1 -> x1"),
        #        (GCNConv(channels, out_channels), "x1, edge_index -> x2"),
        #        (nn.SiLU(), "x2 -> x2"),
        #    ],
        #)
        self.in_conv = GCNConv(channels, out_channels)
        self.out_conv = Sequential(
            "x1, edge_index",
            [
                #(nn.BatchNorm1d(out_channels), "x -> x1"),
                (nn.Dropout(p=0.0), "x1 -> x1"),
                (GCNConv(out_channels, out_channels), "x1, edge_index -> x2"),
                (nn.SiLU(), "x2 -> x2"),
            ],
        )
        if d_t_emb is not None:
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
        x: Tensor,
        edge_index: Tensor,
        t_emb: Optional[Tensor] = None,
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
