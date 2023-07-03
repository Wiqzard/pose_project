from typing import Optional, Tuple, Union, List
import math
import itertools
from enum import Enum, auto

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
from models.custom_layers import GeGLU, LearnedPositionalEmbeddings, GraphResNetBlock
from subdivision import SubdivideMeshes

class AttentionMode(Enum):
    """
    The attention mechanism to use.
    """
    GAT = auto()
    TRANS = auto()


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs

    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.layer_names = [layer[1].__class__.__name__ for layer in self._modules.items()]

    def forward(self, x, faces, edge_index, t_emb, cond=None):
        for layer in self:
            bs, n, c = x.shape
            if isinstance(layer, GraphResNetBlock):
                x = x.reshape(-1, x.shape[-1])
                x = layer(x=x, edge_index=edge_index, t_emb=t_emb)
                x = x.reshape(bs, n, -1)
            elif isinstance(layer, Gatv2CrossAttention) or isinstance(layer, TransformerCrossAttention):
                x = x.reshape(-1, x.shape[-1])
                x = layer(x=x, edge_index=edge_index, cond=cond)
                x = x.reshape(bs, n, -1)
            else:
                x = layer(verts=x, faces=faces, edges=edge_index.T)
        return x


class GraphUnpoolingConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.squeeze(0)
        x = x.permute(1, 0)
        return x    

class GraphUnpoolingMesh(nn.Module):
    """
    ### Graph Unpooling Mesh
    Find a method to implement another unpooling method for remaining
    features. 
    """
    def __init__(self)-> None :
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        #new_features_coords = torch.mean(x[edge_index][:, :, :3], dim=1)
        #remaining_features = torch.mean(x[edge_index][:, :, 3:], dim=1)
        #new_features = torch.cat([new_features_coords, remaining_features], dim=-1)
        new_edge_index, new_features = edge_based_unpooling(edge_index, x)

        return new_features, new_edge_index

def edge_based_unpooling(edge_index, feature_matrix):
    edge_index = edge_index.T
    num_nodes = feature_matrix.shape[0]
    num_edges = edge_index.shape[0]

    # Calculate new node coordinates
    new_node_coords = torch.mean(feature_matrix[edge_index], dim=1)

    # Append new node coordinates to feature matrix
    new_feature_matrix = torch.cat([feature_matrix, new_node_coords], dim=0)
    # check for duplicates
    #new_feature_matrix = torch.unique(new_feature_matrix, dim=0)
    new_node_indices = torch.arange(num_nodes, new_feature_matrix.shape[0]).unsqueeze(-1).to(feature_matrix.device)
    new_edges_1 = torch.cat([edge_index[:, 0].unsqueeze(-1), new_node_indices], dim=1)
    new_edges_2 = torch.cat([new_node_indices, edge_index[:, 1].unsqueeze(-1)], dim=1)
    # Initialize new edge index with old edges and new edges
    new_edge_index = torch.cat([edge_index, new_edges_1, new_edges_2], dim=0)

    # Connect the three new vertices for each old triangle
    for i in range(num_nodes, new_feature_matrix.shape[0]-2, 3):
        triangle_edges = torch.tensor([[i, i + 1], [i + 1, i + 2], [i + 2, i]]).to(feature_matrix.device)
        new_edge_index = torch.cat([new_edge_index, triangle_edges], dim=0)

    return new_edge_index.T, new_feature_matrix

class GraphNetv2(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels: int,
        out_channels: int,
        d_model: int,
        n_res_blocks: int,
        attention_levels: List[int],
        channel_multipliers: List[int],
        unpooling_levels: List[int],
        n_heads: int,
        channels: int= 16,
        d_cond: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.d_model = d_model 
        self.in_proj = nn.Linear(in_channels, d_model) 

        self.spatial_transformer = SpatialTransformer(channels=d_model, n_heads=8, n_layers=4, d_cond=d_cond)
        # print paramters
        print("Number of parameters: ", sum(p.numel() for p in self.spatial_transformer.parameters() if p.requires_grad))
        self.unpooling_layer = SubdivideMeshes() #GraphUnpoolingMesh()
    
        levels = len(channel_multipliers)
        d_time_emb = None
        channels_list = [channels * m for m in channel_multipliers] # [1, 2, 4, 8]
        
        self.output_blocks = nn.ModuleList([])
        channels = d_model
        for i in reversed(range(levels)):
            layers = [
                GraphResNetBlock(
                    channels, 
                    d_time_emb,
                    out_channels=channels_list[i],
                )
            ]
            channels = channels_list[i]
            if n_res_blocks > 0:
                for _ in range(n_res_blocks-1):
                    layers.append(GraphResNetBlock(channels, d_time_emb))
            if i in attention_levels:
                layers.append(Gatv2CrossAttention(channels,channels, n_heads, d_cond, concat=False))

            if i in unpooling_levels:
                layers.append(self.unpooling_layer)
            

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_proj = nn.Linear(channels, out_channels)

    def forward(self, input_data:dict[Tensor]):#x: Tensor, edge_index: Tensor, cond=None, t_emb=None) -> Tensor:
#        x = input_data["init_features"]
#        edge_index = input_data["init_edge_index"]#.T
#        faces = input_data["init_faces"].squeeze(0)
        #mesh = input_data["init_mesh"]
        #x = mesh.verts_padded()
        x = input_data["init_features"]
        edge_index = input_data["init_edges"]
        faces = input_data["init_faces"]
        bs, nv, fdim = x.shape
        #x = x.reshape(-1, fdim)
        img = input_data["img"].unsqueeze(0)
        t_emb = None
        
        cond = self.backbone.forward_features(img)#[0]
        cond = cond.flatten(2)
        cond = cond.permute(0, 2, 1)
        x = self.in_proj(x)
        x = self.spatial_transformer(x, edge_index, cond=cond)
        for module in self.output_blocks:
            # if modeule unpool update mesh, feed mesh to module, extract vertices
            x = module(x=x, faces=faces, edge_index=edge_index, t_emb=t_emb, cond=cond)
            if isinstance(x, tuple):
                x, faces, edge_index = x
            #print(x.shape)
        return self.out_proj(x), edge_index ,faces#self.out(x, edge_index), edge_index    
            
   
    
class GraphNet(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels: int,
        out_channels: int,
        channels: int,
        n_res_blocks: int,
        attention_levels: List[int],
        attention_mode: AttentionMode,
        channel_multipliers: List[int],
        unpooling_levels: List[int],
        n_heads: int,
        d_cond: int = 1024,
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
        self.backbone = backbone
        self.channels = channels
        levels = len(channel_multipliers)
        d_time_emb = None
        
        self.pos_emb = LearnedPositionalEmbeddings(d_model=d_cond)
        

        if attention_mode == AttentionMode.GAT:
            attention_layer = Gatv2CrossAttention
        elif attention_mode == AttentionMode.TRANS:
            attention_layer = TransformerCrossAttention
        else:
            raise ValueError(f"Unknown attention mode: {attention_mode}")

        self.unpooling_layer = GraphUnpoolingMesh()
        self.in_proj = GCNConv(in_channels, channels)

        self.input_blocks = nn.ModuleList([])
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]
        for i in range(levels):
            layers = [
                GraphResNetBlock(channels, d_time_emb, out_channels=channels_list[i])
            ]
            channels = channels_list[i]

            if i in attention_levels:
                layers.append(attention_layer(channels,channels, n_heads, d_cond, concat=False))
            if n_res_blocks > 0:
                for _ in range(n_res_blocks):
                    layers.append(GraphResNetBlock(channels, d_time_emb, out_channels=channels_list[i]))

            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            GraphResNetBlock(channels, d_time_emb),
            attention_layer(channels, channels, n_heads, d_cond, concat=False),
            GraphResNetBlock(channels, d_time_emb),
        )

        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            layers = [
                GraphResNetBlock(
                    channels, #+ input_block_channels.pop(),
                    d_time_emb,
                    out_channels=channels_list[i],
                )
            ]
            channels = channels_list[i]
            if n_res_blocks > 0:
                for _ in range(n_res_blocks-1):
                    layers.append(GraphResNetBlock(channels, d_time_emb))
            if i in attention_levels:
                layers.append(attention_layer(channels,channels, n_heads, d_cond, concat=False))

            if i in unpooling_levels:
                layers.append(self.unpooling_layer)
            

            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_norm =nn.GroupNorm(16, channels)
        self.out_act = nn.SiLU()
        self.out = nn.Linear(channels, out_channels)
        #self.out = GCNConv(channels, out_channels)
        self.z_extra = SpatialTransformer(channels, 8, 8, d_cond=d_cond)

    def forward(self, x, edge_index,  cond=None,t_emb=None):
        cond = self.backbone(cond)#[0]
        cond = cond.flatten(2)
        cond = cond.permute(0, 2, 1)

        x = self.in_proj(x, edge_index)
        x = self.z_extra(x, edge_index, cond)
        
        for module in self.input_blocks:
            
            x = module(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)
            #x_input_block.append(x)
            #print(x.shape)
        x = self.middle_block(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)

        for module in self.output_blocks:
            x = module(x=x, edge_index=edge_index, t_emb=t_emb, cond=cond)
            if isinstance(x, tuple):
                x, edge_index = x
            #print(x.shape)
        return self.out(x), edge_index #self.out(x, edge_index), edge_index    
            
class Gatv2CrossAttention(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_heads: int = 1,
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
        self.n_heads = n_heads
        self.concat = concat
        self.d_cond = d_cond
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        self.lin_l = Linear(
            in_channels,
            n_heads * out_channels,
            bias=bias,
            weight_initializer="glorot",
        )
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(
                in_channels,
                n_heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )

        self.cross_attention = CrossAttention(
            in_channels, d_cond, n_heads, out_channels, return_heads=True
        )
        # self.infusion_transformer = SpatialTransformer(out_channels, heads)

        self.att = Parameter(torch.Tensor(1, n_heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(n_heads * out_channels))
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
        H, C = self.n_heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None

        assert x.dim() == 2, "x should be [N_sum, d_in]"
        x_l = self.lin_l(x).view(-1, H, C)

        if cond is not None:
            assert cond.dim() == 3, "cond should be [BS, N, d_cond]"
            x_r = self.cross_attention(x, cond)
        else:
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)

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
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"params={self.__num_parameters__}"
        )


class TransformerCrossAttention(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        n_heads: int = 1,
        d_cond: int = 1024,
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
        self.n_heads = n_heads
        self.d_cond = d_cond
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_query = Linear(in_channels, n_heads * out_channels)
        self.lin_key = Linear(in_channels, n_heads * out_channels)
        self.lin_value = Linear(in_channels, n_heads * out_channels)
        #self.cross_attention = CrossAttention(out_channels, )
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, n_heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels, n_heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * n_heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
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
        cond: Optional[Tensor] = None,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        H, C = self.n_heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        if cond is not None:
            cond = self.cross_attention(x, cond)
            cond = cond.view(*cond[:2], -1)
            key = self.lin_key(cond).view(-1, H, C)
            value = self.lin_value(cond).view(-1, H, C)
        else:
            key = self.lin_key(x).view(-1, H, C)
            value = self.lin_value(x).view(-1, H, C)

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

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

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
            edge_attr = self.lin_edge(edge_attr).view(-1, self.n_heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.n_heads, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int) -> None:
        super().__init__()
        # self.layers = nn.ModuleList([SpatTransformer(channels, channels, n_heads, d_cond) for _ in range(n_layers)])
        self.norm = torch.nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)#nn.BatchNorm1d(channels)
        self.proj_in = nn.Linear(channels, channels)#GCNConv(channels, channels)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    channels, n_heads, channels // n_heads, d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )
        self.proj_out = nn.Linear(channels, channels) #GCNConv(channels, channels)

    def forward(self, x: Tensor, edge_index: Tensor, cond: Tensor) -> Tensor:
        """ x is (batch_size, channels, seq_len)"""
        #assert x.ndim == 2, "x must be 2d"
        assert cond.ndim == 3, "cond must be (batch_size, seq_len, cond_dim)"
        if x.ndim == 3:
            x = x.permute(0,2,1)
            cond = cond.repeat(x.shape[0], 1,1)
        x_in = x.permute(0,2,1)
        x = self.norm(x).permute(0,2,1)
        x = self.proj_in(x)#, edge_index)
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = self.proj_out(x)#, edge_index)
        return x + x_in


class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int) -> None:
        super().__init__()

#        self.attn1 = CrossAttention(d_model,d_model, n_heads, d_head)
#        self.norm1 = nn.LayerNorm(d_model)
#
        self.attn2 = CrossAttention(
            d_model, d_cond, n_heads, d_head, return_heads=False
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.net = nn.Sequential(
            GeGLU(d_model, d_model * 4),
            nn.Dropout(0.0),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """For now ignore self attention"""
        #x = x + self.attn1(self.norm1(x))
        a =  self.attn2(self.norm2(x), cond)
        if x.ndim == 3:
            a = a.reshape(x.shape[0], x.shape[1], -1)

        x = x +a# + self.attn2(self.norm2(x), cond)
        x = x + self.net(self.norm3(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        d_head: int,
        return_heads: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.return_heads = return_heads

        self.scale = d_head**-0.5

        d_attn = n_heads * d_head
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(d_cond, d_attn, bias=False)
        self.v_proj = nn.Linear(d_cond, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_model, bias=False)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        if cond is not None:
            bs = cond.shape[0]
        assert x.shape[0] % bs == 0, "batch size must be a divisor of x.shape[0]"

        if x.ndim == 2:
            x = torch.stack(torch.chunk(x, bs, dim=0))
        cond = x if cond is None else cond
        q = self.q_proj(x).view(*x.shape[:2], self.n_heads, -1)
        k = self.k_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
        v = self.v_proj(cond).view(*cond.shape[:2], self.n_heads, -1)
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", attn_weights, v)

        if self.return_heads:
            return out.reshape(-1, self.n_heads, self.d_head)
        else:
            out = out.reshape(*out.shape[:2], -1)
            out = self.out_proj(out)
            return out.reshape(-1, self.d_model)


import torch
from data_tools.graph_tools.graph import Graph


def main() -> int:
    graph = Graph.create_random_graph(100, 16)
    graph.set_edge_index()
    x = torch.from_numpy(graph.feature_matrix)
    edge_index = torch.from_numpy(graph.edge_index).T
    cond = torch.randn(10, 49, 1024)
    #gat_v2 = Gatv2CrossAttention(
    #    16, 32, n_heads=8, concat=False
    #)  # concat True, returns 128
    #trans = TransformerCrossAttention(16, 32, n_heads=8, concat=False)

    graph_net = GraphNet(
        in_channels=16, 
        out_channels=3, 
        channels=16,
        n_res_blocks=2,
        attention_levels=[1, 2],
        attention_mode=AttentionMode.GAT,
        channel_multipliers=[1, 1, 2],
        n_heads=4,
        d_cond=1024,
    )
    print("graph_net params: ", sum(p.numel() for p in graph_net.parameters()))
    out = graph_net(x, edge_index, cond=cond)
    # out = gat_v2(x, edge_index, cond)
    #out2 = trans(x, edge_index, cond)
    # cross = CrossAttention(16, 1024, 4, 32)
    # cross_out = cross(x, cond)
    # spat_trans = SpatialTransformer(16, 4, 4, 1024)
    # print number of parameters
    #print("gatv2: ", sum(p.numel() for p in gat_v2.parameters()))
    #print("transformer: ", sum(p.numel() for p in trans.parameters()))

    # print( sum(p.numel() for p in spat_trans.parameters()))
    # out = spat_trans(x, edge_index, cond)
    #   out = cross(x, cond)
    # out2 = SpatTransformer(16, 32, 4, 1024)(x, cond)

    return 0


if __name__ == "__main__":
    main()
