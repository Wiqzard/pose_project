from __future__ import division

import torch
import numpy as np
import random
import subprocess
from torch_scatter import scatter_add
import pdb
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import time

import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    bound = math.sqrt(6 / ((1 + a**2) * fan))
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, "reset_parameters"):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, "children") and list(nn.children()):
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return deg_inv_sqrt, row, col


def to_heterogeneous(
    edge_index, num_nodes, n_id, edge_type, num_edge, device="cuda", args=None
):
    # edge_index = adj[0]
    # num_nodes = adj[2][0]
    edge_type_indices = []
    # pdb.set_trace()
    for k in range(edge_index.shape[1]):
        edge_tmp = edge_index[:, k]
        e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
        edge_type_indices.append(e_type)
    edge_type_indices = np.array(edge_type_indices)
    A = []
    for e_type in range(num_edge):
        edge_tmp = edge_index[:, edge_type_indices == e_type]
        #################################### j -> i ########################################
        edge_tmp = torch.flip(edge_tmp, [0])
        #################################### j -> i ########################################
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        if args.model == "FastGTN":
            edge_tmp, value_tmp = add_self_loops(
                edge_tmp, edge_weight=value_tmp, fill_value=1e-20, num_nodes=num_nodes
            )
            deg_inv_sqrt, deg_row, deg_col = _norm(
                edge_tmp.detach(), num_nodes, value_tmp.detach()
            )
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp.to(device), value_tmp.to(device)))
    edge_tmp = torch.stack(
        (torch.arange(0, n_id.shape[0]), torch.arange(0, n_id.shape[0]))
    ).type(torch.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
    A.append([edge_tmp.to(device), value_tmp.to(device)])
    return A


def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x @ x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:, :K]
    D_topk_value = D_.t()[
        torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices
    ]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = (
        torch.arange(D_.shape[0])
        .unsqueeze(-1)
        .expand(D_.shape[0], K)
        .reshape(-1)
        .to(H.device)
    )
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]
