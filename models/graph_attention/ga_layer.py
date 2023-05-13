import torch
from torch import nn

# from labml_helpers.module import Module


class GraphAttentionV2Layer(nn.Module):
    """
    A Graph Attention V2 (GATv2) layer, which is a single layer within a GATv2 network.
    A GATv2 network consists of multiple instances of this layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False,
    ) -> None:
        """
        Initializes a GraphAttentionV2Layer instance.

        Args:
            in_features (int): The number of input features per node (F).
            out_features (int): The number of output features per node (F').
            n_heads (int): The number of attention heads (K).
            is_concat (bool, optional): Whether to concatenate or average the multi-head results. Defaults to True.
            dropout (float, optional): The dropout probability. Defaults to 0.6.
            leaky_relu_negative_slope (float, optional): The negative slope for the leaky ReLU activation function. Defaults to 0.2.
            share_weights (bool, optional): If True, the same matrix will be applied to the source and target nodes of every edge. Defaults to False.
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        The forward function for the GraphAttentionV2Layer.

        Args:
            h (torch.Tensor): The input node embeddings with shape `[n_nodes, in_features]`.
            adj_mat (torch.Tensor): The adjacency matrix with shape `[n_nodes, n_nodes, n_heads]`.
                Typically, the shape is `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
                The adjacency matrix represents the edges (or connections) among nodes. `adj_mat[i][j]` is `True`
                if there is an edge from node `i` to node `j`.

        Returns:
            torch.Tensor: The output tensor after applying the Graph Attention V2 layer. If `is_concat` is True,
                the shape will be `[n_nodes, n_heads * n_hidden]`. Otherwise, the shape will be `[n_nodes, n_hidden]`.
        """

        n_nodes = h.shape[0]
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] in [1, n_nodes]
        assert adj_mat.shape[1] in [1, n_nodes]
        assert adj_mat.shape[2] in [1, self.n_heads]
        e = e.masked_fill(adj_mat == 0, float("-inf"))

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("ijh,jhf->ihf", a, g_r)
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
