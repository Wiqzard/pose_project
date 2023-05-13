import torch
from torch import nn
from .ga_layer import GraphAttentionV2Layer

Tensor = torch.Tensor


"""
    Graph Attention Network v2 (GATv2) implementation based on the paper:
    How Attentive are Graph Attention Networks?" by Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm
    https://arxiv.org/abs/2105.14491

    PLAN:

    https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
    BasicAttentionBlock 
        -remove the softmax
        - add residual connection
        - add layer norm
        - add conditional informatoin infusion    

"""


class GraphAttentionNetwork(nn.Module):
    """
    A Graph Attention Network v2 (GATv2) implementation.
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        n_classes: int,
        n_heads: int,
        dropout: float,
        share_weights: bool = True,
    ):
        super().__init__()
        """
        Initializes a GraphAttentionNetwork instance.

        Args:
            in_features (int): The number of features per node.
            n_hidden (int): The number of features in the first graph attention layer.
            n_classes (int): The number of classes.
            n_heads (int): The number of heads in the graph attention layers.
            dropout (float): The dropout probability.
            share_weights (bool, optional): If set to True, the same matrix will be applied to the source and the 
                target node of every edge. Defaults to True.
        """
        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(
            in_features,
            n_hidden,
            n_heads,
            is_concat=True,
            dropout=dropout,
            share_weights=share_weights,
        )
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionV2Layer(
            n_hidden,
            n_classes,
            1,
            is_concat=False,
            dropout=dropout,
            share_weights=share_weights,
        )
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        The forward function for the GraphAttentionNetwork.

        Args:
            x (torch.Tensor): The feature vectors with shape `[n_nodes, in_features]`.
            adj_mat (torch.Tensor): The adjacency matrix of the form `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`.

        Returns:
            torch.Tensor: The output tensor after applying the Graph Attention Network v2.
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)
