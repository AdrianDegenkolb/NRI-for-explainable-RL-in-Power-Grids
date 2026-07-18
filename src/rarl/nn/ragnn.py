"""
Relation-Aware GNN (RAGNN) and standard baseline GNN.

RAGNN performs message passing conditioned on predicted edge-type posterior
probabilities from the NRI encoder. A separate GCNConv layer is maintained
per edge type (excluding the "no edge" type). The per-type outputs are
summed, normalized, and passed through a residual block.

For a plain GNN baseline (no edge-type conditioning) use BaselineGNN.
"""

import torch
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool

from .mlp import MLP


def _mean_l2_norm(t: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean per-node L2 norm of feature vectors."""
    return torch.linalg.vector_norm(t, dim=-1).mean().clamp_min(eps)


class RAGNN(nn.Module):
    """
    Relation-Aware Graph Neural Network.

    For each message-passing layer and each edge type k < K-1, a distinct
    GCNConv is applied using ``edge_type_posterior[:, k]`` as edge weights.
    The K-1 outputs are summed and fed through batch norm + ELU + dropout.
    The last edge type (index K-1) is interpreted as "no edge" and is
    skipped (controlled by *skip_last*).

    Input shapes
    ------------
    x                   : [N, x_dim]
    edge_index          : [2, E]
    edge_type_posterior : [E, K]
    batch               : [N]

    Output shape
    ------------
    [B, x_out_dim]  — graph-level embedding after global mean pooling.

    :param x_dim: Input node feature dimension.
    :param hidden_dim: Hidden dimension.
    :param x_out_dim: Output graph-level embedding dimension.
    :param num_layers: Number of message-passing layers.
    :param num_edge_types: Number of edge types K.
    :param skip_last: Skip the last edge-type (the "no edge" type).
    :param dropout_prob: Dropout probability.
    :param residual: Use residual connections between layers.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        x_out_dim: int,
        num_layers: int = 3,
        num_edge_types: int = 2,
        skip_last: bool = True,
        dropout_prob: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.edge_type_range = (
            range(num_edge_types - 1) if skip_last else range(num_edge_types)
        )

        self.node_proj = MLP(
            x_dim, hidden_dim, hidden_dim, dropout_prob=dropout_prob, do_batch_norm=False
        )
        self.bn_node_proj = BatchNorm(hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
                for _ in self.edge_type_range
            ])
            for _ in range(num_layers)
        ])
        self.bn_mp = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(
            hidden_dim, hidden_dim, x_out_dim, dropout_prob=dropout_prob, do_batch_norm=False
        )
        self.stats: dict = {}

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type_posterior: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        :param x: Node features [N, x_dim].
        :param edge_index: Graph connectivity [2, E].
        :param edge_type_posterior: Soft or hard edge-type assignments [E, K].
        :param batch: Batch vector [N].
        :return: Graph-level embeddings [B, x_out_dim].
        """
        assert edge_type_posterior.size(0) == edge_index.size(1)

        h = self.bn_node_proj(self.node_proj(x))

        for l, mp_list in enumerate(self.layers):
            outs = [
                mp_list[k](x=h, edge_index=edge_index, edge_weight=edge_type_posterior[:, k])
                for k in self.edge_type_range
            ]
            h_new = torch.stack(outs).sum(0)
            h_new = self.bn_mp[l](h_new)
            h_new = self.act(h_new)
            self.stats[f"msg_ratio_layer_{l}"] = (
                _mean_l2_norm(h_new) / _mean_l2_norm(h)
            )
            h = self.dropout(h)
            h = h + h_new if self.residual else h_new

        return global_mean_pool(self.final(h), batch)


class BaselineGNN(nn.Module):
    """
    Standard GNN without edge-type conditioning — baseline for comparison.

    Uses a single GCNConv per layer applied to all edges equally.

    Input/output shapes are identical to :class:`RAGNN` except
    *edge_type_posterior* is not required.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        x_out_dim: int,
        num_layers: int = 3,
        dropout_prob: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        self.node_proj = MLP(
            x_dim, hidden_dim, hidden_dim, dropout_prob=dropout_prob, do_batch_norm=False
        )
        self.bn_node_proj = BatchNorm(hidden_dim)

        self.layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
            for _ in range(num_layers)
        ])
        self.bn_mp = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(
            hidden_dim, hidden_dim, x_out_dim, dropout_prob=dropout_prob, do_batch_norm=False
        )
        self.stats: dict = {}

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        :param x: Node features [N, x_dim].
        :param edge_index: Graph connectivity [2, E].
        :param batch: Batch vector [N].
        :return: Graph-level embeddings [B, x_out_dim].
        """
        h = self.bn_node_proj(self.node_proj(x))

        for l, conv in enumerate(self.layers):
            h_new = conv(h, edge_index)
            h_new = self.bn_mp[l](h_new)
            h_new = self.act(h_new)
            self.stats[f"msg_ratio_layer_{l}"] = _mean_l2_norm(h_new) / _mean_l2_norm(h)
            h = self.dropout(h)
            h = h + h_new if self.residual else h_new

        return global_mean_pool(self.final(h), batch)
