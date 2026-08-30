"""
Relation-Aware GNN (RAGNN) and standard baseline GNN.

RAGNN performs message passing conditioned on predicted edge-type posterior
probabilities from the NRI encoder. A separate conv layer is maintained per
edge type (excluding the "no edge" type). The per-type outputs are summed,
normalized, and passed through a residual block.

Two message-passing kernels are supported via ``conv_type``:

* ``"gcn"`` (default) — PyG :class:`~torch_geometric.nn.GCNConv` with
  ``improved=True`` and ``add_self_loops=True``.  Edge weights are passed as
  the ``edge_weight`` argument.
* ``"gin"`` — :class:`WeightedGINConv`: GIN-style weighted sum aggregation
  without self-loops (the residual connection handles the identity path).

For a plain GNN baseline (no edge-type conditioning) use BaselineGNN.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool

from .mlp import MLP

ConvType = Literal["gcn", "gin", "gat"]


def _mean_l2_norm(t: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean per-node L2 norm of feature vectors."""
    return torch.linalg.vector_norm(t, dim=-1).mean().clamp_min(eps)


class RAGNN(nn.Module):
    """
    Relation-Aware Graph Neural Network.

    For each message-passing layer and each edge type k < K-1, a distinct
    conv layer is applied using ``edge_type_posterior[:, k]`` as edge weights.
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
    :param sparsify_threshold: Edges whose summed probability across active types
        is below this value are dropped before message passing. 0.0 disables
        sparsification (default). Recommended value: 0.03.
    :param diagnose_every: Run the self-loop diagnostic every N eval calls.
        0 disables diagnostics entirely (default).
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
            sparsify_threshold: float = 0.0,
            diagnose_every: int = 0,
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
        self.sparsify_threshold = sparsify_threshold
        self.diagnose_every = diagnose_every
        self._eval_call_count: int = 0
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

        # --- Optional threshold-based edge sparsification ---
        if self.sparsify_threshold > 0.0:
            active_cols = list(self.edge_type_range)
            interaction_prob = edge_type_posterior[:, active_cols].sum(dim=-1)
            keep = interaction_prob >= self.sparsify_threshold
            edge_index = edge_index[:, keep]
            edge_type_posterior = edge_type_posterior[keep]

        h = self.bn_node_proj(self.node_proj(x))

        if not self.training:
            self._eval_call_count += 1

        _run_diag = not self.training and self.diagnose_every > 0 and self._eval_call_count % self.diagnose_every == 0

        for layer_ind, layer in enumerate(self.layers):
            outs = [
                layer[k](x=h, edge_index=edge_index, edge_weight=edge_type_posterior[:, k])
                for k in self.edge_type_range
            ]
            h_new = torch.stack(outs).sum(0)
            h_new = self.bn_mp[layer_ind](h_new)
            h_new = self.act(h_new)

            if _run_diag:
                # 1. how much does the signal change in this layer
                self.stats[f"msg_ratio_layer_{layer_ind}"] = _mean_l2_norm(h_new) / _mean_l2_norm(h)
                # 2. how much does the information rely on self loops
                with torch.no_grad():
                    outs_only_self_loops = [
                        layer[k](x=h, edge_index=edge_index, edge_weight=torch.zeros_like(edge_type_posterior[:, k]))
                        for k in self.edge_type_range
                    ]
                    h_new_self_loops = torch.stack(outs_only_self_loops).sum(0)
                    h_new_self_loops = self.bn_mp[layer_ind](h_new_self_loops)
                    h_new_self_loops = self.act(h_new_self_loops)
                    self.stats[f"self_loop_usage_{layer_ind}"] = _mean_l2_norm(h_new_self_loops - h) / _mean_l2_norm(
                        h_new - h)

            h = self.dropout(h)
            h = h + h_new if self.residual else h_new

        return global_mean_pool(self.final(h), batch)


class BaselineGNN(nn.Module):
    """
    Standard GNN without edge-type conditioning — baseline for comparison.

    Uses a single conv layer per message-passing step applied to all edges
    equally. Supports two convolution kernels via *conv_type*:

    * ``"gcn"`` (default) — GCNConv with ``improved=True``. Edge attributes
      are projected to scalar edge weights via a small MLP.
    * ``"gat"`` — GATConv with multi-head attention (``concat=False`` so
      output dim stays *hidden_dim*). Edge attributes are fed directly into
      the attention mechanism; no MLP projection is required.

    Input/output shapes are identical to :class:`RAGNN` except
    *edge_type_posterior* is not required.

    :param x_dim: Node feature dimensionality.
    :param hidden_dim: Hidden node dimensionality.
    :param x_out_dim: Output graph-level embedding dimensionality.
    :param num_layers: Number of message-passing layers.
    :param num_edge_types: Number of edge types (default 1). When > 1 edges
        must be annotated with integer type indices in the forward call.
    :param dropout_prob: Dropout probability.
    :param edge_dim: Dimensionality of edge attributes. Auto-detected from the
        observation space by the RLlib model wrapper; pass ``None`` to disable
        edge feature support. For GCN a projection MLP maps
        ``edge_attr [E, edge_dim] → scalar [E]``; for GAT edge attributes are
        passed directly into the attention kernel.
    :param residual: If ``True``, h = h + h_new each layer; else h = h_new.
    :param conv_type: Convolution kernel — ``"gcn"`` or ``"gat"``.
    :param num_heads: Number of attention heads (GAT only; ignored for GCN).
    """

    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            x_out_dim: int,
            num_layers: int = 3,
            num_edge_types: int = 1,
            dropout_prob: float = 0.0,
            edge_dim: Optional[int] = None,
            residual: bool = True,
            conv_type: ConvType = "gcn",
            num_heads: int = 4,
    ):
        super().__init__()
        self.residual = residual
        self.edge_dim = edge_dim
        self.conv_type = conv_type

        self.node_proj = MLP(x_dim, hidden_dim, hidden_dim, dropout_prob=dropout_prob, do_batch_norm=False)
        self.bn_node_proj = BatchNorm(hidden_dim)

        if conv_type == "gcn":
            if edge_dim is not None:
                self.edge_proj = MLP(edge_dim, hidden_dim, 1, dropout_prob=dropout_prob, do_batch_norm=False)
            self.layers = nn.ModuleList([nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True)
                for _ in range(num_edge_types)])
                for _ in range(num_layers)])
        elif conv_type == "gat":
            # GAT incorporates edge_attr natively via the attention kernel —
            # no MLP projection needed.
            self.layers = nn.ModuleList([nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                        edge_dim=edge_dim, add_self_loops=True)
                for _ in range(num_edge_types)])
                for _ in range(num_layers)])
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r}. Expected 'gcn' or 'gat'.")

        self.bn_mp = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers)])
        self.activation_function = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob)

        self.final = MLP(hidden_dim, hidden_dim, x_out_dim, dropout_prob=dropout_prob, do_batch_norm=False)
        self.stats: dict = {}

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            batch: Tensor,
            edge_types: Optional[Tensor] = None,
            edge_weights: Optional[Tensor] = None,
            edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param x: Node features [N, x_dim].
        :param edge_index: Graph connectivity [2, E].
        :param batch: Batch vector [N].
        :param edge_types: Integer edge-type index per edge [E]. Defaults to
            all-zero (single type) when not provided.
        :param edge_weights: Scalar edge weights [E]. GCN only; mutually
            exclusive with *edge_attr*.
        :param edge_attr: Edge attribute matrix [E, edge_dim]. For GCN,
            projected to scalar weights; for GAT, fed into attention directly.
        :return: Graph-level embeddings [B, x_out_dim].
        """
        assert edge_weights is None or edge_attr is None, \
            "edge_weights and edge_attr are mutually exclusive"

        if edge_types is None:
            edge_types = torch.zeros(edge_index.shape[1], device=x.device, dtype=torch.long)

        # GCN: project edge attributes to scalar weights once before the loop.
        if self.conv_type == "gcn" and edge_attr is not None and edge_weights is None:
            assert self.edge_dim is not None, "edge_dim must be set at construction to use edge_attr"
            edge_weights = self.edge_proj(edge_attr).squeeze(-1)

        h = self.bn_node_proj(self.node_proj(x))

        for i, layer in enumerate(self.layers):
            h_new = torch.zeros_like(h)
            for j, edge_type_conv in enumerate(layer):
                mask = edge_types == j
                ei_masked = edge_index[:, mask]
                if self.conv_type == "gat":
                    ea_masked = edge_attr[mask] if edge_attr is not None else None
                    h_new += edge_type_conv(x=h, edge_index=ei_masked, edge_attr=ea_masked)
                else:
                    h_new += edge_type_conv(
                        x=h,
                        edge_index=ei_masked,
                        edge_weight=edge_weights[mask] if edge_weights is not None else None,
                    )

            h_new = self.bn_mp[i](h_new)
            h_new = self.activation_function(h_new)
            self.stats[f"msg_ratio_layer_{i}"] = _mean_l2_norm(h_new) / _mean_l2_norm(h)
            h = self.dropout(h)
            h = h + h_new if self.residual else h_new

        return global_mean_pool(self.final(h), batch)
