"""
Graphormer-based NRI encoder.

Predicts edge-type logits p(z | x, A) using global self-attention (Graphormer)
rather than local NRI message passing.  This gives every node access to the
full graph context before computing per-edge predictions.

Inspired by Ying et al. "Do Transformers Really Perform Bad for Graph Representation?"
(NeurIPS 2021) and adapted for the RARL edge-inference task.
"""

from typing import Optional

import torch
from torch import nn, Tensor, LongTensor
from torch_geometric.data import Data

from .cacher import GraphDataCache
from .layers import (
    CentralityEncoding,
    SpatialEncoding,
    GraphormerAttentionHead,
    GraphormerEncoderLayer,
)
from ...graph import fully_connected_edge_index_per_batch


class GraphormerNRIEncoder(nn.Module):
    """
    Graphormer-based encoder for edge-type prediction.

    :param x_dim: Input node feature dimension.
    :param hidden_dim: Internal embedding dimension.
    :param num_layers: Total number of Graphormer layers (≥ 2).
    :param num_attention_heads: Number of attention heads per layer.
    :param num_edge_types: Number of edge types K.
    :param max_degree: Maximum node degree (for centrality encoding table size).
    :param max_path_distance: Maximum shortest-path distance (for spatial encoding).
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        num_edge_types: int,
        max_degree: int,
        max_path_distance: int,
    ):
        assert num_layers >= 2, "GraphormerNRIEncoder requires at least 2 layers."
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        self._cache = GraphDataCache()
        self.node_in_lin = nn.Linear(x_dim, hidden_dim)
        self.centrality = CentralityEncoding(max_degree, hidden_dim)
        self.spatial = SpatialEncoding(max_path_distance)

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(hidden_dim, num_attention_heads, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        # Per-edge-type attention heads for the final prediction layer
        self.spatial_per_type = nn.ModuleList([
            SpatialEncoding(max_path_distance) for _ in range(num_edge_types)
        ])
        self.edge_prob_heads = nn.ModuleList([
            GraphormerAttentionHead(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_edge_types)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        powerline_edge_index: Tensor,
        edge_set: Optional[Tensor] = None,
        batch: Optional[LongTensor] = None,
    ) -> Tensor:
        """
        Predict edge-type logits.

        :param x: Node features [B*N, x_dim].
        :param powerline_edge_index: Known graph edges [2, B*E'] — used to
            compute node degrees and path lengths for structural encodings.
        :param edge_set: Edges to predict [2, B*E]. Defaults to fully-connected.
        :param batch: Batch vector [B*N].
        :return: Edge-type logits [B*E, K].
        """
        BxN = x.shape[0]
        if batch is None:
            batch = torch.zeros(BxN, dtype=torch.long, device=x.device)
        B = int(batch.unique().numel())
        N = BxN // B
        if edge_set is None:
            edge_set = fully_connected_edge_index_per_batch(batch, x.device)

        with torch.no_grad():
            graph_data = Data(x=x, edge_index=powerline_edge_index, batch=batch)
            in_deg, out_deg, path_dists = self._cache.get(graph_data)
            node_deg = torch.max(in_deg, out_deg)

        h = self.node_in_lin(x) + self.centrality(node_deg)
        bias = self.spatial(path_dists)
        for layer in self.layers:
            h = layer(h, bias, batch)

        # Predict K attention maps → logits for each edge type
        logits = torch.stack([
            head(h, spa(path_dists), batch, return_attn_logits=True)
            for head, spa in zip(self.edge_prob_heads, self.spatial_per_type)
        ], dim=-1)  # [B, N, N, K]

        # Numerical stability: subtract max before softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values

        # Re-index from dense [B, N, N, K] to sparse [B*E, K] via edge_set
        return logits[batch[edge_set[0]], edge_set[0] % N, edge_set[1] % N]
