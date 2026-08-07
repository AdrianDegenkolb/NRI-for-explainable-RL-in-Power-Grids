"""
RAFeatureExtractor: combines an NRI encoder with RAGNN.

This is the central nn.Module for the RARL pipeline:
  1. Encoder predicts edge-type logits for the fully-connected graph.
  2. Softmax gives the posterior p(z|x) that is used in the KL loss.
  3. Gumbel-Softmax samples differentiable discrete edge-type assignments.
  4. RAGNN performs conditioned message passing to produce a graph embedding.
"""

import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.utils import to_dense_batch

from .encoder import GraphormerNRIEncoder
from .ragnn import RAGNN
from .sampling import GumbelSoftmax
from .sparsification import sparse_top_k_posterior
from ..graph import fully_connected_edge_index_per_batch


class RAFeatureExtractor(nn.Module):
    """
    Relation-Aware feature extractor.

    :param x_dim: Input node feature dimension.
    :param graph_max_degree: Maximum node degree in the input graph (used for graph transformer encodings).
    :param graph_max_path_distance: Maximum shortest-path distance in the input graph (used for graph transformer encodings).
    :param hidden_dim_enc: Hidden dimension for the encoder.
    :param num_layers_enc: Number of encoder layers.
    :param num_attention_heads_enc: Number of attention heads in the encoder (except last enc layer).
    :param num_edge_types: Number of edge types K (and therefore number of attention head in the last layer)
    :param hidden_dim_gnn: Hidden dimension for the RAGNN.
    :param num_layers_gnn: Number of RAGNN message-passing layers.
    :param x_out_dim: Graph-level output embedding dimension.
    :param dropout_prob: Dropout probability.
    :param tau: Initial Gumbel-Softmax temperature.
    :param residual: Use residual connections in RAGNN.
    :param top_k_budget: Number of edges per sample passed to RAGNN (0 = disabled, use full FC graph).
    :param sparsify_threshold: Passed to RAGNN — edges whose summed interaction probability is below
        this value are dropped before message passing. 0.0 disables (default).
    :param diagnose_every: Passed to RAGNN — run self-loop diagnostics every N eval calls.
        0 disables diagnostics (default).
    """

    def __init__(
        self,
        x_dim: int,
        graph_max_degree: int,
        graph_max_path_distance: int,
        hidden_dim_enc: int,
        num_layers_enc: int,
        num_attention_heads_enc: int,
        num_edge_types: int,
        hidden_dim_gnn: int,
        num_layers_gnn: int,
        x_out_dim: int,
        dropout_prob: float = 0.0,
        tau: float = 1.0,
        residual: bool = True,
        top_k_budget: int = 0,
        conv_type: str = "gcn",
        sparsify_threshold: float = 0.0,
        diagnose_every: int = 0,
    ):
        super().__init__()

        self.encoder: nn.Module = GraphormerNRIEncoder(
            x_dim=x_dim,
            hidden_dim=hidden_dim_enc,
            num_edge_types=num_edge_types,
            max_degree=graph_max_degree,
            max_path_distance=graph_max_path_distance,
            num_layers=num_layers_enc,
            num_attention_heads=num_attention_heads_enc
        )
        self.gumbel_softmax = GumbelSoftmax(tau=tau)
        self.gnn = RAGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim_gnn,
            x_out_dim=x_out_dim,
            num_layers=num_layers_gnn,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
            residual=residual,
            skip_last=True,
            conv_type=conv_type,
            sparsify_threshold=sparsify_threshold,
            diagnose_every=diagnose_every,
        )
        self.x_out_dim = x_out_dim
        self.top_k_budget = top_k_budget
        self._sparsification_stats: dict = {}
        self._timings: dict[str, float] = {}

    def set_tau(self, tau: float) -> None:
        """Update the Gumbel-Softmax temperature (called by the annealing callback)."""
        self.gumbel_softmax.tau = tau

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        powerline_edge_index: Optional[Tensor] = None,
        edge_set: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        :param x: Node features [B*N, x_dim].
        :param batch: Batch vector [B*N]. Defaults to single graph.
        :param powerline_edge_index: Known graph edges [2, B*E'] for the
            graph-edge marker and the prior mask at loss time.
        :param edge_set: Edges to infer [2, B*E]. Must have a consistent
            number of edges per graph element. Defaults to fully-connected.
        :return:
            - ``embeddings``: Graph-level features [B, x_out_dim].
            - ``batched_posterior``: Soft edge-type posterior [B, E, K].
        """
        BxN = x.shape[0]
        if batch is None:
            batch = torch.zeros(BxN, dtype=torch.long, device=x.device)
        if edge_set is None:
            edge_set = fully_connected_edge_index_per_batch(batch, x.device)

        # --- Encoder: predict edge-type logits ---
        t0 = time.perf_counter()
        logits: Tensor = self.encoder(
            x=x, batch=batch, edge_set=edge_set, powerline_edge_index=powerline_edge_index
        )  # [B*E, K]
        self._timings["encoder_ms"] = (time.perf_counter() - t0) * 1000

        posterior: Tensor = F.softmax(logits, dim=-1)          # [B*E, K]
        sampled: Tensor = self.gumbel_softmax(logits, hard=self.training)  # [B*E, K]

        # --- Optional top-K sparsification before RAGNN ---
        if self.top_k_budget > 0:
            B = int(batch.max()) + 1
            N = BxN // B
            t0 = time.perf_counter()
            gnn_sampled, gnn_edge_set, self._sparsification_stats = sparse_top_k_posterior(
                posterior_flat=posterior,
                sampled_flat=sampled,
                edge_set=edge_set,
                K_budget=self.top_k_budget,
                B=B,
                N=N,
            )
            self._timings["sparsification_ms"] = (time.perf_counter() - t0) * 1000
        else:
            gnn_sampled, gnn_edge_set = sampled, edge_set
            self._sparsification_stats = {}
            self._timings["sparsification_ms"] = 0.0

        # --- RAGNN: conditioned message passing ---
        t0 = time.perf_counter()
        embeddings: Tensor = self.gnn(
            x=x, edge_index=gnn_edge_set, edge_type_posterior=gnn_sampled, batch=batch
        )  # [B, x_out_dim]
        self._timings["gnn_ms"] = (time.perf_counter() - t0) * 1000

        # Collect sub-module timings
        self._timings.update(self.encoder._timings)
        self._timings.update(self.gumbel_softmax._timings)

        # --- Reshape full posterior to [B, E, K] for KL loss (always uses all E edges) ---
        edge_batch = batch[edge_set[0]]
        batched_posterior, mask = to_dense_batch(posterior, edge_batch)
        assert torch.all(mask), (
            "Inconsistent number of edges across batch elements — "
            "ensure edge_set has the same edge count per graph."
        )

        return embeddings, batched_posterior
