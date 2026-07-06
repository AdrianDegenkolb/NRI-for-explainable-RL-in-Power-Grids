"""
Top-K edge sparsification for RAGNN.

Reduces the fully-connected edge set to the K highest-scoring edges per sample
before message passing, cutting GCNConv memory and compute from O(B·N²) to O(B·K).

The selection is based on P(interaction) = sum of non-"no-edge" posterior mass.
Posterior weights of selected edges are passed unchanged to RAGNN (no re-normalisation),
so GCNConv gradients flow through the selected edges exactly as in the dense case.
Unselected edges receive no RAGNN gradient; they are still regularised by the KL loss.
"""

from typing import Tuple, Dict

import torch
from torch import Tensor


def sparse_top_k_posterior(
    posterior_flat: Tensor,
    sampled_flat: Tensor,
    edge_set: Tensor,
    K_budget: int,
    B: int,
    N: int,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Select the top-K edges per sample by P(interaction) and return sparse tensors.

    The function assumes a uniform FC graph: every sample in the batch has exactly
    E = N*(N-1) directed edges, stored contiguously in the flat tensors
    (sample 0 occupies indices [0, E), sample 1 occupies [E, 2E), ...).

    :param posterior_flat: Softmax posterior [B*E, K].
    :param sampled_flat: Gumbel-Softmax sample [B*E, K].
    :param edge_set: Batched FC edge index [2, B*E] with per-graph node offsets.
    :param K_budget: Number of edges to keep per sample.
    :param B: Batch size.
    :param N: Number of nodes per graph.
    :return:
        - ``sparse_sampled``: Gumbel-Softmax values for selected edges [B*K_budget, K].
        - ``sparse_edge_set``: Edge indices for selected edges [2, B*K_budget].
        - ``stats``: Dict of diagnostic scalars (detached).
    """
    E = N * (N - 1)
    K_budget = min(K_budget, E)
    device = posterior_flat.device
    K = posterior_flat.shape[-1]

    # Reshape flat tensors to per-sample [B, E, *]
    posterior = posterior_flat.view(B, E, K)   # [B, E, K]
    sampled   = sampled_flat.view(B, E, K)     # [B, E, K]

    # P(interaction): sum over all non-"no-edge" types (last index is "no edge")
    scores = posterior[:, :, :-1].sum(dim=-1)  # [B, E]

    # Per-sample global top-K
    topk_idx = scores.topk(K_budget, dim=1, sorted=False).indices  # [B, K_budget]

    # --- Sparse Gumbel-Softmax sample [B*K_budget, K] ---
    idx_k = topk_idx.unsqueeze(-1).expand(-1, -1, K)   # [B, K_budget, K]
    sparse_sampled = sampled.gather(1, idx_k).reshape(B * K_budget, K)

    # --- Sparse edge index [2, B*K_budget] ---
    # edge_set is [2, B*E]; reshape to [2, B, E] then index per sample
    edge_set_3d = edge_set.reshape(2, B, E)             # [2, B, E]
    idx_e = topk_idx.unsqueeze(0).expand(2, -1, -1)     # [2, B, K_budget]
    sparse_edge_set = edge_set_3d.gather(2, idx_e).reshape(2, B * K_budget)

    # --- Diagnostics (detached, no grad) ---
    with torch.no_grad():
        # Per-node out-degree in the sparse graph
        src_global = sparse_edge_set[0].view(B, K_budget)
        offsets = (torch.arange(B, device=device) * N).unsqueeze(1)  # [B, 1]
        src_local = src_global - offsets                              # [B, K_budget]
        out_deg = torch.zeros(B, N, dtype=torch.float32, device=device)
        out_deg.scatter_add_(1, src_local, torch.ones_like(src_local, dtype=torch.float32))

        # Score mass near the cutoff: edges within ε=0.01 of the K-th selected score
        kth_scores = scores.gather(1, topk_idx).min(dim=1).values    # [B]
        boundary_mass = (
            (scores - kth_scores.unsqueeze(1)).abs() < 0.01
        ).float().sum(dim=1).mean()

    stats: Dict[str, Tensor] = {
        "sparsification/mean_out_degree": out_deg.mean().detach(),
        "sparsification/max_out_degree":  out_deg.max().detach(),
        "sparsification/boundary_mass":   boundary_mass.detach(),
        "sparsification/k_budget":        torch.tensor(float(K_budget), device=device),
    }

    return sparse_sampled, sparse_edge_set, stats
