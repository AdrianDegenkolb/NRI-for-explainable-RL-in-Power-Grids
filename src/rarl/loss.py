"""
KL-divergence regularization loss for RARL.

This module is framework-agnostic: it only depends on PyTorch.
The computed loss is added on top of any base RL algorithm's loss.
"""

from typing import Tuple, Dict

import torch
from torch import Tensor


def compute_ra_kl_loss(
    posteriors: Tensor,
    prior_tensor: Tensor,
    graph_edge_masks: Tensor,
    beta: float,
    beta_non_graph: float,
    eps: float = 1e-7,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute the relation-aware KL regularization loss.

    The loss pushes the predicted edge-type posterior toward a structured
    prior that treats graph edges (known topology) differently from latent
    edges (potential hidden dependencies).

    ::

        kl_per_edge = sum_k posterior[k] * log(posterior[k] / prior[k])
        loss = beta * mean(kl_graph) + beta_non_graph * mean(kl_latent)

    Each group is weighted equally regardless of edge count. With the original
    count-weighted formulation ``f_graph * beta * kl_graph + f_latent * beta_ng * kl_latent``
    (which equals ``beta * mean_KL_all_edges`` when betas are equal), latent edges
    dominate the gradient by a factor of ~79× for case14 (40 graph vs 3152 latent edges).

    :param posteriors: Predicted edge-type distributions [B, E, K].
    :param prior_tensor: Prior distributions for each edge [B, E, K] or [E, K].
    :param graph_edge_masks: Boolean mask, True for graph edges [B, E].
    :param beta: KL weight for graph edges.
    :param beta_non_graph: KL weight for latent (non-graph) edges.
    :param eps: Small constant for numerical stability inside log.
    :return:
        - ``kl_loss`` : Scalar weighted KL loss ready to add to RL loss.
        - ``stats``   : Dict with per-component diagnostics for logging.
    """
    if prior_tensor.dim() == 2:
        prior_tensor = prior_tensor.unsqueeze(0).expand_as(posteriors)

    # KL divergence per edge: [B, E]
    kl_per_edge = (
        posteriors * (torch.log(posteriors + eps) - torch.log(prior_tensor + eps))
    ).sum(dim=-1)

    # Split graph / latent edges
    has_graph = graph_edge_masks.any()
    has_latent = (~graph_edge_masks).any()

    kl_graph = (
        kl_per_edge[graph_edge_masks].mean()
        if has_graph
        else torch.zeros((), device=posteriors.device)
    )
    kl_latent = (
        kl_per_edge[~graph_edge_masks].mean()
        if has_latent
        else torch.zeros((), device=posteriors.device)
    )

    num_graph = graph_edge_masks.sum().float()
    num_latent = (~graph_edge_masks).sum().float()
    total = num_graph + num_latent
    f_graph = num_graph / total if total > 0 else torch.zeros((), device=posteriors.device)
    f_latent = num_latent / total if total > 0 else torch.zeros((), device=posteriors.device)
    total_interaction_probability_mass = posteriors[:,:,:-1].sum(dim=[1, 2]).mean()

    kl_loss = beta * kl_graph + beta_non_graph * kl_latent

    stats: Dict[str, Tensor] = {
        "kl_loss": kl_loss.detach(),
        "kl_graph_edges": kl_graph.detach(),
        "kl_latent_edges": kl_latent.detach(),
        "kl_unweighted": (f_graph * kl_graph + f_latent * kl_latent).detach(),
        "fraction_graph_edges": f_graph.detach(),
        "fraction_latent_edges": f_latent.detach(),
        "total_interaction_probability_mass": total_interaction_probability_mass.detach()
    }

    return kl_loss, stats
