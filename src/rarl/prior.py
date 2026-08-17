"""
Prior distribution construction for the RARL KL regularization term.

The key idea: the prior separates edges that are part of the observed graph
(e.g. power-grid transmission lines) from *latent* edges (potential hidden
dependencies). Graph edges receive a high prior existence probability; the
prior for latent edges is derived so that the *average* existence probability
across all edges equals ``(1 + temperature) * num_graph_edges / num_total_edges``.

This construction is limited to K=2 edge types ("exists" / "doesn't exist").
Extension to K>2 types would require a different prior parametrization.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def get_priors(
    prob_graph_edges_exist: float,
    num_graph_edges: int,
    num_non_graph_edges: int,
    temperature: float = 0.2,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the two-class prior for graph edges and non-graph (latent) edges.

    The *temperature* controls how many latent edges the prior expects:
    ``(1 + temperature) * num_graph_edges`` edges in total on average.
    Temperature must satisfy ``0 <= temperature < num_non_graph_edges / num_graph_edges``.

    :param prob_graph_edges_exist: Prior probability that a graph edge is active (e.g. 0.9).
    :param num_graph_edges: Number of known graph edges.
    :param num_non_graph_edges: Number of possible latent (non-graph) edges.
    :param temperature: Controls expected number of discovered latent edges.
    :return: ``(prior_graph_edges, prior_non_graph_edges)``, each of shape [2].
    """
    assert num_non_graph_edges > 0
    assert num_graph_edges >= 0
    assert 0 <= temperature < num_non_graph_edges / (num_graph_edges + 1)
    assert 0.0 <= prob_graph_edges_exist <= 1.0

    num_total = num_graph_edges + num_non_graph_edges
    p1 = np.array([prob_graph_edges_exist, 1.0 - prob_graph_edges_exist], dtype=np.float64)
    avg_exist = (1.0 + temperature) * num_graph_edges / num_total
    p_hat = np.array([avg_exist, 1.0 - avg_exist], dtype=np.float64)
    p2 = (num_total * p_hat - num_graph_edges * p1) / num_non_graph_edges

    eps = 1e-6
    p1 = np.clip(p1, 0.0, 1.0).astype(np.float32)
    p1 /= p1.sum()
    p2 = np.clip(p2, 0.0, 1.0).astype(np.float32)
    p2 /= p2.sum()

    assert np.all(p1 >= -eps) and np.isclose(p1.sum(), 1.0, atol=1e-5), \
        f"Prior for graph edges is invalid: {p1}"
    assert np.all(p2 >= -eps) and np.isclose(p2.sum(), 1.0, atol=1e-5), \
        f"Prior for non-graph edges is invalid: {p2}"

    return Tensor(p1), Tensor(p2)


def create_graph_edge_mask(graph_edges: Tensor, all_edges: Tensor) -> Tensor:
    """
    Boolean mask identifying which edges in *all_edges* are graph edges.

    Both directions of each graph edge are matched (undirected semantics).

    :param graph_edges: Known graph edge index [2, E_graph].
    :param all_edges: Full edge index to mask [2, E_all].
    :return: Bool mask [E_all].
    """
    device = all_edges.device
    all_edges_T = all_edges.T  # [E_all, 2]
    graph_edges = graph_edges.to(device)
    mask = torch.zeros(all_edges_T.shape[0], dtype=torch.bool, device=device)
    reversed_edges = graph_edges[[1, 0], :]

    for e in range(graph_edges.shape[1]):
        mask |= torch.all(all_edges_T == graph_edges[:, e], dim=1)
        mask |= torch.all(all_edges_T == reversed_edges[:, e], dim=1)

    return mask


def get_prior_tensor(
    graph_edges: Tensor,
    all_edges: Tensor,
    prior_for_graph_edges: Tensor,
    prior_for_non_graph_edges: Tensor,
    num_edge_types: int = 2,
    return_mask: bool = False,
) -> Tensor:
    """
    Build a per-edge prior tensor of shape [E_all, K].

    Graph edges receive *prior_for_graph_edges*; the rest receive
    *prior_for_non_graph_edges*. When K > 2, the "exists" probability
    (index 0) is split equally across the first K-1 types, while the last
    type represents "no edge".

    :param graph_edges: Known graph edge index [2, E_graph].
    :param all_edges: Full edge index [2, E_all].
    :param prior_for_graph_edges: 2-element prior for graph edges.
    :param prior_for_non_graph_edges: 2-element prior for non-graph edges.
    :param num_edge_types: Number of edge types K.
    :param return_mask: If True, also return the graph-edge boolean mask.
    :return: Prior tensor [E_all, K], optionally ``(prior, mask)``.
    """
    assert prior_for_graph_edges.shape == prior_for_non_graph_edges.shape

    device = all_edges.device
    mask = create_graph_edge_mask(graph_edges, all_edges)
    E = mask.shape[0]
    prior = torch.zeros((E, num_edge_types), dtype=torch.float32, device=device)
    prior_for_graph_edges = prior_for_graph_edges.to(device)
    prior_for_non_graph_edges = prior_for_non_graph_edges.to(device)

    # Spread "exists" probability evenly over the first (K-1) types
    prior[mask, : num_edge_types - 1] = prior_for_graph_edges[0] / (num_edge_types - 1)
    prior[mask, -1] = prior_for_graph_edges[1]
    prior[~mask, : num_edge_types - 1] = prior_for_non_graph_edges[0] / (num_edge_types - 1)
    prior[~mask, -1] = prior_for_non_graph_edges[1]

    return (prior, mask) if return_mask else prior
