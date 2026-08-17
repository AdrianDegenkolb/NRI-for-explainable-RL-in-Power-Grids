"""Shared helpers for Relation-Aware (RA) RLlib policies.

This module provides the building blocks that both RAPPOTorchPolicy and
RASACTorchPolicy rely on:

- **init_ra_config**: reads the ``relation_awareness`` config block and stores
  annealing attributes (beta, tau) on the policy instance.
- **apply_ra_kl_loss**: computes the KL regularization between the encoder's
  posterior and the prior, adds it to a base algorithm loss, and records
  tower stats for TensorBoard logging.
- **build_prior_and_graph_masks**: constructs the batched prior tensor and
  graph-edge affiliation masks needed by the KL computation.
- **build_ra_stats_dict / store_ra_tower_stats**: aggregate per-tower RA
  metrics into the format expected by RLlib's stats pipeline.
"""

import logging
import time
from typing import Dict, Tuple

import torch
from gymnasium import spaces
from ray.rllib import SampleBatch
from ray.rllib.utils.typing import TensorType
from torch import Tensor

logger = logging.getLogger(__name__)

# Profiling state — tracks cumulative time across gradient steps.
_prof_calls: int = 0
_prof_total_prior: float = 0.0
_prof_total_kl: float = 0.0
_PROF_LOG_EVERY: int = 1000  # log once every N gradient steps


from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES
from rarl import compute_ra_kl_loss, fully_connected_edge_index, get_prior_tensor
from rarl.prior import get_priors
from rarl_rllib.model import RARLModel


def init_ra_config(policy, config: dict) -> None:
    """Store RA annealing attributes on *policy* from its config dict."""
    ra_cfg = config["relation_awareness"]
    policy._loss_cfg = ra_cfg["loss"]
    policy._sampling_cfg = ra_cfg["sampling"]
    policy._prior_cfg = ra_cfg["prior"]
    policy._latent_cfg = ra_cfg["latent_space"]

    policy.current_beta_graph = _get_from_conf_with_fallback(
        policy._loss_cfg, "beta_graph_edges", "beta_graph_edges_end")
    policy.current_beta_non_graph = _get_from_conf_with_fallback(
        policy._loss_cfg, "beta_non_graph_edges", "beta_non_graph_edges_end")
    policy.current_tau = _get_from_conf_with_fallback(
        policy._sampling_cfg, "tau", "tau_end")
    policy.num_edge_types = policy._latent_cfg["num_edge_types"]
    policy.prior_prob_for_graph_edge = policy._prior_cfg["prior_prob_for_graph_edge"]
    policy.temperature = policy._prior_cfg["temperature"]



def apply_ra_kl_loss(
    policy,
    model: RARLModel,
    train_batch: SampleBatch,
    base_loss: TensorType,
) -> TensorType:
    """Add KL regularization term to *base_loss* and store RA tower stats."""
    global _prof_calls, _prof_total_prior, _prof_total_kl

    obs = train_batch["obs"]

    t0 = time.perf_counter()
    prior_tensor, graph_edge_masks = build_prior_and_graph_masks(
        policy, obs[EDGE_INDEX], obs[EDGE_MASK]
    )
    t1 = time.perf_counter()

    posteriors: Tensor = model.get_posterior()
    device = posteriors.device
    prior_tensor = prior_tensor.to(device)
    graph_edge_masks = graph_edge_masks.to(device)

    kl_loss, kl_stats = compute_ra_kl_loss(
        posteriors=posteriors,
        prior_tensor=prior_tensor,
        graph_edge_masks=graph_edge_masks,
        beta=policy.current_beta_graph,
        beta_non_graph=policy.current_beta_non_graph,
    )
    t2 = time.perf_counter()

    _prof_calls += 1
    _prof_total_prior += t1 - t0
    _prof_total_kl += t2 - t1

    if _prof_calls % _PROF_LOG_EVERY == 0:
        batch_size = obs[EDGE_INDEX].shape[0]
        logger.warning(
            "[RA profiling] grad-step=%d  batch=%d  "
            "build_prior=%.3fs (avg %.3fs)  kl_loss=%.3fs (avg %.3fs)",
            _prof_calls, batch_size,
            t1 - t0, _prof_total_prior / _prof_calls,
            t2 - t1, _prof_total_kl / _prof_calls,
        )

    store_ra_tower_stats(model, kl_loss, kl_stats, prior_tensor, posteriors, policy)

    if isinstance(base_loss, tuple):
        # SAC returns (actor_loss, *critic_losses, alpha_loss).
        # KL regularizes the encoder, so it is added to the actor loss only.
        return (base_loss[0] + kl_loss,) + base_loss[1:]
    return base_loss + kl_loss


def build_ra_stats_dict(towers: list) -> Dict[str, TensorType]:
    """Build the RA-specific metrics dict from tower stats."""
    prior = _tower_stack_mean(towers, "ra_mean_prior")
    posterior = _tower_stack_mean(towers, "ra_mean_posterior")
    posteriors = _tower_cat_stat(towers, "ra_posteriors")

    candidates = {
        "relation_awareness/kl_loss":               _tower_mean(towers, "ra_kl_loss"),
        "relation_awareness/kl_unweighted":         _tower_mean(towers, "ra_kl_unweighted"),
        "relation_awareness/kl_graph_edges":        _tower_mean(towers, "ra_kl_graph"),
        "relation_awareness/kl_latent_edges":       _tower_mean(towers, "ra_kl_latent"),
        "relation_awareness/fraction_graph_edges":  _tower_mean(towers, "ra_fraction_graph"),
        "relation_awareness/fraction_latent_edges": _tower_mean(towers, "ra_fraction_latent"),
        "relation_awareness/total_interaction_probability_mass": _tower_mean(towers, "ra_total_interaction_prob_mass"),
        "relation_awareness/current_beta":          _tower_mean(towers, "ra_current_beta"),
        "relation_awareness/current_beta_non_graph":_tower_mean(towers, "ra_current_beta_non_graph"),
        "relation_awareness/current_tau":           _tower_mean(towers, "ra_current_tau"),
        "relation_awareness/prior_existence_probs": (
            prior[:, :-1].sum(dim=-1).cpu().tolist() if prior is not None else None
        ),
        "relation_awareness/posterior_existence_probs": (
            posterior[:, :-1].sum(dim=-1).cpu().tolist() if posterior is not None else None
        ),
        "relation_awareness/posterior_mean": (
            posteriors.mean(dim=0).cpu().tolist() if posteriors is not None else None
        ),
        "relation_awareness/posterior_var": (
            posteriors.var(dim=0).cpu().tolist() if posteriors is not None else None
        ),
    }
    stats = {k: v for k, v in candidates.items() if v is not None}

    gnn_stats = towers[0].tower_stats.get("ra_gnn_stats")
    if gnn_stats is not None and all("ra_gnn_stats" in t.tower_stats for t in towers):
        for key in gnn_stats:
            if all(key in t.tower_stats["ra_gnn_stats"] for t in towers):
                stats[f"relation_awareness/gnn/{key}"] = torch.mean(
                    torch.stack([t.tower_stats["ra_gnn_stats"][key].detach().cpu() for t in towers])
                ).item()

    sparsif_stats = towers[0].tower_stats.get("ra_sparsification_stats")
    if sparsif_stats is not None and all(
        "ra_sparsification_stats" in t.tower_stats for t in towers
    ):
        for key in sparsif_stats:
            stats[f"relation_awareness/{key}"] = torch.mean(
                torch.stack([
                    t.tower_stats["ra_sparsification_stats"][key].detach().cpu()
                    for t in towers
                ])
            ).item()

    return stats


def store_ra_tower_stats(
    model: RARLModel,
    kl_loss: Tensor,
    kl_stats: dict,
    prior_tensor: Tensor,
    posteriors: Tensor,
    policy,
) -> None:
    """Write RARL metrics into model.tower_stats for later aggregation."""
    global _sparsif_log_calls

    model.tower_stats.update({
        "ra_kl_loss":                  kl_loss,
        "ra_kl_graph":                 kl_stats["kl_graph_edges"],
        "ra_kl_latent":                kl_stats["kl_latent_edges"],
        "ra_kl_unweighted":            kl_stats["kl_unweighted"],
        "ra_fraction_graph":           kl_stats["fraction_graph_edges"],
        "ra_fraction_latent":          kl_stats["fraction_latent_edges"],
        "ra_total_interaction_prob_mass": kl_stats["total_interaction_probability_mass"],
        "ra_mean_prior":               prior_tensor.mean(0),
        "ra_mean_posterior":           posteriors.mean(0),
        "ra_posteriors":               posteriors,
        "ra_current_beta":             torch.as_tensor(policy.current_beta_graph,     dtype=torch.float32),
        "ra_current_beta_non_graph":   torch.as_tensor(policy.current_beta_non_graph, dtype=torch.float32),
        "ra_current_tau":              torch.as_tensor(policy.current_tau,            dtype=torch.float32),
        "ra_gnn_stats":                model.ragnn.gnn.stats,
    })

    # Sparsification diagnostics — written every iteration (stats already computed in forward pass)
    sparsif_stats = getattr(model.ragnn, "_sparsification_stats", {})
    if sparsif_stats:
        model.tower_stats["ra_sparsification_stats"] = sparsif_stats


def _get_from_conf_with_fallback(config: dict, key: str, fallback_key: str) -> float:
    """Return *key* from config, falling back to *fallback_key*.

    :raises AssertionError: If neither key is present.
    """
    value = config.get(key, config.get(fallback_key))
    assert value is not None, (
        f"Config must contain '{key}' or '{fallback_key}', got: {list(config.keys())}"
    )
    return value


def build_prior_and_graph_masks(
    policy,
    all_graph_edges: Tensor,
    edge_masks_obs: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return batched prior tensor and graph-edge affiliation masks.

    The prior is fully determined by fixed config values (``prior_prob_for_graph_edge``,
    ``temperature``) and the static grid topology, so it is computed once on the first
    call and cached on the policy.  Line disconnections change which edges are active
    but their effect on the prior is negligible and intentionally ignored.
    """
    if not hasattr(policy, "_cached_prior"):
        N, _ = policy.observation_space[NODES].shape
        all_edges = fully_connected_edge_index(N)
        # Use the full powerline topology from the first batch item (no mask applied).
        graph_edges = all_graph_edges[0]
        g_prior, ng_prior = get_priors(
            prob_graph_edges_exist=policy.prior_prob_for_graph_edge,
            num_graph_edges=graph_edges.shape[1],
            num_non_graph_edges=all_edges.shape[1] - graph_edges.shape[1],
            temperature=policy.temperature,
        )
        prior, mask = get_prior_tensor(
            graph_edges=graph_edges,
            all_edges=all_edges,
            prior_for_graph_edges=g_prior,
            prior_for_non_graph_edges=ng_prior,
            num_edge_types=policy.num_edge_types,
            return_mask=True,
        )
        policy._cached_prior = prior.to(device=policy.device, dtype=torch.float32)
        policy._cached_graph_mask = mask.to(device=policy.device)

    B = all_graph_edges.shape[0]
    return (
        policy._cached_prior.unsqueeze(0).expand(B, -1, -1),
        policy._cached_graph_mask.unsqueeze(0).expand(B, -1),
    )


def _tower_mean(towers: list, key: str) -> float:
    """Average a scalar tower stat across all GPU towers, or None if key is missing."""
    if not all(key in t.tower_stats for t in towers):
        return None
    return torch.mean(
        torch.stack([t.tower_stats[key].detach().cpu() for t in towers])
    ).item()


def _tower_stack_mean(towers: list, key: str, dim: int = 0) -> Tensor:
    """Stack a tensor tower stat across towers and reduce by mean, or None if key is missing."""
    if not all(key in t.tower_stats for t in towers):
        return None
    return torch.mean(
        torch.stack([t.tower_stats[key].detach().cpu() for t in towers]),
        dim=dim,
    )


def _tower_cat_stat(towers: list, key: str, dim: int = 0) -> Tensor:
    """Concatenate a tensor tower stat across towers along *dim*, or None if key is missing."""
    if not all(key in t.tower_stats for t in towers):
        return None
    return torch.cat([t.tower_stats[key].detach().cpu() for t in towers], dim=dim)


def assert_graph_obs_space_and_get_x_dim(obs_space: spaces.Dict) -> int:
    """
    Checks that the given dict space is a graph obs space and returns the node feature dimension.
    :param obs_space: the observation space to check,
    :return: the node feature dimension (x_dim) if the checks pass
    :raise AssertionError: if the obs_space does not contain node features, edge index and edge mask subspaces
    """

    assert NODES in obs_space.spaces, f"obs_space must contain '{NODES}' key for node features"
    assert EDGE_INDEX in obs_space.spaces, f"obs_space must contain '{EDGE_INDEX}' key for edge indices"
    assert EDGE_MASK in obs_space.spaces, f"obs_space must contain '{EDGE_MASK}' key for edge masks"

    _, x_dim = obs_space[NODES].shape
    return x_dim
