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

from typing import Dict, Tuple, List

import torch
from gymnasium import spaces
from ray.rllib import SampleBatch
from ray.rllib.utils.typing import TensorType
from torch import Tensor

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
    obs = train_batch["obs"]
    prior_tensor, graph_edge_masks = build_prior_and_graph_masks(
        policy, obs[EDGE_INDEX], obs[EDGE_MASK]
    )
    posteriors: Tensor = model.get_posterior()

    kl_loss, kl_stats = compute_ra_kl_loss(
        posteriors=posteriors,
        prior_tensor=prior_tensor,
        graph_edge_masks=graph_edge_masks,
        beta=policy.current_beta_graph,
        beta_non_graph=policy.current_beta_non_graph,
    )
    store_ra_tower_stats(model, kl_loss, kl_stats, prior_tensor, posteriors, policy)

    if isinstance(base_loss, tuple):
        # SAC returns (actor_loss, *critic_losses, alpha_loss).
        # KL regularizes the encoder, so it is added to the actor loss only.
        return (base_loss[0] + kl_loss,) + base_loss[1:]
    return base_loss + kl_loss


def build_ra_stats_dict(towers: list) -> Dict[str, TensorType]:
    """Build the RA-specific metrics dict from tower stats."""
    stats = {
        "relation_awareness/kl_loss":               _tower_mean(towers, "ra_kl_loss"),
        "relation_awareness/kl_unweighted":         _tower_mean(towers, "ra_kl_unweighted"),
        "relation_awareness/kl_graph_edges":        _tower_mean(towers, "ra_kl_graph"),
        "relation_awareness/kl_latent_edges":       _tower_mean(towers, "ra_kl_latent"),
        "relation_awareness/fraction_graph_edges":  _tower_mean(towers, "ra_fraction_graph"),
        "relation_awareness/fraction_latent_edges": _tower_mean(towers, "ra_fraction_latent"),
        "relation_awareness/current_beta":          _tower_mean(towers, "ra_current_beta"),
        "relation_awareness/current_beta_non_graph":_tower_mean(towers, "ra_current_beta_non_graph"),
        "relation_awareness/current_tau":           _tower_mean(towers, "ra_current_tau"),
        "relation_awareness/prior_existence_probs": (
            _tower_stack_mean(towers, "ra_mean_prior")[:, 0].cpu().tolist()
        ),
        "relation_awareness/posterior_existence_probs": (
            _tower_stack_mean(towers, "ra_mean_posterior")[:, 0].cpu().tolist()
        ),
        "relation_awareness/posterior_mean": (
            _tower_cat_stat(towers, "ra_posteriors").mean(dim=0).cpu().tolist()
        ),
        "relation_awareness/posterior_var": (
            _tower_cat_stat(towers, "ra_posteriors").var(dim=0).cpu().tolist()
        ),
    }
    for key in towers[0].tower_stats["ra_gnn_stats"]:
        stats[f"relation_awareness/gnn/{key}"] = torch.mean(
            torch.stack([t.tower_stats["ra_gnn_stats"][key].detach() for t in towers])
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
    model.tower_stats.update({
        "ra_kl_loss":                  kl_loss,
        "ra_kl_graph":                 kl_stats["kl_graph_edges"],
        "ra_kl_latent":                kl_stats["kl_latent_edges"],
        "ra_kl_unweighted":            kl_stats["kl_unweighted"],
        "ra_fraction_graph":           kl_stats["fraction_graph_edges"],
        "ra_fraction_latent":          kl_stats["fraction_latent_edges"],
        "ra_mean_prior":               prior_tensor.mean(0),
        "ra_mean_posterior":           posteriors.mean(0),
        "ra_posteriors":               posteriors,
        "ra_current_beta":             torch.as_tensor(policy.current_beta_graph,     dtype=torch.float32),
        "ra_current_beta_non_graph":   torch.as_tensor(policy.current_beta_non_graph, dtype=torch.float32),
        "ra_current_tau":              torch.as_tensor(policy.current_tau,            dtype=torch.float32),
        "ra_gnn_stats":                model.ragnn.gnn.stats,
    })


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
    """Compute batched prior tensor and graph-edge affiliation masks."""
    N, _ = policy.observation_space[NODES].shape
    all_edges = fully_connected_edge_index(N)

    batched_priors: List[Tensor] = []
    batched_graph_masks: List[Tensor] = []

    for i in range(all_graph_edges.shape[0]):
        valid_edges = all_graph_edges[i][:, edge_masks_obs[i].bool()]

        g_prior, ng_prior = get_priors(
            prob_graph_edges_exist=policy.prior_prob_for_graph_edge,
            num_graph_edges=valid_edges.shape[1],
            num_non_graph_edges=all_edges.shape[1] - valid_edges.shape[1],
            temperature=policy.temperature,
        )
        prior, edge_is_powerline_edge_mask = get_prior_tensor(
            graph_edges=valid_edges,
            all_edges=all_edges,
            prior_for_graph_edges=g_prior,
            prior_for_non_graph_edges=ng_prior,
            num_edge_types=policy.num_edge_types,
            return_mask=True,
        )
        batched_priors.append(prior.to(device=policy.device, dtype=torch.float32))
        batched_graph_masks.append(edge_is_powerline_edge_mask.to(device=policy.device))

    return torch.stack(batched_priors), torch.stack(batched_graph_masks)


def _tower_mean(towers: list, key: str) -> float:
    """Average a scalar tower stat across all GPU towers."""
    return torch.mean(
        torch.stack([t.tower_stats[key].detach() for t in towers])
    ).item()


def _tower_stack_mean(towers: list, key: str, dim: int = 0) -> Tensor:
    """Stack a tensor tower stat across towers and reduce by mean."""
    return torch.mean(
        torch.stack([t.tower_stats[key].detach() for t in towers]),
        dim=dim,
    )


def _tower_cat_stat(towers: list, key: str, dim: int = 0) -> Tensor:
    """Concatenate a tensor tower stat across towers along *dim*."""
    return torch.cat([t.tower_stats[key].detach() for t in towers], dim=dim)


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
