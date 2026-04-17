"""
RAPPOTorchPolicy and RASACTorchPolicy — concrete RA-augmented RLlib policies.

Both policies augment their respective base algorithm with:
  - KL regularization on the discrete latent edge distribution (from the RAGNN encoder)
  - Annealing-ready attributes (current_beta_graph, current_beta_non_graph, current_tau)
  - RA-specific TensorBoard metrics

RAPPOTorchPolicy extends PPOTorchPolicy (TorchPolicyV2, on-policy).
RASACTorchPolicy is built via build_policy_class (TorchPolicy V1 API, off-policy).

Config schema (under ``relation_awareness``)::

    relation_awareness:
      sampling:
        tau: 0.2
      prior:
        prior_prob_for_graph_edge: 0.9
        temperature: 0.2
      loss:
        beta_graph_edges: 5.0
        beta_non_graph_edges: 5.0
      latent_space:
        num_edge_types: 2
"""

from typing import Dict, List, Tuple

import ray.rllib.algorithms.sac.sac
import torch
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.sac.sac_torch_policy import (
    ComputeTDErrorMixin,
    TargetNetworkMixin,
    _get_dist_class,
    action_distribution_fn,
    actor_critic_loss,
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    optimizer_fn,
    postprocess_trajectory,
    setup_late_mixins,
    stats as _sac_stats,
    validate_spaces,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from torch import Tensor

from src.grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES
from src.rarl.graph import fully_connected_edge_index
from src.rarl.loss import compute_ra_kl_loss
from src.rarl.prior import get_prior_tensor, get_priors
from src.rarl_rllib.model import RASACTorchModel


# ---------------------------------------------------------------------------
# Shared RA helpers
# ---------------------------------------------------------------------------

def _init_ra_config(policy, config: dict) -> None:
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


def _build_prior_and_graph_masks(
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
        prior, gmask = get_prior_tensor(
            graph_edges=valid_edges,
            all_edges=all_edges,
            prior_for_graph_edges=g_prior,
            prior_for_non_graph_edges=ng_prior,
            num_edge_types=policy.num_edge_types,
            return_mask=True,
        )
        batched_priors.append(prior.to(device=policy.device, dtype=torch.float32))
        batched_graph_masks.append(gmask.to(device=policy.device))

    return torch.stack(batched_priors), torch.stack(batched_graph_masks)


def _store_ra_tower_stats(
    model: ModelV2,
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


def _apply_ra_kl_loss(
    policy,
    model: ModelV2,
    train_batch: SampleBatch,
    base_loss: TensorType,
) -> TensorType:
    """Add KL regularization term to *base_loss* and store RA tower stats."""
    obs = train_batch["obs"]
    prior_tensor, graph_edge_masks = _build_prior_and_graph_masks(
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
    _store_ra_tower_stats(model, kl_loss, kl_stats, prior_tensor, posteriors, policy)
    return base_loss + kl_loss


def _build_ra_stats_dict(towers: list) -> Dict[str, TensorType]:
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


# ---------------------------------------------------------------------------
# RAPPOTorchPolicy
# ---------------------------------------------------------------------------

class RAPPOTorchPolicy(PPOTorchPolicy):
    """PPO policy augmented with RAGNN KL regularization and RA metrics."""

    def __init__(self, observation_space, action_space, config):
        _init_ra_config(self, config)
        super().__init__(observation_space, action_space, config)

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class,
        train_batch: SampleBatch,
    ) -> TensorType:
        base_loss = super().loss(model, dist_class, train_batch)
        return _apply_ra_kl_loss(self, model, train_batch, base_loss)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().stats_fn(train_batch)
        stats.update(_build_ra_stats_dict(self.model_gpu_towers))
        return stats


# ---------------------------------------------------------------------------
# RASACTorchPolicy helpers
# ---------------------------------------------------------------------------

def _build_ra_sac_model_and_action_dist(policy, obs_space, action_space, config):
    """Model builder for RASACTorchPolicy — constructs RASACTorchModel directly."""
    policy_model_config = {**config["model"], **config["policy_model_config"]}
    q_model_config = {**config["model"], **config["q_model_config"]}
    custom_cfg = config["model"].get("custom_model_config", {})

    def _make(name: str) -> RASACTorchModel:
        return RASACTorchModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
            model_config=config["model"],
            name=name,
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=config["twin_q"],
            initial_alpha=config["initial_alpha"],
            target_entropy=config["target_entropy"],
            **custom_cfg,
        )

    model = _make("ra_sac_model")
    policy.target_model = _make("target_ra_sac_model")
    return model, _get_dist_class(policy, config, action_space)


def _ra_sac_actor_critic_loss(policy, model, dist_class, train_batch):
    """SAC actor-critic loss augmented with RA KL regularization.

    actor_critic_loss calls model.forward() on both CUR_OBS and NEXT_OBS,
    leaving the cached posterior pointing to NEXT_OBS. We re-run the encoder
    on CUR_OBS afterwards so that get_posterior() returns the correct
    distribution for the KL loss.
    """
    base_loss = actor_critic_loss(policy, model, dist_class, train_batch)

    # Re-run encoder on current obs to get the correct posterior for KL.
    model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None)

    return _apply_ra_kl_loss(policy, model, train_batch, base_loss)


def _ra_sac_before_loss_init(policy, obs_space, action_space, config):
    """Initialize RA annealing attributes, then run SAC's standard mixin setup."""
    _init_ra_config(policy, config)
    setup_late_mixins(policy, obs_space, action_space, config)


def _ra_sac_stats(policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Combine base SAC stats with RA-specific metrics."""
    stats = _sac_stats(policy, train_batch)
    stats.update(_build_ra_stats_dict(policy.model_gpu_towers))
    return stats


# ---------------------------------------------------------------------------
# RASACTorchPolicy
# ---------------------------------------------------------------------------

RASACTorchPolicy = build_policy_class(
    name="RASACTorchPolicy",
    framework="torch",
    loss_fn=_ra_sac_actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.SACConfig(),
    stats_fn=_ra_sac_stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=_ra_sac_before_loss_init,
    make_model_and_action_dist=_build_ra_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)


# ---------------------------------------------------------------------------
# Tower-stat aggregation helpers
# ---------------------------------------------------------------------------

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


def _get_from_conf_with_fallback(config: dict, key: str, fallback_key: str) -> float:
    """Return *key* from config, falling back to *fallback_key*.

    :raises AssertionError: If neither key is present.
    """
    value = config.get(key, config.get(fallback_key))
    assert value is not None, (
        f"Config must contain '{key}' or '{fallback_key}', got: {list(config.keys())}"
    )
    return value
