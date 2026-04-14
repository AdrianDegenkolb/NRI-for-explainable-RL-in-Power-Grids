"""
make_rarl_policy: factory that wraps any RLlib TorchPolicy to add RARL.

How it works
------------
The factory creates a subclass that overrides ``loss()`` to append the
KL regularization term from :func:`rarl.loss.compute_ra_kl_loss`, and
overrides ``stats_fn()`` to expose all RA-specific metrics for TensorBoard.

The wrapped policy expects:
  - ``model`` to be a :class:`rarl_rllib.model.RARLModel` (i.e. exposes
    ``get_posterior()`` and ``set_tau()``).
  - ``self.config["relation_awareness"]`` with the keys described below.
  - ``self.observation_space`` to have a ``num_nodes`` attribute.

Config schema (under ``relation_awareness``)::

    relation_awareness:
      sampling:
        tau: 0.2                      # or tau_end for annealing
      prior:
        prior_prob_for_graph_edge: 0.9
        temperature: 0.2
      loss:
        beta_graph_edges: 5.0         # or beta_graph_edges_end for annealing
        beta_non_graph_edges: 5.0     # or beta_non_graph_edges_end for annealing
      latent_space:
        num_edge_types: 2

Example::

    from ray.rllib.algorithms.ppo import PPOTorchPolicy
    from rarl_rllib import make_rarl_policy, RARLModel

    RARLPPOPolicy = make_rarl_policy(PPOTorchPolicy)
    # Register and use just like a regular RLlib policy.
"""

from typing import Dict, List, Type

import gymnasium as gym
import torch
from ray.rllib import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType
from torch import Tensor

from src.core.observation_space import EDGE_INDEX, EDGE_MASK, NODES
from src.rarl.graph import fully_connected_edge_index
from src.rarl.loss import compute_ra_kl_loss
from src.rarl.prior import get_prior_tensor, get_priors
from src.rarl_rllib import RARLModel

# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def make_rarl_policy(base_policy_cls: Type[TorchPolicyV2]) -> Type[TorchPolicyV2]:
    """
    Return a relation-aware subclass of *base_policy_cls*.

    The subclass extends ``loss()`` with KL regularization and ``stats_fn()``
    with RARL-specific metrics. Optional annealing of beta/tau is handled separately
    by :class:`rarl_rllib.callback.AnnealingCallback`.

    :param base_policy_cls: Any RLlib TorchPolicyV2 subclass (e.g. PPOTorchPolicy).
    :return: New policy class with RARL loss augmentation.
    """

    class RARLPolicy(base_policy_cls):

        def __init__(
            self,
            observation_space: gym.spaces.Dict,
            action_space: gym.spaces.Discrete,
            config: AlgorithmConfigDict,
            *args,
            **kwargs,
        ):
            """Unpack RARL sub-configs and initialize the base policy."""
            ra_cfg = config["relation_awareness"]

            # Convenience references to nested sub-configs
            self._loss_cfg = ra_cfg["loss"]
            self._sampling_cfg = ra_cfg["sampling"]
            self._prior_cfg = ra_cfg["prior"]
            self._latent_cfg = ra_cfg["latent_space"]

            # Annealed scalars — support both fixed and end-of-annealing keys
            self.current_beta_graph = _get_from_conf_with_fallback(
                self._loss_cfg, "beta_graph_edges", "beta_graph_edges_end")
            self.current_beta_non_graph = _get_from_conf_with_fallback(
                self._loss_cfg, "beta_non_graph_edges", "beta_non_graph_edges_end"
            )
            self.current_tau = _get_from_conf_with_fallback(
                self._sampling_cfg, "tau", "tau_end"
            )

            self.num_edge_types: int = self._latent_cfg["num_edge_types"]
            self.prior_prob_for_graph_edge: float = self._prior_cfg["prior_prob_for_graph_edge"]
            self.temperature: float = self._prior_cfg["temperature"]

            super().__init__(
                observation_space=observation_space,
                action_space=action_space,
                config=config,
                *args,
                **kwargs,
            )

        # -------------------------------------------------------------------
        # Loss
        # -------------------------------------------------------------------

        @override(base_policy_cls)
        def loss(
            self,
            model: RARLModel,
            dist_class,
            train_batch: SampleBatch,
        ) -> TensorType:
            """
            Augment the base policy loss with RARL KL regularization.

            Adds a weighted KL divergence between the posterior edge-type
            distribution (from the GNN) and the prior derived from the
            observed graph structure.
            """
            total_loss = super().loss(model, dist_class, train_batch)

            obs = train_batch["obs"]
            all_graph_edges: Tensor = obs[EDGE_INDEX]   # [B, 2, E_max]
            edge_masks_obs: Tensor = obs[EDGE_MASK]     # [B, E_max]

            # RLlib calls loss() once with a dummy batch during initialization.
            # That batch may have all-false edge masks, which would produce empty
            # tensors downstream. We force at least one valid edge to avoid this.
            #if not edge_masks_obs.any():
            #    return total_loss

            prior_tensor, graph_edge_masks = self._build_prior_and_graph_masks(
                all_graph_edges, edge_masks_obs
            )
            posteriors: Tensor = model.get_posterior()  # [B, E, K]

            kl_loss, stats = compute_ra_kl_loss(
                posteriors=posteriors,
                prior_tensor=prior_tensor,
                graph_edge_masks=graph_edge_masks,
                beta=self.current_beta_graph,
                beta_non_graph=self.current_beta_non_graph,
            )

            self._store_tower_stats(model, kl_loss, stats, prior_tensor, posteriors)

            return total_loss + kl_loss

        def _build_prior_and_graph_masks(
            self,
            all_graph_edges: Tensor,
            edge_masks_obs: Tensor,
        ) -> tuple[Tensor, Tensor]:
            """
            Compute the batched prior tensor and graph-edge affiliation masks.

            For each sample in the batch, valid graph edges are extracted using
            *edge_masks_obs*, then prior probabilities over edge types are derived
            from those edges against the fully-connected reference graph.

            :param all_graph_edges: Padded edge indices of shape [B, 2, E_max].
            :param edge_masks_obs:  Binary validity mask of shape [B, E_max].
            :return: Tuple of (prior_tensor [B, E, K], graph_edge_masks [B, E]).
            """
            N, _ = self.observation_space[NODES].shape
            all_edges = fully_connected_edge_index(N)  # same for every sample

            batched_priors: List[Tensor] = []
            batched_graph_masks: List[Tensor] = []

            for i in range(all_graph_edges.shape[0]):
                prior, gmask = self._prior_for_sample(
                    all_graph_edges[i],
                    edge_masks_obs[i],
                    all_edges,
                )
                batched_priors.append(prior.to(device=self.device, dtype=torch.float32))
                batched_graph_masks.append(gmask.to(device=self.device))

            prior_tensor = torch.stack(batched_priors)       # [B, E, K]
            graph_edge_masks = torch.stack(batched_graph_masks)  # [B, E]
            return prior_tensor, graph_edge_masks

        def _prior_for_sample(
            self,
            graph_edges_padded: Tensor,
            edge_mask: Tensor,
            all_edges: Tensor,
        ) -> tuple[Tensor, Tensor]:
            """
            Compute the prior and graph-affiliation mask for a single observation.

            :param graph_edges_padded: Padded edge index of shape [2, E_max].
            :param edge_mask:          Boolean validity mask of shape [E_max].
            :param all_edges:          Full edge index for the N-node graph [2, E_full].
            :return: Tuple of (prior [E_full, K], graph_mask [E_full]).
            """
            valid_edges = graph_edges_padded[:, edge_mask.bool()]  # [2, E_valid]

            g_prior, ng_prior = get_priors(
                prob_graph_edges_exist=self.prior_prob_for_graph_edge,
                num_graph_edges=valid_edges.shape[1],
                num_non_graph_edges=all_edges.shape[1] - valid_edges.shape[1],
                temperature=self.temperature,
            )

            return get_prior_tensor(
                graph_edges=valid_edges,
                all_edges=all_edges,
                prior_for_graph_edges=g_prior,
                prior_for_non_graph_edges=ng_prior,
                num_edge_types=self.num_edge_types,
                return_mask=True,
            )

        def _store_tower_stats(
            self,
            model: RARLModel,
            kl_loss: Tensor,
            stats: dict,
            prior_tensor: Tensor,
            posteriors: Tensor,
        ) -> None:
            """
            Write all RARL metrics into *model.tower_stats* for later aggregation
            in :meth:`stats_fn`.
            """
            model.tower_stats.update({
                "ra_kl_loss":                  kl_loss,
                "ra_kl_graph":                 stats["kl_graph_edges"],
                "ra_kl_latent":                stats["kl_latent_edges"],
                "ra_kl_unweighted":            stats["kl_unweighted"],
                "ra_fraction_graph":           stats["fraction_graph_edges"],
                "ra_fraction_latent":          stats["fraction_latent_edges"],
                "ra_mean_prior":               prior_tensor.mean(0),
                "ra_mean_posterior":           posteriors.mean(0),
                "ra_posteriors":               posteriors,
                "ra_current_beta":             torch.as_tensor(self.current_beta_graph,     dtype=torch.float32),
                "ra_current_beta_non_graph":   torch.as_tensor(self.current_beta_non_graph, dtype=torch.float32),
                "ra_current_tau":              torch.as_tensor(self.current_tau,            dtype=torch.float32),
                "ra_gnn_stats":                model.ragnn.gnn.stats,
            })

        # -------------------------------------------------------------------
        # Stats
        # -------------------------------------------------------------------

        @override(base_policy_cls)
        def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
            """Extend base stats with all RARL-specific TensorBoard metrics."""
            stats = super().stats_fn(train_batch)
            towers = self.model_gpu_towers

            stats.update({
                # Scalar KL terms and annealing parameters
                "relation_awareness/kl_loss":              _tower_mean(towers, "ra_kl_loss"),
                "relation_awareness/kl_unweighted":        _tower_mean(towers, "ra_kl_unweighted"),
                "relation_awareness/kl_graph_edges":       _tower_mean(towers, "ra_kl_graph"),
                "relation_awareness/kl_latent_edges":      _tower_mean(towers, "ra_kl_latent"),
                "relation_awareness/fraction_graph_edges": _tower_mean(towers, "ra_fraction_graph"),
                "relation_awareness/fraction_latent_edges":_tower_mean(towers, "ra_fraction_latent"),
                "relation_awareness/current_beta":         _tower_mean(towers, "ra_current_beta"),
                "relation_awareness/current_beta_non_graph":_tower_mean(towers, "ra_current_beta_non_graph"),
                "relation_awareness/current_tau":          _tower_mean(towers, "ra_current_tau"),

                # Per-edge-type prior / posterior distributions
                "relation_awareness/prior_existence_probs": (
                    _tower_stack_mean(towers, "ra_mean_prior")[:, 0].cpu().tolist()
                ),
                "relation_awareness/posterior_existence_probs": (
                    _tower_stack_mean(towers, "ra_mean_posterior")[:, 0].cpu().tolist()
                ),

                # Full posterior statistics across all samples and towers
                "relation_awareness/posterior_mean": (
                    _tower_cat_stat(towers, "ra_posteriors").mean(dim=0).cpu().tolist()
                ),
                "relation_awareness/posterior_var": (
                    _tower_cat_stat(towers, "ra_posteriors").var(dim=0).cpu().tolist()
                ),
            })

            # Per-key GNN stats (keys vary by model, so added dynamically)
            for key in towers[0].tower_stats["ra_gnn_stats"]:
                stats[f"relation_awareness/gnn/{key}"] = torch.mean(
                    torch.stack([
                        t.tower_stats["ra_gnn_stats"][key].detach() for t in towers
                    ])
                ).item()

            return stats

    RARLPolicy.__name__ = f"RARL{base_policy_cls.__name__}"
    RARLPolicy.__qualname__ = f"RARL{base_policy_cls.__qualname__}"
    return RARLPolicy


# ---------------------------------------------------------------------------
# Tower-stat aggregation helpers (stateless, live outside the policy class)
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
    """
    Return the value for *key*, falling back to *fallback_key*.

    Supports configs that store a single fixed value (``key``) or an end
    value used after annealing (``fallback_key``).

    :raises AssertionError: If neither key is present.
    """
    value = config.get(key, config.get(fallback_key))
    assert value is not None, (
        f"Config must contain '{key}' or '{fallback_key}', got: {list(config.keys())}"
    )
    return value
