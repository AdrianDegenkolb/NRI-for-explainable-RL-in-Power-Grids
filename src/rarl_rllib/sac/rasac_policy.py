"""RASACTorchPolicy — SAC policy augmented with RAGNN KL regularization.

Built via RLlib's ``build_policy_class`` (TorchPolicy V1 API, off-policy).
Compared to the PPO variant in ``rappo.py``, the SAC version:
  - Uses a dedicated encoder optimizer (appended after the standard SAC
    actor/critic/alpha optimizers) so that the shared encoder receives
    gradients from actor loss + critic losses + KL, without interfering
    with the entropy-temperature alpha parameter.
  - Re-runs the encoder forward pass on ``CUR_OBS`` after the base
    ``actor_critic_loss`` (which also processes ``NEXT_OBS``), ensuring the
    cached posterior matches the observations used for the KL computation.
  - Constructs both the policy model and a separate target model via
    ``RASACTorchModel``.
"""

from typing import Dict

import ray.rllib
import torch
from ray.rllib import SampleBatch
from ray.rllib.algorithms.sac.sac_tf_policy import postprocess_trajectory, validate_spaces
from ray.rllib.algorithms.sac.sac_torch_policy import ComputeTDErrorMixin, action_distribution_fn, actor_critic_loss, \
    optimizer_fn, stats as _sac_stats, setup_late_mixins, _get_dist_class
from ray.rllib.policy import build_policy_class
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import apply_grad_clipping, concat_multi_gpu_td_errors
from ray.rllib.utils.typing import TensorType
from torch import Tensor

from core.constants import RASAC_POLICY
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK
from rarl import compute_ra_kl_loss
from rarl_rllib.sac.rasac_model import RASACTorchModel
from rarl_rllib.common import build_prior_and_graph_masks, store_ra_tower_stats, build_ra_stats_dict, \
    init_ra_config


def _ra_sac_actor_critic_loss(policy, model, dist_class, train_batch):
    """SAC actor-critic loss with a dedicated encoder loss appended.

    actor_critic_loss returns (actor_loss, Q1_loss, [Q2_loss], alpha_loss).
    We append a 5th (or 4th for non-twin) encoder_loss that aggregates all
    loss signals relevant to the shared encoder:
        encoder_loss = actor_loss + Q1_loss + [Q2_loss] + kl_loss

    This matches the optimizer tuple built by _ra_sac_optimizer_fn, which
    appends a dedicated encoder optimizer after the standard SAC optimizers.

    actor_critic_loss calls model.forward() on both CUR_OBS and NEXT_OBS,
    leaving the cached posterior pointing to NEXT_OBS. We re-run the encoder
    on CUR_OBS afterward so that get_posterior() returns the correct
    distribution for the KL loss.
    """
    base_loss = actor_critic_loss(policy, model, dist_class, train_batch)

    # Re-run encoder on current obs so the posterior reflects CUR_OBS.
    model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True), [], None)

    obs = train_batch[SampleBatch.CUR_OBS]
    prior_tensor, graph_edge_masks = build_prior_and_graph_masks(policy, obs[EDGE_INDEX], obs[EDGE_MASK])
    posteriors: Tensor = model.get_posterior()

    kl_loss, kl_stats = compute_ra_kl_loss(
        posteriors=posteriors,
        prior_tensor=prior_tensor,
        graph_edge_masks=graph_edge_masks,
        beta=policy.current_beta_graph,
        beta_non_graph=policy.current_beta_non_graph,
    )
    store_ra_tower_stats(model, kl_loss, kl_stats, prior_tensor, posteriors, policy)

    # base_loss = (actor_loss, Q1_loss, [Q2_loss], alpha_loss)
    # encoder_loss: actor + Q critics + KL, excluding alpha (irrelevant to encoder)
    losses_without_alpha = base_loss[:-1]
    encoder_loss = sum(losses_without_alpha) + kl_loss

    return base_loss + (encoder_loss,)


def _ra_sac_optimizer_fn(policy, config):
    """SAC optimizers + a dedicated encoder optimizer as the last entry.

    The encoder optimizer covers ragnn parameters exclusively, so it receives
    gradient signals from encoder_loss (actor + Q-critics + KL) while the
    actor and critic optimizers update only their respective head parameters.
    """
    base_optimizers = optimizer_fn(policy, config)  # (actor, Q1[, Q2], alpha)
    policy.encoder_optim = torch.optim.Adam(
        params=policy.model.encoder_variables(),
        lr=config["optimization"]["actor_learning_rate"],
        eps=1e-7,
    )
    return base_optimizers + (policy.encoder_optim,)


def _ra_sac_stats(policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Combine base SAC stats with RA-specific metrics."""
    stats = _sac_stats(policy, train_batch)
    stats.update(build_ra_stats_dict(policy.model_gpu_towers))
    return stats


def _ra_sac_before_loss_init(policy, obs_space, action_space, config):
    """Initialize RA annealing attributes, then run SAC's standard mixin setup."""
    init_ra_config(policy, config)
    setup_late_mixins(policy, obs_space, action_space, config)


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


RASACTorchPolicy = build_policy_class(
    name=RASAC_POLICY,
    framework="torch",
    loss_fn=_ra_sac_actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.SACConfig(),
    stats_fn=_ra_sac_stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=_ra_sac_optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=_ra_sac_before_loss_init,
    make_model_and_action_dist=_build_ra_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
