"""RADQNTorchPolicy -- DQN policy augmented with RAGNN KL regularization.

Built via RLlib's ``build_policy_class`` (TorchPolicy V1 API, off-policy).
Mirrors the SAC variant in ``rasac.py`` but uses DQN's Q-learning loss
and model builder instead of SAC's actor-critic setup.
"""

from typing import Dict

import ray
from ray.rllib import SampleBatch
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_stats
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_action_dist import get_torch_categorical_class_with_temperature
from ray.rllib.policy import build_policy_class
from ray.rllib.algorithms.dqn.dqn_torch_policy import ComputeTDErrorMixin, build_q_losses, \
    get_distribution_inputs_and_class, grad_process_and_td_error_fn, extra_action_out_fn, setup_early_mixins, \
    before_loss_init, adam_optimizer
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors
from torch import Tensor
from ray.rllib.utils.typing import TensorType

from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK
from rarl import compute_ra_kl_loss
from rarl_rllib import RADQNTorchModel
from rarl_rllib.policies.common import build_prior_and_graph_masks, store_ra_tower_stats, build_ra_stats_dict, \
    init_ra_config
from rarl_rllib.policies.dqn_postprocessing import postprocess_nstep_and_prio


def _ra_build_q_losses(policy, model, dist_class, train_batch):
    base_loss = build_q_losses(policy, model, dist_class, train_batch)

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

    return base_loss + kl_loss


def _ra_dqn_stats(policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Combine base DQN stats with RA-specific metrics."""
    stats = build_q_stats(policy, train_batch)
    stats.update(build_ra_stats_dict(policy.model_gpu_towers))
    return stats


def _ra_dqn_before_loss_init(policy, obs_space, action_space, config):
    """Initialize RA annealing attributes, then run DQN's standard mixin setup."""
    init_ra_config(policy, config)
    before_loss_init(policy, obs_space, action_space, config)


def _build_ra_dqn_model_and_action_dist(policy, obs_space, action_space, config):
    """Model builder for RADQNTorchPolicy -- constructs RADQNTorchModel directly."""
    custom_cfg = config["model"].get("custom_model_config", {})

    def _make(name: str) -> DQNTorchModel:
        return RADQNTorchModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,  # RADQNTorchModel overrides this with gnn_out_dim
            model_config=config["model"],
            name=name,
            q_hiddens=config["hiddens"],
            dueling=config["dueling"],
            num_atoms=config["num_atoms"],
            use_noisy=config["noisy"],
            v_min=config["v_min"],
            v_max=config["v_max"],
            sigma0=config["sigma0"],
            **custom_cfg,
        )

    model = _make("ra_dqn_model")
    policy.target_model = _make("target_ra_dqn_model")

    temperature = config["categorical_distribution_temperature"]
    return model, get_torch_categorical_class_with_temperature(temperature)


RADQNTorchPolicy = build_policy_class(
    name="RADQNTorchPolicy",
    framework="torch",
    loss_fn=_ra_build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DQNConfig(),
    make_model_and_action_dist=_build_ra_dqn_model_and_action_dist,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=_ra_dqn_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=_ra_dqn_before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)
