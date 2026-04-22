"""GNNBaselineDQNPolicy -- DQN policy with GNN baseline model builder.

Standard DQN policy that uses a custom model builder to instantiate
GNNBaselineDQNModel (which wraps BaselineGNN inside DQNTorchModel heads).
No KL regularization -- this is the fixed-graph baseline.
"""

import ray
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_action_dist import get_torch_categorical_class_with_temperature
from ray.rllib.policy import build_policy_class
from ray.rllib.algorithms.dqn.dqn_torch_policy import (
    build_q_losses, build_q_stats, ComputeTDErrorMixin,
    get_distribution_inputs_and_class, grad_process_and_td_error_fn,
    extra_action_out_fn, setup_early_mixins, before_loss_init, adam_optimizer,
)
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors

from core.constants import DQN_GNN_POLICY
from rarl_rllib.dqn.gnn_dqn_model import GNNBaselineDQNModel
from rarl_rllib.dqn.mlp_dqn_policy import postprocess_nstep_and_prio


def _build_gnn_dqn_model_and_action_dist(policy, obs_space, action_space, config):
    """Model builder for GNNBaselineDQNPolicy."""
    custom_cfg = config["model"].get("custom_model_config", {})

    def _make(name: str) -> DQNTorchModel:
        return GNNBaselineDQNModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=None,
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

    model = _make("gnn_dqn_model")
    policy.target_model = _make("target_gnn_dqn_model")

    temperature = config["categorical_distribution_temperature"]
    return model, get_torch_categorical_class_with_temperature(temperature)


GNNBaselineDQNPolicy = build_policy_class(
    name=DQN_GNN_POLICY,
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DQNConfig(),
    make_model_and_action_dist=_build_gnn_dqn_model_and_action_dist,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)
