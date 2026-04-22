"""Dict-observation-aware n-step postprocessing for DQN.

RLlib's built-in ``adjust_nstep`` assumes observations are numpy arrays and
uses ``batch[OBS][n_step:]`` slicing, which fails when observations are dicts
(e.g. graph observation spaces with NODES / EDGE_INDEX / EDGE_MASK).

This module provides drop-in replacements that handle both flat arrays and
dict observation spaces.
"""

import numpy as np
import ray
from ray.rllib import SampleBatch
from ray.rllib.algorithms.dqn.dqn_torch_policy import (
    build_q_model_and_distribution, build_q_losses, build_q_stats,
    ComputeTDErrorMixin, get_distribution_inputs_and_class,
    grad_process_and_td_error_fn, extra_action_out_fn, setup_early_mixins,
    before_loss_init, adam_optimizer,
)
from ray.rllib.evaluation.postprocessing import adjust_nstep as _adjust_nstep_orig
from ray.rllib.policy import build_policy_class
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS


def _slice_obs_dict(obs_dict: dict, slc: slice) -> dict:
    """Apply a slice to every array in an observation dict."""
    return {k: v[slc] for k, v in obs_dict.items()}


def _stack_obs_dict(obs_dict: dict, n: int) -> dict:
    """Repeat the last element of each array in an observation dict *n* times and stack."""
    return {k: np.stack([v[-1]] * n) for k, v in obs_dict.items()}


def _concat_obs_dicts(a: dict, b: dict) -> dict:
    """Concatenate two observation dicts key-wise along axis 0."""
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def adjust_nstep_dict_obs(n_step: int, gamma: float, batch: SampleBatch) -> None:
    """Like ``adjust_nstep`` but supports dict observation spaces."""
    obs = batch[SampleBatch.OBS]

    # If obs is not a dict, delegate to the original implementation.
    if not isinstance(obs, dict):
        _adjust_nstep_orig(n_step, gamma, batch)
        return

    assert batch.is_single_trajectory(), (
        "Unexpected terminated|truncated in middle of trajectory!"
    )

    len_ = len(batch)

    # Shift NEXT_OBS using dict-aware helpers.
    batch[SampleBatch.NEXT_OBS] = _concat_obs_dicts(
        _slice_obs_dict(obs, slice(n_step, None)),
        _stack_obs_dict(batch[SampleBatch.NEXT_OBS], min(n_step, len_)),
    )

    # TERMINATEDS
    batch[SampleBatch.TERMINATEDS] = np.concatenate([
        batch[SampleBatch.TERMINATEDS][n_step - 1:],
        np.tile(batch[SampleBatch.TERMINATEDS][-1], min(n_step - 1, len_)),
    ], axis=0)

    # TRUNCATEDS (only if present)
    if SampleBatch.TRUNCATEDS in batch:
        batch[SampleBatch.TRUNCATEDS] = np.concatenate([
            batch[SampleBatch.TRUNCATEDS][n_step - 1:],
            np.tile(batch[SampleBatch.TRUNCATEDS][-1], min(n_step - 1, len_)),
        ], axis=0)

    # Change rewards in place.
    for i in range(len_):
        for j in range(1, n_step):
            if i + j < len_:
                batch[SampleBatch.REWARDS][i] += (
                    gamma ** j * batch[SampleBatch.REWARDS][i + j]
                )


def postprocess_nstep_and_prio(
    policy: Policy, batch: SampleBatch, other_agent=None, episode=None,
) -> SampleBatch:
    """Drop-in replacement for ``dqn_tf_policy.postprocess_nstep_and_prio``."""
    if policy.config["n_step"] > 1:
        adjust_nstep_dict_obs(policy.config["n_step"], policy.config["gamma"], batch)

    if PRIO_WEIGHTS not in batch:
        batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])

    return batch


DictObsDQNTorchPolicy = build_policy_class(
    name="DictObsDQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DQNConfig(),
    make_model_and_action_dist=build_q_model_and_distribution,
    postprocess_fn=postprocess_nstep_and_prio,
    stats_fn=build_q_stats,
    action_distribution_fn=get_distribution_inputs_and_class,
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
