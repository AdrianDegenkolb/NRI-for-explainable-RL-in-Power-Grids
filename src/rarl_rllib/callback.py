"""
AnnealingCallback for RLlib: anneals beta (KL weight) and tau (Gumbel temperature).

The callback uses the framework-agnostic :class:`rarl.annealing.AnnealingState`
internally and propagates updated values to all RLlib workers after each
training iteration.

Config keys read from ``policy.config``::

    relation_awareness:
      beta_start: 0.0
      beta_end: 5.0
      beta_non_graph_edges_start: 0.0
      beta_non_graph_edges_end: 5.0
      beta_anneal_timesteps: null    # defaults to total_timesteps
      tau_anneal_timesteps: null     # defaults to total_timesteps

    model:
      custom_model_config:
        sampling:
          tau_start: 2.0
          tau_end: 0.5

Designed to be composed with other RLlib callbacks::

    from ray.rllib.algorithms.callbacks import make_multi_callbacks
    callbacks = make_multi_callbacks([AnnealingCallback, MyOtherCallback])
"""

import time
from typing import Dict, Optional, List, Any

import grid2op
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ray._private.dict import unflattened_lookup
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.experiment import Trial
from ray.tune.experimental.output import (
    TuneReporterBase,
    get_air_verbosity,
    _get_time_str,
    _current_best_trial,
)
from tabulate import tabulate

from core.constants import RL_POLICY, Style
from grid2op_env.observation_converter import GraphObservationConverter
from rarl.annealing import AnnealingState
from visualization import PlottingArgs, visualize_graph, get_node_styles

_POLICY_ID = "default_policy"


def _get_policy(algorithm: Algorithm, policy_id: str = _POLICY_ID):
    """Try multiple policy IDs to handle different algorithm setups."""
    for pid in (policy_id, RL_POLICY, "default_policy"):
        p = algorithm.get_policy(pid)
        if p is not None:
            return p
    return None


class AnnealingCallback(DefaultCallbacks):
    """
    Anneals ``current_beta``, ``current_beta_non_graph``, and ``current_tau``
    on all workers using a three-phase cosine schedule.

    See :func:`rarl.annealing.cosine_decay_schedule` for the schedule details.
    """

    _state: AnnealingState | None = None

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        policy = _get_policy(algorithm)
        if policy is None or not hasattr(policy, "current_beta"):
            return

        ra_cfg = policy.config.get("relation_awareness", {})
        samp_cfg = policy.config["model"]["custom_model_config"].get("sampling", ra_cfg)
        total = algorithm.config.get("total_timesteps", 1_000_000)

        self._state = AnnealingState(
            beta_start=ra_cfg.get("beta_start", 0.0),
            beta_end=ra_cfg.get("beta_end", ra_cfg.get("beta", 1.0)),
            beta_non_graph_start=ra_cfg.get("beta_non_graph_edges_start", 0.0),
            beta_non_graph_end=ra_cfg.get("beta_non_graph_edges_end", ra_cfg.get("beta", 1.0)),
            tau_start=samp_cfg.get("tau_start", 2.0),
            tau_end=samp_cfg.get("tau_end", samp_cfg.get("temperature", 1.0)),
            total_steps=total,
            beta_anneal_steps=ra_cfg.get("beta_anneal_timesteps", total),
            tau_anneal_steps=ra_cfg.get("tau_anneal_timesteps", total),
        )

        # Set initial values on all workers
        self._sync(algorithm, step=0)

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        if self._state is None:
            return

        current_step = result.get("timesteps_total", 0)
        self._state.step(current_step)
        self._sync(algorithm, step=current_step)

    def _sync(self, algorithm: Algorithm, step: int) -> None:
        """Push current annealing values to local + remote workers."""
        if self._state is None:
            return

        beta = self._state.beta
        beta_ng = self._state.beta_non_graph
        tau = self._state.tau

        def _update(worker):
            p = worker.policy_map.get("default_policy") or worker.policy_map.get(RL_POLICY)
            if p is None or not hasattr(p, "current_beta"):
                return
            p.current_beta_graph = beta
            p.current_beta_non_graph = beta_ng
            p.current_tau = tau
            if hasattr(p, "model") and hasattr(p.model, "set_tau"):
                p.model.set_tau(tau)

        algorithm.workers.local_worker().call(_update)
        algorithm.workers.foreach_worker(_update)


class TuneCallback(TuneReporterBase):
    def __init__(
            self,
            log_level: int,
            metric: str,
            mode: str = "max",
            heartbeat_freq: int = 30,
            eval_freq: int = 1,
    ):
        super().__init__(get_air_verbosity(0))
        self._start_end_verbosity = 1
        self._heartbeat_freq = heartbeat_freq
        self.log_level = log_level
        self._last_res_it = 0
        self._eval_freq = eval_freq
        self._metric = metric
        self._mode = mode
        self._best_trial = None

    def print_heartbeat(self, trials, *args, force: bool = False):
        if force or time.time() - self._last_heartbeat_time >= self._heartbeat_freq:
            self._print_heartbeat(trials, *args, force=force)
            self._last_heartbeat_time = time.time()

    def _print_heartbeat(self, trials, *sys_args, force: bool = False):
        result = list()
        # Trial status: 1 RUNNING | 7 PENDING
        result.append(self._get_overall_trial_progress_str(trials))
        # Current time: 2023-02-24 12:35:39 (running for 00:00:37.40)
        result.append(self._time_heartbeat_str)
        # Logical resource usage: 8.0/64 CPUs, 0/0 GPUs
        result.extend(sys_args)
        # *** Current BEST TRIAL: 6c81141f  ***  | with SCORE: 267 found at TIMESTEP: 529
        current_best_trial, metric_val = _current_best_trial(
            trials, self._metric, self._mode
        )
        if current_best_trial:
            best_trial_str = f"Current BEST TRIAL: {current_best_trial.trial_id} | " \
                             f"with SCORE: " \
                             f"{unflattened_lookup(self._metric, current_best_trial.last_result)} " \
                             f"at TIMESTEP: {current_best_trial.last_result['timesteps_total']} "
            result.append(best_trial_str)
        for line in result:
            print(line)

    def on_trial_result(
            self,
            iteration: int,
            trials: List[Trial],
            trial: Trial,
            result: Dict,
            **info,
    ):
        if self.log_level:
            # start printing after first evaluation
            if result['training_iteration'] % self._eval_freq == 0:
                print(Style.BOLD + " ------ TRAIL RESULTS -------" + Style.END)
                self._start_block(f"trial_{trial}_result_{result['training_iteration']}")
                curr_time_str, running_time_str = _get_time_str(self._start_time, time.time())
                print(
                    f"{self._addressing_tmpl.format(trial)} "
                    f"finished iteration {result['training_iteration']} "
                    f"at {curr_time_str}. Total running time: " + running_time_str
                )
                # print intermediate results for trial:
                self._print_result(trial, result)
                self._last_res_it = result['training_iteration']

    def _print_result(self, trial: Trial, result: Optional[Dict] = None, force: bool = False):
        result = result or trial.last_result
        trial_id = str(trial)
        eval_res = result.get("evaluation", {})
        train_res = result.get("custom_metrics", {})
        # Print the table
        headers = ["trial name", "iter", "total time", "ts", "agent_interactions", "EVAL survival duration", "EVAL return",
                   "TRAIN survival duration", "TRAIN return", "episodes_this_iter"]

        table = [[trial_id,
                  result.get('training_iteration', "N/A"),
                  _get_time_str(self._start_time, time.time())[1],
                  result.get("timesteps_total", "N/A"),
                  result.get("custom_metrics", {}).get("total_agent_interact", "N/A"),
                  eval_res.get("custom_metrics", {}).get("grid2op_end_mean", "N/A"),
                  eval_res.get("episode_reward_mean", "N/A"),
                  train_res.get("grid2op_end_mean", "N/A"),
                  result.get("episode_reward_mean", "N/A"),
                  result.get("episodes_this_iter", "N/A"),]]

        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))


class CustomMetricsCallback(DefaultCallbacks):

    def on_algorithm_init(
            self,
            *,
            algorithm: Algorithm,
            **kwargs,
    ) -> None:
        self.log_level = algorithm.my_log_level
        self.curr_level = 0
        self.node_styles = get_node_styles(grid2op.make("l2rpn_case14_sandbox"), GraphObservationConverter)
        env = grid2op.make("l2rpn_case14_sandbox")
        obs_space = GraphObservationConverter(env.observation_space)
        self.powerline_edge_index = obs_space._get_edge_index(env.reset())
        if hasattr(algorithm, "curriculum_training") and algorithm.curriculum_training:
            print(f"Start with curriculum level {self.curr_level}")

    def on_episode_end(
            self,
            *,
            episode: EpisodeV2,
            worker: Optional[RolloutWorker] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Policy] = None,
            env_index: Optional[int] = None,
            **kwargs: Dict[str, Any],
    ) -> None:
        """
        Collect extra metrics such as:
         - grid2op episode length - RLlib counts extra steps because of high level agent.
         - chronic id.
        """
        agents_steps = {k: len(v) for k, v in episode._agent_reward_history.items()}

        episode.custom_metrics["corrected_ep_len"] = agents_steps["high_level_agent"]
        envs = base_env.get_sub_environments()
        grid2op_end = np.array([env.env_g2op.current_obs.current_step for env in envs]).mean()
        chron_id = envs[0].env_g2op.chronics_handler.get_name()
        episode.custom_metrics["grid2op_end"] = grid2op_end
        episode.media["chronic_id"] = chron_id

        # New extra metrics:
        interact_count = np.array([env.interact_count for env in envs]).mean()
        active_dn_count = np.array([env.active_dn_count for env in envs]).mean()
        reconnect_count = np.array([env.reconnect_count for env in envs]).mean()
        disconnect_count = np.array([env.disconnect_count for env in envs]).mean()
        reset_count = np.array([env.reset_count for env in envs]).mean()

        episode.custom_metrics["interact_count"] = interact_count
        episode.custom_metrics["active_dn_count"] = active_dn_count
        episode.custom_metrics["reconnect_count"] = reconnect_count
        episode.custom_metrics["disconnect_count"] = disconnect_count
        episode.custom_metrics["reset_count"] = reset_count

    def on_evaluate_end(
            self,
            *,
            algorithm: "Algorithm",
            evaluation_metrics: dict,
            **kwargs,
    ) -> None:
        data = evaluation_metrics["evaluation"]
        # Save summarized results
        data["custom_metrics"]["grid2op_end_min"] = int(np.min(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_mean"] = int(np.mean(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_max"] = int(np.max(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_std"] = np.std(data["custom_metrics"]["grid2op_end"])
        # Extra metrics:
        data["custom_metrics"]["mean_interact_count"] = np.mean(data["custom_metrics"]["interact_count"])
        data["custom_metrics"]["total_agent_interact"] = np.sum(data["custom_metrics"]["interact_count"])
        data["custom_metrics"]["mean_active_dn_count"] = np.mean(data["custom_metrics"]["active_dn_count"])
        data["custom_metrics"]["mean_reconnect_count"] = np.mean(data["custom_metrics"]["reconnect_count"])
        data["custom_metrics"]["mean_disconnect_count"] = np.mean(data["custom_metrics"]["disconnect_count"])
        data["custom_metrics"]["mean_reset_count"] = np.mean(data["custom_metrics"]["reset_count"])

        if self.log_level > 1:
            print(f" Showing results for evaluated chronics:")
            overview = {
                "chronic_id": data["episode_media"]["chronic_id"],
                "grid2op_end": data["custom_metrics"]["grid2op_end"],
                "reward": data["hist_stats"]["episode_reward"]}
            print(tabulate(overview, headers="keys", tablefmt="rounded_grid"))

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        custom = result.get("custom_metrics", {})
        if "grid2op_end" in custom:
            result["custom_metrics"]["grid2op_end_mean"] = int(np.mean(custom["grid2op_end"]))
            result["custom_metrics"]["grid2op_end_std"] = np.var(custom["grid2op_end"])
            del result["custom_metrics"]["grid2op_end"]
        if "corrected_ep_len" in custom:
            result["custom_metrics"]["corrected_ep_len_mean"] = int(np.mean(custom["corrected_ep_len"]))
            del result["custom_metrics"]["corrected_ep_len"]
        if "interact_count" in custom:
            result["custom_metrics"]["mean_interact_count"] = np.mean(custom["interact_count"])
            result["custom_metrics"]["total_agent_interact"] = np.sum(custom["interact_count"])
            del result["custom_metrics"]["interact_count"]
        if "active_dn_count" in custom:
            result["custom_metrics"]["mean_active_dn_count"] = np.mean(custom["active_dn_count"])
            del result["custom_metrics"]["active_dn_count"]
        if "reconnect_count" in custom:
            result["custom_metrics"]["mean_reconnect_count"] = np.mean(custom["reconnect_count"])
            del result["custom_metrics"]["reconnect_count"]
        if "disconnect_count" in custom:
            result["custom_metrics"]["mean_disconnect_count"] = np.mean(custom["disconnect_count"])
            del result["custom_metrics"]["disconnect_count"]
        if "reset_count" in custom:
            result["custom_metrics"]["mean_reset_count"] = np.mean(custom["reset_count"])
            del result["custom_metrics"]["reset_count"]

        learner_stats = (result.get("info", {})
                         .get("learner", {})
                         .get(RL_POLICY, {})
                         .get("learner_stats", {}))
        posterior_mean = learner_stats.get("relation_awareness/posterior_mean")
        if posterior_mean is not None:
            result["relation_awareness/latent_graph_mean"] = fig_to_chw_uint8(
                visualize_graph(PlottingArgs(
                    num_nodes=57,
                    node_styles=self.node_styles,
                    latent_edge_probs=np.array(posterior_mean),
                    powerline_edge_index=self.powerline_edge_index,
            )))

        posterior_var = learner_stats.get("relation_awareness/posterior_var")
        if posterior_var is not None:
            result["relation_awareness/latent_graph_var"] = fig_to_chw_uint8(
                visualize_graph(PlottingArgs(
                    num_nodes=57,
                    node_styles=self.node_styles,
                    latent_edge_probs=np.array(posterior_var),
                    powerline_edge_index=self.powerline_edge_index,
            )))

        if algorithm.curriculum_training:
            if self.curr_level < len(algorithm.curriculum_threshold) and \
                    result['timesteps_total'] > algorithm.curriculum_threshold[self.curr_level]:
                self.curr_level += 1
                algorithm.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.set_curriculum(self.curr_level)
                    )
                )
                print(f"Curriculum level increased to {self.curr_level}")

def fig_to_chw_uint8(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)   # (H, W, 4)
    rgb  = rgba[..., :3]                                     # (H, W, 3)
    chw  = np.transpose(rgb, (2, 0, 1))                       # (3, H, W)
    return np.ascontiguousarray(chw)