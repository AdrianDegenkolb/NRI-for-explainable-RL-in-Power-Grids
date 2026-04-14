"""
Implements callbacks.
"""

import time
from typing import Any, Dict, Optional, List

import grid2op
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ray._private.dict import unflattened_lookup
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.tune.experiment import Trial
# from grid2op.Environment import BaseEnv
from ray.tune.experimental.output import (
    TuneReporterBase,
    get_air_verbosity,
    _get_time_str,
    _current_best_trial,
)
from tabulate import tabulate

from src.grid2op_env.observation_converter import GraphObservationConverter
from src.visualization import visualize_graph, PlottingArgs, get_node_styles


def fig_to_chw_uint8(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)   # (H, W, 4)
    rgb  = rgba[..., :3]                                     # (H, W, 3)
    chw  = np.transpose(rgb, (2, 0, 1))                       # (3, H, W)
    return np.ascontiguousarray(chw)


class Style:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class CustomMetricsCallback(DefaultCallbacks):
    # def __init__(self, log_level: int = 0):
    #     super().__init__()
    #     self.log_level = log_level

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
        if algorithm.curriculum_training:
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
        # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
        chron_id = envs[0].env_g2op.chronics_handler.get_name()
        episode.custom_metrics["grid2op_end"] = grid2op_end
        episode.media["chronic_id"] = chron_id

        # New extra metrics:
        interact_count = np.array([env.interact_count for env in envs]).mean()
        active_dn_count = np.array([env.active_dn_count for env in envs]).mean()
        reconnect_count = np.array([env.reconnect_count for env in envs]).mean()
        disconnect_count = np.array([env.disconnect_count for env in envs]).mean()
        reset_count = np.array([env.reset_count for env in envs]).mean()
        # print("disconnect_count count: ", [env.disconnect_count for env in envs])
        # print(f"interact_count: {interact_count}, active_dn_count: {active_dn_count}, reconnect_count: {reconnect_count}, disconnect_count: {disconnect_count}, reset_count: {reset_count}")

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
        # print(f'ALL METRICS {result}')
        mean_grid2op_end = int(np.mean(result["custom_metrics"]["grid2op_end"]))
        std_grid2op_end = np.var(result["custom_metrics"]["grid2op_end"])
        mean_episode_duration = int(np.mean(result["custom_metrics"]["corrected_ep_len"]))
        result["custom_metrics"]["grid2op_end_mean"] = mean_grid2op_end
        result["custom_metrics"]["grid2op_end_std"] = std_grid2op_end
        result["custom_metrics"]["corrected_ep_len_mean"] = mean_episode_duration

        # Extra metrics:
        result["custom_metrics"]["mean_interact_count"] = np.mean(result["custom_metrics"]["interact_count"])
        result["custom_metrics"]["total_agent_interact"] = np.sum(result["custom_metrics"]["interact_count"])
        result["custom_metrics"]["mean_active_dn_count"] = np.mean(result["custom_metrics"]["active_dn_count"])
        result["custom_metrics"]["mean_reconnect_count"] = np.mean(result["custom_metrics"]["reconnect_count"])
        result["custom_metrics"]["mean_disconnect_count"] = np.mean(result["custom_metrics"]["disconnect_count"])
        result["custom_metrics"]["mean_reset_count"] = np.mean(result["custom_metrics"]["reset_count"])

        latent_edge_probs = result['info']["learner"]["reinforcement_learning_policy"]["learner_stats"].get("relation_awareness/latent_graph_probs_mean", None)
        if latent_edge_probs is not None:
            result["relation_awareness/latent_graph_mean"] = fig_to_chw_uint8(
                visualize_graph(PlottingArgs(
                    num_nodes=57,
                    node_styles=self.node_styles,
                    latent_edge_probs=np.array(latent_edge_probs),
                powerline_edge_index=self.powerline_edge_index,
            )))

            result["relation_awareness/latent_graph_var"] = fig_to_chw_uint8(
                visualize_graph(PlottingArgs(
                    num_nodes=57,
                    node_styles=self.node_styles,
                    latent_edge_probs=np.array(latent_edge_probs),
                    powerline_edge_index=self.powerline_edge_index,
            )))

        # Delete irrelevant results
        del result["custom_metrics"]["grid2op_end"]
        del result["custom_metrics"]["corrected_ep_len"]
        del result["episode_media"]["chronic_id"]
        # del result["custom_metrics"]["agent_interactions"]
        del result["sampler_results"]
        del result["custom_metrics"]["interact_count"]
        del result["custom_metrics"]["active_dn_count"]
        del result["custom_metrics"]["reconnect_count"]
        del result["custom_metrics"]["disconnect_count"]
        del result["custom_metrics"]["reset_count"]

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


class AnnealingCallback(DefaultCallbacks):
    """Callback that anneals beta and tau parameters during training.

    Uses a cosine decay schedule with three phases:
    - First 10%: Hold constant at start value
    - Next 80%: Cosine decay from start to end value
    - Last 10%: Hold constant at end value
    """

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        """Initialize tau to its starting value at the beginning of training."""
        super().on_algorithm_init(algorithm=algorithm, **kwargs)

        # Get the policy
        policy = algorithm.get_policy("reinforcement_learning_policy")
        if policy is None:
            return

        # Get values
        tau_start = policy.config['relation_awareness']['sampling']['tau_start']
        beta_start = policy.config['relation_awareness']['loss']['beta_graph_edges_start']
        beta_non_graph_edges_start = policy.config['relation_awareness']['loss']['beta_non_graph_edges_start']

        # Set initial tau and beta in model on all workers
        def set_initial_values(worker):
            policy = worker.policy_map.get("reinforcement_learning_policy")
            policy.current_tau = tau_start
            policy.current_beta_graph = beta_start
            policy.current_beta_non_graph_edges = beta_non_graph_edges_start
            if policy and hasattr(policy, 'model') and hasattr(policy.model, 'set_tau'):
                policy.model.set_tau(tau_start)

        # Set on local worker
        set_initial_values(algorithm.workers.local_worker())

        # Set on remote workers
        algorithm.workers.foreach_worker(set_initial_values)

    @staticmethod
    def cosine_decay_schedule(current_step: int, total_steps: int, start_val: float, end_val: float) -> float:
        """
        Compute value using cosine decay schedule.

        Schedule:
        - 0-40% of training: constant at start_val
        - 40-80% of training: cosine decay from start_val to end_val
        - 80-100% of training: constant at end_val

        Args:
            current_step: Current training step
            total_steps: Total training steps for annealing
            start_val: Starting value
            end_val: Ending value

        Returns:
            Current annealed value
        """
        if total_steps <= 0:
            return end_val

        progress = current_step / total_steps

        # First 40%: hold constant at start
        if progress < 0.4:
            return start_val

        # Last 10%: hold constant at end
        if progress > 0.8:
            return end_val

        # Middle 40%: cosine decay
        # Map progress from [0.5, 0.9] to [0, 1]
        decay_progress = (progress - 0.4) / 0.4

        # Cosine decay: starts at 1.0, ends at 0.0
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * decay_progress))

        # Interpolate between start and end using cosine decay
        return end_val + (start_val - end_val) * cosine_decay

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        """Update beta and tau based on training progress using cosine decay."""
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        # Get the policy
        policy = algorithm.get_policy("reinforcement_learning_policy")
        if policy is None or not hasattr(policy, 'current_beta'):
            return

        # Get config
        ra_config = policy.config.get("relation_awareness", {})

        # Get total timesteps for training (max steps, not current)
        total_timesteps = algorithm.config.get("total_timesteps", result.get("timesteps_total", 0))

        # Get annealing parameters
        beta_start = ra_config.get("beta_start", 0.0)
        beta_start_non_graph = ra_config.get("beta_non_graph_edges_start", 0.0)
        beta_end = ra_config.get("beta_end", ra_config.get("beta", 1.0))
        beta_end_non_graph = ra_config.get("beta_non_graph_edges_end", ra_config.get("beta", 1.0))
        beta_anneal_timesteps = ra_config.get("beta_anneal_timesteps", total_timesteps)

        tau_start = ra_config.get("tau_start", 1.0)
        tau_end = ra_config.get("tau_end", ra_config.get("temperature", 1.0))
        tau_anneal_timesteps = ra_config.get("tau_anneal_timesteps", total_timesteps)

        # Get current timesteps
        current_timesteps = result.get("timesteps_total", 0)

        # Compute annealed beta using cosine decay schedule
        new_beta = self.cosine_decay_schedule(
            current_timesteps, beta_anneal_timesteps, beta_start, beta_end
        )

        # compute annealed beta for non-graph edges
        new_beta_non_graph_edges = self.cosine_decay_schedule(
            current_timesteps, beta_anneal_timesteps, beta_start_non_graph, beta_end_non_graph
        )

        # Compute annealed tau using cosine decay schedule
        new_tau = self.cosine_decay_schedule(
            current_timesteps, tau_anneal_timesteps, tau_start, tau_end
        )

        # Update beta and tau in all policies and models
        def update_annealing_params(worker):
            policy = worker.policy_map.get("reinforcement_learning_policy")
            if policy and hasattr(policy, 'current_beta'):
                # Update policy attributes (for logging)
                policy.current_tau = new_tau

                # Update model's GumbelSoftmax tau (for functional effect)
                policy.current_beta_graph = new_beta
                policy.current_beta_non_graph_edges = new_beta_non_graph_edges
                if hasattr(policy, 'model') and hasattr(policy.model, 'set_tau'):
                    policy.model.set_tau(new_tau)

        # Update on local worker
        update_annealing_params(algorithm.workers.local_worker())

        # Update on remote workers
        algorithm.workers.foreach_worker(update_annealing_params)


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
        # print(f'ALL TRIAL METRICS {result}')
        result = result or trial.last_result
        # skip for now since this is already printed after tuning... Perhaps move?
        trial_id = str(trial)
        eval_res = result["evaluation"]
        train_res = result["custom_metrics"]
        # Print the table
        headers = ["trial name",
                   "iter",
                   "total time",
                   "ts",
                   "agent_interactions",
                   "EVAL g2op_end",
                   "EVAL reward",
                   "TRAIN g2op_end",
                   "TRAIN ep_duration",
                   "TRAIN reward",
                   "episodes_this_iter"]
        table = [[trial_id,
                  result['training_iteration'],
                  _get_time_str(self._start_time, time.time())[1],
                  result["timesteps_total"],
                  result["custom_metrics"].get("total_agent_interact", "N/A"),
                  eval_res["custom_metrics"].get("grid2op_end_mean", "N/A"),
                  eval_res["episode_reward_mean"],
                  train_res.get("grid2op_end_mean", "N/A"),
                  train_res.get("corrected_ep_len_mean", "N/A"),
                  result["episode_reward_mean"],
                  result["episodes_this_iter"]]]
        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))
