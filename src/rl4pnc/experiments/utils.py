"""
Utilities in the grid2op experiments.
"""
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, List, OrderedDict, Union

from omegaconf import DictConfig

import grid2op
import numpy as np
import ray
from grid2op.Environment import BaseEnv
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.models import ModelCatalog
from ray.tune.experiment import Trial
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper.stopper import Stopper
from tabulate import tabulate

from src.algorithms.custom_ppo import CustomPPO
from src.algorithms.optuna_search import MyOptunaSearch
from src.rarl_rllib import make_rarl_policy, RARLModel
from src.rarl_rllib.model import GNNBaselineModel
from src.rl4pnc.experiments.callback import Style, TuneCallback
from src.rl4pnc.evaluation.evaluate_rllib_agent import evaluate_rllib_checkpoint

# Configure logging
logger = logging.getLogger(__name__)

# register custom components
POLICIES["rappo_torch_policy"] = make_rarl_policy(PPOTorchPolicy)
ModelCatalog.register_custom_model("gnn_model", GNNBaselineModel)
ModelCatalog.register_custom_model("ragnn_model", RARLModel)


def get_num_available_episodes(env_name: str) -> int:
    """
    Get the number of available episodes for a given environment.

    :param env_name: Name of the Grid2Op environment
    :return: Number of available episodes
    """
    try:
        chronics_path = os.path.join(
            f"{grid2op.get_current_local_dir()}",
            env_name,
            "chronics"
        )
        if os.path.exists(chronics_path):
            num_episodes = len(os.listdir(chronics_path))
            logger.info(f"Found {num_episodes} available episodes for environment {env_name}")
            return num_episodes
        else:
            logger.warning(f"Chronics path not found: {chronics_path}. Defaulting to 50 episodes.")
            return 50
    except Exception as e:
        logger.warning(f"Error counting episodes: {e}. Defaulting to 50 episodes.")
        return 50


def calculate_action_space_asymmetry(env: BaseEnv, add_dn: bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies without symmetries.
    """

    nr_substations = len(env.sub_info)

    logging.info("no symmetries")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )

        alpha = 2 ** (nr_elements - 1) - (2 ** nr_non_lines - 1)
        action_space += alpha if alpha > 1 else 0
        # if alpha > 1:  # without do nothings for single substations
        if (add_dn and alpha > 0) or (alpha > 1):
            controllable_substations[sub] = alpha
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(env: BaseEnv, add_dn: bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following Subrahamian (2021).
    """
    nr_substations = len(env.sub_info)

    logging.info("medha")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        alpha = 2 ** (nr_elements - 1)
        beta = nr_elements - (1 if nr_elements == 2 else 0)
        gamma = 2 ** nr_non_lines - 1 - nr_non_lines
        combined = alpha - beta - gamma
        action_space += combined if combined > 1 else 0
        # if combined > 1:  # without do nothings for single substations
        if (add_dn and combined > 0) or (combined > 1):
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    print(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(env: BaseEnv, add_dn=False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following the proposed action space.
    """
    nr_substations = len(env.sub_info)

    logging.info("TenneT")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        nr_lines = nr_elements - nr_non_lines

        combined = (
                           (
                                   2 ** nr_non_lines - 2
                           )  # configuratations of non-lines except when all lines are same colour
                           * (
                                   2 ** nr_lines  # configurations of lines
                                   - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                                   - 2  # minus case where all lines have the same colour
                                   + (2 if nr_lines == 1 else 0)  # due to doubles with 1 line
                                   + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
                           )
                           + 2  # configurations where non-lines all have the same colour
                           * (
                                   2 ** nr_lines  # configurations of lines
                                   - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                                   - 1  # if all non-lines have the same colour, then if all lines are also this colour, it's allowed
                                   + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
                                   + (1 if nr_lines == 1 else 0)  # due to doubles with 1 line
                           )
                   ) / 2  # remove symmetries

        action_space += int(combined) if combined > 1 else 0
        if (add_dn and combined > 0) or (combined > 1):  # combined > 1: without do nothings for single substations
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def get_capa_substation_id(
        line_info: dict[int, list[int]],
        obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        controllable_substations: dict[int, int],
) -> list[int]:
    """
    Returns the substation id of the substation to act on according to CAPA.
    """
    # calculate the mean rho per substation
    connected_rhos: dict[int, list[float]] = {agent: [] for agent in line_info}
    for sub_idx in line_info:
        for line_idx in line_info[sub_idx]:
            if isinstance(obs_batch, OrderedDict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][0][line_idx]
                    # obs_batch["original_obs"]["rho"][0][line_idx]
                )
            elif isinstance(obs_batch, dict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][line_idx]
                    # obs_batch["original_obs"]["rho"][line_idx]
                )
            else:
                raise ValueError("The observation batch is not supported.")
    for sub_idx in connected_rhos:
        connected_rhos[sub_idx] = [float(np.mean(connected_rhos[sub_idx]))]

    # set non-controllable substations to 0
    for sub_idx in connected_rhos:
        if sub_idx not in list(controllable_substations.keys()):
            connected_rhos[sub_idx] = [0.0]

    # order the substations by the mean rho, maximum first
    connected_rhos = dict(
        sorted(connected_rhos.items(), key=lambda item: item[1], reverse=True)
    )

    # # find substation with max average rho
    # max_value = max(connected_rhos.values())
    # return [key for key, value in connected_rhos.items() if value == max_value][0]

    # return the ordered entries
    # NOTE: When there are two equal max values, the first one is returned first
    return list(connected_rhos.keys())


def find_list_of_agents(env: BaseEnv, action_space: str) -> dict[int, int]:
    """
    Function that returns the number of controllable substations.
    """
    add_dn = "dn" in action_space
    if action_space.startswith("asymmetry"):
        _, _, list_of_agents = calculate_action_space_asymmetry(env, add_dn)
        return list_of_agents
    if action_space.startswith("medha"):
        _, _, list_of_agents = calculate_action_space_medha(env, add_dn)
        return list_of_agents
    if action_space.startswith("tennet"):
        _, _, list_of_agents = calculate_action_space_tennet(env, add_dn)
        return list_of_agents
    raise ValueError("The action space is not supported.")


def find_substation_per_lines(
        env: BaseEnv, list_of_agents: list[int]
) -> dict[int, list[int]]:
    """
    Returns a dictionary connecting line ids to substations.
    """
    line_info: dict[int, list[int]] = {agent: [] for agent in list_of_agents}
    for sub_idx in list_of_agents:
        for or_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_or_id"
        ]:
            line_info[sub_idx].append(or_id)
        for ex_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_ex_id"
        ]:
            line_info[sub_idx].append(ex_id)

    return line_info


def delete_nested_key(d, path):
    keys = path.split('/')
    current = d

    # Traverse through the dictionary using keys from the path
    for key in keys[:-1]:  # Iterate until the second last key
        if key in current:
            current = current[key]
        else:
            return  # If any key is missing, return without making changes

    # Now current points to the dictionary containing the key to be deleted
    last_key = keys[-1]
    if last_key in current:
        del current[last_key]


class MaxCustomMetricStopper(Stopper):
    """Stop trials after reaching a maximum value for the custom metric

    Args:
        metric: Metric to use.
        max_value: If custom metric reaches this value stop trials
    """

    def __init__(self, metric: str, max_value: int):
        self._max_value = max_value
        self._custom_Metric = metric

    def __call__(self, trial_id: str, result: Dict):
        print("current value custom metric: ", result["custom_metric"][self._custom_Metric])
        return result["custom_metric"][self._custom_Metric] >= self._max_value

    def stop_all(self):
        return False


class TimeStopper(Stopper):
    def __init__(self, deadline):
        self._start = time()
        if isinstance(deadline, str):
            self._deadline = int(deadline.split(":")[0]) * 3600 + int(deadline.split(":")[1]) * 60
            print("Run training for ", deadline, " hours.")
        else:
            self._deadline = deadline * 60  # Stop all trials after deadline minutes
            print("Run training for ", deadline, " minutes.")

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time() - self._start > self._deadline


def get_duration(experiment_cfg: DictConfig) -> int | None:
    """Return the wall-clock training budget in seconds, or None to run until nb_timesteps."""
    duration = experiment_cfg.duration
    if duration is None:
        print(f"Run until {experiment_cfg.nb_timesteps} agent time steps.")
        return None
    if isinstance(duration, str):
        h, m = duration.split(":")
        seconds = int(h) * 3600 + int(m) * 60
    else:
        seconds = int(duration) * 60
    if seconds == 0:
        print(f"Run until {experiment_cfg.nb_timesteps} agent time steps.")
        return None
    print(f"Run training for {seconds}s.")
    return seconds


def trial_str_creator(trial: Trial, job_id=""):
    # Don't modify trial.trial_id as it breaks Optuna's internal tracking!
    # Just create a custom display name
    base_id = trial.trial_id.split("_")[0]
    if job_id:
        custom_id = "{}_{}".format(job_id, base_id)
    else:
        custom_id = base_id
    print('Creating trial with ID: ', custom_id)
    return "{}_{}".format(trial.trainable_name, custom_id)


def trial_dir_name(trial: Trial):
    print("Trial name is: ", trial.custom_trial_name)
    return "{}_{}".format(trial.custom_trial_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def print_details(rllib_cfg: Dict[str, Any]) -> None:
    print("Using reward function: ", rllib_cfg["env_config"]["grid2op_kwargs"]["reward_class"].__class__.__name__)
    print("Using action space:    ", rllib_cfg["env_config"]["action_space"])
    print("Using observation space:", rllib_cfg["env_config"]["observation_space"])

def run_training(rllib_cfg: dict[str, Any], cfg: DictConfig, job_id: str) -> ResultGrid:
    """Run RLLib PPO training driven by the Hydra config.

    Args:
        rllib_cfg: Flat RLLib algorithm config dict (built by build_rllib_config).
        cfg:       Full assembled Hydra DictConfig (cfg.experiment, cfg.optimization, …).
        job_id:    Unique identifier for this run (e.g. SLURM job id).
    """
    exp = cfg.experiment
    opt = cfg.optimization

    # --- Init Ray ---
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    print(f"Ray's temporary directory: {ray._private.utils.get_ray_temp_dir()}")
    ray.init(local_mode=exp.ray_local_mode)
    print(f"Ray initialized in {'local' if exp.ray_local_mode else 'cluster'} mode.")

    # --- Optuna search (optional) ---
    algo = None
    asha = None
    if opt.enable:
        algo = MyOptunaSearch(
            metric=opt.score_metric,
            mode=opt.mode,
            points_to_evaluate=[opt.points_to_evaluate] if opt.points_to_evaluate is not None else None,
        )
        if opt.load_from is not None:
            print("Retrieving previous Optuna results from: ", opt.load_from)
            algo.restore_from_dir(opt.load_from)
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(rllib_cfg, key)
                else:
                    rllib_cfg.pop(key, None)

        asha = ASHAScheduler(
            time_attr="timesteps_total",
            max_t=exp.nb_timesteps,
            grace_period=max(1, exp.nb_timesteps // 2),
            reduction_factor=5,
        )

    dur = get_duration(exp)
    time_budget = int(dur * 0.9) if (opt.enable and dur) else None
    if time_budget:
        print(f"Optimization time budget: {time_budget}s ({time_budget / 3600:.2f}h, 10% buffer for cleanup)")

    storage_path = os.path.abspath(os.path.join(os.getcwd(), "results", "experiments"))
    os.makedirs(storage_path, exist_ok=True)

    rllib_cfg["total_timesteps"] = exp.nb_timesteps

    # --- Build TuneConfig ---
    shared_tune_kwargs = dict(
        trial_name_creator=lambda t: trial_str_creator(t, job_id),
        trial_dirname_creator=trial_dir_name,
    )
    if opt.enable:
        tune_config = tune.TuneConfig(
            **shared_tune_kwargs,
            search_alg=algo,
            scheduler=asha,
            metric=opt.score_metric,
            mode=opt.mode,
            num_samples=opt.num_trials or -1,
            time_budget_s=time_budget,
        )
    else:
        tune_config = tune.TuneConfig(**shared_tune_kwargs)

    # --- Build Tuner ---
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=rllib_cfg,
        run_config=air.RunConfig(
            name=exp.experiment_name,
            storage_path=storage_path,
            stop={"timesteps_total": exp.nb_timesteps},
            callbacks=[
                TuneCallback(
                    exp.my_log_level,
                    opt.score_metric,
                    eval_freq=rllib_cfg["evaluation_interval"],
                    heartbeat_freq=60,
                ),
            ],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=exp.checkpoint_freq,
                checkpoint_at_end=True,
                checkpoint_score_attribute=opt.score_metric,
                num_to_keep=5,
            ),
            verbose=exp.verbose,
        ),
        tune_config=tune_config,
    )

    print_details(rllib_cfg)

    # --- Launch ---
    try:
        result_grid = tuner.fit()
    except Exception:
        traceback.print_exc()
        exit()
    finally:
        ray.shutdown()

    # --- Print checkpoint summary for each trial ---
    for i, result in enumerate(result_grid):
        if not result.error:
            checkpoints_tojson = {
                os.path.basename(checkpoint.path): metrics['evaluation']['custom_metrics']
                for checkpoint, metrics in result.best_checkpoints
            }
            with open(os.path.join(result.path, "checkpoint_results.json"), "w") as f:
                json.dump(checkpoints_tojson, f)
            try:
                print(
                    Style.BOLD + f" *---- Trial {i} finished successfully ---*\n" + Style.END +
                    tabulate(
                        [[k] + list(v.values()) for k, v in checkpoints_tojson.items()],
                        headers=['checkpoint'] + list(result.metrics['evaluation']['custom_metrics'].keys()),
                        tablefmt='rounded_grid',
                    )
                )
            except Exception as e:
                print("Could not print checkpoint results table: ", e)
        else:
            print(f"Trial {i} failed with error {result.error}.")

    # --- Find best result ---
    if opt.enable:
        try:
            best_result = result_grid.get_best_result(metric=opt.score_metric, mode=opt.mode)
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}{'=' * 80}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}ERROR: Could not find best trial for metric '{opt.score_metric}'{Style.END}")
            print(f"{Style.RED}Original error: {str(e)}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}{'=' * 80}{Style.END}\n")
            return result_grid

        # Print best hyperparameters
        best_model_cfg = best_result.config["model"]["custom_model_config"]
        best_ra_cfg = best_result.config["relation_awareness"]
        rows = [
            [f"model.custom_model_config.{k}", v]
            for k, v in best_model_cfg.items()
            if not isinstance(v, dict)
        ] + [
            [f"relation_awareness.{k}", v]
            for k, v in best_ra_cfg.items()
            if not isinstance(v, dict)
        ]
        print(f"\n{Style.BOLD}{'=' * 80}{Style.END}")
        print(f"{Style.BOLD}Best hyperparameters found:{Style.END}")
        print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="rounded_grid", floatfmt=".3f"))
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Corresponding checkpoint: ", checkpoint_dir)

        # Save Optuna study
        if algo is not None:
            optuna_path = os.path.join(storage_path, exp.experiment_name, f"optuna_results_{job_id}")
            study_name = f"{exp.experiment_name}_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db_path = algo.save_study(optuna_path, study_name)
            tune_path = os.path.join(storage_path, exp.experiment_name, "tune_results")
            os.makedirs(tune_path, exist_ok=True)
            algo.save_to_dir(tune_path, f"tune_checkpoint_{job_id}")
            print(f"\n{Style.BOLD}Optuna study saved to: {db_path}{Style.END}")
            print(f"  optuna-dashboard sqlite:///{db_path}")
            print(f"{Style.BOLD}{'=' * 80}{Style.END}\n")
    else:
        try:
            best_result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}ERROR: Could not find best trial — {e}{Style.END}\n")
            return result_grid
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Best checkpoint: ", checkpoint_dir)

    # --- Post-training evaluation ---
    post_eval = exp.post_training_evaluation
    if post_eval.enabled:
        print(f"\n{Style.BOLD}{'=' * 80}{Style.END}")
        print(f"{Style.BOLD}Evaluating best checkpoint...{Style.END}")
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            checkpoint_name = os.path.basename(checkpoint_dir)
            num_episodes = (
                get_num_available_episodes(post_eval.env_name)
                if post_eval.num_episodes in ("all", None)
                else int(post_eval.num_episodes)
            )
            print(f"Evaluation environment: {post_eval.env_name}  |  Episodes: {num_episodes}")
            try:
                evaluate_rllib_checkpoint(
                    checkpoint_path=Path(checkpoint_dir).parent,
                    policy_name="reinforcement_learning_policy",
                    checkpoint_name=checkpoint_name,
                    env_name_override=post_eval.env_name,
                    num_episodes=num_episodes,
                    visualize=post_eval.visualize,
                )
                print(f"{Style.BOLD}Evaluation completed successfully!{Style.END}")
            except Exception as e:
                print(f"{Style.BOLD}Warning: Evaluation failed: {e}{Style.END}")
                traceback.print_exc()
        print(f"{Style.BOLD}{'=' * 80}{Style.END}\n")

    return result_grid
