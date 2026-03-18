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

import grid2op
import numpy as np
import ray
from grid2op.Environment import BaseEnv
from ray import air, tune
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.models import ModelCatalog
from ray.tune.experiment import Trial
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper.stopper import Stopper
from tabulate import tabulate

from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.ra_agents.ppo.rllib.rappo.RAPPO import RAPPOTorchPolicy
from src.rl4pnc.algorithms.custom_ppo import CustomPPO
from src.rl4pnc.algorithms.optuna_search import MyOptunaSearch
from src.rl4pnc.experiments.callback import Style, TuneCallback
from src.rl4pnc.evaluation.evaluate_rllib_agent import evaluate_rllib_checkpoint

# Configure logging
logger = logging.getLogger(__name__)

# register custom components
POLICIES["rappo_torch_policy"] = RAPPOTorchPolicy
ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)


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


def get_duration(setup):
    duration = setup.get("duration", None)
    # convert duration to seconds
    if duration is None or duration == 0:
        print(f"Run until {setup['nb_timesteps']} agent time steps.")
        return duration
    if isinstance(duration, str):
        duration = int(duration.split(":")[0]) * 3600 + int(duration.split(":")[1]) * 60
    else:
        duration = duration * 60  # Stop all trials after duration minutes
    print("Run training for ", duration, " seconds.")
    return duration


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


def print_details(custom_model_config: Dict[str, Any], setup: Dict[str, Any]):
    print("Using reward function: ", custom_model_config["env_config"]["grid2op_kwargs"]["reward_class"].__class__.__name__)
    print("Using action space: ", custom_model_config["env_config"]["action_space"])
    print("Using observation space: ", custom_model_config["env_config"]["observation_space"])

def run_training(config: dict[str, Any], setup: dict[str, Any], job_id: str) -> ResultGrid:
    """
    Function that runs the training script.
    """
    # init ray
    # Set the environment variable
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    tmp_dir = ray._private.utils.get_ray_temp_dir()
    print(f"Ray's temporary directory: {tmp_dir}")
    local_mode = setup.get("ray_local_mode", False)
    ray.init(local_mode=local_mode)
    print(f"Ray initialized in {'local' if local_mode else 'cluster'} mode.")

    # whether to perform hyperparameter optimization
    do_optimization = setup['optimization']['enable']

    # Use Optuna search algorithm to find good working parameters
    algo = None
    if do_optimization:
        points_to_eval = setup['optimization'].get('points_to_evaluate', None)
        algo = MyOptunaSearch(
            metric=setup['optimization']["score_metric"],
            mode=setup['optimization']["mode"],
            points_to_evaluate=[points_to_eval] if points_to_eval is not None else None,
        )
        if setup['optimization'].get("load_from", None) is not None:
            print("Retrieving results old experiment from : ", setup['optimization']['load_from'])
            algo.restore_from_dir(setup['optimization']['load_from'])
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(config, key)
                else:
                    del config[key]

        asha = ASHAScheduler(
            time_attr="timesteps_total", # must be monotonic with training iterations
            max_t=setup["nb_timesteps"],  # same unit as time_attr
            grace_period=max(1, setup["nb_timesteps"] // 2),  # or another warmup in timesteps
            reduction_factor=5,
        )

    dur = get_duration(setup)

    # Get time budget for entire optimization (different from per-trial duration)
    time_budget = None
    if do_optimization:
        # For optimization, calculate time budget from duration if specified
        if dur:
            # Leave some buffer time (10%) for cleanup before SLURM kills the job
            time_budget = int(dur * 0.9)
            print(f"Setting time budget for optimization to {time_budget}s ({time_budget/3600:.2f} hours) with 10% buffer for cleanup")

    storage_path = os.path.abspath(os.path.join(setup.get("workdir", "."), "results", "experiments"))
    os.makedirs(storage_path, exist_ok=True)

    # Add total timesteps to config for use in callbacks
    config["total_timesteps"] = setup["nb_timesteps"]

    # Create tuner
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=config,
        run_config=air.RunConfig(
            name=setup["experiment_name"],
            storage_path=storage_path,
            stop={"timesteps_total": setup["nb_timesteps"]},  # Stop condition for individual trials
            callbacks=[
                TuneCallback(
                    setup["my_log_level"],
                    "evaluation/custom_metrics/grid2op_end_mean",
                    eval_freq=config["evaluation_interval"],
                    heartbeat_freq=60,
                ),
            ],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute=setup['optimization']["score_metric"],
                num_to_keep=5,
            ),
            verbose=setup["verbose"],
        ),
        tune_config=tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_dir_name(t),
            search_alg=algo,
            scheduler=asha,
            metric=setup['optimization']["score_metric"],
            mode=setup['optimization']["mode"],
            num_samples=setup['optimization'].get("num_trials", -1) or -1,
            time_budget_s=time_budget,
        ) if do_optimization else
        tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_dir_name(t),
        ),
    )

    print_details(config, setup)

    # Launch tuning
    try:
        result_grid = tuner.fit()
    except Exception as e:
        print("Error during tuning:")
        traceback.print_exc()
        exit()
    finally:
        # Close ray instance
        ray.shutdown()

    for i in range(len(result_grid)):
        result = result_grid[i]
        if not result.error:
            # Print and save available checkpoints
            checkpoints_tojson = {
                os.path.basename(checkpoint.path): metrics['evaluation']['custom_metrics'] for
                checkpoint, metrics in result.best_checkpoints
            }
            with open(os.path.join(result.path, "checkpoint_results.json"), "w") as outfile:
                json.dump(checkpoints_tojson, outfile)
            try:
                print(Style.BOLD + f" *---- Trial {i} finished successfully with evaluation results ---*\n" + Style.END +
                      tabulate(
                          [[k] + list(v.values()) for k, v in checkpoints_tojson.items()],
                          headers=['checkpoint'] + list(result.metrics['evaluation']['custom_metrics'].keys()),
                          tablefmt='rounded_grid')
                      )
            except Exception as e:
                print("Could not print checkpoint results table: ", e)
        else:
            print(f"Trial failed with error {result.error}.")

    # If Optuna optimization was enabled, save results summary
    if do_optimization:
        try:
            best_result = result_grid.get_best_result(metric=setup["optimization"]["score_metric"], mode="max")
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}{'='*80}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}ERROR: Could not find best trial for metric '{setup['optimization']['score_metric']}'{Style.END}")
            print(f"{Style.RED}This usually means:{Style.END}")
            print(f"{Style.RED}  1. No trials completed successfully{Style.END}")
            print(f"{Style.RED}  2. The metric was never reported (check if evaluation is enabled){Style.END}")
            print(f"{Style.RED}  3. All trials failed before reporting any results{Style.END}")
            print(f"\n{Style.RED}Original error: {str(e)}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}{'='*80}{Style.END}\n")

            # Check if any trials completed
            if len(result_grid) == 0:
                print(f"{Style.RED}No trials were run. Check the configuration.{Style.END}")
            else:
                print(f"{Style.YELLOW}Found {len(result_grid)} trial(s), but none reported the required metric.{Style.END}")
                print(f"{Style.YELLOW}Check that evaluation is enabled and runs at least once during training.{Style.END}")

            return result_grid

        config = best_result.config["model"]["custom_model_config"]
        relation_awareness_config = best_result.config["relation_awareness"]
        config.update({"relation_awareness": relation_awareness_config})

        rows = [
            [f"{module}.{param}", value]
            for module, params in config.items()
            for param, value in params.items()
        ]
        table = tabulate(rows, headers=["Parameter", "Value"], tablefmt="rounded_grid", floatfmt=".3f", )
        print(f"\n{Style.BOLD}{'='*80}{Style.END}")
        print(f"{Style.BOLD}Best hyperparameters found:{Style.END}")
        print(table)

        # Print checkpoint location
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Corresponding checkpoint can be found under ", checkpoint_dir)

        # Save the Optuna study to a SQLite database for dashboard access
        if algo is not None:
            optuna_path = os.path.join(storage_path, setup['experiment_name'], f"optuna_results_{job_id}")
            study_name = f"{setup['experiment_name']}_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db_path = algo.save_study(optuna_path, study_name)
            tune_path = os.path.join(storage_path, setup['experiment_name'], f"tune_results")
            os.makedirs(tune_path, exist_ok=True)
            algo.save_to_dir(tune_path, f"tune_checkpoint_{job_id}")
            print(f"\n{Style.BOLD}{'='*80}{Style.END}")
            print(f"{Style.BOLD}Optuna study saved to: {db_path}{Style.END}")
            print(f"{Style.BOLD}To view in Optuna Dashboard, run:{Style.END}")
            print(f"  optuna-dashboard sqlite:///{db_path}")
            print(f"{Style.BOLD}{'='*80}{Style.END}\n")
    else:
        # if no optimization, get best result by episode reward
        try:
            best_result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}{'='*80}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}ERROR: Could not find best trial{Style.END}")
            print(f"{Style.RED}No trials completed successfully or reported metrics.{Style.END}")
            print(f"\n{Style.RED}Original error: {str(e)}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}{'='*80}{Style.END}\n")
            return result_grid

        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Best checkpoint can be found under ", checkpoint_dir)

    # evaluate best checkpoint
    eval_config = setup.get('post_training_evaluation', {})
    if eval_config.get('enabled', True):
        print(f"\n{Style.BOLD}{'='*80}{Style.END}")
        print(f"{Style.BOLD}Evaluating best checkpoint...{Style.END}")

        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print(f"Checkpoint directory: {checkpoint_dir}")
            checkpoint_name = os.path.basename(checkpoint_dir)
            checkpoint_dir = Path(checkpoint_dir).parent

            # Get evaluation environment name
            eval_env_name = eval_config.get('env_name', 'l2rpn_case14_sandbox_val')

            # Calculate number of episodes
            num_episodes_config = eval_config.get('num_episodes', 'all')
            if num_episodes_config == 'all' or num_episodes_config is None:
                num_episodes = get_num_available_episodes(eval_env_name)
            else:
                num_episodes = int(num_episodes_config)

            print(f"Evaluation environment: {eval_env_name}")
            print(f"Number of episodes: {num_episodes}")

            try:
                evaluate_rllib_checkpoint(
                    checkpoint_path=checkpoint_dir,
                    policy_name="reinforcement_learning_policy",
                    checkpoint_name=checkpoint_name,
                    env_name_override=eval_env_name,
                    num_episodes=num_episodes,
                    visualize=eval_config.get('visualize', False)
                )
                print(f"{Style.BOLD}Evaluation completed successfully!{Style.END}")
            except Exception as e:
                print(f"{Style.BOLD}Warning: Evaluation failed: {e}{Style.END}")
                traceback.print_exc()
        print(f"{Style.BOLD}{'='*80}{Style.END}\n")


    return result_grid
