"""
Load configs and agents from checkpoints for evaluation.
"""

import json
import logging
import os
from pathlib import Path

from ray.rllib.models import ModelCatalog

from agents import RllibAgent
from grid2op_env import CustomizedGrid2OpEnvironment

logger = logging.getLogger(__name__)


def load_config(checkpoint_path: Path) -> dict:
    """
    Find and load params.json from the checkpoint directory.

    :param checkpoint_path: Path to the experiment directory or checkpoint subdirectory containing params.json
    :return: Raw params dictionary as loaded from params.json
    """

    params_path = checkpoint_path / "params.json"
    if not params_path.exists():
        params_path = checkpoint_path.parent / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"params.json not found at {checkpoint_path / 'params.json'} or {params_path}")
        logger.info(f"Found params.json in parent directory: {params_path}")

    with open(params_path, 'r') as f:
        return json.load(f)


def preprocess_config(params: dict) -> dict:
    """
    Preprocess a raw params dictionary loaded from params.json.

    Removes serialized object strings that cannot be used directly (they will be
    recreated by the environment). Sets lib_dir to the current working directory,
    syncs the observation_space from the training config into the evaluation config,
    and optionally overrides the environment name.

    :param params: Raw params dictionary as returned by load_config
    :return: Preprocessed params dictionary
    """
    if "env_config" not in params:
        raise ValueError("No env_config found in params.json")

    def _remove_serialized_objects(_env_config: dict) -> None:
        if "grid2op_kwargs" not in _env_config:
            return
        grid2op_kwargs = _env_config["grid2op_kwargs"]
        keys_to_remove = [
            key for key, value in grid2op_kwargs.items()
            if isinstance(value, str) and value.startswith("<")
        ]
        for key in keys_to_remove:
            del grid2op_kwargs[key]
            logger.debug(f"Removing serialized object: {key}")
        logger.debug(f"Cleaned {len(keys_to_remove)} serialized objects from grid2op_kwargs")

    env_config = params["env_config"]
    evaluation_env_config = params["evaluation_config"]["env_config"]

    _remove_serialized_objects(env_config)
    _remove_serialized_objects(evaluation_env_config)

    env_config["lib_dir"] = os.getcwd()
    evaluation_env_config["lib_dir"] = os.getcwd()
    evaluation_env_config["observation_space"] = env_config["observation_space"]

    return params


def load_rllib_agent(
        checkpoint_path: str,
        policy_name: str,
        checkpoint_name: str,
        env_name: str,
        env_config: dict
):
    """
    Load an RLlib agent from a checkpoint.

    :param checkpoint_path: Path to the experiment directory containing checkpoints
    :param policy_name: Name of the policy (defaults to RL_POLICY, specified in core/constants.py)
    :param checkpoint_name: Name of the checkpoint folder (e.g., "checkpoint_000000")
    :param env_name: Name of the Grid2Op environment to evaluate on
    :param env_config: Environment configuration dictionary
    :return: Tuple of (RllibAgent, Grid2Op Environment, gym_wrapper)
    """
    # Add env_name to config for CustomizedGrid2OpEnvironment
    env_config["env_name"] = env_name

    # Create the CustomizedGrid2OpEnvironment to get proper observation/action spaces
    gym_wrapper = CustomizedGrid2OpEnvironment(env_config)

    # Get the underlying Grid2Op environment for evaluation
    g2op_env = gym_wrapper.env_gym.init_env

    # Re-register custom models (ray.shutdown() clears the ModelCatalog registry)
    from rarl_rllib import RAActorCriticModel, RASACTorchModel, RADQNTorchModel
    from rarl_rllib.ppo.gnn_ppo_model import GNNBaselineModel
    from rarl_rllib.sac.gnn_sac_model import GNNBaselineSACModel
    from rarl_rllib.dqn.gnn_dqn_model import GNNBaselineDQNModel
    ModelCatalog.register_custom_model("ra_actor_critic_model", RAActorCriticModel)
    ModelCatalog.register_custom_model("rasac_model", RASACTorchModel)
    ModelCatalog.register_custom_model("radqn_model", RADQNTorchModel)
    ModelCatalog.register_custom_model("gnn_model", GNNBaselineModel)
    ModelCatalog.register_custom_model("gnn_dqn_model", GNNBaselineDQNModel)
    ModelCatalog.register_custom_model("gnn_sac_model", GNNBaselineSACModel)

    # Load the RLlib agent
    agent = RllibAgent(
        action_space=g2op_env.action_space,
        env_config=env_config,
        file_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        gym_wrapper=gym_wrapper
    )

    # Return gym_wrapper to keep it alive and prevent premature cleanup
    return agent, g2op_env, gym_wrapper
