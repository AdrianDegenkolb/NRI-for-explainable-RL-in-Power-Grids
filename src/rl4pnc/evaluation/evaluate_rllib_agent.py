"""
Script to load and evaluate an RLlib agent on Grid2Op environment.
Displays results in a boxplot similar to visualize_agent_eval_results.ipynb

IMPORTANT: This script loads a trained RLlib policy and evaluates it.
The key challenge is that RLlib's Policy.from_checkpoint() needs the correct
observation space structure. For multi-agent environments, the policy was trained
with a specific observation space (e.g., BusConnectivityGraphObsSpace), but it's
wrapped in a multi-agent Dict space during training.

When loading for evaluation, we need to provide the exact observation space structure
that the policy expects.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from ray.rllib.models import ModelCatalog

from src.common.baseline_agent import evaluate_agent
from src.common.constants import SEED
from src.ra_agents.RAFeatureExtractor import RLlibGNNModel, RLlibRAGNNModel, RLlibNRIGNNModel
from src.rl4pnc.evaluation.evaluation_agents import RllibAgent
from src.rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from src.visualization import get_evaluation_metrics, visualize_agent_survival

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(checkpoint_path: str) -> dict:
    """
    Load configuration from params.json in the checkpoint directory.

    :param checkpoint_path: Path to the experiment directory or checkpoint subdirectory containing params.json
    :return: Environment configuration dictionary
    """
    checkpoint_path = Path(checkpoint_path)

    # Try to find params.json in checkpoint_path or its parent
    params_path = checkpoint_path / "params.json"
    if not params_path.exists():
        # If checkpoint_path is a checkpoint subdirectory (e.g., checkpoint_000000),
        # look in the parent directory (the experiment directory)
        params_path = checkpoint_path.parent / "params.json"
        if not params_path.exists():
            raise FileNotFoundError(f"params.json not found at {checkpoint_path / 'params.json'} or {params_path}")
        logger.info(f"Found params.json in parent directory: {params_path}")

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Get the base training environment config
    if "env_config" not in params:
        raise ValueError("No env_config found in params.json")

    env_config = params["env_config"]

    # Clean up the config - remove serialized object strings that can't be used directly
    # These will be recreated by the environment
    if "grid2op_kwargs" in env_config:
        grid2op_kwargs = env_config["grid2op_kwargs"]
        # Remove serialized class/object references as they need to be recreated
        keys_to_remove = []
        for key, value in grid2op_kwargs.items():
            if isinstance(value, str) and (value.startswith("<class") or value.startswith("<")):
                keys_to_remove.append(key)
                logger.debug(f"Removing serialized object: {key} = {value}")

        for key in keys_to_remove:
            del grid2op_kwargs[key]

        logger.debug(f"Cleaned {len(keys_to_remove)} serialized objects from grid2op_kwargs")

    evaluation_env_config = params["evaluation_config"]["env_config"]
    # Clean up the config - remove serialized object strings that can't be used directly
    # These will be recreated by the environment
    if "grid2op_kwargs" in evaluation_env_config:
        grid2op_kwargs = evaluation_env_config["grid2op_kwargs"]
        # Remove serialized class/object references as they need to be recreated
        keys_to_remove = []
        for key, value in grid2op_kwargs.items():
            if isinstance(value, str) and (value.startswith("<class") or value.startswith("<")):
                keys_to_remove.append(key)
                logger.debug(f"Removing serialized object: {key} = {value}")

        for key in keys_to_remove:
            del grid2op_kwargs[key]

        logger.debug(f"Cleaned {len(keys_to_remove)} serialized objects from grid2op_kwargs")

    env_config["lib_dir"] = os.getcwd()
    evaluation_env_config["lib_dir"] = os.getcwd()
    params["env_config"] = env_config
    params["evaluation_config"]["env_config"] = evaluation_env_config
    params["evaluation_config"]["env_config"]["observation_space"] = params["env_config"]["observation_space"]
    return params


def create_gym_wrapper_from_config(env_config: dict):
    """
    Create a CustomizedGrid2OpEnvironment and return its gym wrapper.

    :param env_config: Environment configuration dictionary
    :return: GymEnv instance from CustomizedGrid2OpEnvironment
    """
    # Create the custom environment which includes the properly configured gym wrapper
    custom_env = CustomizedGrid2OpEnvironment(env_config)
    return custom_env.env_gym


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
    :param policy_name: Name of the policy (e.g., "reinforcement_learning_policy")
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


def evaluate_rllib_checkpoint(
    checkpoint_path: str,
    policy_name: str = "reinforcement_learning_policy",
    checkpoint_name: str = "checkpoint_000000",
    env_name_override: str = None,
    num_episodes: int = 50,
    visualize: bool = True,
    save_to_path: Optional[Path] = None
):
    """
    Evaluate an RLlib checkpoint on a Grid2Op environment.

    :param checkpoint_path: Path to the experiment directory containing checkpoints
    :param policy_name: Name of the policy (default: "reinforcement_learning_policy")
    :param checkpoint_name: Name of the checkpoint folder (default: "checkpoint_000000")
    :param env_name_override: Override environment name for evaluation (default: None, uses params.json)
    :param num_episodes: Number of evaluation episodes (default: 50)
    :param visualize: Whether to show visualization after evaluation (default: True)
    :param save_to_path: Optional path to save results (default: None, saves in checkpoint directory)
    :return: Path to results directory
    """
    # Register custom models before loading checkpoint
    ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
    ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
    ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)

    # Load environment configuration from params.json
    try:
        params = load_config(checkpoint_path)
        env_config = params["env_config"]

        # Override env_name if specified
        if env_name_override:
            env_config["env_name"] = env_name_override
            logger.info(f"Overriding env_name to: {env_name_override}")

        env_name = env_config["env_name"]

    except Exception as e:
        logger.error(f"Failed to load env_config from params.json: {e}")
        logger.info("Falling back to manual configuration")

        # Fallback to manual configuration
        env_name = "l2rpn_case14_sandbox_val"
        env_config = {
            "env_name": env_name,
            "action_space": "medha",
            "mask": 5,
            "lib_dir": ".",
            "grid2op_kwargs": {},
            "seed": SEED,
            "rho_threshold": 0.95,
            "n_history": 1,
            "g2op_input": ["r", "t"],
            "custom_input": ["d"],
            "observation_space": "BusConnectivityGraphObsSpace",
            "danger": 0.9,
            "prio": False,
            "use_ffw": False,
            "reset_topo": 0.9,
            "line_reco": True,
            "line_disc": False,
            "penalty_game_over": 0,
            "reward_finish": 0,
            "curriculum_training": False,
            "rules": {
                "activation_threshold": 0.95,
                "line_reco": True,
                "line_disc": False,
                "reset_topo": 0.9,
                "simulate": True
            }
        }

    # Results path
    results_path = Path(checkpoint_path) / "evaluations" if save_to_path is None else save_to_path
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading agent from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Policy: {policy_name}")

    gym_wrapper = None
    try:
        # Load the agent and keep gym_wrapper alive to prevent double-close
        agent, g2op_env, gym_wrapper = load_rllib_agent(
            checkpoint_path=checkpoint_path,
            policy_name=policy_name,
            checkpoint_name=checkpoint_name,
            env_name=env_name,
            env_config=env_config
        )

        logger.info(f"Agent loaded successfully!")

        # Validate observation space restoration
        from src.common.observation_space import BusConnectivityGraphObsSpace
        obs_space = agent._rllib_agent.observation_space

        if hasattr(obs_space, 'spaces') and 'reinforcement_learning_agent' in obs_space.spaces:
            rl_obs_space = obs_space.spaces['reinforcement_learning_agent']

            if isinstance(rl_obs_space, BusConnectivityGraphObsSpace):
                logger.info(f"✓ Observation space correctly restored as BusConnectivityGraphObsSpace")
                logger.info(f"  - x_dim: {rl_obs_space.x_dim}")
                logger.info(f"  - num_nodes: {rl_obs_space.num_nodes}")
                logger.info(f"  - max_num_edges: {rl_obs_space.max_num_edges}")
                logger.info(f"  - e_dim: {rl_obs_space.e_dim}")

                # Check for normalization parameters
                if hasattr(rl_obs_space, 'normalization_min') and rl_obs_space.normalization_min is not None:
                    logger.info(f"  - Normalization parameters present: Yes")
                else:
                    logger.info(f"  - Normalization parameters present: No")
            else:
                logger.warning(f"⚠ Observation space is {type(rl_obs_space).__name__}, expected BusConnectivityGraphObsSpace")
                logger.warning(f"  Custom attributes may be missing!")

        logger.info(f"Evaluating on environment: {env_name}")
        logger.info(f"Number of episodes: {num_episodes}")

        # Evaluate the agent
        evaluate_agent(
            agent=agent,
            env=g2op_env,
            path_results=results_path,
            num_episodes=num_episodes,
            verbose=True
        )

        logger.info("Evaluation completed!")

        # Load and visualize results
        if visualize:
            logger.info("Generating visualization...")
            metrics = get_evaluation_metrics(results_path, "RLlib Agent")
            visualize_agent_survival([metrics], show=True)

        logger.info(f"Results saved to: {results_path}")

        return results_path

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
    finally:
        # Clean up gym_wrapper which will handle environment cleanup
        if gym_wrapper is not None:
            try:
                # Close the gym environment properly
                gym_wrapper.env_gym.close()
            except Exception as cleanup_error:
                # Ignore errors during cleanup (e.g., already closed)
                logger.debug(f"Environment cleanup error (ignored): {cleanup_error}")


def main():
    """Main evaluation script when running as standalone."""

    # Configuration
    checkpoint_path = "/home/adrian/Schreibtisch/1301_THIS_rappo_with_anneal_2/CustomPPO_0_107a0_2026-01-14_22-36-19/"
    policy_name = "reinforcement_learning_policy"
    checkpoint_name = "checkpoint_000020"

    # Optional: override env_name for evaluation (otherwise uses the one from params.json)
    env_name_override = "l2rpn_case14_sandbox_val"  # Set to "l2rpn_case14_sandbox_val" to override
    num_episodes = 50  # Number of evaluation episodes

    # Call the reusable evaluation function
    evaluate_rllib_checkpoint(
        checkpoint_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        env_name_override=env_name_override,
        num_episodes=num_episodes,
        visualize=True
    )


if __name__ == "__main__":
    ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
    ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
    ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)
    main()

