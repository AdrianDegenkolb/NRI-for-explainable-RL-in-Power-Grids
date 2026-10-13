"""
Trains an RL agent using Hydra for config composition.

Algorithm is selected via the training config group:
  training=ppo   (default) — CustomPPO, on-policy
  training=sac             — CustomSAC, off-policy
  training=dqn             — CustomDQN, off-policy, discrete actions

Usage examples
--------------
# Default RAGNN/PPO run:
  python train.py

# SAC with MLP model:
  python train.py training=sac model=mlp relation_awareness=disabled

# Quick test run (minimal timesteps, local Ray mode):
  python train.py experiment=test_minimal training=ppo_test

# MLP baseline:
  python train.py model=mlp obs_space=flat relation_awareness=disabled

# Optuna hyperparameter search:
  python train.py optimization=optuna

# Single-key override via CLI:
  python train.py experiment.nb_timesteps=500000 training.lr=3e-4
"""

import logging
import os
from typing import Any

import grid2op
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms import ppo, sac, dqn
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.policy.policy import PolicySpec

from core.constants import DO_NOTHING_POLICY, RL_POLICY, HIGH_LEVEL_POLICY, RAPPO_POLICY, RASAC_POLICY, RADQN_POLICY, \
    DQN_GNN_POLICY, DQN_MLP_POLICY, set_seed
from core.rllib_config import build_env_config, build_model_config, register_custom_models
from core.train import run_training
from grid2op_env import policy_mapping_fn
from grid2op_env.env import CustomizedGrid2OpEnvironment
from grid2op_env.multi_agent_policies.do_nothing_policy import DoNothingPolicy
from grid2op_env.multi_agent_policies.select_agent_policy import SelectAgentPolicy
from rarl_rllib import RADQNTorchPolicy
from rarl_rllib import RAPPOTorchPolicy, RASACTorchPolicy
from rarl_rllib.dqn.gnn_dqn_policy import GNNBaselineDQNPolicy
from rarl_rllib.dqn.mlp_dqn_policy import DictObsDQNTorchPolicy

logger = logging.getLogger(__name__)

register_custom_models()

POLICIES[DQN_GNN_POLICY] = GNNBaselineDQNPolicy
POLICIES[DQN_MLP_POLICY] = DictObsDQNTorchPolicy
POLICIES[RAPPO_POLICY] = RAPPOTorchPolicy
POLICIES[RASAC_POLICY] = RASACTorchPolicy
POLICIES[RADQN_POLICY] = RADQNTorchPolicy
POLICIES[DO_NOTHING_POLICY] = DoNothingPolicy
POLICIES[HIGH_LEVEL_POLICY] = SelectAgentPolicy

_ALGORITHM_CONFIG_CLS = {
    "ppo": ppo.PPOConfig,
    "sac": sac.SACConfig,
    "dqn": dqn.DQNConfig,
}


# ---------------------------------------------------------------------------
# Config → RLLib translation
# ---------------------------------------------------------------------------

def _build_policies(cfg: DictConfig, algorithm: str) -> dict:
    """Build the multi-agent policies dict."""
    custom_model = cfg.model.custom_model
    model_override = {}

    if custom_model == "ragnn_model" and algorithm == "ppo":
        policy_class = RAPPOTorchPolicy
        model_override = {"model": {"custom_model": "ra_actor_critic_model"}}
    elif custom_model == "ragnn_model" and algorithm == "sac":
        policy_class = RASACTorchPolicy
        model_override = {"model": {"custom_model": "rasac_model"}}
    elif custom_model == "ragnn_model" and algorithm == "dqn":
        policy_class = RADQNTorchPolicy
        model_override = {"model": {"custom_model": "radqn_model"}}
    elif custom_model == "gnn_model" and algorithm == "ppo":
        policy_class = PPOTorchPolicy
        model_override = {"model": {"custom_model": "gnn_model"}}
    elif custom_model == "gnn_model" and algorithm == "sac":
        policy_class = SACTorchPolicy
        model_override = {"model": {"custom_model": "gnn_sac_model"}}
    elif custom_model == "gnn_model" and algorithm == "dqn":
        policy_class = GNNBaselineDQNPolicy
        model_override = {"model": {"custom_model": "gnn_dqn_model"}}
    elif custom_model is None or custom_model == "mlp_model":
        model_override = {}
        if algorithm == "ppo":
            policy_class = PPOTorchPolicy
        elif algorithm == "sac":
            policy_class = SACTorchPolicy
        elif algorithm == "dqn":
            policy_class = DictObsDQNTorchPolicy
        else:
            raise ValueError(f"Unsupported algorithm-model combination '{algorithm}'+'{custom_model}")
    else:
        raise ValueError(f"Unsupported algorithm-model combination '{algorithm}'+'{custom_model}")

    logger.info(f"Using algorithm {policy_class}")
    logger.info(f"Using model {model_override.get('model', {}).get('custom_model', {}) or 'default model'}")

    return {
        HIGH_LEVEL_POLICY: PolicySpec(
            policy_class=SelectAgentPolicy,
            config=(
                AlgorithmConfig()
                .training(model={"custom_model_config": {"rho_threshold": cfg.env.rho_threshold}})
                .rollouts(preprocessor_pref=None)
            ),
        ),
        RL_POLICY: PolicySpec(
            policy_class=policy_class,
            config=model_override,
        ),
        DO_NOTHING_POLICY: PolicySpec(
            policy_class=DoNothingPolicy,
            config=AlgorithmConfig(),
        ),
    }


def build_rllib_config(cfg: DictConfig) -> dict[str, Any]:
    """Translate the assembled Hydra config into a flat RLLib algorithm config dict."""
    # Extract algorithm name before merging (it is a meta-key, not an RLLib param).
    training = OmegaConf.to_container(cfg.training, resolve=True)
    algorithm = training.pop("algorithm", "ppo")

    algo_config_cls = _ALGORITHM_CONFIG_CLS.get(algorithm)
    if algo_config_cls is None:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {list(_ALGORITHM_CONFIG_CLS)}")

    rllib_cfg = algo_config_cls().to_dict()
    rllib_cfg["_disable_preprocessor_api"] = True
    rllib_cfg["seed"] = cfg.experiment.seed

    # --- Training hyperparameters ---
    rllib_cfg.update(training)

    # --- Model ---
    rllib_cfg["model"] = build_model_config(cfg)
    rllib_cfg["_enable_learner_api"] = cfg.model._enable_learner_api
    rllib_cfg["_enable_rl_module_api"] = False

    # --- Relation awareness (passed through to policy config) ---
    rllib_cfg["relation_awareness"] = OmegaConf.to_container(cfg.relation_awareness, resolve=True)

    # --- Imitation pretraining (read by ImitationInitCallback) ---
    imitation = OmegaConf.to_container(cfg.imitation, resolve=True)
    if imitation.get("weights_path"):
        # Tune runs each trial in its own directory, so a relative path would not resolve there.
        imitation["weights_path"] = os.path.abspath(imitation["weights_path"])
    rllib_cfg["imitation"] = imitation

    # --- Environment ---
    train_env_config = build_env_config(cfg, "train")
    val_env_config = build_env_config(cfg, "val")
    rllib_cfg["env_config"] = train_env_config
    rllib_cfg["env"] = CustomizedGrid2OpEnvironment

    # --- Evaluation ---
    eval_cfg = cfg.evaluation
    rllib_cfg["evaluation_interval"] = eval_cfg.evaluation_interval
    rllib_cfg["evaluation_duration_unit"] = eval_cfg.evaluation_duration_unit
    rllib_cfg["always_attach_evaluation_results"] = True
    rllib_cfg["evaluation_num_workers"] = eval_cfg.evaluation_num_workers
    rllib_cfg["evaluation_config"] = {"env_config": val_env_config, "explore": False}

    # Number of eval episodes = number of available chronics for the val env
    val_env_name = val_env_config["env_name"]
    chronics_path = os.path.join(grid2op.get_current_local_dir(), val_env_name, "chronics")
    if os.path.exists(chronics_path):
        rllib_cfg["evaluation_duration"] = len(os.listdir(chronics_path))
    else:
        rllib_cfg["evaluation_duration"] = 50
        logger.warning("Val chronics path not found (%s); defaulting evaluation_duration=50", chronics_path)

    # --- Rollouts / resources ---
    # The training config may override count_steps_by (e.g. SAC uses env_steps).
    rollouts = cfg.rollouts
    rllib_cfg["num_rollout_workers"] = rollouts.num_rollout_workers
    rllib_cfg["num_learner_workers"] = rollouts.num_learner_workers
    rllib_cfg["num_gpus_per_learner_worker"] = rollouts.num_gpus_per_learner_worker
    rllib_cfg["batch_mode"] = rollouts.batch_mode
    # This project runs the old RLlib API stack (_enable_learner_api=False), which has
    # no separate learner-worker actors: training happens on the single driver process
    # via multi_gpu_train_one_step(). `num_learner_workers`/`num_gpus_per_learner_worker`
    # are new-API-stack fields that RLlib's old-API resource request ignores entirely
    # (see Algorithm.default_resource_request: driver GPU = config.num_gpus). The only
    # knob that actually grants the driver a GPU is `rollouts.num_gpus` below.
    rllib_cfg["num_gpus"] = rollouts.num_gpus
    rllib_cfg["count_steps_by"] = training.get("count_steps_by", rollouts.count_steps_by)
    rllib_cfg["keep_per_episode_custom_metrics"] = rollouts.keep_per_episode_custom_metrics
    rllib_cfg["framework"] = rollouts.framework
    rllib_cfg["exploration_config"] = OmegaConf.to_container(rollouts.exploration_config, resolve=True)

    # --- Callbacks ---
    callback_classes = [get_class(cb["_target_"]) for cb in cfg.callbacks.callbacks]
    rllib_cfg["callbacks"] = make_multi_callbacks(callback_classes)

    # --- Multi-agent ---
    rllib_cfg["policies"] = _build_policies(cfg, algorithm)
    rllib_cfg["policy_mapping_fn"] = policy_mapping_fn
    rllib_cfg["policies_to_train"] = [RL_POLICY]

    # --- Misc ---
    rllib_cfg["my_log_level"] = cfg.experiment.my_log_level
    rllib_cfg["trial_info"] = "trial_id"
    rllib_cfg["total_timesteps"] = cfg.experiment.nb_timesteps

    return rllib_cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/rllib", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # fail fast on missing values

    set_seed(cfg.experiment.seed)

    rllib_cfg = build_rllib_config(cfg)
    run_training(rllib_cfg, cfg)


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
