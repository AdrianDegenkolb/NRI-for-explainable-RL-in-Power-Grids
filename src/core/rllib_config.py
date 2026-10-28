"""
Translate the Hydra config into the env and model parts of the RLlib config.

Shared by RL training (`experiments/train.py`) and imitation pretraining
(`experiments/pretrain_policy.py`), so both build exactly the same env and policy network.
"""
import os
from typing import Any

import grid2op
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from ray.rllib.models import ModelCatalog

from core.utils import getl
from rarl_rllib import RAActorCriticModel, RADQNTorchModel, RASACTorchModel
from rarl_rllib.dqn.gnn_dqn_model import GNNBaselineDQNModel
from rarl_rllib.ppo.gnn_ppo_model import GNNBaselineModel
from rarl_rllib.ppo.mlp_ppo_model import FlatMLPModel
from rarl_rllib.sac.gnn_sac_model import GNNBaselineSACModel

# Custom model name used with PPO per `cfg.model.custom_model`, see `_build_policies` in experiments/train.py.
PPO_CUSTOM_MODELS = {"ragnn_model": "ra_actor_critic_model", "gnn_model": "gnn_model"}


def register_custom_models() -> None:
    """Register the project's custom models in RLlib's ModelCatalog."""
    ModelCatalog.register_custom_model("ra_actor_critic_model", RAActorCriticModel)
    ModelCatalog.register_custom_model("rasac_model", RASACTorchModel)
    ModelCatalog.register_custom_model("radqn_model", RADQNTorchModel)
    ModelCatalog.register_custom_model("gnn_model", GNNBaselineModel)
    ModelCatalog.register_custom_model("gnn_dqn_model", GNNBaselineDQNModel)
    ModelCatalog.register_custom_model("gnn_sac_model", GNNBaselineSACModel)
    ModelCatalog.register_custom_model("flat_mlp_model", FlatMLPModel)


def build_env_config(cfg: DictConfig, split: str) -> dict[str, Any]:
    """Build an env_config dict for the given split ('train' or 'val')."""
    env = cfg.env
    obs = cfg.obs_space
    reward_instance = get_class(cfg.reward._target_)()

    env_config = {
        "env_name": f"{env.env_name}_{split}",
        "action_space": env.action_space,
        "observation_space": obs.observation_space,
        "g2op_input": obs.get("g2op_input", []),
        "custom_input": obs.get("custom_input", []),
        "lib_dir": os.getcwd(),
        "grid2op_kwargs": {"reward_class": reward_instance},
        "seed": cfg.experiment.seed,
        "rho_threshold": env.rho_threshold,
        "reset_topo": env.reset_topo,
        "line_reco": env.line_reco,
        "line_disc": env.line_disc,
        "penalty_game_over": env.penalty_game_over,
        "reward_finish": env.reward_finish,
        # Missing in the configs of runs from before the switch existed.
        "normalize_reward_by_episode_length": cfg.reward.get("normalize_by_episode_length", False),
        "curriculum_training": env.curriculum_training,
        "curriculum_thresholds": list(env.curriculum_thresholds),
        "chronics_dir": env.get("chronics_dir", None),
        # Node features; use_forecasts=false drops the forecast features of every converter.
        "use_forecasts": obs.get("use_forecasts", True),
        "attr_to_observe": OmegaConf.to_container(obs.attr_to_observe) if obs.get("attr_to_observe") else None,
        # "complete": the node features of CompleteGraphObsSpace for every attribute-list converter.
        "node_features": obs.get("node_features", "default"),
        # Add each node's substation(s) to the observation (needed by the per-substation GNN readout).
        "node_substations": obs.get("node_substations", False),
        "use_chronics_cache": env.get("use_chronics_cache", False),
    }

    if cfg.opponent.enabled:
        opp = cfg.opponent
        env_config["grid2op_kwargs"].update(
            {
                "opponent_attack_cooldown": opp.opponent_attack_cooldown,
                "opponent_attack_duration": opp.opponent_attack_duration,
                "opponent_budget_per_ts": opp.opponent_budget_per_ts,
                "opponent_init_budget": opp.opponent_init_budget,
                "opponent_action_class": get_class(opp.opponent_action_class),
                "opponent_class": get_class(opp.opponent_class),
                "opponent_budget_class": get_class(opp.opponent_budget_class),
                "kwargs_opponent": OmegaConf.to_container(opp.kwargs_opponent, resolve=True),
            }
        )

    if split == "val":
        overrides = OmegaConf.to_container(cfg.evaluation.eval_env_overrides, resolve=True)
        env_config.update(overrides)

    return env_config


def build_model_config(cfg: DictConfig) -> dict[str, Any]:
    """Build the RLLib model dict from cfg.model and cfg.relation_awareness."""
    model = cfg.model
    custom_model_config: dict[str, Any] = {}

    if model.encoder is not None:
        custom_model_config["encoder"] = OmegaConf.to_container(model.encoder, resolve=True)

    if model.gnn is not None:
        custom_model_config["gnn"] = OmegaConf.to_container(model.gnn, resolve=True)

    if model.get("mlp") is not None:
        custom_model_config["mlp"] = OmegaConf.to_container(model.mlp, resolve=True)

    # sampling and latent_space live in relation_awareness but are also needed by the model
    custom_model_config["sampling"] = OmegaConf.to_container(
        cfg.relation_awareness.sampling, resolve=True
    )
    custom_model_config["latent_space"] = OmegaConf.to_container(
        cfg.relation_awareness.latent_space, resolve=True
    )

    # sparsification: always pass to custom_model_config; temperature is bridged from prior.
    # n_powerlines_directed is only derivable when top_k_multiplier > 0.
    sparse_cfg = OmegaConf.to_container(cfg.relation_awareness.sparsification, resolve=True)
    if getl(sparse_cfg, "top_k_multiplier", 0) > 0:
        env_tmp = grid2op.make(cfg.env.env_name + "_train")
        sparse_cfg["n_powerlines_directed"] = 2 * env_tmp.n_line
        env_tmp.close()
    custom_model_config["sparsification"] = sparse_cfg

    return {
        "fcnet_hiddens": list(model.fcnet_hiddens),
        "fcnet_activation": model.fcnet_activation,
        "post_fcnet_hiddens": list(model.post_fcnet_hiddens),
        "custom_model": model.custom_model if model.custom_model != "null" else None,
        "custom_model_config": custom_model_config,
    }
