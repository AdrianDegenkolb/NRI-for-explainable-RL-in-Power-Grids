"""
Rollout evaluation of a policy network outside RLlib, with the RL env's control logic: do-nothing
while `rho.max() <= rho_threshold`, the policy's argmax action above it, every action enriched by
`reconnection_rule` (env `line_reco`), and the observation converter updated on every step.

Each chronic is seeded with `imitation.tutor.chronic_seed`, so a policy faces exactly the
maintenance schedule the Tutor faced on that chronic and survival can be compared per chronic.
Results use the shared episode table of `analysis.episodes`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from analysis.episodes import make_episodes
from core.heuristic_actions import reconnection_rule
from grid2op_env.observation_converter import ObservationConverter
from imitation.pretrain import policy_logits
from imitation.tutor import DO_NOTHING_INDEX, chronic_seed

logger = logging.getLogger(__name__)


def resolve_chronic_ids(env: BaseEnv, names: list[str]) -> list[str]:
    """Map chronic names (as in split files and episode tables) to the env's chronic paths."""
    by_name = {Path(p).name: p for p in env.chronics_handler.subpaths}
    missing = [n for n in names if Path(n).name not in by_name]
    if missing:
        raise KeyError(f"{len(missing)} chronics not in {env.env_name}, e.g. {missing[:3]}")
    return [by_name[Path(n).name] for n in names]


@torch.no_grad()
def choose_action(model: TorchModelV2, gym_obs: dict, device: torch.device) -> int:
    """Greedy (argmax) action of the policy for one converted observation."""
    obs = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device) for k, v in gym_obs.items()}
    return int(policy_logits(model, obs).argmax(dim=-1).item())


def run_policy_chronic(
    env: BaseEnv,
    converter: ObservationConverter,
    model: TorchModelV2,
    actions: list[BaseAction],
    chronic_id: str,
    base_seed: int,
    rho_threshold: float,
    line_reco: bool,
    device: torch.device,
) -> dict:
    """Roll out one chronic and return its episode row (without `agent`)."""
    seed = chronic_seed(base_seed, chronic_id)
    env.seed(seed)
    env.set_id(chronic_id)
    obs = env.reset()
    gym_obs = converter.to_gym(obs)
    n_decisions = 0
    done = False
    while not done:
        if obs.rho.max() > rho_threshold:
            action = actions[choose_action(model, gym_obs, device)]
            n_decisions += 1
        else:
            action = actions[DO_NOTHING_INDEX]
        if line_reco:
            action = reconnection_rule(obs, action, env.action_space)
        obs, _, done, _ = env.step(action)
        gym_obs = converter.to_gym(obs)
    return {
        "chronic": Path(chronic_id).name,
        "seed": seed,
        "steps_survived": int(obs.current_step),
        "max_step": int(obs.max_step),
        "n_decisions": n_decisions,
    }


def evaluate_policy(
    env: BaseEnv,
    converter: ObservationConverter,
    model: TorchModelV2,
    actions: list[BaseAction],
    chronic_names: list[str],
    agent: str,
    base_seed: int,
    rho_threshold: float,
    line_reco: bool,
    device: torch.device,
) -> pd.DataFrame:
    """Roll out several chronics; returns their episode table."""
    model.eval()
    rows = []
    for chronic_id in resolve_chronic_ids(env, chronic_names):
        row = run_policy_chronic(env, converter, model, actions, chronic_id, base_seed, rho_threshold, line_reco, device)
        logger.info(f"{row['chronic']}: survived {row['steps_survived']}/{row['max_step']} ({row['n_decisions']} decisions)")
        rows.append({"agent": agent, **row})
    return make_episodes(rows)
