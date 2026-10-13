"""
Survival of a pretrained policy ("junior") on held-out chronics, as an episode table
(`src/analysis/episodes.py`) comparable chronic by chronic with the Tutor's.

Rebuilds env, converter and network from the `config.yaml` that experiments/pretrain_policy.py
wrote next to `policy_weights.pt`. Chronics come from a split file, sharded over SLURM array
tasks with a stride (task t of n takes chronics t, t+n, ...).

Example:
    PYTHONPATH=$(pwd)/src python experiments/imitation_rollout.py \\
        --run-dir results/pretraining/pretrain_gnn_graph_soft --split-name test \\
        --task-id 0 --n-tasks 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from grid2op.gym_compat import GymEnv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.episodes import save_episodes
from core.rllib_config import build_env_config, register_custom_models
from grid2op_env.observation_converter import make_observation_converter
from grid2op_env.utils import make_g2op_env
from imitation.dataset import load_split
from imitation.pretrain import build_policy_model
from imitation.rollout import evaluate_policy
from imitation.tutor import load_policy_actions
from pretrain_policy import policy_model_config


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    register_custom_models()
    env_config = build_env_config(cfg, "train")
    device = torch.device(args.device)

    chronics = load_split(Path(args.split_file or cfg.imitation.split))[args.split_name][args.task_id::args.n_tasks]
    env = make_g2op_env(env_config)
    try:
        converter = make_observation_converter(GymEnv(env, with_forecast=True), env_config, backend=env.backend)
        actions = load_policy_actions(env, ROOT / f"data/action_spaces/{env.env_name}/{env_config['action_space']}.json")
        model = build_policy_model(converter.observation_space, len(actions), policy_model_config(cfg))
        model.load_state_dict(torch.load(run_dir / "policy_weights.pt", map_location=device))
        model.to(device)
        episodes = evaluate_policy(
            env, converter, model, actions, chronics, agent=args.agent, base_seed=args.seed,
            rho_threshold=env_config["rho_threshold"], line_reco=env_config["line_reco"], device=device,
        )
    finally:
        env.close()

    out = run_dir / "rollout" / f"episodes_{args.split_name}_{args.task_id:03d}.csv"
    save_episodes(episodes, out)
    print(f"Wrote {len(episodes)} episodes ({int(episodes['survived'].sum())} survived) to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Output dir of experiments/pretrain_policy.py")
    parser.add_argument("--split-file", default=None, help="Default: the split the policy was trained with")
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--agent", default="junior", help="Value of the episode table's agent column")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; must equal the Tutor's for paired comparison")
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_parser().parse_args())
