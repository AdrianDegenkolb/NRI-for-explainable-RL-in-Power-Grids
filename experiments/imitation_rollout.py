"""
Survival of a pretrained policy ("junior") on held-out chronics, as an episode table
(`src/analysis/episodes.py`) comparable chronic by chronic with the Tutor's.

Rebuilds env, converter and network from the `config.yaml` that experiments/pretrain_policy.py
wrote next to `policy_weights.pt` (or experiments/export_policy_weights.py, for RL policies).
With `--do-nothing` the agent always plays do-nothing (plus line reconnection) instead; the run
directory then only provides the env config, and `--out-dir` keeps its episode table apart.
Chronics come from a split file of the train env, or with `--env-split val|test` are all chronics
of that Grid2Op env, which no stage of the imitation pipeline has seen. They are sharded over SLURM
array tasks with a stride (task t of n takes chronics t, t+n, ...).

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
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.episodes import save_episodes
from core.rllib_config import build_env_config, register_custom_models
from grid2op_env.observation_converter import ObservationConverter, make_observation_converter
from grid2op_env.utils import make_g2op_env
from imitation.dataset import load_split
from imitation.pretrain_run import CONFIG_FILE, WEIGHTS_FILE, build_model
from imitation.rollout import ActionChooser, do_nothing_chooser, evaluate_policy, greedy_chooser
from imitation.tutor import load_policy_actions


def select_chronics(env: BaseEnv, cfg: DictConfig, args: argparse.Namespace) -> tuple[list[str], str]:
    """This task's chronics and the tag of the output file.

    Returns:
        `(chronic names, tag)`; tag is the split name for the train env, else `<env_split>env`.
    """
    if args.env_split == "train":
        all_chronics = load_split(Path(args.split_file or cfg.imitation.split))[args.split_name]
        tag = args.split_name
    else:
        all_chronics = [Path(c).name for c in env.chronics_handler.subpaths]
        tag = f"{args.env_split}env"
    return all_chronics[args.task_id::args.n_tasks], tag


def load_policy(
    cfg: DictConfig, run_dir: Path, converter: ObservationConverter, n_actions: int, device: torch.device,
) -> ActionChooser:
    """Greedy chooser of the network stored in `run_dir`."""
    model = build_model(cfg, converter.observation_space, n_actions)
    model.load_state_dict(torch.load(run_dir / WEIGHTS_FILE, map_location=device))
    model.to(device)
    return greedy_chooser(model, device)


def main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / CONFIG_FILE)
    register_custom_models()
    env_config = build_env_config(cfg, args.env_split)
    device = torch.device(args.device)

    env = make_g2op_env(env_config)
    chronics, tag = select_chronics(env, cfg, args)
    try:
        actions = load_policy_actions(env, ROOT / f"data/action_spaces/{env.env_name}/{env_config['action_space']}.json")
        if args.do_nothing:
            converter, chooser = None, do_nothing_chooser
        else:
            converter = make_observation_converter(GymEnv(env, with_forecast=True), env_config, backend=env.backend)
            chooser = load_policy(cfg, run_dir, converter, len(actions), device)
        episodes = evaluate_policy(
            env, converter, chooser, actions, chronics, agent=args.agent, base_seed=args.seed,
            rho_threshold=env_config["rho_threshold"], line_reco=env_config["line_reco"],
        )
    finally:
        env.close()

    out = Path(args.out_dir or run_dir / "rollout") / f"episodes_{tag}_{args.task_id:03d}.csv"
    save_episodes(episodes, out)
    print(f"Wrote {len(episodes)} episodes ({int(episodes['survived'].sum())} survived) to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Output dir of experiments/pretrain_policy.py")
    parser.add_argument("--split-file", default=None, help="Default: the split the policy was trained with")
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"],
                        help="Split of --split-file; only used with --env-split train")
    parser.add_argument("--env-split", default="train", choices=["train", "val", "test"],
                        help="Grid2Op env to roll out in; val/test use all of its chronics")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-tasks", type=int, default=1)
    parser.add_argument("--agent", default="junior", help="Value of the episode table's agent column")
    parser.add_argument("--do-nothing", action="store_true",
                        help="Always play do-nothing instead of the run's policy (reference agent)")
    parser.add_argument("--out-dir", default=None, help="Directory of the episode table; default <run-dir>/rollout")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; must equal the Tutor's for paired comparison")
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_parser().parse_args())
