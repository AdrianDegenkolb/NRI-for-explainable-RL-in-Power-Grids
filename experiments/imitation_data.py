"""
CLI to record imitation-learning data with the greedy Tutor in `src/imitation/tutor.py`.

Runs in the main `L2RPN` conda environment (no `curriculumagent` needed), so states, actions and
heuristics are exactly those of the RL env.

  list-chronics        List chronic ids, to size a SLURM array.
  run-tutor-chronics   Run the Tutor on a batch of chronics, one `.npz` per chronic — the unit of
                       work for one task of slurm/unicluster/imitation_tutor_array.sh.
  build-dataset        Convert a Tutor's raw per-chronic output (either source) into the shared
                       dataset format of `src/imitation/dataset.py`, building the policy targets.
  summarize            Write plot-ready summary files of a Tutor run (`src/imitation/summary.py`):
                       episodes.csv, states.npz, actions.csv, substations.csv.
  teacher-episodes     Recover per-chronic survival of a Teacher array run from its logs, in the
                       shared episode format of `src/analysis/episodes.py`.

Example:
    PYTHONPATH=$(pwd)/src python experiments/imitation_data.py run-tutor-chronics --grid case36 \\
        --action-space data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json \\
        --chronic-ids Scenario_april_000 --save-dir results/imitation/smoketest/experience
    PYTHONPATH=$(pwd)/src python experiments/imitation_data.py build-dataset --source greedy_tutor \\
        --experience-dir results/imitation/<experiment>/experience --target soft --temperature 0.01 \\
        --out data/imitation/case36_k208_greedy_soft_t0.01.npz
    PYTHONPATH=$(pwd)/src python experiments/imitation_data.py summarize --grid case36 \
        --action-space data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json \
        --experience-dir results/imitation/<experiment>/experience --out-dir results/imitation/<experiment>/summary
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.episodes import save_episodes, teacher_episodes_from_logs
from imitation.dataset import GENERAL_TUTOR, GREEDY_TUTOR, HARD, SOFT, from_general_tutor, from_greedy_tutor
from imitation.tutor import (
    DO_NOTHING_MARGIN,
    RHO_THRESHOLD,
    TutorConfig,
    list_chronic_ids,
    run_tutor_chronics,
)
from imitation.summary import summarize_tutor_run

# Grid alias -> Grid2Op env base name, matching configs/rllib/env/case36.yaml / case118.yaml.
GRID_ENV_NAMES = {
    "case36": "l2rpn_wcci_2020",
    "case118": "l2rpn_wcci_2022",
}


def _env_name(grid: str, split: str) -> str:
    """Resolve a grid alias + split (train/val/test) to a Grid2Op environment name."""
    return f"{GRID_ENV_NAMES[grid]}_{split}"


def cmd_list_chronics(args: argparse.Namespace) -> None:
    """Print chronic ids, one per line."""
    for chronic_id in list_chronic_ids(_env_name(args.grid, args.split)):
        print(chronic_id)


def cmd_run_tutor_chronics(args: argparse.Namespace) -> None:
    """Run the Tutor on a batch of chronics."""
    config = TutorConfig(
        env_name=_env_name(args.grid, args.split),
        action_space_path=Path(args.action_space),
        save_dir=Path(args.save_dir),
        seed=args.seed,
        rho_threshold=args.rho_threshold,
        do_nothing_margin=args.do_nothing_margin,
        line_reco=not args.no_line_reco,
        max_steps=args.max_steps,
    )
    run_tutor_chronics(config, chronic_ids=args.chronic_ids)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Convert raw Tutor output into the shared dataset format."""
    experience_dir = Path(args.experience_dir)
    if args.source == GENERAL_TUTOR:
        if args.target != HARD:
            raise SystemExit("The GeneralTutor records no alternatives; only --target hard is possible.")
        data = from_general_tutor(experience_dir, n_actions=args.n_actions, action_space=args.action_space_name)
    else:
        data = from_greedy_tutor(experience_dir, target=args.target, temperature=args.temperature)
    data.save(Path(args.out))
    n_dn = int((data.policy_target.argmax(axis=1) == 0).sum())
    print(f"Wrote {len(data)} samples from {len(set(data.chronic))} chronics "
          f"({n_dn} with do-nothing as top target) to {args.out}: {data.metadata}")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Write plot-ready summary files of a Tutor run."""
    summarize_tutor_run(
        experience_dir=Path(args.experience_dir),
        env_name=_env_name(args.grid, args.split),
        action_space_path=Path(args.action_space),
        out_dir=Path(args.out_dir),
    )
    print(f"Wrote episodes.csv, states.npz, actions.csv, substations.csv to {args.out_dir}")


def cmd_teacher_episodes(args: argparse.Namespace) -> None:
    """Write a Teacher run's per-chronic survival as an episode table."""
    episodes = teacher_episodes_from_logs(Path(args.log_dir), agent=args.agent, seed=args.seed)
    save_episodes(episodes, Path(args.out))
    print(f"Wrote {len(episodes)} episodes ({int(episodes['truncated'].sum())} truncated) to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_chronics = subparsers.add_parser("list-chronics", help="List chronic ids of a grid/split")
    list_chronics.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    list_chronics.add_argument("--split", default="train", choices=["train", "val", "test"])
    list_chronics.set_defaults(func=cmd_list_chronics)

    run_tutor = subparsers.add_parser("run-tutor-chronics", help="Run the Tutor on a batch of chronics")
    run_tutor.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_tutor.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_tutor.add_argument("--action-space", required=True, help="Action-space JSON, as loaded by the RL env")
    run_tutor.add_argument("--chronic-ids", nargs="+", required=True)
    run_tutor.add_argument("--save-dir", required=True, help="Directory for the per-chronic .npz files")
    run_tutor.add_argument("--seed", type=int, default=42)
    run_tutor.add_argument("--rho-threshold", type=float, default=RHO_THRESHOLD)
    run_tutor.add_argument("--do-nothing-margin", type=float, default=DO_NOTHING_MARGIN)
    run_tutor.add_argument("--no-line-reco", action="store_true", help="Disable reconnection_rule (env line_reco: false)")
    run_tutor.add_argument("--max-steps", type=int, default=None, help="Cap steps per chronic (smoke tests)")
    run_tutor.set_defaults(func=cmd_run_tutor_chronics)

    build = subparsers.add_parser("build-dataset", help="Convert raw Tutor output into the shared dataset format")
    build.add_argument("--source", required=True, choices=[GENERAL_TUTOR, GREEDY_TUTOR])
    build.add_argument("--experience-dir", required=True, help="Directory of per-chronic Tutor files")
    build.add_argument("--target", default=HARD, choices=[HARD, SOFT])
    build.add_argument("--temperature", type=float, default=0.0, help="Soft-target temperature in rho units")
    build.add_argument("--out", required=True, help="Output .npz path")
    build.add_argument("--n-actions", type=int, default=209, help="general_tutor only: policy actions incl. do-nothing")
    build.add_argument("--action-space-name", default="teacher_n1_k208.json", help="general_tutor only: metadata")
    build.set_defaults(func=cmd_build_dataset)

    summarize = subparsers.add_parser("summarize", help="Write plot-ready summary files of a Tutor run")
    summarize.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    summarize.add_argument("--split", default="train", choices=["train", "val", "test"])
    summarize.add_argument("--action-space", required=True, help="Action-space JSON the run used")
    summarize.add_argument("--experience-dir", required=True, help="Directory of per-chronic Tutor .npz files")
    summarize.add_argument("--out-dir", required=True)
    summarize.set_defaults(func=cmd_summarize)

    teacher = subparsers.add_parser("teacher-episodes", help="Per-chronic survival of a Teacher run from its logs")
    teacher.add_argument("--log-dir", required=True, help="Teacher experiment directory with shard_*.err logs")
    teacher.add_argument("--out", required=True, help="Output episodes .csv")
    teacher.add_argument("--agent", default="teacher")
    teacher.add_argument("--seed", type=int, default=42, help="Seed the Teacher used for every chronic")
    teacher.set_defaults(func=cmd_teacher_episodes)

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
