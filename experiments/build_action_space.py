"""
CLI to build reduced topology action spaces for IEEE-36 / IEEE-118 with the N-1 Teacher.

Must run in the `curriculum` conda environment (see `environment_curriculum.yaml`), never in
the main `L2RPN` training environment — see `src/action_space_generation/__init__.py`.

Stages (run in order; see the Obsidian note "Teacher Action Space Reduction for IEEE-36 and
IEEE-118" for the full pipeline):

  select-lines        Pick `lines_to_attack` for a grid from a do-nothing rollout.
  list-chronics       List chronic ids, to size a SLURM array (one task per chronic).
  run-teacher         Run the N-1 Teacher over an entire grid, parallelized locally across `jobs`.
  run-teacher-chronic Run the N-1 Teacher on a single chronic — one SLURM array task.
  aggregate           Rank teacher_experience CSVs by frequency, report substation spread, and
                      (with --out) export the top-k actions to this repo's action-space JSON.

Example:
    python experiments/build_action_space.py select-lines --grid case36
    python experiments/build_action_space.py run-teacher --grid case36 \\
        --lines-to-attack 3 12 27 --save-path results/teacher/case36/shard_00.csv
    python experiments/build_action_space.py aggregate --grid case36 \\
        --experience results/teacher/case36/shard_*.csv --top-k 300 \\
        --out data/action_spaces/l2rpn_wcci_2020/teacher_n1_k300.json
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import grid2op
from lightsim2grid import LightSimBackend

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from action_space_generation.aggregation import (
    action_frequency,
    load_experience,
    select_top_k_actions,
    substation_distribution,
)
from action_space_generation.export import export_action_space, verify_round_trip
from action_space_generation.line_selection import select_lines_to_attack
from action_space_generation.teacher_runner import (
    TeacherConfig,
    list_chronic_ids,
    run_teacher,
    run_teacher_single_chronic,
)

# Grid alias -> Grid2Op env base name, matching configs/rllib/env/case36.yaml / case118.yaml.
GRID_ENV_NAMES = {
    "case36": "l2rpn_wcci_2020",
    "case118": "l2rpn_wcci_2022",
}


def _env_name(grid: str, split: str) -> str:
    """Resolve a grid alias + split (train/val/test) to a Grid2Op environment name."""
    if grid not in GRID_ENV_NAMES:
        raise ValueError(f"Unknown grid '{grid}', expected one of {list(GRID_ENV_NAMES)}")
    return f"{GRID_ENV_NAMES[grid]}_{split}"


def cmd_select_lines(args: argparse.Namespace) -> None:
    """Print the `lines_to_attack` selected for a grid, for pasting into a run-teacher call."""
    lines = select_lines_to_attack(
        env_name=_env_name(args.grid, args.split),
        n_lines=args.n_lines,
        n_chronics=args.n_chronics,
    )
    print(f"lines_to_attack for {args.grid}: {lines}")


def cmd_run_teacher(args: argparse.Namespace) -> None:
    """Run the N-1 Teacher over one grid/chronic-shard, appending results to a CSV."""
    config = TeacherConfig(
        env_name=_env_name(args.grid, args.split),
        lines_to_attack=args.lines_to_attack,
        save_path=Path(args.save_path),
        seed=args.seed,
        jobs=args.jobs,
    )
    run_teacher(config)


def cmd_list_chronics(args: argparse.Namespace) -> None:
    """Print chronic ids, one per line — use to size a SLURM array (one task per chronic)."""
    for chronic_id in list_chronic_ids(_env_name(args.grid, args.split)):
        print(chronic_id)


def cmd_run_teacher_chronic(args: argparse.Namespace) -> None:
    """Run the N-1 Teacher on a single chronic — the unit of work for one SLURM array task."""
    config = TeacherConfig(
        env_name=_env_name(args.grid, args.split),
        lines_to_attack=args.lines_to_attack,
        save_path=Path(args.save_path),
        seed=args.seed,
    )
    run_teacher_single_chronic(config, chronics_id=args.chronic_id)


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Rank teacher_experience CSVs by frequency, plot the curve, and report substation spread."""
    experience_paths = [Path(p) for pattern in args.experience for p in sorted(glob.glob(pattern))]
    if not experience_paths:
        raise FileNotFoundError(f"No experience files matched: {args.experience}")

    data = load_experience(experience_paths)
    freq = action_frequency(data)
    print(f"{len(freq)} unique actions across {len(experience_paths)} experience file(s)")

    env = grid2op.make(_env_name(args.grid, args.split), backend=LightSimBackend())
    try:
        actions = select_top_k_actions(data, env, k=args.top_k)
        substations = substation_distribution(actions)
        print(f"Top-{args.top_k} actions touch {len(substations)} distinct substations:")
        print(substations.to_string())

        if args.out is not None:
            if not verify_round_trip(actions, env):
                raise RuntimeError("Serialize/deserialize round trip failed for at least one action")
            export_action_space(actions, Path(args.out))
            print(f"Wrote {len(actions)} actions to {args.out}")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser with one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_lines = subparsers.add_parser("select-lines", help="Pick lines_to_attack for a grid")
    select_lines.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    select_lines.add_argument("--split", default="train", choices=["train", "val", "test"])
    select_lines.add_argument("--n-lines", type=int, default=10)
    select_lines.add_argument("--n-chronics", type=int, default=20)
    select_lines.set_defaults(func=cmd_select_lines)

    run_teacher_parser = subparsers.add_parser("run-teacher", help="Run the N-1 Teacher on one grid/shard")
    run_teacher_parser.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_teacher_parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_teacher_parser.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    run_teacher_parser.add_argument("--save-path", required=True)
    run_teacher_parser.add_argument("--seed", type=int, default=42)
    run_teacher_parser.add_argument("--jobs", type=int, default=-1)
    run_teacher_parser.set_defaults(func=cmd_run_teacher)

    list_chronics = subparsers.add_parser("list-chronics", help="List chronic ids, for sizing a SLURM array")
    list_chronics.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    list_chronics.add_argument("--split", default="train", choices=["train", "val", "test"])
    list_chronics.set_defaults(func=cmd_list_chronics)

    run_teacher_chronic = subparsers.add_parser(
        "run-teacher-chronic", help="Run the N-1 Teacher on a single chronic (one SLURM array task)"
    )
    run_teacher_chronic.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_teacher_chronic.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_teacher_chronic.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    run_teacher_chronic.add_argument("--chronic-id", required=True)
    run_teacher_chronic.add_argument("--save-path", required=True)
    run_teacher_chronic.add_argument("--seed", type=int, default=42)
    run_teacher_chronic.set_defaults(func=cmd_run_teacher_chronic)

    aggregate = subparsers.add_parser("aggregate", help="Rank experience CSVs and export the top-k actions")
    aggregate.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    aggregate.add_argument("--split", default="train", choices=["train", "val", "test"])
    aggregate.add_argument("--experience", nargs="+", required=True, help="CSV path(s) or glob pattern(s)")
    aggregate.add_argument("--top-k", type=int, required=True)
    aggregate.add_argument("--out", default=None, help="Output JSON path, e.g. data/action_spaces/.../teacher_n1_k300.json")
    aggregate.set_defaults(func=cmd_aggregate)

    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
