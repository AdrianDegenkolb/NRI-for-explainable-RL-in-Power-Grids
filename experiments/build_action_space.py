"""
CLI to build reduced topology action spaces for IEEE-36 / IEEE-118 with the N-1 Teacher.

Must run in the `curriculum` conda environment (see `environment_curriculum.yaml`), never in
the main `L2RPN` training environment — see `src/action_space_generation/__init__.py`.

Stages (run in order; see the Obsidian note "Teacher Action Space Reduction for IEEE-36 and
IEEE-118" for the full pipeline):

  select-lines         Pick `lines_to_attack` for a grid from a do-nothing rollout.
  list-chronics        List chronic ids, to size a SLURM array.
  run-teacher          Run the N-1 Teacher over an entire grid, parallelized locally across `jobs`.
  run-teacher-chronic  Run the N-1 Teacher on a single chronic — for a quick smoke test before
                       committing to the full array (see run-teacher-chronics below).
  run-teacher-chronics Run the N-1 Teacher on a batch of chronics, appended to one shard — the
                       unit of work for one SLURM array task. Chronic counts can run into the
                       thousands (e.g. 2592 for l2rpn_wcci_2020_train), which exceeds a typical
                       cluster's SLURM MaxArraySize if sharded one-chronic-per-task; batching
                       several chronics per task keeps the array width bounded.
  aggregate            Rank teacher_experience CSVs by frequency, report substation spread, and
                       (with --out) export the top-k actions to this repo's action-space JSON.
  diagnose-n1          Explain why a grid yields few/no rows, by separating the two filters
                       inside the N-1 search that can each silently reject every candidate.
                       Run this before committing a long array on a grid whose smoke test hits
                       the N-1 branch but writes nothing.

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
import json
import logging
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
from action_space_generation.n1_conversion_diagnostic import diagnose, write_result
from action_space_generation.teacher_runner import (
    TeacherConfig,
    list_chronic_ids,
    run_teacher,
    run_teacher_chronics,
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
    """Print the `lines_to_attack` selected for a grid, space-separated for direct shell reuse."""
    lines = select_lines_to_attack(
        env_name=_env_name(args.grid, args.split),
        n_lines=args.n_lines,
        n_chronics=args.n_chronics,
    )
    # Space-separated (not a Python list repr) so the line can be pasted straight into
    # --lines-to-attack / the SLURM launchers' <lines_to_attack...> argument.
    print(" ".join(str(line_id) for line_id in lines))


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
    """Run the N-1 Teacher on a single chronic — for a smoke test before the full array."""
    config = TeacherConfig(
        env_name=_env_name(args.grid, args.split),
        lines_to_attack=args.lines_to_attack,
        save_path=Path(args.save_path),
        seed=args.seed,
    )
    run_teacher_single_chronic(config, chronics_id=args.chronic_id)


def cmd_run_teacher_chronics(args: argparse.Namespace) -> None:
    """Run the N-1 Teacher on a batch of chronics — the unit of work for one SLURM array task."""
    config = TeacherConfig(
        env_name=_env_name(args.grid, args.split),
        lines_to_attack=args.lines_to_attack,
        save_path=Path(args.save_path),
        seed=args.seed,
    )
    run_teacher_chronics(config, chronics_ids=args.chronic_ids)


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


def cmd_diagnose_n1(args: argparse.Namespace) -> None:
    """Report how often the N-1 search's two filters reject every candidate on a grid."""
    result = diagnose(
        env_name=_env_name(args.grid, args.split),
        lines_to_attack=args.lines_to_attack,
        n_probes=args.n_probes,
        seed=args.seed,
    )
    summary = result.summary()
    print(json.dumps(summary, indent=2))
    if summary["n_probes"] and summary["n_would_save"] == 0:
        print(
            "\nNo probe would have saved a row. Dominant failure: "
            + (
                "Stage 1 — no single rewiring pulls rho below rho_n0."
                if summary["n_failed_stage1_no_candidate_fixes_rho"]
                >= summary["n_failed_stage2_all_candidates_blackout"]
                else "Stage 2 — every candidate blacks out under some lines_to_attack contingency."
            )
        )
    write_result(result, Path(args.out))
    print(f"Wrote full diagnostic to {args.out}")


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
        "run-teacher-chronic", help="Run the N-1 Teacher on a single chronic (smoke test)"
    )
    run_teacher_chronic.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_teacher_chronic.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_teacher_chronic.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    run_teacher_chronic.add_argument("--chronic-id", required=True)
    run_teacher_chronic.add_argument("--save-path", required=True)
    run_teacher_chronic.add_argument("--seed", type=int, default=42)
    run_teacher_chronic.set_defaults(func=cmd_run_teacher_chronic)

    run_teacher_chronics_parser = subparsers.add_parser(
        "run-teacher-chronics", help="Run the N-1 Teacher on a batch of chronics (one SLURM array task)"
    )
    run_teacher_chronics_parser.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_teacher_chronics_parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_teacher_chronics_parser.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    run_teacher_chronics_parser.add_argument("--chronic-ids", nargs="+", required=True)
    run_teacher_chronics_parser.add_argument("--save-path", required=True)
    run_teacher_chronics_parser.add_argument("--seed", type=int, default=42)
    run_teacher_chronics_parser.set_defaults(func=cmd_run_teacher_chronics)

    aggregate = subparsers.add_parser("aggregate", help="Rank experience CSVs and export the top-k actions")
    aggregate.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    aggregate.add_argument("--split", default="train", choices=["train", "val", "test"])
    aggregate.add_argument("--experience", nargs="+", required=True, help="CSV path(s) or glob pattern(s)")
    aggregate.add_argument("--top-k", type=int, required=True)
    aggregate.add_argument("--out", default=None, help="Output JSON path, e.g. data/action_spaces/.../teacher_n1_k300.json")
    aggregate.set_defaults(func=cmd_aggregate)

    diagnose_n1 = subparsers.add_parser(
        "diagnose-n1", help="Separate the two filters that make the N-1 search save nothing"
    )
    diagnose_n1.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    diagnose_n1.add_argument("--split", default="train", choices=["train", "val", "test"])
    diagnose_n1.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    diagnose_n1.add_argument("--n-probes", type=int, default=6)
    diagnose_n1.add_argument("--seed", type=int, default=42)
    diagnose_n1.add_argument("--out", required=True, help="Output JSON path for the diagnostic")
    diagnose_n1.set_defaults(func=cmd_diagnose_n1)

    return parser


if __name__ == "__main__":
    # curriculumagent logs progress via `logging.info` (e.g. "Looking for N-1 action"); without
    # this, a multi-hour run-teacher/run-teacher-chronic job would write nothing to its SLURM
    # log until it finished, making a stalled vs. working job indistinguishable.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
