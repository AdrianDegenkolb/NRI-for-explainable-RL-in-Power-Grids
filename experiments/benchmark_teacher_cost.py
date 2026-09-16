"""
Phase 0 cost-per-interaction benchmark for the N-1 Teacher.

Measures wall-clock cost of the N-1 Teacher's two expensive branches (N-1 search, greedy
fallback) directly, without running full chronics to completion — see the "Cost model & Phase
0 benchmark plan" section of the Obsidian note "Teacher Action Space Reduction for IEEE-36 and
IEEE-118". The cheap "active search" branch (~10 sims) is not timed separately; it is negligible
next to a full sweep over the unitary action set.

The step loop below deliberately mirrors `NMinusOneTeacher.n_minus_one_agent`'s branch logic
(curriculumagent==1.1.2) with `active_search=True`, matching `teacher_runner.run_teacher`'s
production config. It exists only to add an early-stop after `n_interactions`, which the
package's own loop (built for full chronics) does not support.

Combine this script's output with `interact_rate x expected_episode_length`, estimated from
existing training logs (see the note), to size SLURM walltime/array width.

Must run in the `curriculum` conda environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import grid2op
import numpy as np
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.topology_action_search import topology_search_topk
from curriculumagent.teacher.teachers.teacher_n_minus_1 import NMinusOneTeacher
from lightsim2grid import LightSimBackend

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from action_space_generation.teacher_runner import RHO_MAX_THRESHOLD, RHO_N0_THRESHOLD


def _summarize(values: list[float]) -> dict:
    """Reduce a list of per-interaction timings to the statistics needed for SLURM sizing."""
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def benchmark(env_name: str, lines_to_attack: list[int], n_interactions: int, seed: int) -> dict:
    """Time N-1-search and greedy-fallback branches over `n_interactions` real teacher decisions.

    Args:
        env_name: Grid2Op environment name or path.
        lines_to_attack: Line ids used as N-1 contingencies (from `select_lines_to_attack`).
        n_interactions: Stop once this many N-1-search + greedy interactions have been timed.
            The cheap active-search branch does not count towards this budget.
        seed: Grid2Op environment seed.

    Returns:
        Dict with per-branch raw timings (seconds), summary statistics, and metadata (unitary
        action-set size, chronics consumed), ready to `json.dump`.
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    env.seed(seed)

    teacher = NMinusOneTeacher(
        lines_to_attack=lines_to_attack,
        rho_n0_threshold=RHO_N0_THRESHOLD,
        rho_max_threshold=RHO_MAX_THRESHOLD,
        seed=seed,
    )

    timings: dict[str, list[float]] = {"n_minus_one_search": [], "greedy": []}
    n_chronics = 0

    def n_timed() -> int:
        return len(timings["n_minus_one_search"]) + len(timings["greedy"])

    try:
        while n_timed() < n_interactions:
            obs = env.reset()
            done = False
            n_chronics += 1
            while not done and n_timed() < n_interactions:
                rho_max = obs.rho.max()
                if rho_max < teacher.rho_n0:
                    if all(obs.line_status):
                        action = teacher.do_nothing_and_run_through_lines_action(env=env, obs=obs)
                    else:
                        action = find_best_line_to_reconnect(obs, env.action_space({}))
                elif rho_max < teacher.rho_threshold and all(obs.line_status):
                    t0 = time.perf_counter()
                    action, _ = teacher.search_best_n_minus_one_action(env=env, obs=obs)
                    timings["n_minus_one_search"].append(time.perf_counter() - t0)
                    action = find_best_line_to_reconnect(obs, action)
                else:
                    all_actions = teacher.get_all_actions(env)
                    t0 = time.perf_counter()
                    greedy_set = topology_search_topk(env, obs, all_actions, top_k=teacher.top_k)
                    timings["greedy"].append(time.perf_counter() - t0)
                    action = greedy_set[0][1] if greedy_set else env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)
                obs, _, done, _ = env.step(action)
    finally:
        n_unitary_actions = len(teacher.all_actions) if teacher.all_actions is not None else None
        env.close()

    return {
        "env_name": env_name,
        "lines_to_attack": lines_to_attack,
        "n_chronics_used": n_chronics,
        "n_unitary_actions": n_unitary_actions,
        "timings_sec": timings,
        "summary_sec": {branch: _summarize(values) for branch, values in timings.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-name", required=True, help="Grid2Op env, e.g. l2rpn_wcci_2020_train")
    parser.add_argument("--lines-to-attack", type=int, nargs="+", required=True)
    parser.add_argument("--n-interactions", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path for the timing results")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = benchmark(
        env_name=args.env_name,
        lines_to_attack=args.lines_to_attack,
        n_interactions=args.n_interactions,
        seed=args.seed,
    )
    print(json.dumps(result["summary_sec"], indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote full results to {out_path}")
