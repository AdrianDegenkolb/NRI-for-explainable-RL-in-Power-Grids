"""
Diagnose *why* the N-1 Teacher saves so few rows on a given grid.

Motivation: on IEEE-118 the teacher's N-1 branch was observed to fire but still write nothing
(2 branch hits, 0 rows in the first 400 steps of a smoke test), whereas IEEE-36 converted its
first 2 hits into 2 rows. Reaching the branch is necessary but not sufficient — a row is only
written when `NMinusOneTeacher.search_best_n_minus_one_action` sets `n_1_action_found=True`,
which requires passing *two* independent filters:

  Stage 1  Of the top-`top_k` greedy candidates, keep those that pull rho back below `rho_n0`.
           Can come back empty if no single substation rewiring is a big enough lever.
  Stage 2  Of those survivors, keep those that stay finite in `calculate_attacked_max_rho`,
           i.e. survive disconnection of every line in `lines_to_attack` without a game over.
           A candidate that blacks out under any contingency scores `inf` and is disqualified.

Both failures look identical from the outside (branch fires, no row), but they have different
fixes — Stage 1 failing points at `rho_n0` / the action set, Stage 2 failing points at
`lines_to_attack` being too severe. This script separates them.

The probe below mirrors `search_best_n_minus_one_action` (curriculumagent==1.1.2, lines 154-180)
exactly, but counts candidates at each filter instead of returning an action.

Note on sampling: stressed states are reached here by stepping a *do-nothing* agent, not by
replaying the teacher's own trajectory — vastly cheaper, since it skips a ~340 s greedy sweep per
step. The distribution of in-band states therefore differs somewhat from the teacher's, so read
the output as "can these filters pass at all on this grid, and roughly how often", not as a
precise reproduction of the teacher's conversion rate.

Must run in the `curriculum` conda environment.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import grid2op
import numpy as np
from curriculumagent.common.utilities import find_best_line_to_reconnect
from curriculumagent.teacher.submodule.topology_action_search import topology_search_topk
from curriculumagent.teacher.teachers.teacher_n_minus_1 import NMinusOneTeacher
from lightsim2grid import LightSimBackend

from action_space_generation.teacher_runner import RHO_MAX_THRESHOLD, RHO_N0_THRESHOLD


@dataclass
class ProbeSample:
    """One instrumented evaluation of the N-1 search at a stressed observation.

    Attributes:
        chronic: Name of the chronic the sample was taken from.
        step: Timestep within the chronic.
        rho_max: Worst line loading at the probed observation.
        n_greedy: Candidates returned by `topology_search_topk` (Stage 0).
        n_stage1: Candidates left after requiring rho to drop below `rho_n0` (Stage 1).
        n_stage2_finite: Stage-1 survivors that also scored finite under all contingencies.
        best_attacked_rho: Best (lowest) worst-case rho achieved, or None if none was finite.
        would_save: Whether the real teacher would have written a row here.
    """

    chronic: str
    step: int
    rho_max: float
    n_greedy: int
    n_stage1: int
    n_stage2_finite: int
    best_attacked_rho: float | None
    would_save: bool


@dataclass
class DiagnosticResult:
    """Aggregate outcome across all probes for one grid."""

    env_name: str
    lines_to_attack: list[int]
    n_unitary_actions: int | None = None
    samples: list[ProbeSample] = field(default_factory=list)

    def summary(self) -> dict:
        """Reduce the samples to the counts that decide whether this grid can yield rows."""
        n = len(self.samples)
        n_save = sum(sample.would_save for sample in self.samples)
        n_stage1_empty = sum(sample.n_stage1 == 0 for sample in self.samples)
        n_stage2_empty = sum(
            sample.n_stage1 > 0 and sample.n_stage2_finite == 0 for sample in self.samples
        )
        return {
            "n_probes": n,
            "n_would_save": n_save,
            "conversion_rate": (n_save / n) if n else None,
            "n_failed_stage1_no_candidate_fixes_rho": n_stage1_empty,
            "n_failed_stage2_all_candidates_blackout": n_stage2_empty,
        }


def _probe(teacher: NMinusOneTeacher, env, obs, chronic: str) -> ProbeSample:
    """Run the two N-1 filters at one observation, counting survivors at each stage."""
    all_actions = teacher.get_all_actions(env)
    greedy_action_set = topology_search_topk(env, obs, all_actions, top_k=teacher.top_k)

    rho_max = float(obs.rho.max())
    # Stage 1 — mirrors search_best_n_minus_one_action's action_set comprehension.
    action_set = [
        action for rho_delta, action in greedy_action_set if (rho_max - rho_delta) < teacher.rho_n0
    ]

    # Stage 2 — mirrors the loop that only accepts a finite calculate_attacked_max_rho.
    n_finite = 0
    best_attacked = np.inf
    for action in action_set:
        attacked_rho = teacher.calculate_attacked_max_rho(obs=obs, action=action)
        if np.isfinite(attacked_rho):
            n_finite += 1
            best_attacked = min(best_attacked, attacked_rho)

    return ProbeSample(
        chronic=chronic,
        step=int(obs.current_step),
        rho_max=rho_max,
        n_greedy=len(greedy_action_set),
        n_stage1=len(action_set),
        n_stage2_finite=n_finite,
        best_attacked_rho=None if not np.isfinite(best_attacked) else float(best_attacked),
        would_save=n_finite > 0,
    )


def diagnose(env_name: str, lines_to_attack: list[int], n_probes: int, seed: int) -> DiagnosticResult:
    """Probe the N-1 search at up to `n_probes` stressed, fully-connected observations.

    Args:
        env_name: Grid2Op environment name or path.
        lines_to_attack: Line ids used as N-1 contingencies.
        n_probes: How many stressed observations to evaluate before stopping.
        seed: Grid2Op environment seed.

    Returns:
        A `DiagnosticResult` whose `summary()` separates Stage-1 from Stage-2 failures.
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    env.seed(seed)

    teacher = NMinusOneTeacher(
        lines_to_attack=lines_to_attack,
        rho_n0_threshold=RHO_N0_THRESHOLD,
        rho_max_threshold=RHO_MAX_THRESHOLD,
        seed=seed,
    )
    result = DiagnosticResult(env_name=env_name, lines_to_attack=lines_to_attack)

    try:
        while len(result.samples) < n_probes:
            obs = env.reset()
            chronic = env.chronics_handler.get_name()
            done = False
            while not done and len(result.samples) < n_probes:
                rho_max = obs.rho.max()
                # The exact gate the teacher's N-1 branch uses.
                if RHO_N0_THRESHOLD <= rho_max < RHO_MAX_THRESHOLD and all(obs.line_status):
                    sample = _probe(teacher, env, obs, chronic)
                    result.samples.append(sample)
                    logging.info(
                        "probe %d/%d [%s@%d] rho=%.4f greedy=%d stage1=%d stage2_finite=%d save=%s",
                        len(result.samples), n_probes, chronic, sample.step, sample.rho_max,
                        sample.n_greedy, sample.n_stage1, sample.n_stage2_finite, sample.would_save,
                    )
                # Advance cheaply with do-nothing (+ reconnection), never the expensive search.
                obs, _, done, _ = env.step(find_best_line_to_reconnect(obs, env.action_space({})))
    finally:
        result.n_unitary_actions = len(teacher.all_actions) if teacher.all_actions is not None else None
        env.close()

    return result


def write_result(result: DiagnosticResult, out_path: Path) -> None:
    """Serialize a `DiagnosticResult` (samples plus summary) to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_name": result.env_name,
        "lines_to_attack": result.lines_to_attack,
        "n_unitary_actions": result.n_unitary_actions,
        "summary": result.summary(),
        "samples": [sample.__dict__ for sample in result.samples],
    }
    with open(out_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
