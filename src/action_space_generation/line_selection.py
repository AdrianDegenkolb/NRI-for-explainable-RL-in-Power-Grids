"""
Pick which power lines the N-1 Teacher should treat as contingencies (`lines_to_attack`).

`curriculumagent.teacher.teachers.teacher_n_minus_1.NMinusOneTeacher` defaults `lines_to_attack`
to a hardcoded list of line indices from `l2rpn_neurips_2020_track1_small` — meaningless on other
grids. This module replaces that default with a cheap, grid-specific estimate: run a do-nothing
rollout over a handful of chronics and rank lines by how often they approach their thermal limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import grid2op
import numpy as np
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend


@dataclass
class LineStressStats:
    """Per-line stress statistics accumulated over a do-nothing rollout.

    Attributes:
        n_line: Number of lines in the grid.
        near_limit_count: Per-line count of timesteps with `rho > near_limit_threshold`.
        max_rho: Per-line maximum observed `rho`.
    """

    n_line: int
    near_limit_threshold: float = 0.8
    near_limit_count: np.ndarray = field(init=False)
    max_rho: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.near_limit_count = np.zeros(self.n_line, dtype=np.int64)
        self.max_rho = np.zeros(self.n_line, dtype=np.float64)

    def update(self, rho: np.ndarray) -> None:
        """Fold one timestep's per-line `rho` values into the running statistics."""
        self.near_limit_count += rho > self.near_limit_threshold
        np.maximum(self.max_rho, rho, out=self.max_rho)

    def rank_lines(self, n_lines: int) -> list[int]:
        """Return the `n_lines` most-stressed line ids.

        Ranks first by `near_limit_count` (how often a line got close to its limit — the
        statistic we actually want), falling back to `max_rho` to break ties or to still
        produce a sensible ranking if no chronic ever pushed a line above the threshold
        (e.g. very short do-nothing episodes on a fragile grid).

        Args:
            n_lines: How many line ids to return.

        Returns:
            Line ids sorted from most to least stressed, length `min(n_lines, n_line)`.
        """
        order = np.lexsort((-self.max_rho, -self.near_limit_count))
        return [int(line_id) for line_id in order[:n_lines]]


def _rollout_do_nothing(env: BaseEnv, stats: LineStressStats, n_chronics: int) -> None:
    """Step a do-nothing agent through `n_chronics` chronics, updating `stats` per timestep."""
    for _ in range(n_chronics):
        obs = env.reset()
        done = False
        stats.update(obs.rho)
        while not done:
            obs, _, done, _ = env.step(env.action_space({}))
            stats.update(obs.rho)


def select_lines_to_attack(
        env_name: str,
        n_lines: int = 10,
        n_chronics: int = 20,
        near_limit_threshold: float = 0.8,
) -> list[int]:
    """Select the `n_lines` lines most often near their thermal limit under a do-nothing agent.

    Args:
        env_name: Grid2Op environment name or path (e.g. "l2rpn_wcci_2020_train").
        n_lines: Number of line ids to return.
        n_chronics: Number of chronics to roll out. Kept small — this only needs to rank
            lines relative to each other, not measure absolute stress precisely.
        near_limit_threshold: `rho` threshold counted as "near limit" (below the teacher's own
            0.95/1.0 thresholds, since a do-nothing agent that never intervenes may die or
            plateau before reaching them).

    Returns:
        Line ids sorted from most to least stressed, to pass as `lines_to_attack` to
        `NMinusOneTeacher`.
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        stats = LineStressStats(n_line=env.n_line, near_limit_threshold=near_limit_threshold)
        _rollout_do_nothing(env, stats, n_chronics)
    finally:
        env.close()
    return stats.rank_lines(n_lines)
