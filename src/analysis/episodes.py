"""
Shared per-episode table, so survival can be compared across agents (Teacher, Tutor, Junior,
Senior, RL policies) with one loader.

One row per (agent, chronic) episode, stored as CSV:
    agent           str    Label for plots, e.g. "teacher", "tutor", "junior", "senior".
    chronic         str    Chronic name (not path).
    seed            int    Env seed used for this chronic; -1 if unknown. Equal seeds on the same
                           chronic mean the same random maintenance schedule.
    steps_survived  int    Last step reached. For a truncated episode, a lower bound.
    max_step        int    Chronic length.
    survived        bool   steps_survived == max_step.
    truncated       bool   The run was cut off externally (e.g. SLURM walltime), not by a game over.
    n_decisions     int    States in which the agent was asked to act (rho above threshold); -1 if unknown.
    n_actions       int    Decisions in which the agent's own topology action (not do-nothing) was executed;
                           heuristic line reconnections alone do not count. -1 if unknown.
    n_illegal       int    Decisions in which the agent chose a topology action that the env rejected as
                           illegal (grid2op then plays do-nothing). -1 if unknown.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

EPISODE_COLUMNS = ["agent", "chronic", "seed", "steps_survived", "max_step", "survived", "truncated", "n_decisions",
                   "n_actions", "n_illegal"]
# Columns added later; older tables get them as -1 (unknown) when loaded.
_COUNT_COLUMNS = ("n_decisions", "n_actions", "n_illegal")

# Teacher (curriculumagent) log lines: "[<chronic>] Search stopped at <step>/<max_step>" at the end of
# a chronic, "[<chronic>] (<step>/<max_step>) ..." while it runs.
_TEACHER_STOP = re.compile(r"\[(?P<chronic>[^\]]+)\] Search stopped at (?P<step>\d+)/(?P<max>\d+)")
_TEACHER_STEP = re.compile(r"\[(?P<chronic>[^\]]+)\] \((?P<step>\d+)/(?P<max>\d+)\)")


def make_episodes(rows: list[dict]) -> pd.DataFrame:
    """Build an episode table from row dicts, filling `survived` and defaults.

    Args:
        rows: Dicts with at least agent, chronic, steps_survived, max_step.

    Returns:
        DataFrame with exactly `EPISODE_COLUMNS`.
    """
    df = pd.DataFrame(rows)
    for col, default in (("seed", -1), ("truncated", False), *((c, -1) for c in _COUNT_COLUMNS)):
        df[col] = df[col].fillna(default).astype(type(default)) if col in df else default
    df["survived"] = df["steps_survived"] >= df["max_step"]
    return df[EPISODE_COLUMNS]


def save_episodes(df: pd.DataFrame, path: Path) -> None:
    """Write an episode table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[EPISODE_COLUMNS].to_csv(path, index=False)


def load_episodes(*paths: Path) -> pd.DataFrame:
    """Read and concatenate episode tables, e.g. one per agent, for a joint plot.

    Tables written before a count column existed get it as -1 (unknown).
    """
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for col in _COUNT_COLUMNS:
        df[col] = df[col].fillna(-1).astype(int) if col in df else -1
    return df[EPISODE_COLUMNS]


def teacher_episodes_from_logs(log_dir: Path, agent: str = "teacher", seed: int = -1) -> pd.DataFrame:
    """Recover per-chronic survival of a Teacher array run from its `shard_*.err` logs.

    A chronic without a "Search stopped" line was cut off by the walltime; its last logged step
    is kept as a lower bound with `truncated=True`. The Teacher logs only states it acts in, so
    that bound can be far below the step the run actually reached.

    Args:
        log_dir: Teacher experiment directory (e.g. results/teacher/2026_09_17_teacher_n1_case36).
        agent: Label for the `agent` column.
        seed: Seed the Teacher used for every chronic, if known.

    Returns:
        Episode table; `n_decisions` is unknown (-1).
    """
    stopped: dict[str, tuple[int, int]] = {}
    last_seen: dict[str, tuple[int, int]] = {}
    for log in sorted(log_dir.glob("*.err")):
        for line in log.read_text(errors="replace").splitlines():
            if m := _TEACHER_STOP.search(line):
                stopped[m["chronic"]] = (int(m["step"]), int(m["max"]))
            elif m := _TEACHER_STEP.search(line):
                last_seen[m["chronic"]] = (int(m["step"]), int(m["max"]))
    rows = [{"agent": agent, "chronic": c, "seed": seed, "steps_survived": s, "max_step": m, "truncated": False}
            for c, (s, m) in stopped.items()]
    rows += [{"agent": agent, "chronic": c, "seed": seed, "steps_survived": s, "max_step": m, "truncated": True}
             for c, (s, m) in last_seen.items() if c not in stopped]
    return make_episodes(rows).sort_values("chronic", ignore_index=True)
