"""
Plot-ready summary of a Tutor run (`imitation.tutor`), so notebooks need neither the 2,592 raw
per-chronic files nor a grid2op env for the common plots.

`summarize_tutor_run` writes to `<out_dir>/`:
    episodes.csv     Survival per chronic in the shared format of `analysis.episodes`.
    states.npz       Every recorded state, without observations (n states, A actions):
                       chronic [n] str, step [n] int32, action [n] int16 (executed),
                       max_rho [n, A] float32 (NaN where invalid), valid [n, A] bool,
                       rho_current [n] float32 (max rho before acting),
                       meta_action_space, meta_rho_threshold, meta_do_nothing_margin.
                     Improvement of action a over do-nothing: max_rho[:, 0] - max_rho[:, a].
    actions.csv      One row per policy action: index (0 = do-nothing), substation (-1 for
                     do-nothing), n_elements (set_bus entries), teacher_rank (1 = most frequent
                     Teacher action, the action-space JSON order; 0 for do-nothing),
                     tutor_picks (times executed), tutor_best (times lowest valid max rho), action (JSON).
    substations.csv  One row per substation: n_actions (in the action space), tutor_picks,
                     tutor_pick_share (of all topology picks), tutor_best.

For a printed example state, `load_tutor_state` returns one state's full record from the raw
files; rebuild the observation with `env.observation_space.from_vect(state["obs"])` and its
labels with `imitation.dataset.soft_targets`.
"""
from __future__ import annotations

import json
from pathlib import Path

import grid2op
import numpy as np
import pandas as pd
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

from analysis.episodes import make_episodes, save_episodes
from imitation.tutor import DO_NOTHING_INDEX, load_policy_actions


def rho_slice(obs: BaseObservation) -> slice:
    """Position of `rho` in `obs.to_vect()` (the same for every observation of an env)."""
    start = 0
    for name, size in zip(obs.attr_list_vect, obs.shapes()):
        if name == "rho":
            return slice(start, start + int(size))
        start += int(size)
    raise KeyError("rho is not part of the observation vector")


def tutor_episodes(experience_dir: Path, agent: str = "tutor") -> pd.DataFrame:
    """Survival per chronic from the Tutor's per-chronic files."""
    rows = []
    for path in sorted(experience_dir.glob("*.npz")):
        with np.load(path) as data:
            rows.append({
                "agent": agent,
                "chronic": str(data["chronic"]),
                "seed": int(data["seed"]),
                "steps_survived": int(data["steps_survived"]),
                "max_step": int(data["max_step"]),
                "n_decisions": len(data["action"]),
                # The Tutor executes only actions whose simulation was legal.
                "n_actions": int((data["action"] != DO_NOTHING_INDEX).sum()),
                "n_illegal": 0,
            })
    return make_episodes(rows)


def tutor_states(experience_dir: Path, rho_idx: slice) -> dict[str, np.ndarray]:
    """Concatenate the per-state arrays of all chronics, dropping observations (see module docstring)."""
    parts: dict[str, list] = {k: [] for k in ("chronic", "step", "action", "max_rho", "valid", "rho_current")}
    meta = {}
    for path in sorted(experience_dir.glob("*.npz")):
        with np.load(path) as data:
            meta = {"meta_action_space": data["action_space"], "meta_rho_threshold": data["rho_threshold"],
                    "meta_do_nothing_margin": data["do_nothing_margin"]}
            n = len(data["action"])
            if n == 0:
                continue
            parts["chronic"].append(np.full(n, str(data["chronic"])))
            for key in ("step", "action", "max_rho", "valid"):
                parts[key].append(data[key])
            parts["rho_current"].append(data["obs"][:, rho_idx].max(axis=1))
    return {**{k: np.concatenate(v) for k, v in parts.items()}, **meta}


def best_action(max_rho: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Index of the lowest valid max rho per state; -1 if no action is valid."""
    rho = np.where(valid, max_rho, np.inf)
    best = rho.argmin(axis=1)
    return np.where(np.isfinite(rho.min(axis=1)), best, -1)


def action_table(actions: list, action_dicts: list[dict], states: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per policy action with its substation and how often the Tutor picked it.

    Args:
        actions: Policy actions (grid2op), do-nothing first, as from `load_policy_actions`.
        action_dicts: The action-space JSON entries, in file order (without do-nothing).
        states: Output of `tutor_states`.
    """
    n = len(actions)
    substation = np.full(n, -1)
    for i, action in enumerate(actions):
        if i == DO_NOTHING_INDEX:
            continue
        subs = np.flatnonzero(action.get_topological_impact()[1])
        if len(subs) != 1:
            raise ValueError(f"Action {i} changes {len(subs)} substations, expected 1")
        substation[i] = subs[0]
    best = best_action(states["max_rho"], states["valid"])
    return pd.DataFrame({
        "index": np.arange(n),
        "substation": substation,
        "n_elements": [0] + [sum(len(v) for v in d.get("set_bus", {}).values()) for d in action_dicts],
        "teacher_rank": np.arange(n),
        "tutor_picks": np.bincount(states["action"], minlength=n),
        "tutor_best": np.bincount(best[best >= 0], minlength=n),
        "action": ["{}"] + [json.dumps(d) for d in action_dicts],
    })


def substation_table(actions: pd.DataFrame, n_sub: int) -> pd.DataFrame:
    """Aggregate `action_table` per substation (substations without actions included, with zeros)."""
    topo = actions[actions["index"] != DO_NOTHING_INDEX]
    per_sub = topo.groupby("substation").agg(
        n_actions=("index", "size"), tutor_picks=("tutor_picks", "sum"), tutor_best=("tutor_best", "sum"))
    per_sub = per_sub.reindex(range(n_sub), fill_value=0).rename_axis("substation").reset_index()
    per_sub["tutor_pick_share"] = per_sub["tutor_picks"] / max(int(per_sub["tutor_picks"].sum()), 1)
    return per_sub[["substation", "n_actions", "tutor_picks", "tutor_pick_share", "tutor_best"]]


def summarize_tutor_run(experience_dir: Path, env_name: str, action_space_path: Path, out_dir: Path) -> None:
    """Write the plot-ready summary files described in the module docstring."""
    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        actions = load_policy_actions(env, action_space_path)
        states = tutor_states(experience_dir, rho_slice(env.reset()))
        if states["max_rho"].shape[1] != len(actions):
            raise ValueError(f"Run has {states['max_rho'].shape[1]} actions, {action_space_path} gives {len(actions)}")
        action_df = action_table(actions, json.loads(Path(action_space_path).read_text()), states)
        sub_df = substation_table(action_df, env.n_sub)
    finally:
        env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_episodes(tutor_episodes(experience_dir), out_dir / "episodes.csv")
    np.savez_compressed(out_dir / "states.npz", **states)
    action_df.to_csv(out_dir / "actions.csv", index=False)
    sub_df.to_csv(out_dir / "substations.csv", index=False)


def load_states(path: Path) -> dict[str, np.ndarray]:
    """Read `states.npz` into a dict of arrays."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_tutor_state(experience_dir: Path, chronic: str, step: int) -> dict[str, np.ndarray]:
    """Full raw record of one state (obs, max_rho, valid, action, forecasts) for printing an example."""
    with np.load(experience_dir / f"{chronic}.npz") as data:
        rows = np.flatnonzero(data["step"] == step)
        if len(rows) == 0:
            raise KeyError(f"No recorded state at step {step} in chronic {chronic}")
        i = rows[0]
        per_state = [k for k in data.files if data[k].ndim >= 1 and len(data[k]) == len(data["step"])]
        return {**{k: data[k][i] for k in per_state}, "chronic": np.str_(chronic)}
