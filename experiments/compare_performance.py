"""Compare RAPPO vs baseline performance for IEEE14 and IEEE36 experiments.

Produces one figure per grid system with three panels:
  - Boxplot of per-episode survived steps (final eval)
  - Completed-episode percentage (final eval)
  - Training curve: grid2op_end_mean vs timesteps (from TensorBoard logs)

Data sources per trial:
  analysis/survival/survived_steps.npy  – per-episode steps (final eval)
  analysis/survival/summary.json        – aggregate final-eval stats
  events.out.tfevents.*                 – full training curves (~100 points/trial)

Usage:
    python experiments/compare_performance.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Common step grid for interpolation of training curves
_STEP_GRID = np.linspace(0, 200_000, 300)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

BASE = Path(__file__).parent.parent / "results"

EXPERIMENTS: dict[str, Path] = {
    "IEEE14": BASE / "2026_07_18_IEEE14",
    "IEEE36": BASE / "2026_07_18_IEEE36",
}

MAX_STEPS: dict[str, int] = {
    "IEEE14": 8064,
    "IEEE36": 8062,
}

# Display order (RAPPO variants first, then baselines)
MODEL_ORDER = [
    "rappo_multiseed_factored_kl",
    "rappo_multiseed_gcnconv",
    "rappo_multiseed",
    "ppo_gnn",
    "ppo_gnn_gcnconv",
    "ppo_mlp",
]

LABEL: dict[str, str] = {
    "rappo_multiseed_factored_kl": "RAPPO\n(factored KL)",
    "rappo_multiseed_gcnconv":     "RAPPO\n(GCN)",
    "rappo_multiseed":             "RAPPO",
    "ppo_gnn":                     "PPO-GNN",
    "ppo_gnn_gcnconv":             "PPO-GNN\n(GCN)",
    "ppo_mlp":                     "PPO-MLP",
}

COLOR: dict[str, str] = {
    "rappo_multiseed_factored_kl": "#1976D2",
    "rappo_multiseed_gcnconv":     "#42A5F5",
    "rappo_multiseed":             "#1976D2",
    "ppo_gnn":                     "#F57C00",
    "ppo_gnn_gcnconv":             "#FFC107",
    "ppo_mlp":                     "#D32F2F",
}


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

@dataclass
class TrialData:
    seed: Optional[int]
    survived_steps: Optional[np.ndarray]   # shape (n_episodes,)
    summary: Optional[dict]
    tb_steps: Optional[np.ndarray]         # training timesteps from TensorBoard
    tb_grid2op: Optional[np.ndarray]       # grid2op_end_mean at each step


@dataclass
class ModelData:
    key: str
    trials: list[TrialData] = field(default_factory=list)


def _get_seed(algo_dir: Path, trial_name: str) -> Optional[int]:
    for state_file in sorted(algo_dir.glob("experiment_state-*.json")):
        try:
            content = state_file.read_text().strip()
            if not content:
                continue
            d = json.loads(content)
        except Exception:
            continue
        for td in d.get("trial_data", []):
            if not isinstance(td, list):
                continue
            for item in td:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue
                logdir = item.get("relative_logdir") or item.get("logdir", "")
                if Path(logdir).name == trial_name:
                    return item.get("config", {}).get("seed")
    return None


def _load_tb_curve(trial_dir: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load (timesteps_total, grid2op_end_mean) arrays from TensorBoard event file."""
    ev_files = sorted(trial_dir.glob("events.out.tfevents*"))
    if not ev_files:
        return None, None
    ea = event_accumulator.EventAccumulator(str(ev_files[0]))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    step_tag = "ray/tune/timesteps_total"
    grid_tag = "ray/tune/custom_metrics/grid2op_end_mean"
    if step_tag not in tags or grid_tag not in tags:
        return None, None
    steps = np.array([s.value for s in ea.Scalars(step_tag)])
    grid = np.array([s.value for s in ea.Scalars(grid_tag)])
    return steps, grid


def _load_trial(trial_dir: Path, algo_dir: Path) -> Optional[TrialData]:
    surv_dir = trial_dir / "analysis" / "survival"

    # Skip trials without survival analysis (aborted early runs)
    if not surv_dir.exists():
        return None

    survived_steps: Optional[np.ndarray] = None
    npy_file = surv_dir / "survived_steps.npy"
    if npy_file.exists():
        survived_steps = np.load(npy_file).astype(float)

    summary: Optional[dict] = None
    summary_file = surv_dir / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())

    tb_steps, tb_grid2op = _load_tb_curve(trial_dir)

    return TrialData(
        seed=_get_seed(algo_dir, trial_dir.name),
        survived_steps=survived_steps,
        summary=summary,
        tb_steps=tb_steps,
        tb_grid2op=tb_grid2op,
    )


def load_experiment(exp_dir: Path) -> dict[str, ModelData]:
    models: dict[str, ModelData] = {}
    for algo_dir in sorted(exp_dir.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name not in LABEL:
            continue
        model = ModelData(key=algo_dir.name)
        for trial_dir in sorted(algo_dir.glob("CustomPPO_*")):
            trial = _load_trial(trial_dir, algo_dir)
            if trial is not None:
                model.trials.append(trial)
        if model.trials:
            models[algo_dir.name] = model
    return models


def ordered_models(models: dict[str, ModelData]) -> list[tuple[str, ModelData]]:
    return [(k, models[k]) for k in MODEL_ORDER if k in models]


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────

_SMOOTH_WINDOW = 15  # rolling average window (points on the 300-pt grid)


def _smooth(arr: np.ndarray) -> np.ndarray:
    kernel = np.ones(_SMOOTH_WINDOW) / _SMOOTH_WINDOW
    return np.convolve(arr, kernel, mode="same")


def _interp_curve(trial: TrialData) -> Optional[np.ndarray]:
    """Interpolate trial's training curve onto the common step grid."""
    if trial.tb_steps is None or trial.tb_grid2op is None:
        return None
    return np.interp(_STEP_GRID, trial.tb_steps, trial.tb_grid2op)


def _training_curve_stats(model: ModelData) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean ± std of grid2op_end_mean across seeds on common step grid."""
    curves = [_interp_curve(t) for t in model.trials]
    curves = [c for c in curves if c is not None]
    arr = np.stack(curves)  # (n_seeds, 300)
    return arr.mean(axis=0), arr.std(axis=0), len(curves)


# ──────────────────────────────────────────────
# Per-panel plot functions
# ──────────────────────────────────────────────

def plot_boxplot(ax: plt.Axes, ordered: list[tuple[str, ModelData]], max_steps: int) -> None:
    """Boxplot of per-episode survived steps (concatenated across seeds per model)."""
    all_steps = []
    labels = []
    colors = []

    for key, model in ordered:
        arrays = [t.survived_steps for t in model.trials if t.survived_steps is not None]
        if not arrays:
            continue
        combined = np.concatenate(arrays)
        all_steps.append(combined)
        n = len(model.trials)
        n_eps = len(combined)
        lbl = LABEL[key]
        labels.append(f"{lbl}\n(n={n} seeds,\n{n_eps} ep)")
        colors.append(COLOR[key])

    bp = ax.boxplot(
        all_steps,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4, linestyle="none"),
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax.axhline(max_steps, color="#888888", lw=1, ls="--", zorder=0)
    ax.text(len(all_steps) + 0.45, max_steps, "max", ha="right", va="bottom",
            fontsize=7, color="#888888")

    _, ymax = ax.get_ylim()
    for i, vals in enumerate(all_steps, start=1):
        pct = 100.0 * (vals >= max_steps).mean()
        ax.text(i, ymax * 0.98, f"{pct:.0f}%", ha="center", va="top", fontsize=8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Survived steps / episode", fontsize=9)
    ax.set_title("Final eval — episode distribution", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def plot_completion_bars(ax: plt.Axes, ordered: list[tuple[str, ModelData]]) -> None:
    """Bar chart of completed-episode percentage (final eval)."""
    labels, means, stds, colors = [], [], [], []

    for key, model in ordered:
        pcts = [t.summary["completed_episodes_pct"]
                for t in model.trials if t.summary]
        if not pcts:
            continue
        labels.append(LABEL[key])
        means.append(float(np.mean(pcts)))
        stds.append(float(np.std(pcts)))
        colors.append(COLOR[key])

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.6,
           error_kw={"linewidth": 1.5})

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(x[i], m + s + 0.5, f"{m:.0f}%", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Completed episodes (%)", fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("Final eval — completion rate", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_training_curves(ax: plt.Axes, ordered: list[tuple[str, ModelData]]) -> None:
    """Training curve: grid2op_end_mean vs agent timesteps (from TensorBoard)."""
    half = _SMOOTH_WINDOW // 2  # trim edge artefacts from convolution
    x = _STEP_GRID[half:-half] / 1_000

    for key, model in ordered:
        means, stds, n = _training_curve_stats(model)
        sm_mean = _smooth(means)[half:-half]
        color = COLOR[key]
        label = LABEL[key].replace("\n", " ")
        ax.plot(x, sm_mean, color=color, label=label, lw=1.8)
        if n > 1:
            sm_std = _smooth(stds)[half:-half]
            ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                            color=color, alpha=0.15)

    ax.set_xlabel("Agent timesteps (×1 000)", fontsize=9)
    ax.set_ylabel("grid2op_end_mean (steps)", fontsize=9)
    ax.set_title("Training curve (TensorBoard)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def make_figure(system: str, models: dict[str, ModelData], out_dir: Path) -> None:
    ordered = ordered_models(models)
    ms = MAX_STEPS[system]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Performance comparison — {system}", fontsize=13, fontweight="bold")

    plot_boxplot(axes[0], ordered, ms)
    plot_completion_bars(axes[1], ordered)
    plot_training_curves(axes[2], ordered)

    plt.tight_layout()
    for suffix in (".png", ".svg"):
        out = out_dir / f"compare_performance_{system}{suffix}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def main() -> None:
    out_dir = BASE
    for system, exp_dir in EXPERIMENTS.items():
        if not exp_dir.exists():
            print(f"Skipping {system}: {exp_dir} not found")
            continue
        models = load_experiment(exp_dir)
        print(f"\n{system}:")
        for k, m in ordered_models(models):
            seeds = [t.seed for t in m.trials]
            print(f"  {k}: {len(m.trials)} trials, seeds={seeds}")
        make_figure(system, models, out_dir)


if __name__ == "__main__":
    main()
