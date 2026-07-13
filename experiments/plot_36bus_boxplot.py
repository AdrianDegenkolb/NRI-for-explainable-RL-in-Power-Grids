"""Survival-step boxplot: RAPPO vs baselines on IEEE 36-bus.

Per-episode survived steps across 144 test episodes (same scenarios for all models).
Baselines: loaded from evaluations/ folder (post-training eval).
RAPPO+sparse: loaded from analysis/survival/survived_steps.npy.

Usage:
    PYTHONPATH=src conda run -n L2RPN python experiments/plot_36bus_boxplot.py
"""
from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent / "results"
MAX_STEPS = 8062

RUNS = {
    "PPO-MLP": ROOT / "2026_05_26_IEEE36/ppo_mlp/CustomPPO_MLP_4808022_629fd_2026-05-26_00-30-51/evaluations",
    "PPO-GNN": ROOT / "2026_05_26_IEEE36/ppo_gnn/CustomPPO_GNN_4808023_62f07_2026-05-26_00-30-51/evaluations",
    "RAPPO+sparse": None,  # loaded separately
}
RAPPO_SURV = (
    ROOT / "experiments/2026_07_07_IEEE36/rappo_static_download"
    / "CustomPPO_RARL_5835191_669ad_2026-07-07_19-22-16/analysis/survival/survived_steps.npy"
)
COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e"]


def load_from_evaluations(eval_dir: Path) -> np.ndarray:
    steps = []
    for scenario in sorted(eval_dir.iterdir()):
        meta = scenario / "episode_meta.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            steps.append(int(json.load(f)["nb_timestep_played"]))
    return np.array(steps)


def main() -> None:
    datasets: dict[str, np.ndarray] = {}
    for name, eval_dir in RUNS.items():
        if eval_dir is None:
            datasets[name] = np.load(RAPPO_SURV).astype(float)
        else:
            datasets[name] = load_from_evaluations(eval_dir)

    labels = list(datasets.keys())
    data   = [datasets[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5, linestyle="none"),
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for flier, color in zip(bp["fliers"], COLORS):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    # Max-steps reference line
    ax.axhline(MAX_STEPS, color="#888888", lw=1, ls="--", zorder=0)
    ax.text(len(labels) + 0.45, MAX_STEPS, "episode\nmax", ha="right", va="bottom",
            fontsize=7, color="#888888")

    # Completion % above each box
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for i, vals in enumerate(data, start=1):
        pct = 100 * (vals >= MAX_STEPS).mean()
        ax.text(i, ymax - 0.01 * span, f"{pct:.0f}% full",
                ha="center", va="top", fontsize=8, color="#333333")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Survived steps per episode", fontsize=10)
    ax.set_title("IEEE 36-bus — test performance\n(144 episodes, same scenarios)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path("36bus_boxplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
