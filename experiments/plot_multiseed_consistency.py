"""Plot latent graph consistency across seeds.

Two panels:
  Left:  total_interaction_probability_mass over training steps per seed
  Right: current_beta (annealing schedule, same for all seeds — shown once)

Usage:
    python experiments/plot_multiseed_consistency.py
    python experiments/plot_multiseed_consistency.py --sparse   # rappo_sparse_multiseed
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("results/experiments/2026_07_07_IEEE14")
DIRS = {
    "dense": BASE / "rappo_mulitseed",
    "sparse": BASE / "rappo_sparse_multiseed",
}

COL_STEPS = "timesteps_total"
COL_MASS = "info/learner/reinforcement_learning_policy/learner_stats/relation_awareness/total_interaction_probability_mass"
COL_BETA = "info/learner/reinforcement_learning_policy/learner_stats/relation_awareness/current_beta"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_trials(experiment_dir: Path) -> list[tuple[int, pd.DataFrame]]:
    """Return list of (seed, dataframe) sorted by seed."""
    trials = []
    for csv_path in sorted(experiment_dir.glob("CustomPPO_*/progress.csv")):
        params_path = csv_path.parent / "params.json"
        seed = json.loads(params_path.read_text()).get("seed", -1)
        df = pd.read_csv(csv_path, usecols=lambda c: c in {COL_STEPS, COL_MASS, COL_BETA})
        df = df.dropna(subset=[COL_MASS])
        trials.append((seed, df))
    return sorted(trials, key=lambda x: x[0])


def compute_shaded(trials: list) -> tuple:
    """Return (steps, mean, std) with rolling smoothing."""
    all_steps = sorted(set(step for _, df in trials for step in df[COL_STEPS]))
    aligned = pd.DataFrame({
        seed: df.set_index(COL_STEPS)[COL_MASS].reindex(all_steps).interpolate("index")
        for seed, df in trials
    })
    mean = aligned.mean(axis=1).rolling(5, min_periods=1, center=True).mean()
    std  = aligned.std(axis=1).rolling(5, min_periods=1, center=True).mean()
    return all_steps, mean, std


def plot_combined(output: str = "multiseed_consistency_combined.png") -> None:
    """Both variants in one coordinate system."""
    palette = {"dense": "#1f77b4", "sparse": "#d62728"}
    labels  = {"dense": "No sparsification (5 seeds)", "sparse": "Top-K sparse (5 seeds)"}

    fig, ax = plt.subplots(figsize=(9, 4))

    for key in ("dense", "sparse"):
        trials = load_trials(DIRS[key])
        if not trials:
            print(f"No trials found for {key}")
            continue
        steps, mean, std = compute_shaded(trials)
        c = palette[key]
        ax.plot(steps, mean, color=c, linewidth=2, label=labels[key])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=c)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Total interaction probability mass")
    ax.set_title("Active edge mass — mean ± std across 5 seeds")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(output)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="multiseed_consistency_combined.png")
    args = parser.parse_args()
    plot_combined(args.output)


if __name__ == "__main__":
    main()
