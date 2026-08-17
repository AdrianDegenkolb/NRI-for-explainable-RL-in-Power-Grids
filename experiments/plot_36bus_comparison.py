"""Training curve comparison: RAPPO vs baselines on IEEE 36-bus system.

Two panels:
  Top:    episode_reward_mean (training rollout)
  Bottom: custom_metrics/grid2op_end_mean (survival steps)

Both smoothed with a rolling window. Old RAPPO (crashed at 17k) shown dashed.

Usage:
    PYTHONPATH=src conda run -n L2RPN python experiments/plot_36bus_comparison.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent / "results"

RUNS = {
    "PPO-MLP": ROOT / "2026_05_26_IEEE36/ppo_mlp/CustomPPO_MLP_4808022_629fd_2026-05-26_00-30-51/progress.csv",
    "PPO-GNN": ROOT / "2026_05_26_IEEE36/ppo_gnn/CustomPPO_GNN_4808023_62f07_2026-05-26_00-30-51/progress.csv",
    "RAPPO (crashed)": ROOT / "2026_05_26_IEEE36/rappo/CustomPPO_RARL_4808025_62cfb_2026-05-26_00-30-51/progress.csv",
    "RAPPO + sparsification": ROOT / "experiments/2026_07_07_IEEE36/rappo_static_download/CustomPPO_RARL_5835191_669ad_2026-07-07_19-22-16/progress.csv",
}

STYLE = {
    "PPO-MLP":                  dict(color="#1f77b4", lw=1.8, ls="-",  zorder=2),
    "PPO-GNN":                  dict(color="#2ca02c", lw=1.8, ls="-",  zorder=2),
    "RAPPO (crashed)":          dict(color="#d62728", lw=1.5, ls="--", zorder=1),
    "RAPPO + sparsification":   dict(color="#ff7f0e", lw=2.2, ls="-",  zorder=3),
}

SMOOTH = 9   # rolling-mean window (iterations)

# ── helpers ───────────────────────────────────────────────────────────────────

def smooth(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w, min_periods=1, center=True).mean()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timesteps_total"] = pd.to_numeric(df["timesteps_total"], errors="coerce")
    df["episode_reward_mean"] = pd.to_numeric(df["episode_reward_mean"], errors="coerce")
    df["custom_metrics/grid2op_end_mean"] = pd.to_numeric(
        df.get("custom_metrics/grid2op_end_mean", np.nan), errors="coerce"
    )
    return df.dropna(subset=["timesteps_total"]).sort_values("timesteps_total")


# ── plot ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_r, ax_s = axes

    for name, path in RUNS.items():
        df = load(path)
        steps = df["timesteps_total"] / 1_000  # → k steps
        kw = STYLE[name]

        # Training reward
        ax_r.plot(steps, smooth(df["episode_reward_mean"], SMOOTH), label=name, **kw)

        # Survival steps
        col = "custom_metrics/grid2op_end_mean"
        if col in df.columns and df[col].notna().any():
            ax_s.plot(steps, smooth(df[col], SMOOTH), **kw)

    # Crash annotation (drawn after data so y-limits are set)
    crash_steps = 17.4  # k
    for ax in axes:
        ax.axvline(crash_steps, color="#d62728", lw=0.9, ls=":", alpha=0.7)
        ylo, yhi = ax.get_ylim()
        ax.text(crash_steps + 1.5, ylo + 0.92 * (yhi - ylo),
                "crash", fontsize=7, color="#d62728", va="top")

    # Formatting
    ax_r.set_ylabel("Episode reward (mean)", fontsize=10)
    ax_s.set_ylabel("Survival steps (mean)", fontsize=10)
    ax_s.set_xlabel("Environment steps", fontsize=10)

    ax_r.set_title("IEEE 36-bus: training curves — RAPPO vs. baselines", fontsize=11, pad=6)
    ax_r.legend(fontsize=8, loc="upper left")

    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path("36bus_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
