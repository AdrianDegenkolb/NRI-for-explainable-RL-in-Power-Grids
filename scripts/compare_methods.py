"""Cross-method comparison of substation action frequency and line congestion.

Produces three figure types saved to --out-dir:

  action_frequency_bars.{png,svg}   — grouped bar chart, one bar per method per substation
  spearman_actions.{png,svg}        — pairwise Spearman-r heatmap for action frequencies
  spearman_congestion.{png,svg}     — pairwise Spearman-r heatmap for line congestion
  grid_overlay.{png,svg}            — grid view: node color = action freq, edge color = mean ρ

Usage:
    PYTHONPATH=$(pwd)/src python experiments/compare_methods.py \\
        --env-name l2rpn_case14_sandbox_test \\
        --out-dir results/cross_method/ieee14 \\
        --methods \\
            RAPPO:results/2026_07_18_IEEE14/rappo_multiseed_gcnconv \\
            PPO-GNN:results/2026_07_20_IEEE14/ppo_gnn_gcnconv \\
            PPO-MLP:results/2026_07_20_IEEE14/ppo_mlp
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.cross_method.loader import load_method, MethodData
from analysis.cross_method.plots import (
    plot_action_frequency_bars,
    plot_congestion_bars,
    plot_spearman_heatmap,
    plot_grid_overlay,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_method_arg(raw: str) -> tuple[str, Path]:
    """Parse ``'NAME:PATH'`` into ``(name, Path)``."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"Expected 'NAME:PATH', got: {raw!r}. "
            "Example: RAPPO:results/2026_07_18_IEEE14/rappo_multiseed_gcnconv"
        )
    name, path = raw.split(":", maxsplit=1)
    return name.strip(), Path(path.strip())


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s/{png,svg}", stem)


def _plot_bars(methods: list[MethodData], out_dir: Path) -> None:
    n_sub = methods[0].n_sub
    fig, ax = plt.subplots(figsize=(max(10, n_sub * 0.6), 4))
    plot_action_frequency_bars(methods, ax)
    ax.set_title("Substation action frequency by method (mean ± std over seeds)")
    plt.tight_layout()
    _save(fig, out_dir, "action_frequency_bars")


def _plot_congestion_bars(methods: list[MethodData], env, out_dir: Path) -> None:
    n_lines = methods[0].n_lines
    line_labels = list(env.name_line)
    fig, ax = plt.subplots(figsize=(max(12, n_lines * 0.55), 4))
    plot_congestion_bars(methods, ax, line_labels=line_labels)
    ax.set_title("Mean line congestion by method (mean ± std over seeds)")
    plt.tight_layout()
    _save(fig, out_dir, "congestion_bars")


def _plot_heatmap(methods: list[MethodData], out_dir: Path, target: str) -> None:
    n_runs = sum(m.n_seeds for m in methods)
    size = max(6, n_runs * 0.55)
    fig, ax = plt.subplots(figsize=(size, size))
    plot_spearman_heatmap(methods, ax, target=target)
    plt.tight_layout()
    _save(fig, out_dir, f"spearman_{target}")


def _plot_grid(methods: list[MethodData], env, out_dir: Path) -> None:
    n_panels = len(methods) + 1  # one per method + consensus
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    node_sm, edge_sm = plot_grid_overlay(methods, env, axes)

    # Two shared horizontal colorbars below the panels
    fig.subplots_adjust(bottom=0.20, wspace=0.05)
    cbar_node_ax = fig.add_axes([0.10, 0.07, 0.35, 0.03])
    cbar_edge_ax = fig.add_axes([0.55, 0.07, 0.35, 0.03])
    fig.colorbar(node_sm, cax=cbar_node_ax, orientation="horizontal",
                 label="Action frequency (mean over seeds)")
    fig.colorbar(edge_sm, cax=cbar_edge_ax, orientation="horizontal",
                 label="Mean line loading ρ  [green = low, red = high]")
    _save(fig, out_dir, "grid_overlay")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-name", required=True,
        help="Grid2Op environment name (e.g. l2rpn_case14_sandbox_test)",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory where all output figures are written",
    )
    parser.add_argument(
        "--methods", nargs="+", required=True, metavar="NAME:PATH",
        help="One or more 'METHOD_NAME:EXPERIMENT_DIR' pairs",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    methods: list[MethodData] = []
    for raw in args.methods:
        name, path = _parse_method_arg(raw)
        logger.info("Loading '%s' from %s", name, path)
        methods.append(load_method(name, path))

    # ------------------------------------------------------------------ plots 1 & 2
    _plot_bars(methods, out_dir)
    _plot_heatmap(methods, out_dir, target="actions")
    _plot_heatmap(methods, out_dir, target="congestion")

    # ------------------------------------------------------------------ plots 3 & 4 (need env)
    import grid2op
    logger.info("Creating Grid2Op env '%s'", args.env_name)
    env = grid2op.make(args.env_name)
    _plot_congestion_bars(methods, env, out_dir)
    _plot_grid(methods, env, out_dir)

    logger.info("Done. All figures written to %s", out_dir)


if __name__ == "__main__":
    main()
