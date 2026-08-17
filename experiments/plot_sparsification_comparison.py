"""Visualize top-K sparsification effect on the latent graph.

Left panel:  all 3192 latent edges (no sparsification) — many thin edges.
Right panel: only top-K edges — sparse, interpretable graph.

Uses the saved posterior_mean_edge.npy from post-training analysis output.

Usage (from project root):
    PYTHONPATH=src conda run -n L2RPN python experiments/plot_sparsification_comparison.py \
        --analysis-dir <path/to/trial/analysis> [--k 420] [--output sparsification.png]
"""
import argparse
import glob
from pathlib import Path

import grid2op
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from grid2op.PlotGrid import PlotMatplot

from grid2op_env.observation_converter import GraphObservationConverter


# ── node layout helpers ──────────────────────────────────────────────────────

def _node_positions(env) -> dict[int, np.ndarray]:
    """Return {node_idx: [x, y]} for all 57 graph nodes."""
    plot_helper = PlotMatplot(env.observation_space)
    layout = plot_helper._grid_layout
    r = 20.0

    def _offset(sub_id: int, src: np.ndarray) -> np.ndarray:
        target = np.array(layout[f"sub_{sub_id}"])
        vec = target - src
        norm = np.linalg.norm(vec)
        return target if norm == 0 else target - (vec / norm) * r

    pointing = (
        [layout[f"sub_{sid}"] for sid in env.line_ex_to_subid]   # OR nodes point toward EX sub
        + [layout[f"sub_{sid}"] for sid in env.line_or_to_subid]  # EX nodes point toward OR sub
        + [layout.get(f"gen_{sid}_{gid}", layout[f"sub_{sid}"]) for gid, sid in enumerate(env.gen_to_subid)]
        + [layout.get(f"load_{sid}_{lid}", layout[f"sub_{sid}"]) for lid, sid in enumerate(env.load_to_subid)]
    )
    sub_ids = np.concatenate([
        env.line_or_to_subid, env.line_ex_to_subid,
        env.gen_to_subid, env.load_to_subid,
    ])
    return {i: _offset(sid, np.array(src)) for i, (sid, src) in enumerate(zip(sub_ids, pointing))}


def _node_colors(env) -> list[str]:
    n = env.n_line
    return ["#888888"] * 2 * n + ["#2ca02c"] * env.n_gen + ["#ff7f0e"] * env.n_load


def _node_sizes(env) -> list[int]:
    return [40] * 2 * env.n_line + [120] * env.n_gen + [100] * env.n_load


def _powerline_edges(env) -> list[tuple[int, int]]:
    """Edges connecting OR node i → EX node i+n_line for each powerline."""
    n = env.n_line
    return [(i, i + n) for i in range(n)]


# ── drawing ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, pos, node_colors, node_sizes, powerline_edges, probs, title, min_lw=0.3, max_lw=4.0):
    """Draw one panel: powerlines (dashed) + latent edges (solid, width ∝ prob)."""
    from rarl import fully_connected_edge_index

    G = nx.Graph()
    n_nodes = len(pos)
    G.add_nodes_from(range(n_nodes))

    # Latent edges
    edge_index = fully_connected_edge_index(n_nodes).numpy()  # [2, E]
    for e_idx, prob in enumerate(probs):
        if prob <= 0:
            continue
        src, dst = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        lw = min_lw + (max_lw - min_lw) * prob
        G.add_edge(src, dst, weight=lw, prob=float(prob))

    latent_edges = list(G.edges())
    latent_widths = [G[u][v]["weight"] for u, v in latent_edges]
    latent_alphas = [max(0.15, G[u][v]["prob"]) for u, v in latent_edges]

    # Draw latent edges (batched by alpha is tricky → draw all at once with mean alpha)
    if latent_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=latent_edges,
            width=latent_widths, alpha=0.4,
            edge_color="#1f77b4", arrows=False, ax=ax,
        )

    # Draw powerlines on top
    PG = nx.Graph()
    PG.add_nodes_from(range(n_nodes))
    for u, v in powerline_edges:
        PG.add_edge(u, v)
    nx.draw_networkx_edges(
        PG, pos, edgelist=powerline_edges,
        width=2.0, alpha=1.0, edge_color="black",
        style="--", arrows=False, ax=ax,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors, node_size=node_sizes,
        ax=ax, linewidths=0.5, edgecolors="black",
    )

    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir-dense",
        default=(
            "results/experiments/2026_07_07_IEEE14/rappo_mulitseed/"
            "CustomPPO_RARL_5829988_59bfa_2026-07-07_10-53-40/analysis"
        ),
        help="Analysis dir from the non-sparse run (for left panel)",
    )
    parser.add_argument("--k", type=int, default=420, help="Top-K budget for right panel")
    parser.add_argument("--env", default="l2rpn_case14_sandbox")
    parser.add_argument("--output", default="sparsification_comparison.png")
    args = parser.parse_args()

    # Load posterior mean edge probs from the non-sparse model
    files = glob.glob(f"{args.analysis_dir_dense}/**/posterior_mean_edge.npy", recursive=True)
    if not files:
        raise FileNotFoundError(f"posterior_mean_edge.npy not found under {args.analysis_dir_dense}")
    probs_full = np.load(files[0])   # [E] = [3192] for case14
    print(f"Loaded posterior: shape={probs_full.shape}, max={probs_full.max():.3f}")

    # Top-K mask
    k = min(args.k, len(probs_full))
    topk_idx = np.argpartition(probs_full, -k)[-k:]
    probs_sparse = np.zeros_like(probs_full)
    probs_sparse[topk_idx] = probs_full[topk_idx]

    # Grid2Op env for node layout
    env = grid2op.make(args.env)
    pos = _node_positions(env)
    node_colors = _node_colors(env)
    node_sizes = _node_sizes(env)
    pl_edges = _powerline_edges(env)

    n_active_full   = int((probs_full   > 0.05).sum())
    n_active_sparse = int((probs_sparse > 0.05).sum())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Top-K sparsification of the latent graph (RAPPO-14)", fontsize=14, y=1.01)

    _draw_panel(
        axes[0], pos, node_colors, node_sizes, pl_edges, probs_full,
        title=f"No sparsification\n({len(probs_full)} edges, {n_active_full} with p > 0.05)",
        min_lw=0.2, max_lw=3.0,
    )
    _draw_panel(
        axes[1], pos, node_colors, node_sizes, pl_edges, probs_sparse,
        title=f"Top-K sparsification (K={k})\n({n_active_sparse} edges retained)",
        min_lw=0.5, max_lw=5.0,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black",      lw=2,   linestyle="--", label="Power line"),
        Line2D([0], [0], color="#1f77b4",    lw=2,   linestyle="-",  label="Latent edge (width ∝ prob)"),
        Line2D([0], [0], marker="o",         color="#888888", lw=0, markersize=8, label="Line node"),
        Line2D([0], [0], marker="p",         color="#2ca02c", lw=0, markersize=8, label="Generator"),
        Line2D([0], [0], marker="^",         color="#ff7f0e", lw=0, markersize=8, label="Load"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    out = Path(args.output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
