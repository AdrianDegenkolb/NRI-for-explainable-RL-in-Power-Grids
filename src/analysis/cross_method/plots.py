"""Cross-method visualizations: bar chart, Spearman heatmap, and grid overlay."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from analysis.cross_method.loader import MethodData

logger = logging.getLogger(__name__)

# One color per method; extended automatically for more than 3 methods.
_METHOD_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _method_color(i: int) -> str:
    return _METHOD_PALETTE[i % len(_METHOD_PALETTE)]


def _spearman_matrix(mat: npt.NDArray) -> npt.NDArray:
    """Compute pairwise Spearman r for all row pairs of *mat* [n, d] → [n, n]."""
    n = mat.shape[0]
    r = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r_val, _ = spearmanr(mat[i], mat[j])
            r[i, j] = r_val
            r[j, i] = r_val
    return r


# ---------------------------------------------------------------------------
# Plot 1: Grouped bar chart — substation action frequency
# ---------------------------------------------------------------------------

def plot_action_frequency_bars(
    methods: List[MethodData],
    ax: plt.Axes,
) -> None:
    """Grouped bar chart of mean substation action frequency, one group per substation.

    Each method contributes one bar per substation; error bars show std across seeds.

    :param methods: One :class:`MethodData` per method to compare.
    :param ax: Axes to draw on.
    """
    n_methods = len(methods)
    n_sub = methods[0].n_sub
    bar_width = 0.8 / n_methods
    x = np.arange(n_sub)

    for i, method in enumerate(methods):
        mean = method.action_freqs.mean(axis=0) * 100.0  # convert to percent
        std = method.action_freqs.std(axis=0) * 100.0
        offset = (i - (n_methods - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            mean,
            width=bar_width,
            yerr=std,
            label=method.name,
            color=_method_color(i),
            error_kw={"elinewidth": 0.8, "capsize": 2},
            edgecolor="none",
        )

    ax.set_xlabel("Substation index")
    ax.set_ylabel("Action frequency (% of action steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlim(-0.5, n_sub - 0.5)
    ax.legend(frameon=False)


# ---------------------------------------------------------------------------
# Plot 1b: Grouped bar chart — line congestion (mean ρ)
# ---------------------------------------------------------------------------

def plot_congestion_bars(
    methods: List[MethodData],
    ax: plt.Axes,
    line_labels: Optional[List[str]] = None,
) -> None:
    """Grouped bar chart of mean line congestion (mean ρ), one group per line.

    Each method contributes one bar per line; error bars show std across seeds.
    A dashed horizontal line marks the thermal limit at ρ = 1.

    :param methods: One :class:`MethodData` per method to compare.
    :param ax: Axes to draw on.
    :param line_labels: Optional list of line name strings for the x-axis ticks.
                        Falls back to integer indices when ``None``.
    """
    n_methods = len(methods)
    n_lines = methods[0].n_lines
    bar_width = 0.8 / n_methods
    x = np.arange(n_lines)

    for i, method in enumerate(methods):
        mean = method.mean_rhos.mean(axis=0)
        std = method.mean_rhos.std(axis=0)
        offset = (i - (n_methods - 1) / 2.0) * bar_width
        ax.bar(
            x + offset,
            mean,
            width=bar_width,
            yerr=std,
            label=method.name,
            color=_method_color(i),
            error_kw={"elinewidth": 0.8, "capsize": 2},
            edgecolor="none",
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.9, label="Thermal limit")
    ax.set_xlabel("Line")
    ax.set_ylabel("Mean ρ (line loading)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        line_labels if line_labels is not None else [str(i) for i in range(n_lines)],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_xlim(-0.5, n_lines - 0.5)
    ax.legend(frameon=False)


# ---------------------------------------------------------------------------
# Plot 2: Spearman rank-correlation heatmap
# ---------------------------------------------------------------------------

def plot_spearman_heatmap(
    methods: List[MethodData],
    ax: plt.Axes,
    target: str = "actions",
) -> None:
    """Pairwise Spearman-r heatmap across all seeds and methods.

    Rows/columns are individual (method, seed) runs; method boundaries are marked
    with black separator lines; tick labels are colored by method.

    :param methods: One :class:`MethodData` per method to compare.
    :param ax: Axes to draw on.
    :param target: ``'actions'`` uses substation frequencies; ``'congestion'`` uses mean ρ.
    """
    vectors: List[npt.NDArray] = []
    tick_labels: List[str] = []
    tick_colors: List[str] = []
    method_boundaries: List[int] = []  # cumulative run counts at each method boundary

    for i, method in enumerate(methods):
        data = method.action_freqs if target == "actions" else method.mean_rhos
        for seed_idx in range(data.shape[0]):
            vectors.append(data[seed_idx])
            tick_labels.append(f"{method.name} #{seed_idx}")
            tick_colors.append(_method_color(i))
        method_boundaries.append(len(vectors))

    mat = np.stack(vectors, axis=0)       # [total_runs, n_feat]
    r_matrix = _spearman_matrix(mat)      # [total_runs, total_runs]

    im = ax.imshow(r_matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman r", fraction=0.046, pad=0.04)

    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    # Draw separator lines between methods (skip the last boundary = end of matrix)
    for boundary in method_boundaries[:-1]:
        b = boundary - 0.5
        ax.axhline(b, color="black", linewidth=1.0)
        ax.axvline(b, color="black", linewidth=1.0)

    label = "Substation action frequency" if target == "actions" else "Line congestion (mean ρ)"
    ax.set_title(f"Pairwise Spearman r — {label}")


# ---------------------------------------------------------------------------
# Plot 3: Grid overlay via visualize_grid
# ---------------------------------------------------------------------------

def plot_grid_overlay(
    methods: List[MethodData],
    env,
    axes: Sequence[plt.Axes],
    freq_vmax: Optional[float] = None,
) -> Tuple[plt.cm.ScalarMappable, plt.cm.ScalarMappable]:
    """Overlay action frequency (node color) and mean congestion (edge color) on the grid.

    Draws one panel per method (averaged over its seeds) plus a final consensus panel
    (averaged over all seeds and methods). Node colors run white → red for action
    frequency; edge colors run green → red for mean ρ. Edge width is constant.

    :param methods: One :class:`MethodData` per method to compare.
    :param env: Instantiated Grid2Op environment providing the grid layout.
    :param axes: Sequence of ``len(methods) + 1`` axes (one per method + consensus).
    :param freq_vmax: Upper bound for the action-frequency colormap.
                      Defaults to the global mean-frequency maximum across all methods.
    :returns: ``(node_sm, edge_sm)`` — ScalarMappable objects suitable for colorbars.
    """
    from visualization import GridPlottingArgs, visualize_grid

    all_freqs = np.concatenate([m.action_freqs for m in methods], axis=0)  # [total, n_sub]
    all_rhos = np.concatenate([m.mean_rhos for m in methods], axis=0)      # [total, n_lines]

    if freq_vmax is None:
        freq_vmax = float(all_freqs.mean(axis=0).max())
        freq_vmax = max(freq_vmax, 1e-6)

    node_norm = mcolors.Normalize(vmin=0.0, vmax=freq_vmax)
    edge_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    node_cmap = plt.cm.Reds
    edge_cmap = plt.cm.RdYlGn_r

    # Build (title, sub_freq, mean_rho) for each panel
    panels = [
        (m.name, m.action_freqs.mean(axis=0), m.mean_rhos.mean(axis=0))
        for m in methods
    ]
    panels.append(("Consensus", all_freqs.mean(axis=0), all_rhos.mean(axis=0)))

    for ax, (title, sub_freq, mean_rho) in zip(axes, panels):
        node_colors = [
            mcolors.to_hex(node_cmap(node_norm(float(v)))) for v in sub_freq
        ]
        line_colors = [
            mcolors.to_hex(edge_cmap(edge_norm(float(np.clip(v, 0.0, 1.0)))))
            for v in mean_rho
        ]
        visualize_grid(
            GridPlottingArgs(
                env=env,
                node_colors=node_colors,
                line_colors=line_colors,
                line_widths=[2.5] * env.n_line,
                node_size=600,
                font_size=10,
                font_color="black",
                show_legend=False,
            ),
            ax=ax,
        )
        ax.set_title(title)

    node_sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    node_sm.set_array([])
    edge_sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    edge_sm.set_array([])
    return node_sm, edge_sm
