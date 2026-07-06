from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from analysis.analyze_latent_graphs.hypo3_action_effect_coupling import get_reconfigured_nodes
from analysis.post_training.interfaces import EpisodeAnalyzer, StepContext


# ---------------------------------------------------------------------------
# Grid visualisation helper (used by both save() and redraw_plots())
# ---------------------------------------------------------------------------

def _plot_action_freq_grid(env, sub_freq: npt.NDArray, out_dir: Path) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from visualization import GridPlottingArgs, visualize_grid

    reconfig_cmap = mpl.colormaps["YlOrRd"]
    zero_color = (0.85, 0.85, 0.85, 1.0)
    n_sub = env.n_sub
    vmax = float(sub_freq.max()) if sub_freq.max() > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    node_colors = [
        zero_color if sub_freq[s] <= 0.0 else reconfig_cmap(norm(sub_freq[s]))
        for s in range(n_sub)
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    visualize_grid(
        GridPlottingArgs(
            env=env,
            node_size=700,
            font_size=12,
            node_colors=node_colors,
            font_color="black",
            show_legend=False,
        ),
        ax=ax,
    )
    sm = ScalarMappable(cmap=reconfig_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Action frequency (normalised)")
    ax.set_title("Spatial action distribution (grid view)")
    fig.savefig(out_dir / "substation_action_frequency_grid.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "substation_action_frequency_grid.svg", bbox_inches="tight")
    plt.close(fig)


def redraw_plots(data_dir: Path, env_name: Optional[str] = None) -> None:
    """Regenerate all action-frequency plots from saved .npy files.

    :param data_dir: directory written by :meth:`TopologyActionAnalyzer.save`
                     (i.e. ``out_dir/actions``).
    :param env_name: Grid2Op environment name for the grid layout.
                     Falls back to the ``env_name.txt`` file in *data_dir* when omitted.
    """
    import grid2op

    d = data_dir
    if env_name is None:
        env_name_file = d / "env_name.txt"
        if not env_name_file.exists():
            raise FileNotFoundError(
                "env_name not given and env_name.txt not found in data_dir"
            )
        env_name = env_name_file.read_text().strip()

    sub_freq = np.load(d / "sub_action_freq.npy")
    action_steps = int(np.load(d / "action_steps.npy"))
    total_steps = int(np.load(d / "total_steps.npy"))

    n_sub = sub_freq.shape[0]
    acting_pct = sub_freq * 100.0
    vmax = max(float(acting_pct.max()), 1e-6)
    cmap = mpl.colormaps["YlOrRd"]
    zero_color = (0.85, 0.85, 0.85, 1.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    bar_colors = [zero_color if v <= 0.0 else cmap(norm(v)) for v in acting_pct]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 0.04]})
    ax, cax = axes
    ax.bar(np.arange(n_sub), acting_pct, color=bar_colors, edgecolor="none")
    ax.set_xlabel("Substation index")
    ax.set_ylabel("Action frequency (% of action steps)")
    ax.set_title(f"Spatial action distribution ({action_steps} action steps / {total_steps} total)")
    ax.set_xlim(-0.5, n_sub - 0.5)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Action frequency (%)")
    plt.tight_layout()
    fig.savefig(d / "substation_action_frequency.png", dpi=150, bbox_inches="tight")
    fig.savefig(d / "substation_action_frequency.svg", bbox_inches="tight")
    plt.close(fig)

    env = grid2op.make(env_name)
    _plot_action_freq_grid(env, sub_freq, d)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TopologyActionAnalyzer(EpisodeAnalyzer):
    def __init__(self, env_name: Optional[str] = None) -> None:
        self._env_name = env_name
        self._node_action_counts: Optional[npt.NDArray] = None
        self._sub_action_counts: Optional[npt.NDArray] = None
        self._action_steps: int = 0
        self._total_steps: int = 0

    def on_step(self, ctx: StepContext) -> None:
        obs = ctx.obs
        N = 2 * obs.n_line + obs.n_gen + obs.n_load
        n_sub = obs.n_sub
        if self._node_action_counts is None:
            self._node_action_counts = np.zeros(N, dtype=np.int64)
        if self._sub_action_counts is None:
            self._sub_action_counts = np.zeros(n_sub, dtype=np.int64)

        self._total_steps += 1
        reconfigured = get_reconfigured_nodes(ctx.action, obs)
        if not reconfigured:
            return

        self._action_steps += 1
        n_line = obs.n_line
        n_gen = obs.n_gen
        for node_idx in reconfigured:
            if 0 <= node_idx < N:
                self._node_action_counts[node_idx] += 1

        touched_subs: set = set()
        for node_idx in reconfigured:
            if node_idx < n_line:
                touched_subs.add(int(obs.line_or_to_subid[node_idx]))
            elif node_idx < 2 * n_line:
                touched_subs.add(int(obs.line_ex_to_subid[node_idx - n_line]))
            elif node_idx < 2 * n_line + n_gen:
                touched_subs.add(int(obs.gen_to_subid[node_idx - 2 * n_line]))
            else:
                touched_subs.add(int(obs.load_to_subid[node_idx - 2 * n_line - n_gen]))
        for s in touched_subs:
            self._sub_action_counts[s] += 1

    def save(self, out_dir: Path) -> None:
        import grid2op

        d = out_dir / "actions"
        d.mkdir(parents=True, exist_ok=True)

        if self._sub_action_counts is None:
            return

        if self._env_name:
            (d / "env_name.txt").write_text(self._env_name)

        np.save(d / "node_action_counts.npy", self._node_action_counts)
        np.save(d / "sub_action_counts.npy", self._sub_action_counts)
        np.save(d / "action_steps.npy", np.array(self._action_steps))
        np.save(d / "total_steps.npy", np.array(self._total_steps))

        acting = max(self._action_steps, 1)
        sub_freq = self._sub_action_counts.astype(np.float64) / acting
        np.save(d / "sub_action_freq.npy", sub_freq)

        n_sub = sub_freq.shape[0]
        acting_pct = sub_freq * 100.0
        vmax = max(float(acting_pct.max()), 1e-6)
        cmap = mpl.colormaps["YlOrRd"]
        zero_color = (0.85, 0.85, 0.85, 1.0)
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
        bar_colors = [zero_color if v <= 0.0 else cmap(norm(v)) for v in acting_pct]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 0.04]})
        ax, cax = axes
        ax.bar(np.arange(n_sub), acting_pct, color=bar_colors, edgecolor="none")
        ax.set_xlabel("Substation index")
        ax.set_ylabel("Action frequency (% of action steps)")
        ax.set_title(f"Spatial action distribution ({self._action_steps} action steps / {self._total_steps} total)")
        ax.set_xlim(-0.5, n_sub - 0.5)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="Action frequency (%)")
        plt.tight_layout()
        fig.savefig(d / "substation_action_frequency.png", dpi=150, bbox_inches="tight")
        fig.savefig(d / "substation_action_frequency.svg", bbox_inches="tight")
        plt.close(fig)

        if self._env_name:
            env = grid2op.make(self._env_name)
            _plot_action_freq_grid(env, sub_freq, d)
