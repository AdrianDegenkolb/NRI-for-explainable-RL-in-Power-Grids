#!/usr/bin/env python3
"""
Interactive episode viewer for RAPPO/RAGNN agents.

Loads an agent from a checkpoint, runs one episode, then shows a two-panel
matplotlib viewer with a time slider:
  - Left:  Substation-level grid coloured by line congestion (green → red).
           The substation acted upon at each step is highlighted in gold.
  - Right: NRI node graph with latent edges from the encoder posterior
           (only for RL-activated steps; unlabelled for heuristic steps).

Usage (from project root):
    PYTHONPATH=$(pwd)/src python experiments/episode_viewer.py \\
        --checkpoint results/.../trial_dir \\
        --checkpoint-name checkpoint_000020 \\
        --chronic-idx 0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation

from core.constants import RL_POLICY
from core.loading import load_config, preprocess_config, load_rllib_agent
from grid2op_env.observation_converter import (
    GraphObservationConverter, EDGE_INDEX, EDGE_MASK,
)
from visualization.utils import (
    GridPlottingArgs, PlottingArgs,
    get_node_styles, visualize_grid, visualize_graph,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_RHO_CMAP = plt.cm.RdYlGn_r


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """All data captured at one environment step."""

    obs: BaseObservation
    """Grid2Op observation *before* the action was applied."""

    action: BaseAction
    """Action chosen by the agent."""

    posterior: Optional[npt.NDArray]
    """NRI encoder posterior [N*(N-1), K], or None for heuristic/MLP steps."""

    pg_edge_index: npt.NDArray
    """Active powerline + topology edges [2, E_active]."""

    is_rl_step: bool
    """True when the RL policy was activated (rho > threshold)."""

    reward: float
    """Reward received after applying this action."""

    done: bool
    """Whether the episode ended after this step."""

    episode_step: int = 0
    """Original step index within the full episode (including heuristic steps)."""


# ── Episode runner ──────────────────────────────────────────────────────────

def run_episode(agent, g2op_env, chronic_idx: int) -> list[StepRecord]:
    """
    Play one episode and collect per-step records.

    :param agent: Loaded :class:`RllibAgent` instance.
    :param g2op_env: Raw Grid2Op environment (``load_rllib_agent`` third return).
    :param chronic_idx: Index of the chronic (scenario) to play.
    :return: List of :class:`StepRecord`, one per environment step.
    """
    g2op_env.set_id(chronic_idx)
    obs = g2op_env.reset()

    steps: list[StepRecord] = []
    done = False
    total_reward = 0.0

    while not done:
        # agent.act() calls gym_wrapper.update_obs() internally → cur_gym_obs is fresh
        action = agent.act(obs, total_reward, done)
        # activate_agent reads self.rho_max set inside act() → always consistent
        is_rl = agent.activate_agent(obs)

        # Active edges from gym wrapper (already updated above)
        edge_index_padded = agent.gym_wrapper.cur_gym_obs[EDGE_INDEX]
        edge_mask = agent.gym_wrapper.cur_gym_obs[EDGE_MASK]
        pg_edge_index = edge_index_padded[:, edge_mask]  # [2, E_active]

        # NRI posterior (only available for RL steps with an NRI model)
        posterior: Optional[npt.NDArray] = None
        if is_rl:
            try:
                post = agent._rllib_agent.model.get_posterior()  # [B, E, K]
                if post.dim() == 3:
                    post = post[0]
                posterior = post.cpu().detach().numpy()
            except AttributeError:
                pass  # MLP / GNN baseline — no encoder

        new_obs, reward, done, _ = g2op_env.step(action)

        steps.append(StepRecord(
            obs=obs,
            action=action,
            posterior=posterior,
            pg_edge_index=pg_edge_index,
            is_rl_step=is_rl,
            reward=reward,
            done=done,
            episode_step=len(steps),
        ))

        obs = new_obs
        total_reward += reward

    return steps


# ── Colour helpers ──────────────────────────────────────────────────────────

_DISCONNECTED_COLOR = "#aaaaaa"
_TOPO_EDGE_COLOR = "#cccccc"
_ACTED_SUB_COLOR = "#FFD700"   # gold
_DEFAULT_SUB_COLOR = "white"


def _rho_color(rho: float) -> str:
    """Map rho ∈ [0, 1+] to a hex colour via RdYlGn_r (green → red)."""
    return mcolors.to_hex(_RHO_CMAP(min(rho, 1.0)))


def _line_colors_for_step(step: StepRecord, n_line: int) -> list[str]:
    """One colour per powerline: grey if disconnected, rho-mapped otherwise."""
    rho = step.obs.rho
    status = step.obs.line_status
    return [
        _DISCONNECTED_COLOR if not status[i] else _rho_color(rho[i])
        for i in range(n_line)
    ]


def _sub_colors_for_step(step: StepRecord, n_sub: int) -> list[str]:
    """White for every substation except acted ones, which are gold."""
    colors = [_DEFAULT_SUB_COLOR] * n_sub
    try:
        _, subs_impacted = step.action.get_topological_impact()
        for idx in np.where(subs_impacted)[0]:
            colors[int(idx)] = _ACTED_SUB_COLOR
    except Exception:
        pass
    return colors


def _edge_colors_for_step(step: StepRecord, n_line: int) -> list[str]:
    """
    One colour per edge in ``step.pg_edge_index``:
      - Powerline edge (or→ex or ex→or): rho-mapped colour.
      - Topology edge (intra-substation): light grey.
    """
    rho = step.obs.rho
    colors: list[str] = []
    for src, dst in step.pg_edge_index.T:
        if 0 <= src < n_line and dst == src + n_line:
            colors.append(_rho_color(rho[src]))
        elif n_line <= src < 2 * n_line and dst == src - n_line:
            colors.append(_rho_color(rho[src - n_line]))
        else:
            colors.append(_TOPO_EDGE_COLOR)
    return colors


def _acted_substations(step: StepRecord) -> list[int]:
    """Return list of substation indices touched by the action (may be empty)."""
    try:
        _, subs_impacted = step.action.get_topological_impact()
        return [int(i) for i in np.where(subs_impacted)[0]]
    except Exception:
        return []


# ── Interactive viewer ──────────────────────────────────────────────────────

class EpisodeViewer:
    """
    Matplotlib figure with a time slider for stepping through an episode.

    Left panel:  substation-level grid coloured by congestion.
    Right panel: NRI node graph with latent edges (when available).
    """

    def __init__(
        self,
        steps: list[StepRecord],
        env,
        node_styles,
        n_line: int,
        n_sub: int,
    ) -> None:
        self.steps = [s for s in steps if s.is_rl_step]
        self.env = env
        self.node_styles = node_styles
        self.n_line = n_line
        self.n_sub = n_sub
        self.num_nodes = len(node_styles)

        self._build_figure()

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(22, 10))
        self.fig.patch.set_facecolor("#f5f5f5")

        # Two main plot panels
        self.ax_grid = self.fig.add_axes([0.02, 0.12, 0.44, 0.80])
        self.ax_nri  = self.fig.add_axes([0.52, 0.12, 0.44, 0.80])

        # Rho colorbar between panels (thin strip)
        cbar_ax = self.fig.add_axes([0.47, 0.20, 0.02, 0.60])
        sm = ScalarMappable(cmap=_RHO_CMAP, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("ρ (congestion)", fontsize=9, labelpad=4)
        cbar.ax.tick_params(labelsize=8)

        # Status text at top
        self.status_text = self.fig.text(
            0.5, 0.96, "", ha="center", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Slider at bottom
        slider_ax = self.fig.add_axes([0.10, 0.03, 0.80, 0.04])
        self.slider = Slider(
            slider_ax, "RL Step",
            0, len(self.steps) - 1,
            valinit=0, valstep=1, valfmt="%d",
        )
        self.slider.on_changed(self._on_slider)

        # Initial render
        self._draw(0)

    def _draw(self, step_idx: int) -> None:
        step = self.steps[step_idx]

        self.ax_grid.cla()
        self.ax_nri.cla()

        # ── Left: substation grid ──────────────────────────────────────── #
        visualize_grid(
            GridPlottingArgs(
                env=self.env,
                line_colors=_line_colors_for_step(step, self.n_line),
                node_colors=_sub_colors_for_step(step, self.n_sub),
                node_size=700,
                show_legend=False,
            ),
            ax=self.ax_grid,
        )
        self.ax_grid.set_title("Power Grid  (green = safe · red = congested · gold = acted substation)",
                               fontsize=10, pad=6)

        # ── Right: NRI node graph ──────────────────────────────────────── #
        visualize_graph(
            PlottingArgs(
                num_nodes=self.num_nodes,
                node_styles=self.node_styles,
                powerline_edge_index=step.pg_edge_index,
                powerline_edge_colors=_edge_colors_for_step(step, self.n_line),
                latent_edge_probs=step.posterior,
                latent_edge_weight=5.0,
                visualize_edge_prob_threshold=0.5,
                show_legend=(step_idx == 0),
            ),
            ax=self.ax_nri,
        )
        nri_title = (
            "NRI Latent Graph  (coloured arcs = latent edges with P > 0.5)"
            if step.posterior is not None
            else "NRI Latent Graph  (heuristic step — no encoder output)"
        )
        self.ax_nri.set_title(nri_title, fontsize=10, pad=6)

        # ── Status text ────────────────────────────────────────────────── #
        rho_max = step.obs.rho.max()
        acted = _acted_substations(step)
        acted_str = f"Sub {', '.join(str(s) for s in acted)}" if acted else "do-nothing"
        step_type = "RL-activated" if step.is_rl_step else "heuristic"
        has_nri = " + NRI" if step.posterior is not None else ""
        self.status_text.set_text(
            f"RL step {step_idx + 1} / {len(self.steps)}  │  "
            f"episode step {step.episode_step}  │  "
            f"ρ_max = {rho_max:.3f}  │  "
            f"{step_type}{has_nri}  │  "
            f"action: {acted_str}  │  "
            f"reward = {step.reward:.3f}"
            + ("  │  DONE" if step.done else "")
        )

        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        self._draw(int(val))

    def show(self) -> None:
        """Display the interactive viewer (blocks until window is closed)."""
        plt.show()


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive step-by-step episode viewer for RAPPO/RAGNN agents."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the trial directory containing checkpoint_* subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-name", type=str, default=None,
        help="Checkpoint folder name (e.g. checkpoint_000020). Defaults to the latest.",
    )
    parser.add_argument(
        "--chronic-idx", type=int, default=0,
        help="Index of the chronic (scenario) to play. Default: 0.",
    )
    args = parser.parse_args()

    # Auto-select latest checkpoint if not specified
    checkpoint_name = args.checkpoint_name
    if checkpoint_name is None:
        candidates = sorted(args.checkpoint.glob("checkpoint_*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint_* directories found in {args.checkpoint}")
        checkpoint_name = candidates[-1].name
        print(f"Auto-selected checkpoint: {checkpoint_name}")

    print(f"Loading checkpoint: {args.checkpoint / checkpoint_name}")

    params = load_config(args.checkpoint)
    params = preprocess_config(params)
    env_config = params["env_config"]

    agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(args.checkpoint),
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=env_config["env_name"],
        env_config=env_config,
    )

    print(f"Running episode (chronic {args.chronic_idx}) …")
    steps = run_episode(agent, g2op_env, args.chronic_idx)
    print(f"Collected {len(steps)} steps  "
          f"(RL-activated: {sum(s.is_rl_step for s in steps)}, "
          f"with NRI: {sum(s.posterior is not None for s in steps)})")

    node_styles = get_node_styles(g2op_env, GraphObservationConverter)

    viewer = EpisodeViewer(
        steps=steps,
        env=g2op_env,
        node_styles=node_styles,
        n_line=g2op_env.n_line,
        n_sub=g2op_env.n_sub,
    )
    viewer.show()


if __name__ == "__main__":
    main()
