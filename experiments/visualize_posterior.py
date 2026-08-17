"""
Visualize per-timestep encoder posterior predictions for a RAPPO checkpoint.

Runs the RAPPO-14 agent for several episodes and plots the per-step latent
edge probability distributions to reveal whether the encoder is making
coherent input-conditional predictions or collapsing to a fixed pattern.

Usage (from project root):
    PYTHONPATH=$(pwd)/src python experiments/visualize_posterior.py
    PYTHONPATH=$(pwd)/src python experiments/visualize_posterior.py --n_episodes 5 --output out.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation

from analysis.analyze_latent_graphs.agent_analysis_framework import (
    LatentGraphAnalysisAgent,
    PosteriorAnalyzer,
)
from core.constants import RL_POLICY
from core.loading import load_config, preprocess_config, load_rllib_agent
from grid2op_env.observation_converter import GraphObservationConverter
from visualization.utils import PlottingArgs, get_node_styles, visualize_graph

logging.basicConfig(level=logging.WARNING)

CHECKPOINT = Path(
    "results/2026_05_26_IEEE14/rappo/"
    "CustomPPO_RARL_4808021_62fbb_2026-05-26_00-30-52"
)


# ── posterior collector ────────────────────────────────────────────────────

class PosteriorCollector(PosteriorAnalyzer):
    """Accumulates per-step posteriors and powerline edge indices."""

    def __init__(self):
        self.posteriors: list[npt.NDArray] = []       # each: [E, K]
        self.edge_indices: list[npt.NDArray] = []     # each: [2, E'] powerline edges
        self.rho_max: list[float] = []
        self._episode_starts: list[int] = []          # index into posteriors at each episode boundary

    def on_rl_step(
        self,
        posterior: npt.NDArray,       # [E, K]
        prior: npt.NDArray,
        powergrid_graph: npt.NDArray, # [2, E'] real powerline edges
        observation: BaseObservation,
        environment: Environment,
        action: BaseAction,
    ):
        self.posteriors.append(posterior)
        self.edge_indices.append(powergrid_graph)
        self.rho_max.append(float(observation.rho.max()))

    def on_heuristic_step(self, powergrid_graph, observation, environment):
        pass

    def on_new_episode(self, chronic_id: str):
        print(f"  Episode: {chronic_id}")
        # Mark the start index of each new episode so we can sample one step per episode
        self._episode_starts.append(len(self.posteriors))

    def on_evaluation_end(self):
        pass


# ── summary plot ───────────────────────────────────────────────────────────

def plot_summary(posteriors_full: list[npt.NDArray], rho_max: list[float], output: Path):
    """4-panel summary using P(interaction) = posterior[:, :-1].sum(-1)."""
    p_interact = np.stack([p[:, :-1].sum(axis=-1) for p in posteriors_full])  # [T, E]
    T, E = p_interact.shape
    active = p_interact > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"RAPPO-14 encoder posterior  |  {T} RL-activated steps  |  {E} latent edges",
        fontsize=12,
    )

    ax = axes[0, 0]
    n_show = min(100, E)
    top_idx = np.argsort(p_interact.var(axis=0))[-n_show:]
    im = ax.imshow(p_interact[:, top_idx].T, aspect="auto", interpolation="nearest",
                   vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"P(interaction) heatmap — top {n_show} edges by variance")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Edge (ranked by variance)")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    vals = p_interact.ravel()
    ax.hist(vals, bins=60, density=True, color="steelblue", alpha=0.85)
    ax.axvline(0.5, color="red", ls="--", lw=1, label="0.5 threshold")
    ax.set_title(
        f"Distribution of P(interaction)\n"
        f"{(vals > 0.9).mean():.1%} > 0.9   |   {(vals < 0.1).mean():.1%} < 0.1"
    )
    ax.set_xlabel("P(interaction)")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[1, 0]
    n_active = active.sum(axis=1)
    ax.plot(n_active, lw=0.9, color="steelblue", alpha=0.9, label="# active edges")
    ax.axhline(n_active.mean(), color="steelblue", ls="--", lw=1,
               label=f"mean = {n_active.mean():.1f}")
    ax.set_ylabel("# active edges  (P > 0.5)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax.twinx()
    ax2.plot(rho_max, lw=0.7, color="orange", alpha=0.6, label="rho_max")
    ax2.set_ylabel("rho_max", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax.set_title("Active edges & rho_max per RL step")
    ax.set_xlabel("RL step")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax = axes[1, 1]
    mean_per_edge = p_interact.mean(axis=0)
    ax.plot(np.sort(mean_per_edge), color="steelblue")
    ax.fill_between(range(E), np.sort(mean_per_edge), alpha=0.2, color="steelblue")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="0.5 threshold")
    ax.set_title(
        f"Mean P(interaction) per edge (sorted)\n"
        f"{(mean_per_edge > 0.5).sum()} / {E} edges consistently active (mean > 0.5)"
    )
    ax.set_xlabel("Edge rank")
    ax.set_ylabel("Mean P(interaction)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


# ── per-step graph plots ───────────────────────────────────────────────────

def plot_per_step_graphs(
    posteriors_full: list[npt.NDArray],   # list of [E, K]
    edge_indices: list[npt.NDArray],       # list of [2, E'] powerline edges
    rho_max: list[float],
    node_styles,
    n_graphs: int,
    output: Path,
    episode_starts: list[int] | None = None,
):
    """Plot individual per-step latent graphs using the existing visualize_graph function."""
    T = len(posteriors_full)
    N = len(node_styles)

    # Pick one step per episode (midpoint of each episode's RL-activated steps).
    # Falls back to evenly-spaced if episode boundaries aren't available.
    if episode_starts and len(episode_starts) >= 2:
        boundaries = episode_starts + [T]  # add end sentinel
        episode_step_indices = []
        for ep_start, ep_end in zip(boundaries[:-1], boundaries[1:]):
            if ep_end > ep_start:
                episode_step_indices.append((ep_start + ep_end) // 2)
        step_indices = np.array(episode_step_indices[:n_graphs], dtype=int)
    else:
        step_indices = np.linspace(0, T - 1, n_graphs, dtype=int)

    actual_n = len(step_indices)
    ncols = 2
    nrows = (actual_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 6))
    axes = np.array(axes).ravel()
    fig.suptitle(
        f"Per-step latent graphs — one per episode  (threshold P > 0.5)\n"
        f"({actual_n} episodes shown)",
        fontsize=13,
    )

    for panel_i, step_i in enumerate(step_indices):
        posterior = posteriors_full[step_i]   # [E, K]
        edge_index = edge_indices[step_i]     # [2, E']
        n_active = (posterior[:, :-1].sum(axis=-1) > 0.5).sum()

        args = PlottingArgs(
            num_nodes=N,
            node_styles=node_styles,
            powerline_edge_index=edge_index,
            latent_edge_probs=posterior,
            visualize_edge_prob_threshold=0.5,
            show_legend=(panel_i == 0),
        )
        ax = axes[panel_i]
        visualize_graph(args, ax=ax)
        ax.set_title(f"Episode midpoint step {step_i}  |  rho_max={rho_max[step_i]:.2f}  |  {n_active} active edges",
                     fontsize=9)

    for ax in axes[actual_n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--n_graphs", type=int, default=6,
                        help="Number of individual per-step graphs to plot")
    parser.add_argument("--output", type=Path, default=Path("posterior_analysis.png"))
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    params = load_config(args.checkpoint)
    params = preprocess_config(params)
    env_config = params["env_config"]
    env_name = env_config["env_name"]

    agent, _g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(args.checkpoint),
        policy_name=RL_POLICY,
        checkpoint_name="checkpoint_000000",
        env_name=env_name,
        env_config=env_config,
    )

    node_styles = get_node_styles(gym_wrapper.env_gym.init_env, GraphObservationConverter)

    collector = PosteriorCollector()
    analysis_agent = LatentGraphAnalysisAgent(
        rllib_agent=agent,
        gym_wrapper=gym_wrapper,
        analysers=[collector],
    )

    print(f"Running {args.n_episodes} episodes …")
    analysis_agent.analyze(num_episodes=args.n_episodes)

    if not collector.posteriors:
        print("No RL-activated steps recorded — agent may not have been triggered.")
        return

    T = len(collector.posteriors)
    E, K = collector.posteriors[0].shape
    p_interact = np.stack([p[:, :-1].sum(axis=-1) for p in collector.posteriors])

    print(f"\nResults:")
    print(f"  RL-activated steps : {T}")
    print(f"  Latent edges       : {E}  (K={K} edge types)")
    print(f"  Mean P(interaction): {p_interact.mean():.4f}")
    print(f"  Bimodal fraction   : {((p_interact < 0.1) | (p_interact > 0.9)).mean():.3f}  (< 0.1 or > 0.9)")
    print(f"  Mean active / step : {(p_interact > 0.5).sum(axis=1).mean():.1f}")
    print(f"  Consistently active: {(p_interact.mean(axis=0) > 0.5).sum()} edges  (mean > 0.5)")

    plot_summary(collector.posteriors, collector.rho_max, args.output)

    graphs_output = args.output.with_name(args.output.stem + "_graphs" + args.output.suffix)
    plot_per_step_graphs(
        collector.posteriors,
        collector.edge_indices,
        collector.rho_max,
        node_styles,
        n_graphs=args.n_graphs,
        output=graphs_output,
        episode_starts=collector._episode_starts,
    )


if __name__ == "__main__":
    main()
