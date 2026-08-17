"""Load a RAPPO checkpoint and visualize the latent graph for a few environment states.

Creates a grid of plots: each column = one timestep, rows = posterior and grid overlay.

Usage (from project root):
    PYTHONPATH=src conda run -n L2RPN python experiments/plot_latent_graph_states.py \
        --trial results/experiments/2026_07_07_IEEE14/rappo_sparse_multiseed/CustomPPO_RARL_5835192_c976a_2026-07-07_18-49-14 \
        --n-states 4 --output latent_graph_states.png
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)


def load_agent(trial_dir: Path, checkpoint_name: str, env_name: str):
    from core.loading import load_config, preprocess_config, load_rllib_agent
    from core.constants import RL_POLICY

    params = preprocess_config(load_config(trial_dir))
    env_config = params["env_config"]
    agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(trial_dir),
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config,
    )
    return agent, g2op_env, gym_wrapper


def extract_posterior(agent) -> np.ndarray:
    """Return posterior [E, K] after the last forward pass."""
    model = agent._rllib_agent.model
    posterior_t: torch.Tensor = model.get_posterior()
    if posterior_t.dim() == 3:
        posterior_t = posterior_t[0]
    return posterior_t.cpu().detach().numpy()


def collect_states(agent, g2op_env, n_states: int, skip_steps: int = 30):
    """Run the agent and collect (obs, posterior) every skip_steps RL steps."""
    obs = g2op_env.reset()
    collected = []
    reward, done = 0.0, False
    rl_step = 0

    while not done and len(collected) < n_states:
        is_rl = agent.activate_agent(obs)
        action = agent.activation_function(obs, reward, done)
        if is_rl:
            if rl_step % skip_steps == 0:
                posterior = extract_posterior(agent)  # [E, K]
                collected.append((obs.copy(), posterior))
            rl_step += 1
        obs, reward, done, _ = g2op_env.step(action)

    return collected


def draw_grid_state(ax, obs, posterior, env, node_styles, threshold: float = 0.3):
    """Draw one panel: power grid + latent edges from posterior."""
    from visualization.utils import visualize_graph, PlottingArgs
    from grid2op_env.observation_converter import GraphObservationConverter
    from rarl.graph import fully_connected_edge_index

    N = len(node_styles)
    n_line = env.n_line

    # Powerline edge index (OR → EX, both directions for the graph)
    pl_src = list(range(n_line)) + list(range(n_line, 2 * n_line))
    pl_dst = list(range(n_line, 2 * n_line)) + list(range(n_line))
    powerline_edge_index = np.array([pl_src, pl_dst])

    # Posterior interaction prob: sum over all non-null edge types (skip last)
    # posterior shape: [E, K]  →  interaction_prob: [E]
    interaction_prob = posterior[:, :-1].sum(axis=-1)  # [E]

    # Expand to [E, 2] for visualize_graph (col 0 = interaction, col 1 = null)
    null_prob = posterior[:, -1]
    latent_edge_probs = np.stack([interaction_prob, null_prob], axis=-1)  # [E, 2]

    args = PlottingArgs(
        num_nodes=N,
        node_styles=node_styles,
        powerline_edge_index=powerline_edge_index,
        latent_edge_probs=latent_edge_probs,
        latent_edge_weight=6.0,
        skip_last_edge_type=True,
        visualize_edge_prob_threshold=threshold,
        show_legend=False,
    )
    visualize_graph(args, ax=ax)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial",
        default=(
            "results/experiments/2026_07_07_IEEE14/rappo_sparse_multiseed/"
            "CustomPPO_RARL_5835192_c976a_2026-07-07_18-49-14"
        ),
    )
    parser.add_argument("--checkpoint", default="checkpoint_000008")
    parser.add_argument("--env", default="l2rpn_case14_sandbox_test")
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--skip", type=int, default=20,
                        help="Collect one state every N RL steps")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Probability threshold for showing latent edges")
    parser.add_argument("--output", default="latent_graph_states.png")
    args = parser.parse_args()

    trial_dir = Path(args.trial)

    print(f"Loading agent from {trial_dir.name} / {args.checkpoint} ...")
    agent, g2op_env, gym_wrapper = load_agent(trial_dir, args.checkpoint, args.env)

    from visualization.utils import get_node_styles
    from grid2op_env.observation_converter import GraphObservationConverter
    node_styles = get_node_styles(g2op_env, GraphObservationConverter)

    print(f"Collecting {args.n_states} states ...")
    states = collect_states(agent, g2op_env, args.n_states, skip_steps=args.skip)
    print(f"  → Got {len(states)} states")

    fig, axes = plt.subplots(1, len(states), figsize=(6 * len(states), 6))
    if len(states) == 1:
        axes = [axes]
    fig.suptitle(
        f"Latent graph — RAPPO-14 sparse (K=420), {args.checkpoint}",
        fontsize=13, y=1.01,
    )

    for col, (obs, posterior) in enumerate(states):
        ax = axes[col]
        n_active = int((posterior[:, :-1].sum(axis=-1) > args.threshold).sum())
        ax.set_title(f"Step {col * args.skip}\n({n_active} edges > {args.threshold})", fontsize=10)
        draw_grid_state(ax, obs, posterior, g2op_env, node_styles, threshold=args.threshold)

    fig.tight_layout()
    out = Path(args.output)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
