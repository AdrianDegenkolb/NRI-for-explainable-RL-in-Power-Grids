"""Integrated Gradients: node features vs. edge interaction probability importance.

Quantifies how much the RAPPO policy relies on node features (observations)
vs. latent edge probabilities (graph structure) when choosing actions.

Method: Integrated Gradients w.r.t. the logit of the chosen action.
  - Input: (x_flat [N*D], edge_probs [E]) where E = edges in the FC graph.
  - Encoder is *bypassed*: edge_probs are injected directly as edge_type_posterior
    into RAGNN, so gradients flow through both input channels independently.
  - Baseline: (zeros, 0.5·ones) — "no observation, uniform edge prior".
  - Attribution: IG = (input − baseline) × avg_grad over 50 interpolation steps.
  - Aggregation: sum of |IG| for node part vs. edge part → importance ratio.

Usage (from project root):
    PYTHONPATH=src conda run -n L2RPN python experiments/ig_feature_importance.py \\
        [--trial <path>] [--checkpoint checkpoint_XXXXXX] \\
        [--env l2rpn_case14_sandbox] [--n-obs 30] [--n-steps 50] \\
        [--output ig_importance]
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

logging.basicConfig(level=logging.WARNING)

DEFAULT_TRIAL = (
    "results/experiments/2026_07_07_IEEE14/rappo_mulitseed/"
    "CustomPPO_RARL_5829988_59bfa_2026-07-07_10-53-40"
)
DEFAULT_CHECKPOINT = "checkpoint_000007"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(trial_dir: Path, checkpoint: str, env_name: str):
    """Load RAPPO agent via core.loading; return (agent, g2op_env)."""
    import ray
    from core.loading import load_config, preprocess_config, load_rllib_agent
    from core.constants import RL_POLICY

    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False)
    params = preprocess_config(load_config(trial_dir))
    agent, g2op_env, _ = load_rllib_agent(
        checkpoint_path=str(trial_dir),
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint,
        env_name=env_name,
        env_config=params["env_config"],
    )
    return agent, g2op_env


# ── observation collection ────────────────────────────────────────────────────

def collect_observations(
    agent,
    g2op_env,
    gym_wrapper,
    n_obs: int,
    rho_thresh: float = 0.5,
) -> list[dict]:
    """Collect gym obs dicts from agent policy rollouts, keeping only critical states.

    Runs the agent's own policy across chronics and stores observations where
    rho_max > rho_thresh. Iterates over all available chronics until n_obs are
    collected or all chronics are exhausted.

    :param agent: Loaded RllibAgent used to generate actions.
    :param g2op_env: Grid2Op environment.
    :param gym_wrapper: Gym wrapper to convert g2op obs to gym dicts.
    :param n_obs: Target number of observations.
    :param rho_thresh: Minimum rho_max to store an observation.
    :return: List of gym observation dicts (at most n_obs).
    """
    obs_list: list[dict] = []
    n_avail = len(g2op_env.chronics_handler.subpaths)

    for ci in range(n_avail):
        if len(obs_list) >= n_obs:
            break
        g2op_env.set_id(ci)
        g2op_obs = g2op_env.reset()
        done, reward = False, 0.0

        while not done:
            action = agent.act(g2op_obs, reward, done)
            g2op_obs, reward, done, _ = g2op_env.step(action)
            if float(g2op_obs.rho.max()) > rho_thresh:
                gym_wrapper.update_obs(g2op_obs)
                obs_list.append({k: v.copy() for k, v in gym_wrapper.cur_gym_obs.items()})
            if len(obs_list) >= n_obs:
                break

    return obs_list


# ── IG wrapper ────────────────────────────────────────────────────────────────

class _IGWrapper(torch.nn.Module):
    """Differentiable (x_flat, edge_probs) → logit[action] bypassing the encoder."""

    def __init__(self, gnn, mlp, edge_index: Tensor, batch: Tensor, node_dim: int):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("batch", batch)
        self.node_dim = node_dim

    def forward(self, x_flat: Tensor, edge_probs: Tensor, action_idx: int) -> Tensor:
        """
        :param x_flat: Flattened node features [N*D].
        :param edge_probs: Interaction probability per edge [E_fc].
        :param action_idx: Index of the action whose logit is returned.
        :return: Scalar logit.
        """
        x = x_flat.view(-1, self.node_dim)  # [N, D]
        # Build [E, 2] posterior: col-0=interact, col-1=no-edge
        edge_type_posterior = torch.stack([edge_probs, 1.0 - edge_probs], dim=-1)
        embedding = self.gnn(
            x=x,
            edge_index=self.edge_index,
            edge_type_posterior=edge_type_posterior,
            batch=self.batch,
        )  # [1, out_dim]
        logits, _ = self.mlp({"obs": embedding}, [], None)  # [1, n_actions]
        return logits[0, action_idx]


def _preprocess_obs(obs: dict, device: torch.device) -> tuple[Tensor, Tensor, int, int]:
    """Extract (x [N,D], batch [N], N, D) from a gym obs dict.

    Note: edge_index is not taken from the obs.  The obs stores powerline edges,
    but the encoder (and thus the posterior) runs over the fully-connected graph.
    We generate the FC edge_index separately via rarl.graph.fully_connected_edge_index.
    """
    from grid2op_env.observation_converter import NODES

    nodes = torch.tensor(obs[NODES], dtype=torch.float32, device=device)  # [N, D]
    N, D = nodes.shape
    batch = torch.zeros(N, dtype=torch.long, device=device)
    return nodes, batch, N, D


# ── integrated gradients ──────────────────────────────────────────────────────

def integrated_gradients(
    wrapper: _IGWrapper,
    x: Tensor,
    edge_probs: Tensor,
    x_baseline: Tensor,
    edge_baseline: Tensor,
    action_idx: int,
    n_steps: int = 50,
) -> tuple[Tensor, Tensor]:
    """
    Compute Integrated Gradients for (x_flat, edge_probs).

    Model parameters are frozen during IG so that only input gradients are
    accumulated.

    :return: (ig_x [N*D], ig_edge [E_fc]) — signed attributions.
    """
    x_flat = x.view(-1)
    x_base_flat = x_baseline.view(-1)

    # Freeze model params — gradients still pass through them to reach xi/ei
    for p in wrapper.parameters():
        p.requires_grad_(False)

    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x_flat.device)
    sum_grad_x = torch.zeros_like(x_flat)
    sum_grad_e = torch.zeros_like(edge_probs)

    for alpha in alphas:
        xi = (x_base_flat + alpha * (x_flat - x_base_flat)).detach().requires_grad_(True)
        ei = (edge_baseline + alpha * (edge_probs - edge_baseline)).detach().requires_grad_(True)
        out = wrapper(xi, ei, action_idx)
        out.backward()
        sum_grad_x = sum_grad_x + xi.grad
        sum_grad_e = sum_grad_e + ei.grad

    avg_grad_x = sum_grad_x / (n_steps + 1)
    avg_grad_e = sum_grad_e / (n_steps + 1)

    ig_x = (x_flat - x_base_flat) * avg_grad_x
    ig_e = (edge_probs - edge_baseline) * avg_grad_e
    return ig_x.detach(), ig_e.detach()


# ── main analysis ─────────────────────────────────────────────────────────────

def run_ig_analysis(
    agent,
    g2op_env,
    observations: list[dict],
    n_steps: int,
) -> dict:
    """
    Run IG for each observation; return aggregated attributions.

    :return: dict with keys:
        - ``ratio_nodes``: mean fraction of |IG| attributed to node features
        - ``ratio_edges``: mean fraction attributed to edge probs
        - ``per_node_ig``: [T, N, D] signed IG for node features
        - ``per_edge_ig``: [T, E] signed IG for edge probs
        - ``edge_probs_all``: [T, E] actual edge interaction probs
        - ``actions``: [T] chosen action indices
    """
    from core.constants import RL_POLICY

    rllib_agent = agent._rllib_agent
    model = rllib_agent.model
    device = next(model.parameters()).device

    from rarl.graph import fully_connected_edge_index

    gnn = model.ragnn.gnn
    mlp = model.mlp

    per_node_ig_list: list[np.ndarray] = []
    per_edge_ig_list: list[np.ndarray] = []
    edge_probs_list: list[np.ndarray] = []
    action_list: list[int] = []
    ratio_nodes_list: list[float] = []

    # Build wrapper once (N and D are fixed for a given grid/model)
    first_obs = observations[0]
    _, batch0, N, D = _preprocess_obs(first_obs, device)
    edge_index_fc = fully_connected_edge_index(N, device=device, self_loops=False)
    wrapper = _IGWrapper(gnn, mlp, edge_index_fc, batch0, D).to(device)
    wrapper.eval()

    for t, obs in enumerate(observations):
        # Forward pass to get action and posterior
        with torch.no_grad():
            action, _, _ = rllib_agent.compute_single_action(obs, policy_id=RL_POLICY)

        posterior = model.get_posterior()  # [1, E_fc, K]
        edge_probs_t = posterior[0, :, :-1].sum(-1)  # [E_fc] interaction probs

        x, _, _, _ = _preprocess_obs(obs, device)
        E = edge_probs_t.shape[0]  # N*(N-1)

        # Baselines
        x_baseline = torch.zeros_like(x)
        edge_baseline = torch.full((E,), 0.5, device=device)

        ig_x, ig_e = integrated_gradients(
            wrapper, x, edge_probs_t, x_baseline, edge_baseline, int(action), n_steps
        )

        # Per-node: reshape [N*D] → [N, D], sum over feature dim → [N]
        ig_nodes = ig_x.view(N, D)
        total_nodes = ig_nodes.abs().sum().item()
        total_edges = ig_e.abs().sum().item()
        total = total_nodes + total_edges + 1e-12
        ratio_nodes_list.append(total_nodes / total)

        per_node_ig_list.append(ig_nodes.cpu().numpy())
        per_edge_ig_list.append(ig_e.cpu().numpy())
        edge_probs_list.append(edge_probs_t.cpu().numpy())
        action_list.append(int(action))

        if (t + 1) % 5 == 0:
            print(f"  IG {t+1}/{len(observations)}: nodes={100*total_nodes/total:.1f}%  "
                  f"edges={100*total_edges/total:.1f}%  action={action}")

    return {
        "ratio_nodes": float(np.mean(ratio_nodes_list)),
        "ratio_edges": float(1.0 - np.mean(ratio_nodes_list)),
        "ratio_nodes_all": np.array(ratio_nodes_list),
        "per_node_ig": np.stack(per_node_ig_list),   # [T, N, D]
        "per_edge_ig": np.stack(per_edge_ig_list),   # [T, E]
        "edge_probs_all": np.stack(edge_probs_list), # [T, E]
        "actions": np.array(action_list),             # [T]
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, output_prefix: str, trial_name: str) -> None:
    """Three-panel figure: importance ratio, per-node bar, edge scatter."""

    ratio_nodes = results["ratio_nodes"]
    ratio_edges = results["ratio_edges"]
    ratio_all = results["ratio_nodes_all"]

    # Mean |IG| per node (averaged over T and D)
    per_node_ig = results["per_node_ig"]  # [T, N, D]
    node_importance = np.abs(per_node_ig).mean(axis=(0, 2))  # [N]

    # Edge importance: mean |IG| per edge across T
    per_edge_ig = results["per_edge_ig"]   # [T, E]
    edge_ig_mean = np.abs(per_edge_ig).mean(axis=0)  # [E]
    edge_probs_mean = results["edge_probs_all"].mean(axis=0)  # [E]

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(10, 5))

    # ── Panel A: overall importance ratio ────────────────────────────────────
    bars = ax_a.bar(
        ["Node features", "Edge probabilities"],
        [ratio_nodes, ratio_edges],
        color=["#1f77b4", "#ff7f0e"],
        alpha=0.85,
        width=0.5,
    )
    for bar, val in zip(bars, [ratio_nodes, ratio_edges]):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{100*val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax_a.set_ylim(0, 1.15)
    ax_a.set_ylabel("Fraction of total |IG|", fontsize=9)
    ax_a.set_title("A: Overall importance\n(mean across observations)", fontsize=9)
    ax_a.grid(True, axis="y", alpha=0.4)
    ax_a.spines[["top", "right"]].set_visible(False)

    # ── Panel D: edge interaction prob vs. |IG| ───────────────────────────────
    # Subsample if E is large
    E = len(edge_ig_mean)
    idx = np.arange(E)
    if E > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(E, size=2000, replace=False)
    ax_d.scatter(
        edge_probs_mean[idx],
        edge_ig_mean[idx],
        s=4, alpha=0.4, color="#1f77b4",
    )
    ax_d.set_xlabel("Mean interaction probability p(z|x)", fontsize=9)
    ax_d.set_ylabel("Mean |IG| for edge prob", fontsize=9)
    ax_d.set_title("D: Edge importance vs. interaction probability", fontsize=9)
    ax_d.grid(True, alpha=0.4)
    ax_d.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Integrated Gradients — node features vs. edge probabilities\n"
        f"{trial_name}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = Path(f"{output_prefix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out.resolve()}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default=DEFAULT_TRIAL,
                        help="Path to the Ray Tune trial directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint sub-folder name (e.g. checkpoint_000007)")
    parser.add_argument("--env", default="l2rpn_case14_sandbox",
                        help="Grid2Op environment name")
    parser.add_argument("--n-obs", type=int, default=30,
                        help="Number of observations to average IG over")
    parser.add_argument("--rho-thresh", type=float, default=0.5,
                        help="Minimum rho_max to store an observation (default: 0.5)")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of integration steps for IG")
    parser.add_argument("--output", default="ig_importance",
                        help="Output file prefix (without extension)")
    args = parser.parse_args()

    trial_dir = Path(args.trial)
    print(f"Loading agent: {trial_dir.name} / {args.checkpoint}")
    agent, g2op_env = load_model(trial_dir, args.checkpoint, args.env)

    print(f"Collecting {args.n_obs} observations (rho_thresh={args.rho_thresh}) ...")
    gym_wrapper = agent.gym_wrapper
    observations = collect_observations(agent, g2op_env, gym_wrapper, args.n_obs, args.rho_thresh)
    print(f"  Collected {len(observations)} observations")

    print(f"Running Integrated Gradients (n_steps={args.n_steps}) ...")
    results = run_ig_analysis(agent, g2op_env, observations, args.n_steps)

    print(f"\n=== RESULTS ===")
    print(f"  Node features importance: {100*results['ratio_nodes']:.1f}%")
    print(f"  Edge probabilities importance: {100*results['ratio_edges']:.1f}%")
    print(f"  (averaged over {len(observations)} observations)")

    # Save raw arrays
    npy_out = Path(f"{args.output}_arrays.npz")
    np.savez(
        npy_out,
        per_node_ig=results["per_node_ig"],
        per_edge_ig=results["per_edge_ig"],
        edge_probs_all=results["edge_probs_all"],
        ratio_nodes_all=results["ratio_nodes_all"],
        actions=results["actions"],
    )
    print(f"Saved arrays: {npy_out.resolve()}")

    plot_results(results, args.output, trial_dir.name)

    import ray
    ray.shutdown()


if __name__ == "__main__":
    main()
