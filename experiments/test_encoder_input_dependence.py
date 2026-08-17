"""Test whether the NRI encoder's posterior depends on the input observation.

Hypothesis: the encoder converges to a fixed graph regardless of x (observation-independent).

Four experiments on one seed (dense, seed 0 by default):

  E1. Variance analysis:
      Collect 100 diverse observations (many chronics, varied rho), compute posteriors,
      report per-edge std across observations.  If all stds ~0 → input-independent.

  E2. 10×10 grid visualisation:
      Plot latent graphs for 100 diverse observations to visually confirm E1.

  E3. Input ablation (decisive sanity check):
      Compare posterior for a real observation vs. the same obs with node features
      zeroed out.  If outputs are identical → encoder ignores x completely.

  E4. Input-similarity ↔ posterior-similarity correlation:
      Compute pairwise cosine similarities of node-feature matrices and posterior
      vectors.  If the encoder uses x, the two similarity matrices should correlate.

Usage (from project root):
    PYTHONPATH=src conda run -n L2RPN python experiments/test_encoder_input_dependence.py \
        [--trial <path>] [--checkpoint checkpoint_000007] [--n-obs 100] [--rho-thresh 0.7]
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

logging.basicConfig(level=logging.WARNING)

DEFAULT_TRIAL = (
    "results/experiments/2026_07_07_IEEE14/rappo_mulitseed/"
    "CustomPPO_RARL_5829988_59bfa_2026-07-07_10-53-40"
)
DEFAULT_CHECKPOINT = "checkpoint_000007"


# ── observation collection ─────────────────────────────────────────────────────

def collect_diverse_obs(env_name: str, n_target: int, rho_thresh: float = 0.7) -> list:
    """Collect diverse observations spanning many chronics and rho levels.

    Strategy:
    - For each chronic: run do-nothing until rho > rho_thresh, game-over, or step limit.
    - Collect one observation per chronic at the moment rho first exceeds rho_thresh
      (or the last observation before game-over).
    - Also collect every 5th step to get mid-episode variety.
    - Shuffle and cap at n_target.
    """
    import grid2op
    env = grid2op.make(env_name)
    n_chronics = len(env.chronics_handler.subpaths)
    collected = []

    for chronic_id in range(n_chronics):
        if len(collected) >= n_target * 3:  # gather extra, then trim
            break
        env.set_id(chronic_id)
        obs = env.reset()
        done = False
        step = 0
        last_obs = obs

        while not done:
            obs, _, done, _ = env.step(env.action_space({}))
            step += 1
            last_obs = obs
            rho_max = float(obs.rho.max())

            # Collect at high-rho moments
            if rho_max > rho_thresh:
                collected.append((obs.copy(), rho_max, chronic_id, step))
                # Continue collecting within same chronic at higher rho
                if rho_max > 0.95 or step > 200:
                    break

            # Also collect at intermediate steps for diversity
            if step % 20 == 0:
                collected.append((obs.copy(), rho_max, chronic_id, step))

        # If episode ended without high rho, still take the last obs
        if last_obs.rho.max() < rho_thresh:
            collected.append((last_obs.copy(), float(last_obs.rho.max()), chronic_id, step))

    # Shuffle for chronics diversity, then cap
    rng = np.random.default_rng(42)
    rng.shuffle(collected)
    result = collected[:n_target]
    print(f"  Collected {len(result)} observations from {n_chronics} chronics")
    rhos = [r for _, r, _, _ in result]
    print(f"  rho range: [{min(rhos):.2f}, {max(rhos):.2f}], mean={np.mean(rhos):.2f}")
    return result


# ── model loading & posterior extraction ──────────────────────────────────────

def load_model(trial_dir: Path, checkpoint: str, env_name: str):
    import ray
    from core.loading import load_config, preprocess_config, load_rllib_agent
    from core.constants import RL_POLICY

    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False)
    params = preprocess_config(load_config(trial_dir))
    agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(trial_dir),
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint,
        env_name=env_name,
        env_config=params["env_config"],
    )
    return agent, g2op_env


def get_posterior(agent, g2op_obs) -> np.ndarray:
    """Forward one g2op observation through the model; return posterior [E, K]."""
    from core.constants import RL_POLICY
    agent.gym_wrapper.update_obs(g2op_obs)
    agent._rllib_agent.compute_single_action(
        agent.gym_wrapper.cur_gym_obs, policy_id=RL_POLICY
    )
    p = agent._rllib_agent.model.get_posterior()
    if p.dim() == 3:
        p = p[0]
    return p.cpu().detach().numpy()


def get_gym_obs_features(agent, g2op_obs) -> np.ndarray:
    """Return flattened node features from the gym observation."""
    from grid2op_env.observation_converter import NODES
    agent.gym_wrapper.update_obs(g2op_obs)
    return agent.gym_wrapper.cur_gym_obs[NODES].flatten()


# ── E3: ablation ──────────────────────────────────────────────────────────────

def ablation_test(agent, g2op_obs) -> dict:
    """Compare posterior for real obs vs. zeroed node features."""
    from core.constants import RL_POLICY
    from grid2op_env.observation_converter import NODES

    agent.gym_wrapper.update_obs(g2op_obs)
    real_gym_obs = {k: v.copy() if hasattr(v, "copy") else v
                    for k, v in agent.gym_wrapper.cur_gym_obs.items()}

    # Zeroed version
    zero_gym_obs = {k: (np.zeros_like(v) if k == NODES else v)
                    for k, v in real_gym_obs.items()}

    # Real
    agent._rllib_agent.compute_single_action(real_gym_obs, policy_id=RL_POLICY)
    p_real = agent._rllib_agent.model.get_posterior()
    if p_real.dim() == 3:
        p_real = p_real[0]
    p_real = p_real.cpu().detach().numpy()

    # Zeroed
    agent._rllib_agent.compute_single_action(zero_gym_obs, policy_id=RL_POLICY)
    p_zero = agent._rllib_agent.model.get_posterior()
    if p_zero.dim() == 3:
        p_zero = p_zero[0]
    p_zero = p_zero.cpu().detach().numpy()

    interaction_real = p_real[:, :-1].sum(axis=-1)
    interaction_zero = p_zero[:, :-1].sum(axis=-1)
    diff = np.abs(interaction_real - interaction_zero)

    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "cosine_sim": float(
            np.dot(interaction_real, interaction_zero)
            / (np.linalg.norm(interaction_real) * np.linalg.norm(interaction_zero) + 1e-8)
        ),
        "interaction_real": interaction_real,
        "interaction_zero": interaction_zero,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def draw_panel(ax, posterior, node_styles, n_line, threshold, title):
    from visualization.utils import visualize_graph, PlottingArgs
    N = len(node_styles)
    pl_src = list(range(n_line)) + list(range(n_line, 2 * n_line))
    pl_dst = list(range(n_line, 2 * n_line)) + list(range(n_line))
    powerline_edge_index = np.array([pl_src, pl_dst])
    interaction_prob = posterior[:, :-1].sum(axis=-1)
    latent_edge_probs = np.stack([interaction_prob, posterior[:, -1]], axis=-1)
    n_active = int((interaction_prob > threshold).sum())
    ax.set_title(f"{title}\n({n_active} edges)", fontsize=7)
    args = PlottingArgs(
        num_nodes=N,
        node_styles=node_styles,
        powerline_edge_index=powerline_edge_index,
        latent_edge_probs=latent_edge_probs,
        latent_edge_weight=5.0,
        skip_last_edge_type=True,
        visualize_edge_prob_threshold=threshold,
        show_legend=False,
    )
    visualize_graph(args, ax=ax)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env", default="l2rpn_case14_sandbox_test")
    parser.add_argument("--n-obs", type=int, default=100)
    parser.add_argument("--rho-thresh", type=float, default=0.7,
                        help="Minimum rho to target for dangerous observations")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Edge probability threshold for visualisation")
    parser.add_argument("--output-prefix", default="encoder_input_dependence")
    args = parser.parse_args()

    trial_dir = Path(args.trial)
    print(f"Loading agent: {trial_dir.name} / {args.checkpoint}")
    agent, g2op_env = load_model(trial_dir, args.checkpoint, args.env)

    from visualization.utils import get_node_styles
    from grid2op_env.observation_converter import GraphObservationConverter
    node_styles = get_node_styles(g2op_env, GraphObservationConverter)
    n_line = g2op_env.n_line

    # ── Collect observations ───────────────────────────────────────────────────
    print(f"Collecting {args.n_obs} diverse observations (rho_thresh={args.rho_thresh}) ...")
    obs_records = collect_diverse_obs(args.env, args.n_obs, args.rho_thresh)
    obs_list = [o for o, *_ in obs_records]
    rhos = [r for _, r, *_ in obs_records]
    chronic_ids = [c for _, _, c, _ in obs_records]

    # ── E1: Variance analysis ─────────────────────────────────────────────────
    print("E1: Computing posteriors for all observations ...")
    posteriors = []
    node_feat_vecs = []
    for i, obs in enumerate(obs_list):
        posteriors.append(get_posterior(agent, obs))
        node_feat_vecs.append(get_gym_obs_features(agent, obs))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(obs_list)}")

    posteriors_np = np.stack(posteriors)             # [N_obs, E, K]
    interaction_probs = posteriors_np[:, :, :-1].sum(axis=-1)  # [N_obs, E]

    per_edge_std = interaction_probs.std(axis=0)     # [E]
    per_edge_mean = interaction_probs.mean(axis=0)   # [E]

    print(f"\n=== E1 RESULTS ===")
    print(f"  Per-edge interaction_prob std:  mean={per_edge_std.mean():.4f}, "
          f"max={per_edge_std.max():.4f}, median={np.median(per_edge_std):.4f}")
    print(f"  Fraction of edges with std < 0.01: {(per_edge_std < 0.01).mean():.2%}")
    print(f"  Fraction of edges with std < 0.05: {(per_edge_std < 0.05).mean():.2%}")
    print(f"  Fraction of edges with std > 0.10: {(per_edge_std > 0.10).mean():.2%}")

    # ── E3: Ablation ──────────────────────────────────────────────────────────
    print("\n=== E3 ABLATION TEST (real obs vs. zeroed node features) ===")
    ablation = ablation_test(agent, obs_list[0])
    print(f"  max |posterior diff|: {ablation['max_diff']:.6f}")
    print(f"  mean |posterior diff|: {ablation['mean_diff']:.6f}")
    print(f"  cosine similarity real vs. zero: {ablation['cosine_sim']:.6f}")

    # ── E4: Input ↔ posterior similarity correlation ──────────────────────────
    print("\n=== E4 INPUT-SIMILARITY ↔ POSTERIOR-SIMILARITY ===")
    node_feat_matrix = np.stack(node_feat_vecs)      # [N_obs, D]
    N_obs = len(obs_list)

    # Pairwise cosine similarity (upper-triangle only for speed)
    def pairwise_cos(M: np.ndarray) -> np.ndarray:
        M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        return M_norm @ M_norm.T

    cos_input = pairwise_cos(node_feat_matrix)
    cos_post  = pairwise_cos(interaction_probs)

    triu_idx = np.triu_indices(N_obs, k=1)
    rho_corr, p_val = spearmanr(cos_input[triu_idx], cos_post[triu_idx])
    print(f"  Spearman r(input_cos, posterior_cos): {rho_corr:.4f}  (p={p_val:.2e})")
    print(f"  (r≈0: encoder ignores input; r>0.3: encoder uses input)")

    # ── Figures ───────────────────────────────────────────────────────────────

    # Fig 1: E1 — per-edge std histogram
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    axes1[0].hist(per_edge_std, bins=50, edgecolor="black", color="#1f77b4")
    axes1[0].set_xlabel("Per-edge std of interaction_prob across observations")
    axes1[0].set_ylabel("Count")
    axes1[0].set_title("E1: Distribution of per-edge posterior variance\n"
                        "(std≈0 for all edges ⟹ input-independent encoder)")
    axes1[0].axvline(0.05, color="red", linestyle="--", label="std=0.05")
    axes1[0].legend()

    axes1[1].scatter(per_edge_mean, per_edge_std, alpha=0.3, s=3, color="#1f77b4")
    axes1[1].set_xlabel("Mean interaction_prob")
    axes1[1].set_ylabel("Std of interaction_prob")
    axes1[1].set_title("E1: Mean vs. std per edge")
    fig1.suptitle(f"Encoder input-dependence test — {trial_dir.name[:30]}...", fontsize=11)
    fig1.tight_layout()
    out1 = Path(f"{args.output_prefix}_variance.png")
    fig1.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")
    plt.close(fig1)

    # Fig 2: E3 ablation — real vs. zeroed posterior
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].scatter(range(len(ablation["interaction_real"])),
                     np.sort(ablation["interaction_real"])[::-1],
                     s=3, label="Real obs", alpha=0.7)
    axes2[0].scatter(range(len(ablation["interaction_zero"])),
                     np.sort(ablation["interaction_zero"])[::-1],
                     s=3, label="Zero features", alpha=0.7)
    axes2[0].set_xlabel("Edge rank")
    axes2[0].set_ylabel("Interaction probability")
    axes2[0].set_title(f"E3: Posterior — real vs. zeroed input\n"
                       f"cosine sim={ablation['cosine_sim']:.4f}, "
                       f"max diff={ablation['max_diff']:.4f}")
    axes2[0].legend(fontsize=9)

    # E4: scatter of pairwise similarities
    sample = rng = np.random.default_rng(0)
    idx = np.random.default_rng(0).choice(len(triu_idx[0]), size=min(5000, len(triu_idx[0])), replace=False)
    axes2[1].scatter(cos_input[triu_idx][idx], cos_post[triu_idx][idx],
                     alpha=0.2, s=3, color="#d62728")
    axes2[1].set_xlabel("Pairwise cosine sim (node features)")
    axes2[1].set_ylabel("Pairwise cosine sim (posteriors)")
    axes2[1].set_title(f"E4: Input vs. posterior similarity\n"
                       f"Spearman r={rho_corr:.3f}, p={p_val:.1e}")
    fig2.tight_layout()
    out2 = Path(f"{args.output_prefix}_ablation_e4.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close(fig2)

    # Fig 3: E2 — 10×10 grid visualisation of latent graphs
    print("E2: Plotting 10×10 latent graph grid ...")
    n_rows, n_cols = 10, 10
    assert len(posteriors) >= n_rows * n_cols
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            idx_obs = i * n_cols + j
            ax = axes3[i, j]
            rho_str = f"rho={rhos[idx_obs]:.2f}"
            ch_str = f"ch{chronic_ids[idx_obs]}"
            draw_panel(ax, posteriors[idx_obs], node_styles, n_line,
                       threshold=args.threshold,
                       title=f"{ch_str} {rho_str}")
    fig3.suptitle(
        f"E2: 100 diverse observations — {trial_dir.name[:40]}...\n"
        f"threshold={args.threshold}. Input-independent ⟺ all panels identical.",
        fontsize=11, y=1.005,
    )
    fig3.tight_layout()
    out3 = Path(f"{args.output_prefix}_grid100.png")
    fig3.savefig(out3, dpi=100, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close(fig3)

    import ray
    ray.shutdown()


if __name__ == "__main__":
    main()
