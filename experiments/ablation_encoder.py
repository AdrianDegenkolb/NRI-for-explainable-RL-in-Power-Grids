"""Ablation: replace encoder predictions with random/empty graphs.

Tests whether the downstream GNN actually benefits from the encoder's latent graph
by comparing agent performance under three conditions:

  baseline  — real encoder posterior (trained model)
  empty     — zero non-null edge weights → only self-loops (MLP-like)
  random    — fresh random soft posterior at every timestep

Runs all test chronics for each condition on one seed; prints and plots survival stats.

Usage (from project root):
    PYTHONPATH=src conda run -n L2RPN python experiments/ablation_encoder.py \
        [--trial <path>] [--checkpoint checkpoint_000007] [--n-chronics 50]
"""
import argparse
import contextlib
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.WARNING)

DEFAULT_TRIAL = (
    "results/experiments/2026_07_07_IEEE14/rappo_mulitseed/"
    "CustomPPO_RARL_5829988_59bfa_2026-07-07_10-53-40"
)
DEFAULT_CHECKPOINT = "checkpoint_000007"


# ── encoder patch ──────────────────────────────────────────────────────────────

def _make_ablated_forward(fe, ablation_type: str):
    """Return a patched forward for RAFeatureExtractor that overrides the GNN input."""

    def patched_forward(x, batch=None, powerline_edge_index=None, edge_set=None):
        from rarl.graph import fully_connected_edge_index_per_batch
        from torch_geometric.utils import to_dense_batch

        BxN = x.shape[0]
        if batch is None:
            batch = torch.zeros(BxN, dtype=torch.long, device=x.device)
        if edge_set is None:
            edge_set = fully_connected_edge_index_per_batch(batch, x.device)

        logits = fe.encoder(
            x=x, batch=batch, edge_set=edge_set,
            powerline_edge_index=powerline_edge_index,
        )
        posterior = F.softmax(logits, dim=-1)

        if ablation_type == "empty":
            # All weight on null type → GCNConv edge weights = 0 → self-loops only
            gnn_weights = torch.zeros_like(posterior)
            gnn_weights[:, -1] = 1.0
        elif ablation_type == "random":
            # New random soft posterior at each call
            gnn_weights = F.softmax(torch.randn_like(logits), dim=-1)
        elif ablation_type == "full":
            # Uniform weight across non-null types → all edges present with equal weight
            gnn_weights = torch.zeros_like(posterior)
            n_non_null = posterior.shape[-1] - 1
            gnn_weights[:, :-1] = 1.0 / max(n_non_null, 1)
        else:  # baseline
            gnn_weights = fe.gumbel_softmax(logits, hard=fe.training)

        if fe.top_k_budget > 0 and ablation_type != "full":
            from rarl.nn.sparsification import sparse_top_k_posterior
            B = int(batch.max()) + 1
            N = BxN // B
            gnn_sampled, gnn_edge_set, fe._sparsification_stats = sparse_top_k_posterior(
                posterior_flat=posterior,
                sampled_flat=gnn_weights,
                edge_set=edge_set,
                K_budget=fe.top_k_budget,
                B=B,
                N=N,
            )
        else:
            gnn_sampled, gnn_edge_set = gnn_weights, edge_set
            fe._sparsification_stats = {}

        embeddings = fe.gnn(
            x=x, edge_index=gnn_edge_set,
            edge_type_posterior=gnn_sampled, batch=batch,
        )

        edge_batch = batch[edge_set[0]]
        batched_posterior, _ = to_dense_batch(posterior, edge_batch)
        return embeddings, batched_posterior

    return patched_forward


@contextlib.contextmanager
def ablated_encoder(feature_extractor, ablation_type: str):
    """Context manager that temporarily replaces the RAFeatureExtractor forward."""
    original = feature_extractor.forward
    feature_extractor.forward = _make_ablated_forward(feature_extractor, ablation_type)
    try:
        yield
    finally:
        feature_extractor.forward = original


# ── evaluation ─────────────────────────────────────────────────────────────────

def run_episodes(agent, g2op_env, chronic_ids: list[int], max_iter: int | None = None) -> list[dict]:
    """Run agent on given chronic IDs; return list of per-episode result dicts."""
    results = []
    for cid in chronic_ids:
        g2op_env.set_id(cid)
        obs = g2op_env.reset()
        done = False
        reward = 0.0
        total_reward = 0.0
        steps = 0
        max_steps = None

        while not done:
            action = agent.act(obs, reward, done)
            obs, reward, done, info = g2op_env.step(action)
            total_reward += reward
            steps += 1
            if max_steps is None and hasattr(obs, "max_step"):
                max_steps = obs.max_step
            if max_iter is not None and steps >= max_iter:
                break

        if max_steps is None:
            max_steps = steps
        results.append({
            "chronic": cid,
            "steps": steps,
            "max_steps": max_steps,
            "survived_pct": 100.0 * steps / max(max_steps, 1),
            "completed": steps >= max_steps,
            "total_reward": total_reward,
        })
    return results


def summarise(results: list[dict]) -> dict:
    steps = [r["steps"] for r in results]
    survived = [r["survived_pct"] for r in results]
    completed = [r["completed"] for r in results]
    return {
        "mean_steps": float(np.mean(steps)),
        "median_steps": float(np.median(steps)),
        "std_steps": float(np.std(steps)),
        "survived_pct": float(np.mean(survived)),
        "completed_pct": float(100 * np.mean(completed)),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env", default="l2rpn_case14_sandbox_test")
    parser.add_argument("--n-chronics", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Max steps per episode (None = full episode)")
    parser.add_argument("--conv-type", default=None, choices=["gcn", "gin"],
                        help="Override GNN conv type (gin for pre-revert checkpoints).")
    parser.add_argument("--output", default="ablation_encoder.png")
    args = parser.parse_args()

    import ray
    from core.loading import load_config, preprocess_config, load_rllib_agent
    from core.constants import RL_POLICY

    trial_dir = Path(args.trial)
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False)
    params = preprocess_config(load_config(trial_dir))
    agent, g2op_env, _ = load_rllib_agent(
        checkpoint_path=str(trial_dir),
        policy_name=RL_POLICY,
        checkpoint_name=args.checkpoint,
        env_name=args.env,
        env_config=params["env_config"],
        conv_type=args.conv_type,
    )

    # The RAFeatureExtractor instance
    feature_extractor = agent._rllib_agent.model.ragnn

    chronic_ids = list(range(min(args.n_chronics, len(g2op_env.chronics_handler.subpaths))))
    print(f"Running {len(chronic_ids)} chronics per condition ...\n")

    conditions = ["baseline", "full", "empty", "random"]
    all_results = {}

    for condition in conditions:
        print(f"  Condition: {condition} ...")
        with ablated_encoder(feature_extractor, condition):
            results = run_episodes(agent, g2op_env, chronic_ids, args.max_iter)
        all_results[condition] = results
        s = summarise(results)
        print(f"    mean_steps={s['mean_steps']:.0f}  survived={s['survived_pct']:.1f}%  "
              f"completed={s['completed_pct']:.1f}%")

    ray.shutdown()

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'Condition':<12} {'Mean steps':>11} {'Median':>8} {'Survived%':>10} {'Completed%':>11}")
    print("-" * 56)
    baseline_mean = summarise(all_results["baseline"])["mean_steps"]
    for cond in conditions:
        s = summarise(all_results[cond])
        delta = s["mean_steps"] - baseline_mean
        sign = "+" if delta >= 0 else ""
        note = f"  ({sign}{delta:.0f})" if cond != "baseline" else ""
        print(f"{cond:<12} {s['mean_steps']:>11.0f} {s['median_steps']:>8.0f} "
              f"{s['survived_pct']:>10.1f} {s['completed_pct']:>11.1f}{note}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"baseline": "#2ca02c", "full": "#1f77b4", "empty": "#d62728", "random": "#ff7f0e"}
    short_labels = {"baseline": "Baseline\n(encoder)", "full": "Full\n(all edges)",
                    "empty": "Empty\n(self-loops)", "random": "Random\n(noise)"}

    # Box plot of survived_pct per condition
    data = [[r["survived_pct"] for r in all_results[c]] for c in conditions]
    bp = ax.boxplot(data, labels=[short_labels[c] for c in conditions], patch_artist=True)
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(colors[cond])
        patch.set_alpha(0.7)
    ax.set_ylabel("Survived % of episode")
    ax.set_title("Survival distribution per condition")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Encoder ablation — {trial_dir.name[:45]}\n"
        f"(empty=self-loops only, random=noise per step)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
