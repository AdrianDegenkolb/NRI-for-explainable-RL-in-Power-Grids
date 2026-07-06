"""
Profile realized degree distribution of the encoder posterior for sparsification planning.

Runs the RAPPO-14 checkpoint on a few episodes and histograms the P(interaction)
scores across the full N×N edge matrix. Outputs:

  - Score histogram (shows bimodality / diffuseness)
  - Realized edge count vs. global threshold curve
  - Per-node degree distribution at two representative thresholds
  - Same plots for a randomly-initialised encoder (proxy for early training)
  - Hub-degree-vs-K-budget analysis: checks whether the largest hub exhausts the
    global K budget before other nodes receive any edges

This is the gate check before implementing sparse neighbourhood selection:
if realized degree stays well below N under a reasonable threshold, sparsification
pays off; if not, the fix is elsewhere.

Usage (from project root):
    # Full analysis using a trained checkpoint:
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python experiments/profile_degree_distribution.py
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python experiments/profile_degree_distribution.py --n_episodes 10

    # Random-init only (no checkpoint needed — useful for 36-bus N=177):
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python experiments/profile_degree_distribution.py \\
        --random_only --n_nodes 177 --n_edge_types 3 --n_powerlines 118
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.WARNING)

CHECKPOINT = Path(
    "results/2026_05_26_IEEE14/rappo/"
    "CustomPPO_RARL_4808021_62fbb_2026-05-26_00-30-52"
)


# ── collector ─────────────────────────────────────────────────────────────────

def _import_checkpoint_deps():
    """Lazy import of heavy checkpoint-loading deps (not needed in --random_only mode)."""
    global LatentGraphAnalysisAgent, PosteriorAnalyzer, RL_POLICY
    global load_config, preprocess_config, load_rllib_agent
    from analysis.analyze_latent_graphs.agent_analysis_framework import (
        LatentGraphAnalysisAgent,
        PosteriorAnalyzer,
    )
    from core.constants import RL_POLICY
    from core.loading import load_config, preprocess_config, load_rllib_agent


class ScoreCollector:
    """Stores the full [E, K] posterior for each RL-activated step."""

    def __init__(self):
        self.posteriors: list[npt.NDArray] = []  # [E, K] each

    def on_rl_step(self, posterior, prior, powergrid_graph,
                   observation, environment, action):
        self.posteriors.append(posterior)

    def on_heuristic_step(self, powergrid_graph, observation, environment):
        pass

    def on_new_episode(self, chronic_id):
        print(f"  episode {chronic_id}")

    def on_evaluation_end(self):
        pass


# ── analysis helpers ───────────────────────────────────────────────────────────

def p_interact(posterior: npt.NDArray) -> npt.NDArray:
    """P(interaction) per edge: sum of all non-null type probabilities. Shape [E]."""
    return posterior[:, :-1].sum(axis=-1)


def realized_edges_vs_threshold(scores_2d: npt.NDArray,
                                 thresholds: npt.NDArray) -> npt.NDArray:
    """Mean number of surviving edges per sample across a range of thresholds."""
    # scores_2d: [T, E]  thresholds: [G]  → [G]
    return np.array([(scores_2d > t).sum(axis=1).mean() for t in thresholds])


def degree_distribution(scores_2d: npt.NDArray, threshold: float,
                         N: int) -> npt.NDArray:
    """
    Per-node out-degree distribution at a given global threshold.
    scores_2d: [T, E] where E = N*(N-1) directed edges in row-major order
    (node 0→1, 0→2, ..., 0→N-1, 1→0, 1→2, ...).
    Returns degree array of shape [T*N].
    """
    active = scores_2d > threshold  # [T, E]
    T, E = active.shape
    # reshape to [T, N, N-1] → sum over columns → [T, N]
    per_node = active.reshape(T, N, N - 1).sum(axis=-1)  # [T, N]
    return per_node.ravel()


def percentile_threshold(scores_2d: npt.NDArray, keep_fraction: float) -> float:
    """Global threshold that keeps `keep_fraction` of all edges on average."""
    return float(np.percentile(scores_2d, 100 * (1 - keep_fraction)))


# ── random-encoder baseline ───────────────────────────────────────────────────

def random_encoder_scores(N: int, K: int, n_samples: int = 200) -> npt.NDArray:
    """
    Simulate posterior from a randomly-initialised encoder.
    Draws logits ~ N(0,1) and applies softmax → [n_samples, E, K].
    Returns P(interaction) array [n_samples, E].
    """
    E = N * (N - 1)
    logits = np.random.randn(n_samples, E, K).astype(np.float32)
    # softmax
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    posterior = exp / exp.sum(axis=-1, keepdims=True)
    return posterior[:, :, :-1].sum(axis=-1)  # [n_samples, E]


# ── plotting ──────────────────────────────────────────────────────────────────

def hub_degree_vs_budget(scores_2d: npt.NDArray,
                          N: int,
                          n_powerlines: int,
                          k_multipliers: tuple = (1, 2, 3)) -> None:
    """
    Print hub-degree-vs-K-budget table.

    For each K = multiplier × n_powerlines_directed, reports:
      - Mean total surviving edges across samples
      - Mean / p95 / max hub (max-degree) node out-degree
      - Fraction of samples where the hub alone exceeds K/2

    This answers: "does the largest hub consume most of the global K budget?"
    scores_2d: [T, E] where E = N*(N-1)
    n_powerlines: number of DIRECTED powerline edges (= n_lines × 2)
    """
    T, E = scores_2d.shape

    print("\n── Hub-degree vs. global K budget ──────────────────────────────────")
    print(f"  N={N}  directed powerlines={n_powerlines}  samples={T}")
    print(f"  {'K':>6}  {'mult':>4}  {'mean_edges':>10}  "
          f"{'hub_mean':>8}  {'hub_p95':>7}  {'hub_max':>7}  "
          f"{'hub>K/2 (%)':>11}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*11}")

    for mult in k_multipliers:
        K_budget = mult * n_powerlines
        # Global top-K: for each sample, keep the K highest-scoring edges
        # and compute per-node out-degree from those K selections.
        sorted_idx = np.argsort(scores_2d, axis=1)[:, ::-1]  # [T, E] descending
        topk_mask = np.zeros_like(scores_2d, dtype=bool)
        topk_mask[np.arange(T)[:, None], sorted_idx[:, :K_budget]] = True

        # Per-node out-degree: reshape [T, E] → [T, N, N-1], sum over N-1 cols
        per_node_deg = topk_mask.reshape(T, N, N - 1).sum(axis=-1)  # [T, N]
        hub_deg = per_node_deg.max(axis=1)  # [T] — largest hub per sample

        mean_edges = topk_mask.sum(axis=1).mean()
        hub_mean   = hub_deg.mean()
        hub_p95    = float(np.percentile(hub_deg, 95))
        hub_max    = hub_deg.max()
        frac_exceeds = (hub_deg > K_budget / 2).mean() * 100

        print(f"  {K_budget:>6}  {mult:>4}×  {mean_edges:>10.1f}  "
              f"{hub_mean:>8.1f}  {hub_p95:>7.1f}  {hub_max:>7}  "
              f"{frac_exceeds:>10.1f}%")


def plot_all(trained_scores: npt.NDArray,
             random_scores: npt.NDArray,
             N: int,
             n_powerlines: int,
             output: Path):
    """Four-panel summary plot."""
    E = N * (N - 1)
    thresholds = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Encoder score distribution  "
        f"(N={N}, E={E} FC edges, {len(trained_scores)} samples)",
        fontsize=11,
    )

    # ── top-left: score histogram ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(trained_scores.ravel(), bins=80, density=True,
            alpha=0.75, color="steelblue", label="trained")
    ax.hist(random_scores.ravel(), bins=80, density=True,
            alpha=0.5, color="orange", label="random init (proxy: early training)")
    ax.set_title("P(interaction) score distribution")
    ax.set_xlabel("P(interaction)")
    ax.set_ylabel("Density")
    ax.legend()

    # ── top-right: surviving edges vs threshold ────────────────────────────
    ax = axes[0, 1]
    trained_curve = realized_edges_vs_threshold(trained_scores, thresholds)
    random_curve  = realized_edges_vs_threshold(random_scores,  thresholds)
    ax.plot(thresholds, trained_curve, color="steelblue", label="trained")
    ax.plot(thresholds, random_curve,  color="orange",    label="random init")
    ax.axhline(E, color="gray", ls=":", lw=1, label=f"N²−N = {E}")

    # Mark a few candidate thresholds
    for frac, ls in [(0.01, "--"), (0.05, "-.")]:
        t = percentile_threshold(trained_scores, frac)
        n = float(realized_edges_vs_threshold(trained_scores, np.array([t])))
        ax.axvline(t, color="steelblue", ls=ls, lw=1,
                   label=f"top {frac*100:.0f}% → {n:.0f} edges (t={t:.3f})")
    ax.set_title("Mean surviving edges vs. global threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean surviving edges per sample")
    ax.legend(fontsize=7)

    # ── bottom-left: per-node degree at top-1% ────────────────────────────
    ax = axes[1, 0]
    t1 = percentile_threshold(trained_scores, 0.01)
    t5 = percentile_threshold(trained_scores, 0.05)
    deg1 = degree_distribution(trained_scores, t1, N)
    deg5 = degree_distribution(trained_scores, t5, N)
    bins = np.arange(0, N + 1)
    ax.hist(deg1, bins=bins, alpha=0.75, color="steelblue",
            label=f"top 1% (t={t1:.3f})", density=True)
    ax.hist(deg5, bins=bins, alpha=0.5, color="green",
            label=f"top 5% (t={t5:.3f})", density=True)
    ax.set_title("Per-node out-degree distribution (trained)")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Fraction of nodes")
    ax.legend(fontsize=8)

    # ── bottom-right: same for random init ────────────────────────────────
    ax = axes[1, 1]
    rt1 = percentile_threshold(random_scores, 0.01)
    rt5 = percentile_threshold(random_scores, 0.05)
    rdeg1 = degree_distribution(random_scores, rt1, N)
    rdeg5 = degree_distribution(random_scores, rt5, N)
    ax.hist(rdeg1, bins=bins, alpha=0.75, color="orange",
            label=f"top 1% (t={rt1:.3f})", density=True)
    ax.hist(rdeg5, bins=bins, alpha=0.5, color="red",
            label=f"top 5% (t={rt5:.3f})", density=True)
    ax.set_title("Per-node out-degree distribution (random init)")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Fraction of nodes")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")

    # ── text summary ──────────────────────────────────────────────────────
    print("\n── Score summary ───────────────────────────────────────────")
    print(f"  N={N}  E(FC)={E}")
    for label, scores in [("trained", trained_scores), ("random ", random_scores)]:
        vals = scores.ravel()
        print(f"\n  [{label}]")
        print(f"    mean={vals.mean():.4f}  median={np.median(vals):.4f}  "
              f"std={vals.std():.4f}")
        print(f"    >0.5 : {(vals>0.5).mean():.2%} of edges")
        print(f"    >0.9 : {(vals>0.9).mean():.2%} of edges")
        for frac in [0.005, 0.01, 0.05, 0.10]:
            t = percentile_threshold(scores, frac)
            n = float(realized_edges_vs_threshold(scores, np.array([t])))
            print(f"    top {frac*100:.1f}% → threshold={t:.4f}  "
                  f"mean surviving edges={n:.1f}  ({n/E*100:.1f}% of FC)")

    # ── hub-degree-vs-budget ──────────────────────────────────────────────
    for label, scores in [("trained", trained_scores), ("random ", random_scores)]:
        print(f"\n  [{label}] hub-degree analysis")
        hub_degree_vs_budget(scores, N, n_powerlines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("degree_distribution.png"))
    # ── random-only mode ─────────────────────────────────────────────────────
    parser.add_argument(
        "--random_only", action="store_true",
        help="Skip checkpoint loading; run random-init analysis only. "
             "Requires --n_nodes, --n_edge_types, --n_powerlines.",
    )
    parser.add_argument("--n_nodes", type=int, default=None,
                        help="Number of nodes N (e.g. 177 for case36).")
    parser.add_argument("--n_edge_types", type=int, default=None,
                        help="Number of edge types K (including null type).")
    parser.add_argument("--n_powerlines", type=int, default=None,
                        help="Number of DIRECTED powerline edges (n_lines × 2).")
    parser.add_argument("--n_random_samples", type=int, default=500,
                        help="Number of random-init samples to draw (default 500).")
    args = parser.parse_args()

    if args.random_only:
        if args.n_nodes is None or args.n_edge_types is None or args.n_powerlines is None:
            parser.error("--random_only requires --n_nodes, --n_edge_types, --n_powerlines")

        N = args.n_nodes
        K = args.n_edge_types
        E = N * (N - 1)
        n_powerlines = args.n_powerlines
        print(f"Random-only mode: N={N}  E={E}  K={K}  powerlines={n_powerlines}")

        random_scores = random_encoder_scores(N, K, n_samples=args.n_random_samples)
        # Use random scores as both "trained" and "random" slots so plot_all
        # still renders; label will show "trained" but data is identical.
        plot_all(random_scores, random_scores, N, n_powerlines, args.output)
        return

    _import_checkpoint_deps()

    print(f"Loading checkpoint: {args.checkpoint}")
    params = load_config(args.checkpoint)
    params = preprocess_config(params)
    env_config = params["env_config"]

    agent, _g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(args.checkpoint),
        policy_name=RL_POLICY,
        checkpoint_name="checkpoint_000000",
        env_name=env_config["env_name"],
        env_config=env_config,
    )

    collector = ScoreCollector()
    analysis_agent = LatentGraphAnalysisAgent(
        rllib_agent=agent,
        gym_wrapper=gym_wrapper,
        analysers=[collector],
    )

    print(f"Running {args.n_episodes} episodes …")
    analysis_agent.analyze(num_episodes=args.n_episodes)

    if not collector.posteriors:
        print("No RL-activated steps recorded.")
        return

    # Stack into [T, E, K] → [T, E]
    posteriors = np.stack(collector.posteriors)   # [T, E, K]
    T, E, K = posteriors.shape
    trained_scores = p_interact(posteriors.reshape(T * E, K)).reshape(T, E)

    # N: infer from E = N*(N-1)
    N = int(round((1 + (1 + 4 * E) ** 0.5) / 2))
    print(f"\nCollected {T} RL steps  |  E={E} FC edges  |  N={N} nodes  |  K={K}")

    # n_powerlines: use CLI override or default to 2 × (N-1) as a rough estimate
    # (case14: 20 lines → 40 directed; case36: 59 lines → 118 directed)
    if args.n_powerlines is not None:
        n_powerlines = args.n_powerlines
    else:
        # Heuristic: powerlines ≈ N (very rough); caller should pass --n_powerlines
        n_powerlines = N
        print(f"  Warning: --n_powerlines not set; using N={N} as placeholder. "
              f"Pass --n_powerlines <2×n_lines> for accurate hub-budget analysis.")

    print("Generating random-init baseline …")
    random_scores = random_encoder_scores(N, K, n_samples=min(T, 500))

    plot_all(trained_scores, random_scores, N, n_powerlines, args.output)


if __name__ == "__main__":
    main()
