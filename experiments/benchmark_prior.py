"""
Standalone benchmark for build_prior_and_graph_masks.

Measures how long the function takes for realistic IEEE-14 and IEEE-36 inputs
without needing a full training run.

Usage (from project root):
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python experiments/benchmark_prior.py
"""

import time
import types

import torch

from rarl.graph import fully_connected_edge_index
from rarl_rllib.common import build_prior_and_graph_masks

# ── grid parameters ────────────────────────────────────────────────────────────
# n_sub is the number of NODES in the observation graph (buses, not substations).
# For case14: 14 substations × ~2 buses each → 57 nodes in obs graph.
# For case36: 36 substations → 177 nodes in obs graph.
# These come from GraphObservationConverter, not from grid2op env.n_sub.
CASES = {
    "case14": dict(n_sub=57, n_lines=20),
    "case36": dict(n_sub=177, n_lines=59),
}

BATCH_SIZES = [16, 256, 1024]
N_REPEATS = 5


# ── mock policy ────────────────────────────────────────────────────────────────

def _make_policy(n_sub: int, n_lines: int, num_edge_types: int = 2):
    """Minimal policy-like namespace with the attributes build_prior_and_graph_masks needs."""
    from grid2op_env.observation_converter import NODES
    from gymnasium import spaces
    import numpy as np

    policy = types.SimpleNamespace()
    policy.observation_space = {
        NODES: spaces.Box(low=-np.inf, high=np.inf, shape=(n_sub, 1), dtype=np.float32),
    }
    policy.num_edge_types = num_edge_types
    policy.prior_prob_for_graph_edge = 0.9
    policy.temperature = 0.2
    policy.device = "cpu"
    return policy


# ── mock observations ──────────────────────────────────────────────────────────

def _make_obs(batch_size: int, n_lines: int):
    """
    Simulate a batch of graph observations.

    all_graph_edges: [B, 2, E_max]  — directed powerline edges (both directions)
    edge_masks_obs:  [B, E_max]     — all active (no disconnected lines)
    """
    n_directed = n_lines * 2  # both directions

    # Build a fixed directed edge index for the powerlines.
    # Actual node indices don't matter for timing — just need valid shape.
    src = torch.arange(n_lines).repeat(2)
    dst = torch.arange(n_lines, 2 * n_lines).clamp(max=n_lines - 1).repeat(2)
    single_edge_index = torch.stack([src, dst])  # [2, E_max]

    all_graph_edges = single_edge_index.unsqueeze(0).expand(batch_size, -1, -1).clone()
    edge_masks_obs = torch.ones(batch_size, n_directed, dtype=torch.float32)

    return all_graph_edges, edge_masks_obs


# ── benchmark ──────────────────────────────────────────────────────────────────

def benchmark(case_name: str, n_sub: int, n_lines: int):
    n_fc = n_sub * (n_sub - 1)
    print(f"\n{'=' * 60}")
    print(f"{case_name}: {n_sub} nodes  {n_lines} lines  {n_fc} FC edges")
    print(f"{'=' * 60}")

    policy = _make_policy(n_sub, n_lines)

    for batch_size in BATCH_SIZES:
        all_graph_edges, edge_masks_obs = _make_obs(batch_size, n_lines)

        times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            build_prior_and_graph_masks(policy, all_graph_edges, edge_masks_obs)
            times.append(time.perf_counter() - t0)

        mean_t = sum(times) / len(times)
        min_t = min(times)
        print(f"  batch={batch_size:>4}  mean={mean_t:.3f}s  min={min_t:.3f}s  "
              f"per-sample={mean_t/batch_size*1000:.2f}ms")


if __name__ == "__main__":
    for name, params in CASES.items():
        benchmark(name, **params)
