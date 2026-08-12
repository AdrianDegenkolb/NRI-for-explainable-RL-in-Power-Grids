"""
Benchmark RAGNN.forward across three variants:

  1. Fully-connected edge index, eval mode  (diagnostics ON)
  2. Fully-connected edge index, train mode (diagnostics OFF, dropout=0)
  3. Sparse powerline edge index, eval mode (diagnostics ON)

Variants 1 vs 2 isolate the cost of the diagnostic block; 1 vs 3 show the
savings from using a sparse edge set instead of the fully-connected graph.

Usage (from project root):
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python scripts/benchmark_ragnn.py
"""
from __future__ import annotations

import time
from typing import Callable

import torch
from torch import Tensor

from rarl.graph import fully_connected_edge_index
from rarl.nn.ragnn import RAGNN

# ── Grid parameters ─────────────────────────────────────────────────────────────
# n_nodes: total bus nodes produced by GraphObservationConverter (not substations).
# n_lines: number of physical powerlines; each gives 2 directed edges in the
#          sparse case (both directions).
CASES = {
    "case14": dict(n_nodes=57, n_lines=20),
    "case36": dict(n_nodes=177, n_lines=59),
}

# Hyperparameters matching ragnn.yaml / RAFeatureExtractor defaults.
# x_dim=3: one scalar per node for each attribute in obs_space/graph.yaml (r, t, d).
GNN_KWARGS: dict = dict(
    x_dim=3,
    hidden_dim=256,
    x_out_dim=64,
    num_layers=3,
    num_edge_types=2,
    skip_last=True,
    dropout_prob=0.0,   # keep at 0 so train/eval outputs are comparable
    residual=True,
    conv_type="gcn",
)

BATCH_SIZE = 32  # matches typical rollout worker batch size
N_WARMUP = 5
N_REPEATS = 30


# ── Input construction ──────────────────────────────────────────────────────────

def _powerline_edge_index(n_nodes: int, n_lines: int) -> Tensor:
    """
    Synthetic directed powerline edge index (both directions per line).

    Lines are arranged as a chain: node i → node (i+1) % n_nodes, plus reversed.

    :param n_nodes: Number of nodes in the graph.
    :param n_lines: Number of powerlines.
    :return: Edge index [2, 2*n_lines].
    """
    src = torch.arange(n_lines) % n_nodes
    dst = (torch.arange(n_lines) + 1) % n_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def _batch_edge_index(single_ei: Tensor, n_nodes: int, batch_size: int) -> Tensor:
    """
    Build a batched edge index for PyG by offsetting node indices per graph.

    :param single_ei: Edge index for one graph [2, E].
    :param n_nodes: Nodes per graph (offset step).
    :param batch_size: Number of graphs.
    :return: Batched edge index [2, B*E].
    """
    E = single_ei.size(1)
    offsets = torch.arange(batch_size).repeat_interleave(E) * n_nodes
    return single_ei.repeat(1, batch_size) + offsets.unsqueeze(0)


def _make_inputs(
    n_nodes: int,
    n_lines: int,
    batch_size: int,
    sparse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build synthetic RAGNN inputs for one batch.

    :param n_nodes: Nodes per graph.
    :param n_lines: Powerlines per graph.
    :param batch_size: Number of graphs in the batch.
    :param sparse: Use powerline edge index if True, fully-connected if False.
    :return:
        - x: Node features [B*N, x_dim].
        - edge_index: [2, B*E].
        - edge_type_posterior: Normalised soft assignments [B*E, K].
        - batch: Graph membership vector [B*N].
    """
    K = GNN_KWARGS["num_edge_types"]
    single_ei = (
        _powerline_edge_index(n_nodes, n_lines)
        if sparse
        else fully_connected_edge_index(n_nodes)
    )
    edge_index = _batch_edge_index(single_ei, n_nodes, batch_size)

    n_total_nodes = n_nodes * batch_size
    n_total_edges = edge_index.size(1)

    x = torch.randn(n_total_nodes, GNN_KWARGS["x_dim"])
    raw = torch.rand(n_total_edges, K)
    edge_type_posterior = raw / raw.sum(dim=-1, keepdim=True)
    batch = torch.arange(batch_size).repeat_interleave(n_nodes)

    return x, edge_index, edge_type_posterior, batch


# ── Timing helpers ──────────────────────────────────────────────────────────────

def _time_fn(fn: Callable[[], object], n_warmup: int, n_repeats: int) -> list[float]:
    """
    Warm up then time a callable.

    :param fn: Zero-argument callable to benchmark.
    :param n_warmup: Number of warm-up calls (discarded).
    :param n_repeats: Number of timed calls.
    :return: Wall-clock time per call in seconds.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _fmt(times: list[float]) -> str:
    """Format timing statistics as a compact string (milliseconds)."""
    mean_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    return f"mean={mean_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms"


# ── Main benchmark ──────────────────────────────────────────────────────────────

def benchmark_case(case_name: str, n_nodes: int, n_lines: int) -> None:
    """
    Run all three variants for one grid case and print a comparison table.

    :param case_name: Human-readable case label.
    :param n_nodes: Number of bus nodes.
    :param n_lines: Number of powerlines.
    """
    n_fc = n_nodes * (n_nodes - 1)
    n_sp = n_lines * 2
    print(f"\n{'=' * 70}")
    print(f"{case_name}: {n_nodes} nodes | FC={n_fc} edges | sparse={n_sp} edges")
    print(f"batch_size={BATCH_SIZE}  warmup={N_WARMUP}  repeats={N_REPEATS}")
    print(f"{'=' * 70}")

    model = RAGNN(**GNN_KWARGS)

    x_fc, ei_fc, etp_fc, batch_vec = _make_inputs(n_nodes, n_lines, BATCH_SIZE, sparse=False)
    x_sp, ei_sp, etp_sp, _ = _make_inputs(n_nodes, n_lines, BATCH_SIZE, sparse=True)

    with torch.no_grad():
        # [1] FC + diagnostics ON (model in eval mode)
        model.eval()
        t1 = _time_fn(lambda: model(x_fc, ei_fc, etp_fc, batch_vec), N_WARMUP, N_REPEATS)

        # [2] FC + diagnostics OFF (model in train mode; dropout=0 keeps outputs comparable)
        model.train()
        t2 = _time_fn(lambda: model(x_fc, ei_fc, etp_fc, batch_vec), N_WARMUP, N_REPEATS)

        # [3] sparse + diagnostics ON (eval mode, powerline edge index)
        model.eval()
        t3 = _time_fn(lambda: model(x_sp, ei_sp, etp_sp, batch_vec), N_WARMUP, N_REPEATS)

    mean1 = sum(t1) / len(t1)
    mean2 = sum(t2) / len(t2)
    mean3 = sum(t3) / len(t3)

    print(f"  [1] FC     + diag ON   {_fmt(t1)}")
    print(f"  [2] FC     + diag OFF  {_fmt(t2)}  speedup vs [1]: {mean1 / mean2:.2f}x")
    print(f"  [3] sparse + diag ON   {_fmt(t3)}  speedup vs [1]: {mean1 / mean3:.2f}x")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    for name, params in CASES.items():
        benchmark_case(name, **params)
