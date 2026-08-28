"""
GNN collapse diagnostic.

Measures how sensitive the model's output is to graph structure by comparing
logits across three conditions on a batch of real observations:

  baseline     — original edge connectivity
  zero_edges   — EDGE_MASK zeroed (only GCNConv self-loops remain)
  shuffled     — src/dst node indices independently permuted

Also hooks into each GCNConv layer to measure the neighbour-message
contribution relative to the self-loop signal at every layer.

Usage
-----
PYTHONPATH=$(pwd)/src python experiments/diagnose_gnn_collapse.py [options]

Examples
--------
python experiments/diagnose_gnn_collapse.py --method Default --steps 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.loading import AgentSpec, load_agent_from_spec
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES
from experiments.graph_ablation import METHODS, best_checkpoint


# ── Helpers ────────────────────────────────────────────────────────────────────

_INT_KEYS = {EDGE_INDEX}

def to_batch(obs: dict) -> dict:
    """Convert a numpy obs dict to a single-item tensor batch.

    EDGE_INDEX is cast to int64 (required by gather/GCNConv); everything
    else is cast to float32.
    """
    return {
        k: torch.tensor(v, dtype=torch.int64 if k in _INT_KEYS else torch.float32).unsqueeze(0)
        for k, v in obs.items()
    }


def get_logits(model: torch.nn.Module, obs: dict) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _ = model({"obs": to_batch(obs)}, [], None)
    return logits.squeeze(0)


def collect_observations(agent, env, gym_env, n: int) -> list[dict]:
    """Roll out n steps and collect gym observations."""
    obs_list = []
    g2op_obs = env.reset()
    done = False
    reward = 0
    for _ in range(n):
        gym_obs = gym_env.observation_converter.to_gym(g2op_obs)
        obs_list.append(gym_obs)
        action = agent.act(g2op_obs, reward)
        g2op_obs, reward, done, _ = env.step(action)
        if done:
            g2op_obs = env.reset()
            done = False
    return obs_list


def shuffle_edges(obs: dict, rng: np.random.Generator) -> dict:
    n_nodes = obs[NODES].shape[0]
    perm_src = rng.permutation(n_nodes)
    perm_dst = rng.permutation(n_nodes)
    valid = obs[EDGE_MASK].astype(bool)
    ei = obs[EDGE_INDEX].copy()
    ei[0, valid] = perm_src[ei[0, valid]]
    ei[1, valid] = perm_dst[ei[1, valid]]
    return {**obs, EDGE_INDEX: ei}


def zero_edges(obs: dict) -> dict:
    return {**obs, EDGE_MASK: np.zeros_like(obs[EDGE_MASK])}


# ── Per-layer neighbour contribution ──────────────────────────────────────────

def measure_layer_contributions(model: torch.nn.Module, obs: dict) -> None:
    """
    For each GCNConv layer in the GNN, compare:
      - output with real edges  (full forward)
      - output with zero edges  (only GCNConv self-loops remain)

    Reports the ratio  ||Δh_neighbours|| / ||h_full||  per layer, where
    Δh_neighbours = h_full - h_self_loops_only.
    A ratio near 0 means the layer ignores neighbours entirely.
    """
    gnn = getattr(model, "gnn", None)
    if gnn is None:
        print("  [skip] model has no .gnn attribute")
        return

    # Capture layer inputs/outputs via hooks
    layer_records: list[dict] = []

    def make_hook(record: dict, key: str):
        def hook(_, __, output):
            record[key] = output.detach().clone()
        return hook

    handles = []
    for layer_modules in gnn.layers:
        for conv in layer_modules:
            rec: dict = {}
            layer_records.append(rec)
            handles.append(conv.register_forward_hook(make_hook(rec, "full")))

    # Forward with real edges
    model.eval()
    with torch.no_grad():
        model({"obs": to_batch(obs)}, [], None)

    # Swap hooks to record self-loop-only pass
    for h in handles:
        h.remove()
    handles.clear()

    for i, (layer_modules, rec) in enumerate(zip(gnn.layers, layer_records)):
        for conv in layer_modules:
            handles.append(conv.register_forward_hook(make_hook(rec, "self_only")))

    obs_zero = zero_edges(obs)
    with torch.no_grad():
        model({"obs": to_batch(obs_zero)}, [], None)

    for h in handles:
        h.remove()

    print(f"\n  Per-layer neighbour contribution  "
          f"(||h_full - h_self_only|| / ||h_full||):")
    for i, rec in enumerate(layer_records):
        if "full" not in rec or "self_only" not in rec:
            continue
        diff  = (rec["full"] - rec["self_only"]).norm().item()
        total = rec["full"].norm().item()
        ratio = diff / (total + 1e-12)
        print(f"    layer {i}: {ratio:.4f}  "
              f"(|Δ|={diff:.4f}, |full|={total:.4f})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GNN collapse diagnostic")
    parser.add_argument("--method",  default="Default", choices=list(METHODS))
    parser.add_argument("--steps",   type=int, default=50,
                        help="Number of observations to average over (default: 50)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    cfg = METHODS[args.method]
    dirs = sorted(ROOT.glob(cfg["glob"]))
    d = dirs[0]
    ckpt = best_checkpoint(d / "checkpoint_results.json")[0]

    agent, env, gym_env = load_agent_from_spec(
        AgentSpec(name=args.method, load_path=d, checkpoint_name=ckpt)
    )
    model = agent._rllib_agent.model

    print(f"\nMethod: {args.method} | checkpoint: {ckpt} | steps: {args.steps}")

    # ── Collect observations ───────────────────────────────────────────────────
    obs_list = collect_observations(agent, env, gym_env, args.steps)
    print(f"Collected {len(obs_list)} observations.")

    # ── Logit sensitivity ─────────────────────────────────────────────────────
    d_zero_list, d_shuf_list, logit_std_list = [], [], []

    for obs in obs_list:
        l_base = get_logits(model, obs)
        l_zero = get_logits(model, zero_edges(obs))
        l_shuf = get_logits(model, shuffle_edges(obs, rng))

        d_zero_list.append((l_base - l_zero).norm().item())
        d_shuf_list.append((l_base - l_shuf).norm().item())
        logit_std_list.append(l_base.std().item())

    avg_std  = np.mean(logit_std_list)
    avg_zero = np.mean(d_zero_list)
    avg_shuf = np.mean(d_shuf_list)

    print(f"\n  Logit sensitivity (averaged over {args.steps} observations):")
    print(f"    logit std (baseline)              : {avg_std:.4f}  ← scale reference")
    print(f"    L2(baseline, zero_edges)  avg/max : {avg_zero:.4f} / {max(d_zero_list):.4f}")
    print(f"    L2(baseline, shuffled)    avg/max : {avg_shuf:.4f} / {max(d_shuf_list):.4f}")
    print(f"    relative sensitivity (zero_edges) : {avg_zero / (avg_std + 1e-12):.4f}  ← ~0 = collapsed")
    print(f"    relative sensitivity (shuffled)   : {avg_shuf / (avg_std + 1e-12):.4f}  ← ~0 = collapsed")

    # ── Per-layer analysis on first observation ────────────────────────────────
    print(f"\n  Per-layer analysis (first observation):")
    measure_layer_contributions(model, obs_list[0])


if __name__ == "__main__":
    main()
