"""
Diagnostic script for graph ablation.

Runs three conditions back-to-back and reports steps survived and RL
activation rate per episode:

  baseline     — original observations, no ablation
  ablation     — EDGE_INDEX shuffled (GraphAblationWrapper)
  zero_edges   — EDGE_MASK zeroed out (no edges at all)

Usage
-----
python experiments/diagnose_graph_ablation.py [options]

Examples
--------
# 5 episodes, RL always active (stress-test the GNN directly):
python experiments/diagnose_graph_ablation.py --episodes 5 --threshold 0.0

# 10 episodes, normal heuristic threshold:
python experiments/diagnose_graph_ablation.py --episodes 10 --threshold 0.95

# Pick a different method:
python experiments/diagnose_graph_ablation.py --method Elements --episodes 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.loading import AgentSpec, load_agent_from_spec
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK
from experiments.graph_ablation import GraphAblationWrapper, METHODS, best_checkpoint


# ── Observation wrappers ───────────────────────────────────────────────────────

def _apply_ablation(gym_env, seed: int = 42) -> None:
    gym_env.observation_converter = GraphAblationWrapper(
        gym_env.observation_converter, seed=seed
    )


def _apply_zero_edges(gym_env) -> None:
    orig = gym_env.observation_converter

    class _ZeroEdgeMask:
        observation_space = orig.observation_space
        def to_gym(self, obs):
            result = orig.to_gym(obs)
            result[EDGE_MASK] = np.zeros_like(result[EDGE_MASK])
            return result
        def reset_obs(self):   orig.reset_obs()
        def normalize(self, o): return orig.normalize(o)
        def close(self):        orig.close()

    gym_env.observation_converter = _ZeroEdgeMask()


# ── Single-condition runner ────────────────────────────────────────────────────

def run_condition(
    method: str,
    cfg: dict,
    label: str,
    wrapper_fn,
    num_episodes: int,
    activation_threshold: float,
) -> None:
    """Load agent, apply wrapper, run num_episodes, print per-episode stats.

    Args:
        method: method display name
        cfg: METHODS entry with 'glob' key
        label: condition label for printing
        wrapper_fn: callable(gym_env) that patches the observation converter
        num_episodes: number of episodes to run
        activation_threshold: rho threshold below which the heuristic acts alone
    """
    dirs = sorted(ROOT.glob(cfg["glob"]))
    if not dirs:
        print(f"[{label}] No checkpoints found for {method}")
        return
    d = dirs[0]
    ckpt = best_checkpoint(d / "checkpoint_results.json")[0]

    agent, env, gym_env = load_agent_from_spec(
        AgentSpec(name=method, load_path=d, checkpoint_name=ckpt)
    )
    if wrapper_fn is not None:
        wrapper_fn(gym_env)

    agent.activation_thresh = activation_threshold

    # Count RL activations by wrapping compute_single_action
    rl_calls = [0]
    orig_compute = agent._rllib_agent.compute_single_action
    def _counted(*a, **kw):
        rl_calls[0] += 1
        return orig_compute(*a, **kw)
    agent._rllib_agent.compute_single_action = _counted

    print(f"\n{'─'*60}")
    print(f"  Condition : {label}")
    print(f"  Method    : {method}  |  checkpoint: {ckpt}")
    print(f"  Episodes  : {num_episodes}  |  threshold: {activation_threshold}")
    print(f"{'─'*60}")

    survived_list = []
    activation_list = []

    for ep in range(num_episodes):
        rl_calls[0] = 0
        obs = env.reset()
        done = False
        reward = 0
        steps = 0
        while not done:
            action = agent.act(obs, reward)
            obs, reward, done, info = env.step(action)
            steps += 1
        survived_list.append(steps)
        activation_list.append(rl_calls[0])
        print(f"  ep {ep+1:>3}: survived {steps:>5} steps | RL activations: {rl_calls[0]:>4}")

    avg_steps = np.mean(survived_list)
    avg_act   = np.mean(activation_list)
    pct       = avg_steps / 8064 * 100
    print(f"  {'─'*54}")
    print(f"  avg: {avg_steps:>6.1f} steps ({pct:.1f}%) | avg RL activations: {avg_act:.1f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Graph ablation diagnostic")
    parser.add_argument("--method",    default="Default", choices=list(METHODS),
                        help="Which trained method to load (default: Default)")
    parser.add_argument("--episodes",  type=int,   default=1,
                        help="Number of episodes per condition (default: 1)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="RL activation threshold — rho must exceed this for "
                             "the RL agent to act. Use 0.0 to always activate. (default: 0.95)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="RNG seed for GraphAblationWrapper (default: 42)")
    args = parser.parse_args()

    cfg = METHODS[args.method]
    common = dict(method=args.method, cfg=cfg,
                  num_episodes=args.episodes,
                  activation_threshold=args.threshold)

    run_condition(label="Baseline   (no ablation)",
                  wrapper_fn=None, **common)
    run_condition(label="Ablation   (shuffled edges)",
                  wrapper_fn=lambda g: _apply_ablation(g, seed=args.seed), **common)
    run_condition(label="Zero edges (EDGE_MASK all zeros)",
                  wrapper_fn=_apply_zero_edges, **common)


if __name__ == "__main__":
    main()
