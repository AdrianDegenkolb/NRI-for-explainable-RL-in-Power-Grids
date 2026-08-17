"""Plot a 5×5 latent graph grid: 5 high-rho observations × 5 seeds.

Rows    = 5 observations with rho_max > --rho-thresh collected under do-nothing.
Columns = 5 trained seeds (sorted by seed index).

Uses visualize_graph from src/visualization/utils.py.

Usage (from project root):
    PYTHONPATH=$(pwd)/src conda run -n L2RPN python experiments/plot_latent_graph_grid.py
    PYTHONPATH=$(pwd)/src conda run -n L2RPN python experiments/plot_latent_graph_grid.py \\
        --experiment-dir results/2026_07_18_IEEE14/rappo_multiseed_gcnconv \\
        --env l2rpn_case14_sandbox_test \\
        --rho-thresh 0.95 --threshold 0.5 \\
        --output latent_graph_grid_ieee14.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.WARNING)


# ── trial / seed discovery ─────────────────────────────────────────────────────

def find_trial_dirs(experiment_dir: Path) -> list[Path]:
    return sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (
            (d / "params.json").exists()
            or any(c.is_dir() and c.name.startswith("checkpoint_") for c in d.iterdir())
        )
    )


def _seed_from_slurm_log(trial_dir: Path) -> Optional[int]:
    job_id = trial_dir.name.split("_")[2]
    out_dir = trial_dir.parent / "out"
    if not out_dir.is_dir():
        return None
    for log_file in out_dir.glob(f"*_s*.{job_id}.log"):
        stem = log_file.name.split(".")[0]
        parts = stem.rsplit("_s", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def _seed_from_experiment_state(trial_dir: Path) -> Optional[int]:
    for state_file in sorted(trial_dir.parent.glob("experiment_state-*.json")):
        try:
            d = json.loads(state_file.read_text().strip())
        except Exception:
            continue
        for td in d.get("trial_data", []):
            if not isinstance(td, list):
                continue
            for item in td:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue
                logdir = item.get("relative_logdir") or item.get("logdir", "")
                if Path(logdir).name == trial_dir.name:
                    return item.get("config", {}).get("seed")
    return None


def get_seed(trial_dir: Path) -> int:
    params_file = trial_dir / "params.json"
    if params_file.exists():
        try:
            seed = json.loads(params_file.read_text()).get("seed")
            if seed is not None:
                return int(seed)
        except Exception:
            pass
    for fn in [_seed_from_experiment_state, _seed_from_slurm_log]:
        seed = fn(trial_dir)
        if seed is not None:
            return int(seed)
    raise KeyError(f"Cannot determine seed for {trial_dir.name}")


def best_checkpoint(trial_dir: Path) -> str:
    ckpt_dirs = sorted(
        [d for d in trial_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints in {trial_dir}")
    return ckpt_dirs[-1].name


# ── observation collection ─────────────────────────────────────────────────────

def collect_critical_obs(g2op_env, n_obs: int, rho_thresh: float) -> list:
    """Collect up to n_obs observations with rho_max > rho_thresh using do-nothing.

    Steps through chronics sequentially; takes the first timestep per chronic
    where the threshold is exceeded so observations come from diverse grid states.
    """
    collected = []
    n_chronics = len(g2op_env.chronics_handler.subpaths)
    for cid in range(n_chronics):
        if len(collected) >= n_obs:
            break
        g2op_env.set_id(cid)
        obs = g2op_env.reset()
        done = False
        while not done:
            obs, _, done, _ = g2op_env.step(g2op_env.action_space({}))
            if float(obs.rho.max()) > rho_thresh:
                collected.append((obs.copy(), float(obs.rho.max()), cid))
                break
    return collected


# ── posterior extraction ───────────────────────────────────────────────────────

def get_posterior(agent, g2op_obs) -> np.ndarray:
    """Run one forward pass; return raw posterior [E, K]."""
    from core.constants import RL_POLICY
    agent.gym_wrapper.update_obs(g2op_obs)
    agent._rllib_agent.compute_single_action(
        agent.gym_wrapper.cur_gym_obs, policy_id=RL_POLICY,
    )
    p = agent._rllib_agent.model.get_posterior()
    if p.dim() == 3:
        p = p[0]
    return p.cpu().detach().numpy()  # [E, K]


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-dir",
                   default="results/2026_07_18_IEEE14/rappo_multiseed_gcnconv")
    p.add_argument("--env", default="l2rpn_case14_sandbox_test")
    p.add_argument("--n-obs", type=int, default=5,
                   help="Number of observations (rows) — default 5")
    p.add_argument("--rho-thresh", type=float, default=0.95,
                   help="Minimum rho_max to consider a state critical (default 0.95)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Edge probability threshold for visualization (default 0.5)")
    p.add_argument("--output", default="latent_graph_grid_ieee14.png")
    p.add_argument("--conv-type", default=None, choices=["gcn", "gin"],
                   help="Override GNN conv type (use 'gin' for pre-revert checkpoints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)

    import ray
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR,
             log_to_driver=False, num_cpus=1,
             object_store_memory=512 * 1024 * 1024)

    from core.loading import load_config, preprocess_config, load_rllib_agent
    from core.constants import RL_POLICY
    from visualization.utils import get_node_styles, PlottingArgs, visualize_graph
    from grid2op_env.observation_converter import GraphObservationConverter

    # ── Discover and sort trials by seed ──────────────────────────────────────
    trial_dirs = find_trial_dirs(experiment_dir)
    seed_to_trial: dict[int, Path] = {}
    for td in trial_dirs:
        try:
            seed_to_trial[get_seed(td)] = td
        except KeyError as e:
            print(f"WARNING: skipping {td.name} — {e}")
    seeds = sorted(seed_to_trial.keys())
    print(f"Seeds found: {seeds}")

    # ── Load first seed to collect observations and get grid metadata ─────────
    first_seed = seeds[0]
    first_trial = seed_to_trial[first_seed]
    first_ckpt = best_checkpoint(first_trial)
    print(f"\nLoading seed {first_seed} ({first_ckpt}) to collect observations ...")
    params = preprocess_config(load_config(first_trial))
    first_agent, g2op_env, _ = load_rllib_agent(
        checkpoint_path=str(first_trial),
        policy_name=RL_POLICY,
        checkpoint_name=first_ckpt,
        env_name=args.env,
        env_config=params["env_config"],
        conv_type=args.conv_type,
    )

    node_styles = get_node_styles(g2op_env, GraphObservationConverter)
    n_line = g2op_env.n_line
    pl_src = list(range(n_line)) + list(range(n_line, 2 * n_line))
    pl_dst = list(range(n_line, 2 * n_line)) + list(range(n_line))
    powerline_edge_index = np.array([pl_src, pl_dst])
    num_nodes = len(node_styles)

    # ── Collect critical observations (do-nothing, rho > rho_thresh) ──────────
    print(f"\nCollecting {args.n_obs} observations with rho > {args.rho_thresh} ...")
    obs_records = collect_critical_obs(g2op_env, args.n_obs, args.rho_thresh)
    print(f"  Got {len(obs_records)} observations")
    if len(obs_records) == 0:
        raise RuntimeError("No critical observations found — try lowering --rho-thresh")

    g2op_obs_list = [r[0] for r in obs_records]
    rhos = [r[1] for r in obs_records]
    chronic_ids = [r[2] for r in obs_records]
    n_obs = len(g2op_obs_list)

    # ── Compute posteriors per seed × observation ──────────────────────────────
    # posteriors[seed] = list of [E, K] arrays, one per observation
    all_posteriors: dict[int, list[np.ndarray]] = {}

    print(f"\nSeed {first_seed}: computing posteriors ...")
    all_posteriors[first_seed] = [get_posterior(first_agent, obs) for obs in g2op_obs_list]

    for seed in seeds[1:]:
        trial_dir = seed_to_trial[seed]
        ckpt = best_checkpoint(trial_dir)
        print(f"Seed {seed}: loading {ckpt} from {trial_dir.name} ...")
        params = preprocess_config(load_config(trial_dir))
        agent, _, _ = load_rllib_agent(
            checkpoint_path=str(trial_dir),
            policy_name=RL_POLICY,
            checkpoint_name=ckpt,
            env_name=args.env,
            env_config=params["env_config"],
            conv_type=args.conv_type,
        )
        print(f"  Computing posteriors ...")
        all_posteriors[seed] = [get_posterior(agent, obs) for obs in g2op_obs_list]

    # ── Plot n_obs × n_seeds grid ─────────────────────────────────────────────
    n_cols = len(seeds)
    cell_size = 3.5
    fig, axes = plt.subplots(
        n_obs, n_cols,
        figsize=(cell_size * n_cols, cell_size * n_obs),
        squeeze=False,
    )

    for col, seed in enumerate(seeds):
        for row in range(n_obs):
            ax = axes[row, col]
            posterior = all_posteriors[seed][row]   # [E, K]
            n_active = int((posterior[:, :-1].sum(-1) > args.threshold).sum())

            if row == 0:
                title = f"seed {seed}\n({n_active} edges > {args.threshold})"
            else:
                title = f"({n_active} edges > {args.threshold})"
            ax.set_title(title, fontsize=7)

            visualize_graph(
                PlottingArgs(
                    num_nodes=num_nodes,
                    node_styles=node_styles,
                    powerline_edge_index=powerline_edge_index,
                    latent_edge_probs=posterior,
                    latent_edge_weight=5.0,
                    skip_last_edge_type=True,
                    visualize_edge_prob_threshold=args.threshold,
                    show_legend=False,
                ),
                ax=ax,
            )

    # Row labels on the left side
    for row in range(n_obs):
        axes[row, 0].set_ylabel(
            f"ch{chronic_ids[row]}  rho={rhos[row]:.2f}", fontsize=7
        )

    fig.suptitle(
        f"Latent graph — {n_obs} critical observations (rho > {args.rho_thresh}) × {n_cols} seeds\n"
        f"{experiment_dir.name} — threshold={args.threshold}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.output)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)

    ray.shutdown()


if __name__ == "__main__":
    main()
