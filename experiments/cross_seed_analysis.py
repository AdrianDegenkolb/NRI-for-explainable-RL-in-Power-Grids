"""
Cross-seed posterior consistency analysis for RAPPO.

Discovers all Ray Tune trial directories in an experiment folder, identifies
the seed each belongs to (via params.json), selects the best available
checkpoint from each, runs inference on a shared set of observations, and
reports pairwise Spearman r of mean posteriors across seeds.

Designed to run automatically after a multi-seed SLURM array completes (see
experiments/slurm/cpu/rappo14_multiseed_with_analysis.sh).

Output (saved to --out-dir):
  spearman_heatmap.png  — S×S pairwise Spearman r matrix
  seed_posteriors.png   — per-seed N×N mean posterior heatmap
  posteriors.npz        — raw data for redrawing without re-running

Usage (from project root):
    PYTHONPATH=$(pwd)/src python experiments/cross_seed_analysis.py \\
        --experiment-dir results/experiments/2026_07_13_IEEE14/rappo_multiseed \\
        --out-dir results/experiments/2026_07_13_IEEE14/rappo_multiseed/cross_seed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(level=logging.WARNING)


# ── trial discovery ────────────────────────────────────────────────────────────

def find_trial_dirs(experiment_dir: Path) -> List[Path]:
    """Return all Ray Tune trial subdirectories (identified by presence of params.json)."""
    return sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (d / "params.json").exists()
    )


def get_seed(trial_dir: Path) -> int:
    """
    Extract the training seed from params.json.

    RLlib serialises AlgorithmConfig to params.json with 'seed' at the top level.

    :raises KeyError: if 'seed' is not present.
    """
    with open(trial_dir / "params.json") as f:
        params = json.load(f)
    seed = params.get("seed")
    if seed is None:
        raise KeyError(
            f"'seed' not found in {trial_dir / 'params.json'}. "
            "Keys present: " + str(list(params.keys())[:10])
        )
    return int(seed)


def best_checkpoint(trial_dir: Path) -> str:
    """
    Return the name of the best available checkpoint directory.

    Strategy:
    1. Parse progress.csv to find the training_iteration with highest
       episode_reward_mean; pick the available checkpoint closest to that
       iteration (from below).
    2. Fall back to the numerically highest checkpoint dir if the CSV is
       missing, unreadable, or no reward column is found.

    :raises FileNotFoundError: if no checkpoint dirs exist.
    """
    ckpt_dirs = sorted(
        [d for d in trial_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint dirs found in {trial_dir}")

    progress_csv = trial_dir / "progress.csv"
    if progress_csv.exists():
        try:
            import csv
            rows = []
            with open(progress_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        reward = float(row.get("episode_reward_mean", ""))
                        iteration = int(float(row.get("training_iteration", "")))
                        rows.append((reward, iteration))
                    except (ValueError, TypeError):
                        continue

            if rows:
                best_iter = max(rows, key=lambda t: t[0])[1]
                # Map checkpoint number to iteration:
                # checkpoint_N is saved at the N-th checkpoint event.
                # Determine checkpoint_freq from available dirs and rows.
                # Simpler: pick the available checkpoint whose number*freq is
                # closest to best_iter from below.
                # We don't know freq, but we can infer it from the last
                # checkpoint dir and the last training_iteration row.
                last_ckpt_num = int(ckpt_dirs[-1].name.split("_")[1])
                last_iter = rows[-1][1]
                if last_ckpt_num > 0 and last_iter > 0:
                    freq = last_iter / last_ckpt_num
                    # Estimate checkpoint number for best_iter
                    target_num = int(best_iter / freq)
                    # Find the closest available checkpoint number <= target_num
                    available_nums = [int(d.name.split("_")[1]) for d in ckpt_dirs]
                    candidates = [n for n in available_nums if n <= target_num]
                    if candidates:
                        chosen_num = max(candidates)
                        return f"checkpoint_{chosen_num:06d}"
        except Exception as exc:
            logging.warning("Could not parse progress.csv (%s); using latest checkpoint.", exc)

    # Fallback: use the last (numerically highest) available checkpoint
    return ckpt_dirs[-1].name


# ── observation collection ─────────────────────────────────────────────────────

def collect_obs(env_name: str, n_target: int, rho_thresh: float) -> List:
    """
    Collect diverse Grid2Op observations spanning many chronics and rho levels.

    For each chronic: run do-nothing until rho > rho_thresh or game-over.
    Also samples every 20th step for variety. Results are shuffled and capped
    at n_target.

    :return: list of (obs, rho_max, chronic_id, step).
    """
    import grid2op
    env = grid2op.make(env_name)
    n_chronics = len(env.chronics_handler.subpaths)
    collected = []

    for chronic_id in range(n_chronics):
        if len(collected) >= n_target * 3:
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

            if rho_max > rho_thresh:
                collected.append((obs.copy(), rho_max, chronic_id, step))
                if rho_max > 0.95 or step > 200:
                    break

            if step % 20 == 0:
                collected.append((obs.copy(), rho_max, chronic_id, step))

        if float(last_obs.rho.max()) < rho_thresh:
            collected.append((last_obs.copy(), float(last_obs.rho.max()), chronic_id, step))

    rng = np.random.default_rng(42)
    rng.shuffle(collected)
    result = collected[:n_target]
    rhos = [r for _, r, _, _ in result]
    print(f"  Collected {len(result)} obs from {n_chronics} chronics  "
          f"rho=[{min(rhos):.2f}, {max(rhos):.2f}]  mean={np.mean(rhos):.2f}")
    return result


# ── model loading & inference ──────────────────────────────────────────────────

def load_agent(trial_dir: Path, checkpoint_name: str, env_name: str):
    """
    Load an RllibAgent from a checkpoint.

    :return: (agent, g2op_env, gym_wrapper) — keep gym_wrapper alive.
    """
    from core.constants import RL_POLICY
    from core.loading import load_config, preprocess_config, load_rllib_agent
    params = preprocess_config(load_config(trial_dir))
    agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=str(trial_dir),
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=params["env_config"],
    )
    return agent, g2op_env, gym_wrapper


def compute_posteriors(agent, obs_list: List) -> np.ndarray:
    """
    Run inference for every observation; return interaction probabilities [N_obs, E].

    interaction_prob[e] = posterior[e, :-1].sum()
    (sum over all edge-types except the final "no-edge" type).

    :param agent: Loaded RllibAgent with an RA model.
    :param obs_list: List of raw Grid2Op BaseObservation objects.
    :return: Float32 array [N_obs, E].
    """
    from core.constants import RL_POLICY
    results = []
    for obs in obs_list:
        agent.gym_wrapper.update_obs(obs)
        agent._rllib_agent.compute_single_action(
            agent.gym_wrapper.cur_gym_obs, policy_id=RL_POLICY
        )
        p = agent._rllib_agent.model.get_posterior()   # [1, E, K] or [E, K]
        if p.dim() == 3:
            p = p[0]                                    # [E, K]
        results.append(p.cpu().detach().numpy()[:, :-1].sum(axis=-1))  # [E]
    return np.stack(results).astype(np.float32)         # [N_obs, E]


def n_nodes_from_env(g2op_env) -> int:
    """Compute num_nodes consistent with GraphObservationConverter."""
    n_storage = g2op_env.n_storage if hasattr(g2op_env, "n_storage") else 0
    return 2 * g2op_env.n_line + g2op_env.n_gen + g2op_env.n_load + n_storage


# ── analysis ───────────────────────────────────────────────────────────────────

def pairwise_spearman(
    seed_posteriors: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """
    Pairwise Spearman r between per-seed mean interaction probability vectors.

    :param seed_posteriors: {seed: [N_obs, E] array}.
    :return: (r_matrix [S, S], sorted seeds list).
    """
    seeds = sorted(seed_posteriors.keys())
    means = np.stack([seed_posteriors[s].mean(axis=0) for s in seeds])  # [S, E]
    S = len(seeds)
    r_mat = np.ones((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            r, _ = spearmanr(means[i], means[j])
            r_mat[i, j] = r_mat[j, i] = float(r)
    return r_mat, seeds


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_spearman_heatmap(
    r_mat: np.ndarray,
    seeds: List[int],
    out_path: Path,
) -> None:
    """S×S annotated heatmap of pairwise Spearman r values."""
    S = len(seeds)
    fig, ax = plt.subplots(figsize=(max(4, S + 1), max(4, S + 1)))
    im = ax.imshow(r_mat, vmin=-1.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    labels = [f"seed {s}" for s in seeds]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(S):
        for j in range(S):
            color = "white" if abs(r_mat[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{r_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label="Spearman r", fraction=0.046, pad=0.04)
    off_diag = r_mat[np.triu_indices(S, k=1)]
    ax.set_title(
        f"Cross-seed posterior consistency (mean posterior per seed)\n"
        f"mean r={off_diag.mean():.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_seed_posteriors(
    seed_posteriors: Dict[int, np.ndarray],
    n_nodes: int,
    n_line: int,
    out_path: Path,
) -> None:
    """One N×N mean posterior heatmap per seed, with powerline edge overlay."""
    from rarl.graph import fully_connected_edge_index
    seeds = sorted(seed_posteriors.keys())
    S = len(seeds)

    fc = fully_connected_edge_index(n_nodes).numpy()  # [2, E]
    pl_src = list(range(n_line)) + list(range(n_line, 2 * n_line))
    pl_dst = list(range(n_line, 2 * n_line)) + list(range(n_line))

    fig, axes = plt.subplots(1, S, figsize=(4 * S, 4.5), squeeze=False)
    im_ref = None
    for col, seed in enumerate(seeds):
        ax = axes[0][col]
        mean_p = seed_posteriors[seed].mean(axis=0)  # [E]
        mat = np.full((n_nodes, n_nodes), np.nan)
        mat[fc[0], fc[1]] = mean_p

        im_ref = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlOrRd", origin="upper")
        for src, dst in zip(pl_src, pl_dst):
            ax.add_patch(plt.Rectangle(
                (dst - 0.5, src - 0.5), 1, 1,
                fill=False, edgecolor="steelblue", linewidth=0.5,
            ))
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("dest node")
        if col == 0:
            ax.set_ylabel("src node")

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes[0], label="P(edge exists) mean", fraction=0.015, pad=0.04)
    fig.suptitle(
        f"Per-seed mean posterior over {len(next(iter(seed_posteriors.values())))} observations\n"
        "Blue rectangles = powerline edges",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment-dir", required=True,
                   help="Ray Tune experiment dir containing per-trial subdirs")
    p.add_argument("--env", default="l2rpn_case14_sandbox_test",
                   help="Grid2Op environment for inference (default: case14 test)")
    p.add_argument("--n-obs", type=int, default=100,
                   help="Number of shared observations (default: 100)")
    p.add_argument("--rho-thresh", type=float, default=0.7,
                   help="Min rho to target in observation collection (default: 0.7)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <experiment-dir>/cross_seed)")
    p.add_argument("--checkpoint-name", default=None,
                   help="Override checkpoint name for all trials "
                        "(e.g. checkpoint_000005). Default: best per trial.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir / "cross_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    import ray
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False)

    # ── Discover trials and map to seeds ──────────────────────────────────────
    trial_dirs = find_trial_dirs(experiment_dir)
    if not trial_dirs:
        raise RuntimeError(f"No trial dirs (with params.json) found in {experiment_dir}")

    print(f"Found {len(trial_dirs)} trial dir(s) in {experiment_dir}\n")
    seed_to_info: Dict[int, Tuple[Path, str]] = {}
    for td in trial_dirs:
        try:
            seed = get_seed(td)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  WARNING: skipping {td.name} — {e}")
            continue
        try:
            ckpt = args.checkpoint_name or best_checkpoint(td)
        except FileNotFoundError as e:
            print(f"  WARNING: skipping {td.name} — {e}")
            continue
        seed_to_info[seed] = (td, ckpt)
        print(f"  seed={seed}  checkpoint={ckpt}  trial={td.name}")

    if not seed_to_info:
        raise RuntimeError("No usable trials found.")

    # ── Collect shared observations ────────────────────────────────────────────
    print(f"\nCollecting {args.n_obs} shared observations "
          f"(env={args.env}, rho_thresh={args.rho_thresh}) ...")
    obs_records = collect_obs(args.env, args.n_obs, args.rho_thresh)
    obs_list = [o for o, *_ in obs_records]

    # ── Per-seed inference ─────────────────────────────────────────────────────
    seed_posteriors: Dict[int, np.ndarray] = {}
    gym_wrappers = []   # keep alive until after inference
    n_nodes = n_line = None

    for seed in sorted(seed_to_info.keys()):
        trial_dir, ckpt_name = seed_to_info[seed]
        print(f"\nSeed {seed}: loading {ckpt_name} from {trial_dir.name} ...")
        agent, g2op_env, gym_wrapper = load_agent(trial_dir, ckpt_name, args.env)
        gym_wrappers.append(gym_wrapper)

        if n_nodes is None:
            n_nodes = n_nodes_from_env(g2op_env)
            n_line = g2op_env.n_line
            print(f"  Grid: n_nodes={n_nodes}  n_line={n_line}")

        print(f"  Running inference on {len(obs_list)} observations ...")
        probs = compute_posteriors(agent, obs_list)   # [N_obs, E]
        seed_posteriors[seed] = probs
        print(f"  mean={probs.mean():.4f}  std={probs.std():.4f}")

    # ── Cross-seed Spearman r ──────────────────────────────────────────────────
    r_mat, seeds = pairwise_spearman(seed_posteriors)
    off_diag = r_mat[np.triu_indices(len(seeds), k=1)]

    print("\n=== CROSS-SEED SPEARMAN r (mean posterior per seed) ===")
    header = "         " + "  ".join(f"seed {s}" for s in seeds)
    print(header)
    for i, si in enumerate(seeds):
        row = f"seed {si}:  " + "  ".join(f"{r_mat[i, j]:+.3f}" for j in range(len(seeds)))
        print(row)
    print(f"\nOff-diagonal: mean={off_diag.mean():.3f}  "
          f"min={off_diag.min():.3f}  max={off_diag.max():.3f}")

    # ── Per-seed input-dependence ──────────────────────────────────────────────
    print("\n=== PER-SEED INPUT-DEPENDENCE (std across observations) ===")
    for seed in sorted(seed_posteriors.keys()):
        probs = seed_posteriors[seed]              # [N_obs, E]
        mean_std = probs.std(axis=0).mean()
        cv = mean_std / (probs.mean() + 1e-8)
        print(f"  seed {seed}: mean_std={mean_std:.4f}  CV={cv:.3f}")

    # ── Save raw data ──────────────────────────────────────────────────────────
    npz_path = out_dir / "posteriors.npz"
    np.savez(
        npz_path,
        seeds=np.array(seeds),
        r_matrix=r_mat,
        **{f"posteriors_seed{s}": seed_posteriors[s] for s in seeds},
    )
    print(f"\nSaved: {npz_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_spearman_heatmap(r_mat, seeds, out_dir / "spearman_heatmap.png")
    plot_seed_posteriors(seed_posteriors, n_nodes, n_line,
                         out_dir / "seed_posteriors.png")

    ray.shutdown()


if __name__ == "__main__":
    main()
