"""
Cross-seed posterior consistency analysis for RAPPO.

Discovers all Ray Tune trial directories in an experiment folder, identifies
the seed each belongs to (via params.json), selects the best available
checkpoint from each, runs inference on a shared set of observations, and
reports pairwise Pearson r and top-K edge overlap of mean posteriors across seeds.

Designed to run automatically after a multi-seed SLURM array completes (see
experiments/slurm/cpu/rappo14_multiseed_with_analysis.sh).

Output (saved to --out-dir):
  pearson_heatmap.png      — S×S pairwise Pearson r matrix
  topk_overlap_heatmap.png — S×S pairwise top-K edge overlap matrix
  seed_posteriors.png      — per-seed N×N mean posterior heatmap
  posteriors.npz           — raw data for redrawing without re-running

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
from scipy.stats import pearsonr

logging.basicConfig(level=logging.WARNING)


# ── trial discovery ────────────────────────────────────────────────────────────

def find_trial_dirs(experiment_dir: Path) -> List[Path]:
    """Return all Ray Tune trial subdirectories.

    Accepts dirs that have either params.json (original runs) or at least one
    checkpoint_* subdirectory (re-run trials that Ray Tune skipped serialising
    params.json for).
    """
    return sorted(
        d for d in experiment_dir.iterdir()
        if d.is_dir() and (
            (d / "params.json").exists()
            or any(c.is_dir() and c.name.startswith("checkpoint_")
                   for c in d.iterdir())
        )
    )


def _seed_from_slurm_log(trial_dir: Path) -> Optional[int]:
    """Infer seed from SLURM log filenames like 'rappo14fk_s1.5887370.log'.

    The SLURM script names output files with the pattern *_s{seed}.{job_id}.log
    and the job ID matches the numeric part of the trial folder name.
    """
    job_id = trial_dir.name.split("_")[2]  # e.g. "5887370" from CustomPPO_RARL_5887370_...
    out_dir = trial_dir.parent / "out"
    if not out_dir.is_dir():
        return None
    for log_file in out_dir.glob(f"*_s*.{job_id}.log"):
        # filename: rappo14fk_s1.5887370.log  →  stem before first dot = "rappo14fk_s1"
        stem = log_file.name.split(".")[0]   # "rappo14fk_s1"
        parts = stem.rsplit("_s", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
    return None


def _seed_from_experiment_state(trial_dir: Path) -> Optional[int]:
    """Look up this trial's seed in the parent's experiment_state-*.json."""
    for state_file in sorted(trial_dir.parent.glob("experiment_state-*.json")):
        try:
            content = state_file.read_text().strip()
            if not content:
                continue
            d = json.loads(content)
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
    """
    Extract the training seed for a trial.

    Tries in order:
    1. params.json at the trial level (original runs serialised by Ray Tune).
    2. experiment_state-*.json in the parent directory (re-run trials where
       Ray Tune did not write params.json).
    3. SLURM log filename in out/ (e.g. rappo14fk_s1.5887370.log).

    :raises KeyError: if the seed cannot be found by any method.
    """
    params_file = trial_dir / "params.json"
    if params_file.exists():
        try:
            params = json.loads(params_file.read_text())
            seed = params.get("seed")
            if seed is not None:
                return int(seed)
        except Exception:
            pass

    seed = _seed_from_experiment_state(trial_dir)
    if seed is not None:
        return int(seed)

    seed = _seed_from_slurm_log(trial_dir)
    if seed is not None:
        return int(seed)

    raise KeyError(
        f"Could not determine seed for {trial_dir.name}. "
        "Checked params.json, experiment_state-*.json, and SLURM log filenames."
    )


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

def load_agent(trial_dir: Path, checkpoint_name: str, env_name: str,
               conv_type: Optional[str] = None):
    """
    Load an RllibAgent from a checkpoint.

    :param conv_type: Override the GNN conv type in the loaded config.
        Use ``"gin"`` for checkpoints trained before the GCN revert.
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
        conv_type=conv_type,
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

def pairwise_pearson(
    seed_posteriors: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """
    Pairwise Pearson r between per-seed mean interaction probability vectors.

    Pearson is preferred over Spearman here because a large fraction of edges
    share near-identical low probabilities, making rank assignments noisy and
    Spearman r unreliable. Pearson operates on the actual values so the
    high-probability minority of edges dominates the correlation signal.

    :param seed_posteriors: {seed: [N_obs, E] array}.
    :return: (r_matrix [S, S], sorted seeds list).
    """
    seeds = sorted(seed_posteriors.keys())
    means = np.stack([seed_posteriors[s].mean(axis=0) for s in seeds])  # [S, E]
    S = len(seeds)
    r_mat = np.ones((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            r, _ = pearsonr(means[i], means[j])
            r_mat[i, j] = r_mat[j, i] = float(r)
    return r_mat, seeds


def top_k_overlap(
    seed_posteriors: Dict[int, np.ndarray],
    k: int,
) -> Tuple[float, np.ndarray, List[int]]:
    """
    Mean pairwise fraction of top-K edges shared between seeds.

    Top-K overlap directly answers "do seeds discover the same latent
    relationships?" independently of probability scale or distribution shape.

    :param seed_posteriors: {seed: [N_obs, E] array}.
    :param k: Number of top edges to compare.
    :return: (mean_overlap, overlap_matrix [S, S], sorted seeds list).
    """
    seeds = sorted(seed_posteriors.keys())
    means = np.stack([seed_posteriors[s].mean(axis=0) for s in seeds])  # [S, E]
    top_k_sets = [set(np.argsort(means[i])[-k:]) for i in range(len(seeds))]
    S = len(seeds)
    mat = np.ones((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            overlap = len(top_k_sets[i] & top_k_sets[j]) / k
            mat[i, j] = mat[j, i] = float(overlap)
    off_diag = mat[np.triu_indices(S, k=1)]
    return float(off_diag.mean()), mat, seeds


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_matrix_heatmap(
    mat: np.ndarray,
    seeds: List[int],
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
) -> None:
    """Generic S×S annotated heatmap for any pairwise seed metric.

    :param mat: [S, S] symmetric matrix with 1s on the diagonal.
    :param title: Figure title (off-diagonal stats are appended automatically).
    :param cbar_label: Colorbar label.
    :param vmin: Colormap lower bound.
    :param vmax: Colormap upper bound.
    :param fmt: Format spec for cell annotations (e.g. ``".2f"`` or ``".0%"``).
    """
    S = len(seeds)
    fig, ax = plt.subplots(figsize=(max(4, S + 1), max(4, S + 1)))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn")
    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    labels = [f"seed {s}" for s in seeds]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    threshold = vmin + 0.7 * (vmax - vmin)
    for i in range(S):
        for j in range(S):
            color = "white" if mat[i, j] > threshold else "black"
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    off_diag = mat[np.triu_indices(S, k=1)]
    ax.set_title(
        f"{title}\n"
        f"mean={off_diag.mean():.3f}  min={off_diag.min():.3f}  max={off_diag.max():.3f}"
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
    p.add_argument("--rho-thresh", type=float, default=0.95,
                   help="Min rho to target in observation collection (default: 0.7)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <experiment-dir>/cross_seed)")
    p.add_argument("--checkpoint-name", default=None,
                   help="Override checkpoint name for all trials "
                        "(e.g. checkpoint_000005). Default: best per trial.")
    p.add_argument("--conv-type", default=None, choices=["gcn", "gin"],
                   help="Override GNN conv type when loading checkpoint "
                        "(gin for pre-revert checkpoints). Default: from config or 'gcn'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    out_dir = Path(args.out_dir) if args.out_dir else experiment_dir / "cross_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    import ray
    ray.init(
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
        log_to_driver=False,
        num_cpus=1,
        object_store_memory=512 * 1024 * 1024,  # 512 MB — inference only, no workers needed
    )

    # ── Discover trials and map to seeds ──────────────────────────────────────
    trial_dirs = find_trial_dirs(experiment_dir)
    if not trial_dirs:
        raise RuntimeError(f"No trial dirs found in {experiment_dir}")

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
        agent, g2op_env, gym_wrapper = load_agent(trial_dir, ckpt_name, args.env,
                                                   conv_type=args.conv_type)
        gym_wrappers.append(gym_wrapper)

        if n_nodes is None:
            n_nodes = n_nodes_from_env(g2op_env)
            n_line = g2op_env.n_line
            print(f"  Grid: n_nodes={n_nodes}  n_line={n_line}")

        print(f"  Running inference on {len(obs_list)} observations ...")
        probs = compute_posteriors(agent, obs_list)   # [N_obs, E]
        seed_posteriors[seed] = probs
        print(f"  mean={probs.mean():.4f}  std={probs.std():.4f}")

    # ── Cross-seed Pearson r ───────────────────────────────────────────────────
    r_mat, seeds = pairwise_pearson(seed_posteriors)
    off_diag = r_mat[np.triu_indices(len(seeds), k=1)]

    print("\n=== CROSS-SEED PEARSON r (mean posterior per seed) ===")
    header = "         " + "  ".join(f"seed {s}" for s in seeds)
    print(header)
    for i, si in enumerate(seeds):
        row = f"seed {si}:  " + "  ".join(f"{r_mat[i, j]:+.3f}" for j in range(len(seeds)))
        print(row)
    print(f"\nOff-diagonal: mean={off_diag.mean():.3f}  "
          f"min={off_diag.min():.3f}  max={off_diag.max():.3f}")

    # ── Top-K edge overlap ─────────────────────────────────────────────────────
    # Use n_line as K: the number of actual powerline edges in the grid.
    k = n_line if n_line is not None else 20
    mean_overlap, overlap_mat, _ = top_k_overlap(seed_posteriors, k=k)

    print(f"\n=== TOP-{k} EDGE OVERLAP (fraction of top-{k} edges shared) ===")
    print(header)
    for i, si in enumerate(seeds):
        row = f"seed {si}:  " + "  ".join(
            f"{overlap_mat[i, j]:.3f}" for j in range(len(seeds))
        )
        print(row)
    print(f"\nMean pairwise top-{k} overlap: {mean_overlap:.3f}")

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
        overlap_matrix=overlap_mat,
        **{f"posteriors_seed{s}": seed_posteriors[s] for s in seeds},
    )
    print(f"\nSaved: {npz_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_matrix_heatmap(
        r_mat, seeds, out_dir / "pearson_heatmap.png",
        title="Cross-seed posterior consistency — Pearson r (mean posterior per seed)",
        cbar_label="Pearson r",
        vmin=-1.0, vmax=1.0,
    )
    plot_matrix_heatmap(
        overlap_mat, seeds, out_dir / "topk_overlap_heatmap.png",
        title=f"Cross-seed top-{k} edge overlap (fraction of top-{k} edges shared)",
        cbar_label=f"Top-{k} overlap",
        vmin=0.0, vmax=1.0,
        fmt=".2f",
    )
    plot_seed_posteriors(seed_posteriors, n_nodes, n_line,
                         out_dir / "seed_posteriors.png")

    ray.shutdown()


if __name__ == "__main__":
    main()
