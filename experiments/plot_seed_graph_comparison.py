"""Compare latent graph consistency across random seeds for RAPPO-14.

For each condition (dense / sparse), loads the best checkpoint of 5 seeds,
collects diverse observations from the Grid2Op training env, and computes
pairwise Spearman-ρ and Jaccard(p>0.5) between each pair of seeds on each
observation.  Results are visualised as 5×5 heatmaps and a boxplot row.

Usage (from project root):
    PYTHONPATH=src python experiments/plot_seed_graph_comparison.py
    PYTHONPATH=src python experiments/plot_seed_graph_comparison.py \\
        --condition sparse --n-obs 50 --output my_comparison.png
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

logging.basicConfig(level=logging.WARNING)

# ── paths ─────────────────────────────────────────────────────────────────────

BASE = Path("results/experiments/2026_07_07_IEEE14")
EXPERIMENT_DIRS: dict[str, Path] = {
    "dense": BASE / "rappo_mulitseed",
    "sparse": BASE / "rappo_sparse_multiseed",
}

CLUSTER_PREFIX = "/pfs/data6/home/ka/ka_iai/ka_hw6998/dev"
LOCAL_PREFIX = "/home/adrian/Dev"

# Cluster dir name uses the correct spelling; local has a typo.
CLUSTER_SUBDIR = "rappo_multiseed"
LOCAL_SUBDIR = "rappo_mulitseed"

N_SEEDS = 5


# ── path helpers ──────────────────────────────────────────────────────────────

def remap_path(cluster_path: str) -> Path:
    """Translate a cluster checkpoint path to its local equivalent."""
    local = cluster_path.replace(CLUSTER_PREFIX, LOCAL_PREFIX)
    local = local.replace(CLUSTER_SUBDIR, LOCAL_SUBDIR)
    return Path(local)


def find_best_checkpoint(log_path: Path) -> Optional[Path]:
    """Parse a log file and return the local path to the best checkpoint."""
    pattern = re.compile(r"Best checkpoint:\s+(\S+)")
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return remap_path(m.group(1))
    return None


def find_log_files(experiment_dir: Path, prefix: str) -> list[Path]:
    """Return sorted log files matching *prefix*_s{i}.*.log."""
    return sorted((experiment_dir / "out").glob(f"{prefix}_s*.log"))


def collect_checkpoints(condition: str) -> list[Optional[Path]]:
    """Return best checkpoint paths for all seeds of *condition*."""
    exp_dir = EXPERIMENT_DIRS[condition]
    prefix = "rappo14sp" if condition == "sparse" else "rappo14"
    logs = find_log_files(exp_dir, prefix)
    if not logs:
        raise FileNotFoundError(f"No log files found in {exp_dir / 'out'}")
    checkpoints: list[Optional[Path]] = []
    for log in sorted(logs)[:N_SEEDS]:
        ckpt = find_best_checkpoint(log)
        checkpoints.append(ckpt)
        status = str(ckpt) if ckpt else "NOT FOUND"
        print(f"  [{condition}] {log.name} → {status}")
    return checkpoints


# ── Ray / RLlib setup ─────────────────────────────────────────────────────────

def init_ray() -> None:
    import ray
    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False,
             logging_level=logging.ERROR)


def register_models() -> None:
    from ray.rllib.models import ModelCatalog
    from rarl_rllib import RAActorCriticModel
    from rarl_rllib.ppo.gnn_ppo_model import GNNBaselineModel
    ModelCatalog.register_custom_model("ra_actor_critic_model", RAActorCriticModel)
    ModelCatalog.register_custom_model("gnn_model", GNNBaselineModel)


def load_policy(checkpoint: Path):
    """Load an RLlib Policy from a best-checkpoint path (trial/checkpoint_XXXXX)."""
    from ray.rllib import Policy
    from core.constants import RL_POLICY
    policy_path = checkpoint / "policies" / RL_POLICY
    return Policy.from_checkpoint(str(policy_path))


# ── observation collection ────────────────────────────────────────────────────

def build_env(checkpoint: Path, env_name: str):
    """Build a CustomizedGrid2OpEnvironment from params.json of a checkpoint's trial dir."""
    from core.loading import load_config, preprocess_config
    from grid2op_env import CustomizedGrid2OpEnvironment

    trial_dir = checkpoint.parent  # strip checkpoint_XXXXXX
    params = preprocess_config(load_config(trial_dir))
    env_config = params["env_config"]
    env_config["env_name"] = env_name
    return CustomizedGrid2OpEnvironment(env_config)


def collect_observations_simple(
    gym_env,
    n_obs: int,
    n_chronics: int = 10,
) -> list[dict]:
    """Collect observations by stepping the underlying g2op env with do-nothing actions.

    Samples every ``max(1, episode_len // (n_obs // n_chronics))`` steps across
    *n_chronics* chronics for diversity.
    """
    target_per_chronic = max(1, n_obs // n_chronics)
    obs_list: list[dict] = []
    g2op_env = gym_env.env_g2op
    n_avail = len(g2op_env.chronics_handler.subpaths)

    for chronic_idx in range(n_chronics):
        g2op_env.set_id(chronic_idx % n_avail)
        g2op_obs = g2op_env.reset()
        gym_env.update_obs(g2op_obs)
        obs_list.append({k: v.copy() for k, v in gym_env.cur_gym_obs.items()})

        episode_len = g2op_env.chronics_handler.max_timestep()
        stride = max(1, episode_len // target_per_chronic)
        done = False
        step = 0

        while not done:
            g2op_obs, _, done, _ = g2op_env.step(g2op_env.action_space({}))
            gym_env.update_obs(g2op_obs)
            step += 1
            if step % stride == 0:
                obs_list.append({k: v.copy() for k, v in gym_env.cur_gym_obs.items()})
            if len(obs_list) >= (chronic_idx + 1) * target_per_chronic:
                break

        if len(obs_list) >= n_obs:
            break

    return obs_list[:n_obs]


# ── posterior extraction ──────────────────────────────────────────────────────

def get_interaction_probs(policy, obs: dict) -> np.ndarray:
    """Run one forward pass; return interaction probs [E]."""
    with torch.no_grad():
        policy.compute_single_action(obs)
    posterior = policy.model.get_posterior().detach().cpu().numpy()  # [1, E, K]
    return posterior[0, :, :-1].sum(-1)  # [E]


def compute_seed_probs(policy, observations: list[dict]) -> np.ndarray:
    """Return interaction probs [T, E] for all observations."""
    probs = [get_interaction_probs(policy, obs) for obs in observations]
    return np.stack(probs)  # [T, E]


# ── metrics ──────────────────────────────────────────────────────────────────

def spearman_per_obs(probs_i: np.ndarray, probs_j: np.ndarray) -> np.ndarray:
    """Return Spearman-ρ between two seed's probs for each observation. Shape [T]."""
    T = probs_i.shape[0]
    rhos = np.empty(T)
    for t in range(T):
        rho, _ = spearmanr(probs_i[t], probs_j[t])
        rhos[t] = rho
    return rhos


def jaccard_per_obs(
    probs_i: np.ndarray, probs_j: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return Jaccard(p>threshold) between two seeds for each observation. Shape [T]."""
    T = probs_i.shape[0]
    jaccards = np.empty(T)
    for t in range(T):
        ai = probs_i[t] > threshold
        aj = probs_j[t] > threshold
        inter = float((ai & aj).sum())
        union = float((ai | aj).sum())
        jaccards[t] = inter / union if union > 0 else 1.0
    return jaccards


def compute_matrices(
    all_probs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric 5×5 Spearman and Jaccard matrices (diagonal=NaN)."""
    n = len(all_probs)
    spearman_mat = np.full((n, n), np.nan)
    jaccard_mat = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i + 1, n):
            sp = spearman_per_obs(all_probs[i], all_probs[j]).mean()
            jc = jaccard_per_obs(all_probs[i], all_probs[j]).mean()
            spearman_mat[i, j] = sp
            spearman_mat[j, i] = sp
            jaccard_mat[i, j] = jc
            jaccard_mat[j, i] = jc

    return spearman_mat, jaccard_mat


# ── plotting ──────────────────────────────────────────────────────────────────

def draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Draw a 5×5 heatmap with annotated cells."""
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels([f"s{i}" for i in range(mat.shape[1])], fontsize=7)
    ax.set_yticklabels([f"s{i}" for i in range(mat.shape[0])], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)


def draw_boxplot(
    ax: plt.Axes,
    data: np.ndarray,
    label: str,
    title: str,
) -> None:
    """Draw a single boxplot for off-diagonal values."""
    flat = data[~np.isnan(data)].ravel()
    ax.boxplot([flat], labels=[label], patch_artist=True,
               medianprops={"color": "black"},
               boxprops={"facecolor": "steelblue", "alpha": 0.7})
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3)


def make_offdiag(mat: np.ndarray) -> np.ndarray:
    """Return flattened off-diagonal values of a square matrix."""
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]


def plot_results(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
    conditions: list[str],
) -> None:
    """Produce the heatmap grid + boxplot row and save."""
    n_cond = len(conditions)
    # Layout: n_cond rows of heatmaps + 1 boxplot row
    fig = plt.figure(figsize=(14, 5 * n_cond + 3))
    gs = fig.add_gridspec(
        n_cond + 1, 2,
        height_ratios=[5] * n_cond + [3],
        hspace=0.50, wspace=0.40,
    )

    boxplot_entries: list[tuple[str, str, np.ndarray]] = []  # (metric, cond, offdiag)

    for row_idx, cond in enumerate(conditions):
        sp_mat, jc_mat = results[cond]
        ax_sp = fig.add_subplot(gs[row_idx, 0])
        draw_heatmap(ax_sp, sp_mat, f"{cond.capitalize()} — Spearman-ρ")
        ax_jc = fig.add_subplot(gs[row_idx, 1])
        draw_heatmap(ax_jc, jc_mat, f"{cond.capitalize()} — Jaccard (p>0.5)")
        boxplot_entries.append(("Spearman", cond, make_offdiag(sp_mat)))
        boxplot_entries.append(("Jaccard", cond, make_offdiag(jc_mat)))

    # Boxplot row: one box per (metric, condition) combination
    n_boxes = len(boxplot_entries)
    bp_gs = gs[n_cond, :].subgridspec(1, n_boxes, wspace=0.45)
    for bp_idx, (metric, cond, offdiag) in enumerate(boxplot_entries):
        ax_bp = fig.add_subplot(bp_gs[0, bp_idx])
        draw_boxplot(ax_bp, offdiag, cond, f"{metric}\n{cond} (off-diag)")

    fig.suptitle(
        "Latent graph consistency across 5 seeds (RAPPO-14)\n"
        "Diagonal = NaN  |  Both triangles filled (symmetric)",
        fontsize=11,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close(fig)


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_npy_files(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Save spearman/jaccard arrays as .npy files next to the plot."""
    stem = output_path.stem
    out_dir = output_path.parent
    for cond, (sp_mat, jc_mat) in results.items():
        sp_path = out_dir / f"{stem}_spearman_{cond}.npy"
        jc_path = out_dir / f"{stem}_jaccard_{cond}.npy"
        np.save(sp_path, sp_mat)
        np.save(jc_path, jc_mat)
        print(f"Saved: {sp_path}")
        print(f"Saved: {jc_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="l2rpn_case14_sandbox_train",
                   help="Grid2Op environment name")
    p.add_argument("--n-obs", type=int, default=100,
                   help="Number of observations to collect per condition")
    p.add_argument("--output", default="seed_graph_comparison.png",
                   help="Output plot path")
    p.add_argument("--condition", choices=["dense", "sparse", "both"], default="both",
                   help="Which condition(s) to evaluate")
    return p.parse_args()


def process_condition(
    condition: str,
    env_name: str,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect observations, run all seeds, compute pairwise metrics."""
    print(f"\n=== Condition: {condition} ===")

    # 1. Find best checkpoints
    checkpoints = collect_checkpoints(condition)
    valid = [(i, ck) for i, ck in enumerate(checkpoints) if ck is not None]
    if not valid:
        raise RuntimeError(f"No valid checkpoints for condition={condition}")

    # 2. Build env from first valid checkpoint (reused for all seeds)
    first_ck = valid[0][1]
    print(f"Building env from {first_ck.parent.name} ...")
    gym_env = build_env(first_ck, env_name)

    # 3. Collect observations once (shared across seeds)
    print(f"Collecting {n_obs} observations ...")
    observations = collect_observations_simple(gym_env, n_obs)
    print(f"  Collected {len(observations)} observations")

    # 4. Compute posteriors per seed
    all_probs: list[np.ndarray | None] = [None] * N_SEEDS
    for seed_idx, ck in valid:
        print(f"  Seed {seed_idx}: loading {ck.name} ...")
        register_models()
        policy = load_policy(ck)
        probs = compute_seed_probs(policy, observations)  # [T, E]
        all_probs[seed_idx] = probs
        print(f"    shape: {probs.shape}, mean interaction prob: {probs.mean():.4f}")

    # Fill missing seeds with NaN arrays of the same shape
    example = next(p for p in all_probs if p is not None)
    for i in range(N_SEEDS):
        if all_probs[i] is None:
            all_probs[i] = np.full_like(example, np.nan)

    # 5. Compute pairwise similarity matrices
    print("  Computing pairwise metrics ...")
    sp_mat, jc_mat = compute_matrices(all_probs)
    return sp_mat, jc_mat


def main() -> None:
    args = parse_args()
    init_ray()

    conditions: list[str] = (
        ["dense", "sparse"] if args.condition == "both" else [args.condition]
    )
    output = Path(args.output)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cond in conditions:
        sp_mat, jc_mat = process_condition(cond, args.env, args.n_obs)
        results[cond] = (sp_mat, jc_mat)

        sp_off = make_offdiag(sp_mat)
        jc_off = make_offdiag(jc_mat)
        print(f"\n  [{cond}] Spearman-ρ: mean={sp_off.mean():.3f}, "
              f"std={sp_off.std():.3f}, min={sp_off.min():.3f}, max={sp_off.max():.3f}")
        print(f"  [{cond}] Jaccard:    mean={jc_off.mean():.3f}, "
              f"std={jc_off.std():.3f}, min={jc_off.min():.3f}, max={jc_off.max():.3f}")

    plot_results(results, output, conditions)
    save_npy_files(results, output)

    import ray
    ray.shutdown()


if __name__ == "__main__":
    main()
