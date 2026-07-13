"""
Experiment: KL-only encoder training.

Trains the GraphormerNRIEncoder using only the KL regularization loss (no RL
signal) to test two things:
  1. Can the encoder converge to the prior distribution for arbitrary inputs?
  2. How does beta affect convergence speed and stability?

Each beta value gets a fresh encoder initialised from the same random seed.
Snapshots of the mean posterior (over a fixed validation batch) are taken at
evenly-spaced steps including the initial state (step 0, before any training)
and the final state (step n_steps).

Output (saved to experiments/kl_only/ by default):
  - loss_curves.png  : Raw + smoothed KL loss per step for every beta.
  - heatmaps.png     : N×N P(edge exists) heatmaps at each snapshot.
                       Rows = snapshot steps, columns = beta values.
                       Blue rectangles mark actual powerline edges (prior 0.9).

Usage (from project root):
    PYTHONPATH=$(pwd)/src python experiments/train_encoder_kl_only.py
    PYTHONPATH=$(pwd)/src python experiments/train_encoder_kl_only.py \\
        --n-steps 300 --betas 0.1 1.0 5.0 --n-snapshots 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from rarl.nn.encoder.encoder import GraphormerNRIEncoder
from rarl.prior import get_priors, get_prior_tensor
from rarl.graph import fully_connected_edge_index


# ── Grid topology ──────────────────────────────────────────────────────────────
#
# case14 node layout (matches GraphObservationConverter ordering):
#   [line_or(0–19) | line_ex(20–39) | gen(40–44) | load(45–55)]
#
# n_line=20, n_gen=5, n_load=11, n_storage=0  →  N = 2*20 + 5 + 11 = 57
#
# Powerline edges connect or-node i to ex-node i+n_line for each line i.
# This matches exactly what GraphObservationConverter._line_edges produces.

N_LINE: int  = 20
N_GEN: int   = 6
N_LOAD: int  = 11
N_NODES: int = 2 * N_LINE + N_GEN + N_LOAD   # 57
X_DIM: int   = 7    # matches _DEFAULT_NODE_FEATURES in observation_converter.py
K: int       = 2    # edge types: index 0 = "exists", index 1 = "no edge"

# Node type boundary indices (for heatmap annotations)
_BOUNDS = {"or": 0, "ex": N_LINE, "gen": 2 * N_LINE, "load": 2 * N_LINE + N_GEN}


def build_powerline_edges() -> Tensor:
    """
    Directed powerline edge index [2, 2*N_LINE].

    Mirrors GraphObservationConverter._line_edges:
      or-node i  →  ex-node i+N_LINE  (and reverse).
    """
    or_nodes = torch.arange(N_LINE)
    ex_nodes = torch.arange(N_LINE, 2 * N_LINE)
    src = torch.cat([or_nodes, ex_nodes])
    dst = torch.cat([ex_nodes, or_nodes])
    return torch.stack([src, dst])


# ── Setup helpers ──────────────────────────────────────────────────────────────

def build_prior(powerline_edges: Tensor, temperature: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Build the per-edge prior tensor and graph-edge boolean mask for the FC graph.

    :param powerline_edges: Directed powerline edge index [2, E_pl].
    :param temperature: Controls expected latent edge count; 0.5 → 1.5× powerlines.
    :return: (prior_tensor [E, K], graph_mask [E]).
    """
    fc      = fully_connected_edge_index(N_NODES)   # [2, E]
    n_graph = powerline_edges.shape[1]
    n_all   = fc.shape[1]
    n_lat   = n_all - n_graph

    prior_graph, prior_lat = get_priors(
        prob_graph_edges_exist=0.9,
        num_graph_edges=n_graph,
        num_non_graph_edges=n_lat,
        temperature=temperature,
    )
    prior_tensor, graph_mask = get_prior_tensor(
        graph_edges=powerline_edges,
        all_edges=fc,
        prior_for_graph_edges=prior_graph,
        prior_for_non_graph_edges=prior_lat,
        num_edge_types=K,
        return_mask=True,
    )
    return prior_tensor, graph_mask


def build_encoder() -> GraphormerNRIEncoder:
    """
    Small encoder for fast CPU experimentation.

    max_degree covers the max in/out degree of the powerline graph.
    In the or→ex structure each node touches exactly one powerline edge (degree=1),
    but gen/load nodes may appear with higher effective degree via topology edges
    in full training — 10 is a safe upper bound.
    """
    return GraphormerNRIEncoder(
        x_dim=X_DIM,
        hidden_dim=32,
        num_layers=2,
        num_attention_heads=2,
        num_edge_types=K,
        max_degree=10,
        max_path_distance=7,
    )


def expand_edges_for_batch(edges: Tensor, B: int) -> Tensor:
    """Tile a single-graph edge index with per-graph node-index offsets → [2, B*E]."""
    return torch.cat([edges + i * N_NODES for i in range(B)], dim=1)


def make_obs(B: int) -> Tuple[Tensor, Tensor]:
    """Random node features [B*N, X_DIM] and batch vector [B*N]."""
    x     = torch.randn(B * N_NODES, X_DIM)
    batch = torch.arange(B).repeat_interleave(N_NODES)
    return x, batch


# ── Snapshot ───────────────────────────────────────────────────────────────────

def take_snapshot(
    encoder: GraphormerNRIEncoder,
    fixed_x: Tensor,
    fixed_batch: Tensor,
    pl_edges_batched: Tensor,
    B: int,
) -> np.ndarray:
    """
    Evaluate encoder on fixed inputs; return mean P(edge exists) as [N, N].
    Diagonal entries are NaN (no self-loops in the FC graph).
    """
    E = N_NODES * (N_NODES - 1)
    encoder.eval()
    with torch.no_grad():
        logits = encoder(x=fixed_x, powerline_edge_index=pl_edges_batched, batch=fixed_batch)
        post   = F.softmax(logits, dim=-1)                              # [B*E, K]
    encoder.train()

    p_exists = post.view(B, E, K)[:, :, 0].mean(dim=0).cpu().numpy()  # [E]
    fc  = fully_connected_edge_index(N_NODES).numpy()                  # [2, E]
    mat = np.full((N_NODES, N_NODES), np.nan)
    mat[fc[0], fc[1]] = p_exists
    return mat


# ── Loss ──────────────────────────────────────────────────────────────────────

def _factored_kl_loss(
    post: Tensor,
    prior_tensor: Tensor,
    graph_mask: Tensor,
    beta: float,
    eps: float = 1e-7,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    KL loss with equal weight per edge group (no f_graph / f_latent scaling).

    ::

        loss = beta * mean(KL over graph edges) + beta * mean(KL over latent edges)

    This gives powerline edges and latent edges equal gradient weight regardless
    of how many edges are in each group, unlike compute_ra_kl_loss which scales
    each group's contribution by its fraction of total edges.

    :param post: Posterior [B, E, K].
    :param prior_tensor: Prior [E, K] or [B, E, K].
    :param graph_mask: Boolean mask, True for graph edges [B, E].
    :param beta: Weight applied to both KL terms.
    :return: (loss, kl_graph_mean, kl_latent_mean) — all scalar tensors.
    """
    if prior_tensor.dim() == 2:
        prior_tensor = prior_tensor.unsqueeze(0).expand_as(post)

    kl_per_edge = (post * (torch.log(post + eps) - torch.log(prior_tensor + eps))).sum(-1)  # [B, E]

    kl_graph  = kl_per_edge[graph_mask].mean()
    kl_latent = kl_per_edge[~graph_mask].mean()
    loss = beta * (kl_graph + kl_latent)
    return loss, kl_graph.detach(), kl_latent.detach()


# ── Training ───────────────────────────────────────────────────────────────────

def train_for_beta(
    beta: float,
    prior_tensor: Tensor,
    graph_mask: Tensor,
    powerline_edges: Tensor,
    n_steps: int,
    snapshot_steps: List[int],
    B: int,
    lr: float,
    seed: int,
) -> Tuple[List[float], Dict[int, np.ndarray]]:
    """
    Train a fresh encoder with KL loss only and record snapshots.

    Snapshots are taken before each listed step (step 0 = initial state) and
    also after the final training step (step n_steps).

    :return: (kl_loss_per_step, {step_index: posterior_N×N_matrix}).
    """
    torch.manual_seed(seed)
    encoder   = build_encoder()
    optimizer = Adam(encoder.parameters(), lr=lr)

    torch.manual_seed(42)                            # fixed validation batch
    fixed_x, fixed_batch = make_obs(B)
    pl_fixed = expand_edges_for_batch(powerline_edges, B)

    E       = N_NODES * (N_NODES - 1)
    snaps   = set(snapshot_steps)
    losses: List[float]              = []
    snapshots: Dict[int, np.ndarray] = {}

    for step in range(n_steps + 1):                  # +1 so we capture final state
        if step in snaps:
            snapshots[step] = take_snapshot(encoder, fixed_x, fixed_batch, pl_fixed, B)
        if step == n_steps:
            break

        x, batch   = make_obs(B)
        pl_batched = expand_edges_for_batch(powerline_edges, B)

        optimizer.zero_grad()
        logits  = encoder(x=x, powerline_edge_index=pl_batched, batch=batch)   # [B*E, K]
        post    = F.softmax(logits, dim=-1).view(B, E, K)                      # [B, E, K]
        mask_3d = graph_mask.unsqueeze(0).expand(B, -1)                        # [B, E]

        loss, kl_graph, kl_latent = _factored_kl_loss(post, prior_tensor, mask_3d, beta)
        loss.backward()
        optimizer.step()
        losses.append((0.5 * (kl_graph + kl_latent)).item())  # equal-weight mean for logging

    return losses, snapshots


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_losses(all_losses: Dict[float, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for beta, losses in all_losses.items():
        ax.plot(losses, label=f"β = {beta}", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL divergence (unweighted, no β)")
    ax.set_title("KL-only encoder training")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmaps(
    all_snaps: Dict[float, Dict[int, np.ndarray]],
    snapshot_steps: List[int],
    powerline_edges: Tensor,
    out_path: Path,
) -> None:
    betas  = list(all_snaps.keys())
    n_cols = len(betas)
    n_rows = len(snapshot_steps)

    pl = powerline_edges.numpy()   # [2, E_pl]
    pl_src, pl_dst = pl[0], pl[1]

    # Node-type boundary positions and labels
    bounds = list(_BOUNDS.values())          # [0, 20, 40, 45]
    bound_labels = list(_BOUNDS.keys())      # ["or", "ex", "gen", "load"]
    # Tick at each boundary + last node
    tick_pos    = bounds + [N_NODES - 1]
    tick_labels = bound_labels + [str(N_NODES - 1)]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    im_ref = None
    for col, beta in enumerate(betas):
        snaps = all_snaps[beta]
        for row, step in enumerate(snapshot_steps):
            ax  = axes[row][col]
            mat = snaps.get(step, np.full((N_NODES, N_NODES), np.nan))

            im_ref = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlOrRd", origin="upper")

            # Blue rectangle outlines on powerline edge cells
            for src, dst in zip(pl_src, pl_dst):
                ax.add_patch(plt.Rectangle(
                    (dst - 0.5, src - 0.5), 1, 1,
                    fill=False, edgecolor="steelblue", linewidth=0.5,
                ))

            # Node-type boundary lines
            for b in bounds[1:]:            # skip 0
                ax.axhline(b - 0.5, color="black", linewidth=0.4, alpha=0.5)
                ax.axvline(b - 0.5, color="black", linewidth=0.4, alpha=0.5)

            label = "initial" if step == 0 else f"step {step}"
            if row == 0:
                ax.set_title(f"β = {beta}", fontsize=10)

            ax.set_xticks(tick_pos)
            ax.set_yticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
            ax.set_yticklabels(tick_labels, fontsize=6)
            if row == n_rows - 1:
                ax.set_xlabel("dest node  [or|ex|gen|load]", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{label}\nsrc node", fontsize=8)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, fraction=0.015, pad=0.02)
        cbar.set_label("P(edge exists)")

    fig.suptitle(
        "Posterior P(edge exists) — blue outlines = powerline edges (prior p=0.9)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_orex_zoom(
    all_snaps: Dict[float, Dict[int, np.ndarray]],
    snapshot_steps: List[int],
    out_path: Path,
) -> None:
    """
    Zoom into the or→ex submatrix (rows 0:N_LINE, cols N_LINE:2*N_LINE).

    This is where all powerline edges live. Each row is an or-node, each column
    an ex-node. The diagonal (or_i → ex_i) is the actual powerline; off-diagonal
    entries are latent. At convergence the diagonal should be ~0.9 and the rest ~0.008.
    """
    betas  = list(all_snaps.keys())
    n_cols = len(betas)
    n_rows = len(snapshot_steps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows), squeeze=False)

    im_ref = None
    for col, beta in enumerate(betas):
        snaps = all_snaps[beta]
        for row, step in enumerate(snapshot_steps):
            ax  = axes[row][col]
            mat = snaps.get(step, np.full((N_NODES, N_NODES), np.nan))
            sub = mat[:N_LINE, N_LINE:2 * N_LINE]   # [N_LINE, N_LINE]

            im_ref = ax.imshow(sub, vmin=0.0, vmax=1.0, cmap="YlOrRd", origin="upper")

            # Mark the diagonal (actual powerline edges)
            for i in range(N_LINE):
                ax.add_patch(plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="steelblue", linewidth=0.8,
                ))

            label = "initial" if step == 0 else f"step {step}"
            if row == 0:
                ax.set_title(f"β = {beta}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{label}\nor-node (src)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("ex-node (dest)", fontsize=8)

            ticks = [0, 5, 10, 15, 19]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=6)
            ax.set_yticklabels(ticks, fontsize=6)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, fraction=0.015, pad=0.02)
        cbar.set_label("P(edge exists)")

    fig.suptitle(
        "or→ex submatrix zoom — diagonal = powerline edges (should reach p=0.9)\n"
        "off-diagonal = latent or→ex edges (should reach p=0.008)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-steps",     type=int,   default=500,
                   help="Training steps per beta (default: 500)")
    p.add_argument("--betas",       type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 5.0],
                   help="Beta values to compare (default: 0.1 0.5 1.0 5.0)")
    p.add_argument("--n-snapshots", type=int,   default=5,
                   help="Snapshot count including initial & final (default: 5)")
    p.add_argument("--batch-size",  type=int,   default=8,
                   help="Observations per training step (default: 8)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Adam learning rate (default: 1e-3)")
    p.add_argument("--seed",        type=int,   default=0,
                   help="Seed for encoder initialisation (default: 0)")
    p.add_argument("--out-dir",     type=str,   default="experiments/kl_only",
                   help="Output directory (default: experiments/kl_only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    powerline_edges          = build_powerline_edges()
    prior_tensor, graph_mask = build_prior(powerline_edges)

    # Snapshot steps: 0 (initial) … n_steps (final), evenly spaced
    snapshot_steps: List[int] = sorted(set(
        np.linspace(0, args.n_steps, args.n_snapshots, dtype=int).tolist()
    ))

    n_graph = powerline_edges.shape[1]
    n_total = N_NODES * (N_NODES - 1)
    p_lat   = float(prior_tensor[~graph_mask, 0].mean())
    print(f"Graph: {n_graph} directed powerline edges  |  FC total: {n_total}")
    print(f"Prior: graph p=0.90  latent p={p_lat:.3f}")
    print(f"Steps: {args.n_steps}  |  Betas: {args.betas}")
    print(f"Snapshots at steps: {snapshot_steps}\n")

    all_losses:  Dict[float, List[float]]             = {}
    all_snapshots: Dict[float, Dict[int, np.ndarray]] = {}

    for beta in args.betas:
        print(f"Training beta={beta} ...", end=" ", flush=True)
        losses, snaps = train_for_beta(
            beta=beta,
            prior_tensor=prior_tensor,
            graph_mask=graph_mask,
            powerline_edges=powerline_edges,
            n_steps=args.n_steps,
            snapshot_steps=snapshot_steps,
            B=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        all_losses[beta]     = losses
        all_snapshots[beta]  = snaps
        print(f"final loss = {losses[-1]:.5f}")

    plot_losses(all_losses, out_dir / "loss_curves.png")
    plot_heatmaps(all_snapshots, snapshot_steps, powerline_edges, out_dir / "heatmaps.png")
    plot_orex_zoom(all_snapshots, snapshot_steps, out_dir / "orex_zoom.png")


if __name__ == "__main__":
    main()
