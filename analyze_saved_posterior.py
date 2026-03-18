"""
Standalone script to load a saved posterior_distribution.pkl and run all MetricVisualizers on it.

Usage:
    python analyze_saved_posterior.py \
        --pkl  results/experiments/2601_compute_metrics/metrics_agg/posterior_distribution.pkl \
        --env  l2rpn_case14_sandbox \
        --out  results/experiments/2601_compute_metrics/metrics_agg_standalone
"""

import argparse
import logging
import traceback
from pathlib import Path

import grid2op
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import torch
from tabulate import tabulate

from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.experiments.analyze_latent_graphs.Metrics import (
    MetricVisualizer,
    DegreeDistributionVisualizer,
    ClusteringCoefficientVisualizer,
    InnerTreeNodeProbabilityVisualizer,
    PosteriorDistributionVisualizer,
    KLDivergenceVisualizer,
    SymmetryMetricVisualizer,
    EdgeNodeTypeVisualizer,
    BetweennessVisualizer,
    AllPairsShortestPathVisualizer,
)
from src.nri.utils import fully_connected_edge_index, get_priors, get_prior_tensor
from src.visualization import get_node_styles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_node_mask(posterior: np.ndarray, fc_edge_index: np.ndarray,
                      num_nodes: int, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean mask [num_nodes] for the biggest connected component."""
    edge_probs = posterior[:, 0:-1].sum(axis=1)
    edges_above = fc_edge_index[:, edge_probs >= threshold]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_above.T.tolist())
    largest_cc = max(nx.connected_components(G), key=len)

    mask = np.zeros(num_nodes, dtype=bool)
    mask[list(largest_cc)] = True
    return mask


def sample_n_graphs(posterior: np.ndarray, fc_edge_index: np.ndarray,
                    n: int = 100) -> list:
    """Sample n edge-index graphs from the posterior Bernoulli distribution."""
    E, _ = posterior.shape
    graphs = []
    for _ in range(n):
        edge_exists = np.random.random(E) < posterior[:, 0]
        graphs.append(fc_edge_index[:, edge_exists])
    return graphs


def build_prior(powergrid_graph: np.ndarray, fc_edge_index: np.ndarray,
                prior_for_graph_edges: float = 0.9,
                temperature: float = 0.5,
                num_edge_types: int = 2) -> np.ndarray:
    """Construct the NRI prior distribution matching the powergrid topology."""
    E = powergrid_graph.shape[1]
    p_graph, p_non = get_priors(
        prob_graph_edges_exist=prior_for_graph_edges,
        num_graph_edges=E,
        num_non_graph_edges=fc_edge_index.shape[1] - E,
        temperature=temperature,
    )
    prior = get_prior_tensor(
        graph_edges=torch.from_numpy(powergrid_graph),
        all_edges=torch.from_numpy(fc_edge_index),
        prior_for_graph_edges=p_graph,
        prior_for_non_graph_edges=p_non,
        num_edge_types=num_edge_types,
    ).detach().cpu().numpy()
    return prior


# ──────────────────────────────────────────────────────────────────────────────
# Summary table (mirrors PosteriorMetrics.print_summary_table logic)
# ──────────────────────────────────────────────────────────────────────────────

_FOUR_VARIANT_KEYS = ("latent_full", "latent_subgraph", "powergrid_full", "powergrid_subgraph")
_AUX_KEYS = {"node_mask", "powergrid_graph", "posterior"}
_BETWEENNESS_KEY_MAP = {
    "latent_full": "betweenness_centrality_latent_full",
    "latent_subgraph": "betweenness_centrality_latent_sub",
    "powergrid_full": "betweenness_centrality_powergrid_full",
    "powergrid_subgraph": "betweenness_centrality_powergrid_sub",
}


def _scalar_stats(arr: np.ndarray, node_mask=None, subgraph_only=False):
    try:
        v = arr.ravel().astype(float)
        if subgraph_only and node_mask is not None:
            v = v[node_mask.ravel()]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return dict(mean=float(np.mean(v)), std=float(np.std(v)),
                    min=float(np.min(v)), max=float(np.max(v)),
                    median=float(np.median(v)))
    except Exception:
        return None


def _extract_four_variant_stats(data):
    if not isinstance(data, dict):
        return None
    if all(k in data for k in _FOUR_VARIANT_KEYS):
        key_map = {v: v for v in _FOUR_VARIANT_KEYS}
    elif all(v in data for v in _BETWEENNESS_KEY_MAP.values()):
        key_map = _BETWEENNESS_KEY_MAP
    else:
        return None
    node_mask = data.get("node_mask")
    result = {}
    for variant, data_key in key_map.items():
        arr = data.get(data_key)
        if arr is None or not isinstance(arr, np.ndarray):
            result[variant] = None
            continue
        is_sub = "subgraph" in variant or "sub" in variant
        result[variant] = _scalar_stats(arr, node_mask=node_mask, subgraph_only=is_sub)
    return result


def _extract_scalars(data):
    try:
        arrays = []
        if isinstance(data, np.ndarray):
            arrays = [data.ravel()]
        elif isinstance(data, (tuple, list)):
            for item in data:
                if isinstance(item, np.ndarray):
                    arrays.append(item.ravel())
        elif isinstance(data, dict):
            for key, val in data.items():
                if key in _AUX_KEYS:
                    continue
                if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
                    arrays.append(val.ravel())
        if not arrays:
            return None
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return dict(mean=float(np.mean(values)), std=float(np.std(values)),
                    min=float(np.min(values)), max=float(np.max(values)),
                    median=float(np.median(values)))
    except Exception:
        return None


def _stats(arr: np.ndarray) -> dict:
    """Compute basic stats over a flat finite array."""
    v = arr.ravel().astype(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return dict(mean=np.mean(v), std=np.std(v), min=np.min(v), max=np.max(v), median=np.median(v))


def _fmt(s: dict) -> list:
    if not s:
        return ["N/A"] * 5
    return [f"{s['mean']:.4f}", f"{s['std']:.4f}", f"{s['min']:.4f}", f"{s['max']:.4f}", f"{s['median']:.4f}"]


def print_distributions_table(metric_names: list, agg_data: dict):
    """
    Print one combined table where every row corresponds to a (metric, distribution)
    pair and columns show mean / std / min / max / median.

    Distributions handled
    ---------------------
    * Four-variant dict metrics  → Latent-Full, Latent-BCC (biggest connected
                                   component), Powergrid-Full, Powergrid-BCC.
      Subgraph variants are filtered to node_mask before computing stats.
    * Posterior distribution     → p(edge exists) and p(no edge)
    * Symmetry analysis          → single scalar score
    * Any remaining ndarray      → flattened
    """
    headers = ["Metric", "Distribution", "Mean", "Std", "Min", "Max", "Median"]
    rows = []

    for metric_name in metric_names:
        data = agg_data.get(metric_name)
        if data is None:
            rows.append([metric_name, "–", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue

        # ── Four-variant dict ──────────────────────────────────────────────
        if isinstance(data, dict):
            node_mask = data.get("node_mask")

            if all(k in data for k in _FOUR_VARIANT_KEYS):
                key_map = {
                    "Latent – Full":      ("latent_full",        False),
                    "Latent – BCC":       ("latent_subgraph",    True),
                    "Powergrid – Full":   ("powergrid_full",     False),
                    "Powergrid – BCC":    ("powergrid_subgraph", True),
                }
            elif all(v in data for v in _BETWEENNESS_KEY_MAP.values()):
                key_map = {
                    "Latent – Full":    (_BETWEENNESS_KEY_MAP["latent_full"],        False),
                    "Latent – BCC":     (_BETWEENNESS_KEY_MAP["latent_subgraph"],    True),
                    "Powergrid – Full": (_BETWEENNESS_KEY_MAP["powergrid_full"],     False),
                    "Powergrid – BCC":  (_BETWEENNESS_KEY_MAP["powergrid_subgraph"], True),
                }
            else:
                key_map = {}

            if key_map:
                for dist_label, (data_key, is_sub) in key_map.items():
                    arr = data.get(data_key)
                    if arr is None or not isinstance(arr, np.ndarray):
                        rows.append([metric_name, dist_label, "N/A", "N/A", "N/A", "N/A", "N/A"])
                        continue
                    v = arr.ravel().astype(float)
                    # Only apply node_mask when array is per-node (same length as mask).
                    # Path-length arrays are already pre-filtered and must NOT be masked.
                    if is_sub and node_mask is not None and len(v) == len(node_mask):
                        v = v[node_mask.ravel()]
                    rows.append([metric_name, dist_label] + _fmt(_stats(v)))
                continue

            # Other dict – report each numeric array key separately
            for k, v in data.items():
                if k in _AUX_KEYS or not isinstance(v, np.ndarray):
                    continue
                if not np.issubdtype(v.dtype, np.number):
                    continue
                rows.append([metric_name, k] + _fmt(_stats(v)))
            continue

        # ── Tuple (array, edge_index): PosteriorDistribution (2D) or KL (1D) ──
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray):
            first = data[0]
            if first.ndim == 2:
                # PosteriorDistributionVisualizer: shape (E, K)
                rows.append([metric_name, "p(edge exists)"] + _fmt(_stats(first[:, 0])))
                rows.append([metric_name, "p(no edge)"]     + _fmt(_stats(first[:, -1])))
            else:
                # KLDivergenceVisualizer or similar: 1-D values per edge
                rows.append([metric_name, "values"] + _fmt(_stats(first)))
            continue

        # ── Plain scalar (e.g. SymmetryAnalysis) ──────────────────────────
        if isinstance(data, (float, int, np.floating, np.integer)):
            rows.append([metric_name, "score", f"{data:.4f}", "–", "–", "–", "–"])
            continue

        # ── Plain ndarray ──────────────────────────────────────────────────
        if isinstance(data, np.ndarray):
            rows.append([metric_name, "values"] + _fmt(_stats(data)))
            continue

        rows.append([metric_name, "–", "N/A", "N/A", "N/A", "N/A", "N/A"])

    print("\n── Distribution Statistics ──")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def print_summary_table(metric_names, agg_data):
    four_variant_rows = {v: [] for v in _FOUR_VARIANT_KEYS}
    other_rows = []
    headers = ["Metric", "Mean", "Std", "Min", "Max", "Median"]

    for metric_name in metric_names:
        data = agg_data.get(metric_name)
        if data is None:
            na = [metric_name, "N/A", "N/A", "N/A", "N/A", "N/A"]
            for v in _FOUR_VARIANT_KEYS:
                four_variant_rows[v].append(na)
            other_rows.append(na)
            continue

        variant_stats = _extract_four_variant_stats(data)
        if variant_stats is not None:
            for variant, stats in variant_stats.items():
                row = [metric_name] + (
                    ["N/A"] * 5 if stats is None else [
                        f"{stats['mean']:.4f}", f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}", f"{stats['max']:.4f}",
                        f"{stats['median']:.4f}",
                    ]
                )
                four_variant_rows[variant].append(row)
        else:
            scalars = _extract_scalars(data)
            row = [metric_name] + (
                ["N/A"] * 5 if scalars is None else [
                    f"{scalars['mean']:.4f}", f"{scalars['std']:.4f}",
                    f"{scalars['min']:.4f}", f"{scalars['max']:.4f}",
                    f"{scalars['median']:.4f}",
                ]
            )
            other_rows.append(row)

    variant_titles = {
        "latent_full":        "Latent Graph – Full",
        "latent_subgraph":    "Latent Graph – Biggest Connected Component",
        "powergrid_full":     "Power Grid – Full (Baseline)",
        "powergrid_subgraph": "Power Grid – Biggest Connected Component (Baseline)",
    }
    for variant, title in variant_titles.items():
        rows = four_variant_rows[variant]
        if rows:
            print(f"\n── {title} ──")
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    if other_rows:
        print("\n── Other Metrics ──")
        print(tabulate(other_rows, headers=headers, tablefmt="rounded_outline"))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute & visualize metrics from a saved posterior pickle.")
    parser.add_argument("--pkl", type=Path,
                        default=Path("results/experiments/2601_compute_metrics/metrics_agg/posterior_distribution.pkl"),
                        help="Path to the posterior_distribution.pkl file.")
    parser.add_argument("--env", type=str, default="l2rpn_case14_sandbox",
                        help="grid2op environment name (needed for node styles).")
    parser.add_argument("--out", type=Path,
                        default=Path("results/experiments/2601_compute_metrics/metrics_agg_standalone"),
                        help="Directory where figures and aggregated pkl files are saved.")
    parser.add_argument("--prior_prob", type=float, default=0.9,
                        help="Prior probability for power-grid edges to exist.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature used when building the prior.")
    parser.add_argument("--num_edge_types", type=int, default=2,
                        help="Number of edge types in the NRI model.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of posterior graph samples to draw.")
    args = parser.parse_args()

    # ── Load posterior & power-grid edge index ────────────────────────────────
    logger.info(f"Loading {args.pkl} …")
    with open(args.pkl, "rb") as f:
        posterior, _ = pickle.load(f)

    logger.info(f"  posterior shape : {posterior.shape}")

    # ── Environment / node styles ─────────────────────────────────────────────
    logger.info(f"Loading grid2op env '{args.env}' for node styles …")
    env = grid2op.make(args.env)
    node_styles = get_node_styles(env, BusConnectivityGraphObsSpace)
    num_nodes = len(node_styles)
    obs = env.reset()
    powergrid_graph = BusConnectivityGraphObsSpace(env.observation_space).to_gym(obs)[EDGE_INDEX]
    logger.info(f"  num_nodes = {num_nodes}")

    # ── Derived quantities ────────────────────────────────────────────────────
    fc_edge_index = fully_connected_edge_index(num_nodes).detach().cpu().numpy()
    node_mask = compute_node_mask(posterior, fc_edge_index, num_nodes)
    samples = sample_n_graphs(posterior, fc_edge_index, n=args.num_samples)
    prior = build_prior(powergrid_graph, fc_edge_index,
                        prior_for_graph_edges=args.prior_prob,
                        temperature=args.temperature,
                        num_edge_types=args.num_edge_types)


    # ── Metric visualizers ────────────────────────────────────────────────────
    metrics: dict[str, MetricVisualizer] = {
        "Node Degree":                    DegreeDistributionVisualizer(node_styles=node_styles),
        "Clustering Coefficient":         ClusteringCoefficientVisualizer(node_styles=node_styles),
        "Inner Tree Node Probability":    InnerTreeNodeProbabilityVisualizer(node_styles=node_styles),
        "Posterior Distribution":         PosteriorDistributionVisualizer(node_styles=node_styles),
        "KL Divergence":                  KLDivergenceVisualizer(node_styles=node_styles),
        "Path Length":                    PathLengthVisualizer(node_styles=node_styles),
        "All-Pairs Shortest Path":        AllPairsShortestPathVisualizer(node_styles=node_styles),
        "Symmetry Analysis":              SymmetryMetricVisualizer(node_styles=node_styles),
        "Connected Node Types":           EdgeNodeTypeVisualizer(node_styles=node_styles),
        "Betweenness Centrality":         BetweennessVisualizer(node_styles=node_styles),
    }

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common kwargs passed to every __call__
    call_kwargs = dict(
        posterior=posterior,
        prior=prior,
        samples=samples,
        powergrid_graph=powergrid_graph,
        edge_index_fully_connected=fc_edge_index,
        node_mask=node_mask,
        observation=obs,
        show_figure=False,
    )

    agg_data: dict = {}

    for metric_name, vis in metrics.items():
        slug = metric_name.replace(" ", "_").lower()
        logger.info(f"Computing '{metric_name}' …")
        try:
            figure, data = vis(**call_kwargs)

            # Save the figure for this single step
            if figure is not None:
                fig_path = out_dir / f"{slug}_single.svg"
                figure.savefig(fig_path)
                plt.close(figure)
                logger.info(f"  → figure saved to {fig_path}")

            # Save the raw data
            pkl_path = out_dir / f"{slug}.pkl"
            vis.save_data(data, pkl_path)
            logger.info(f"  → data saved to {pkl_path}")

            agg_data[metric_name] = data

        except Exception:
            logger.error(f"Error in '{metric_name}':")
            traceback.print_exc()

    # ── Visualize (aggregated = single step here, since we have one posterior) ─
    logger.info("\nGenerating visualizations …")
    for metric_name, vis in metrics.items():
        data = agg_data.get(metric_name)
        if data is None:
            continue
        slug = metric_name.replace(" ", "_").lower()
        try:
            figure = vis._visualize(computation_result=data, aggregated=True, show_figure=False)
            if figure is not None:
                fig_path = out_dir / f"{slug}.svg"
                figure.savefig(fig_path)
                plt.close(figure)
                logger.info(f"  Saved {fig_path}")
        except Exception:
            logger.error(f"Visualization error for '{metric_name}':")
            traceback.print_exc()

    # ── Summary tables ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("METRIC SUMMARY")
    print("=" * 70)
    print_summary_table(list(metrics.keys()), agg_data)

    print("\n" + "=" * 70)
    print("PER-DISTRIBUTION STATISTICS")
    print("=" * 70)
    print_distributions_table(list(metrics.keys()), agg_data)


if __name__ == "__main__":
    main()
