import abc
import pickle
from enum import Enum
from pathlib import Path
from typing import List, Tuple, TypeVar, Generic, Dict, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
from grid2op.Observation import BaseObservation
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.visualization import visualize_graph, PlottingArgs
from src.visualization.utils import NodeStyle

T = TypeVar("T")  # generic type for computed metric

# ---------------------------------------------------------------------------
# Global colour palette for the three graph-histogram variants
# ---------------------------------------------------------------------------
COLOR_POWERGRID = "steelblue"
COLOR_LATENT_FULL = "darkorange"
COLOR_LATENT_SUB = "seagreen"

# Ordered (color, title) pairs used by _plot_three_histograms
_HIST_SPECS = [
    (COLOR_POWERGRID,   "Powergrid (Full)"),
    (COLOR_LATENT_FULL, "Latent Graph (Full)"),
    (COLOR_LATENT_SUB,  "Latent Subgraph"),
]


def _compute_shared_bins(
    arrays: List[npt.NDArray],
    n_bins: int = 20,
    integer_bins: bool = False,
) -> npt.NDArray:
    """Return a common bin array computed from all non-empty *arrays*.

    Parameters
    ----------
    arrays:
        List of 1-D data arrays (NaN values are stripped automatically).
    n_bins:
        Number of bins when ``integer_bins=False``.
    integer_bins:
        When *True* bins are integer-centred (useful for path-length data).
    """
    clean = [a[~np.isnan(a)] for a in arrays if len(a) > 0]
    clean = [a for a in clean if len(a) > 0]
    if not clean:
        return np.linspace(0, 1, n_bins + 1)

    all_data = np.concatenate(clean)
    lo, hi = float(all_data.min()), float(all_data.max())

    if integer_bins:
        max_d = int(np.ceil(hi))
        return np.arange(0.5, max_d + 1.5, 1.0)

    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, n_bins + 1)


def _plot_three_histograms(
    fig: Figure,
    axs,
    data_triple: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    xlabel: str,
    suptitle: str,
    aggregated: bool = False,
    integer_bins: bool = False,
    n_bins: int = 20,
) -> None:
    """Render three comparable histograms onto *axs* (length-3 sequence of Axes).

    All panels share the same bin edges and the same x/y axis limits.
    A vertical dashed red line marks the mean and a light horizontal grid
    is drawn on every panel.

    Parameters
    ----------
    fig:          Parent figure (used for suptitle).
    axs:          Sequence of three ``Axes`` objects.
    data_triple:  ``(powergrid_full, latent_full, latent_subgraph)`` 1-D arrays.
                  NaN values are stripped before plotting.
    xlabel:       Label for the x-axis of every panel.
    suptitle:     Figure-level title; "(Aggregated)" is appended when *aggregated*.
    integer_bins: Forwarded to :func:`_compute_shared_bins`.
    n_bins:       Forwarded to :func:`_compute_shared_bins`.
    """
    pg, lf, ls = data_triple
    # Strip NaN
    pg = pg[~np.isnan(pg)] if len(pg) > 0 else pg
    lf = lf[~np.isnan(lf)] if len(lf) > 0 else lf
    ls = ls[~np.isnan(ls)] if len(ls) > 0 else ls

    bins = _compute_shared_bins([pg, lf, ls], n_bins=n_bins, integer_bins=integer_bins)

    for ax, data, (color, title) in zip(axs, [pg, lf, ls], _HIST_SPECS):
        if len(data) > 0:
            ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.75, density=True)
            mean_val = float(np.mean(data))
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.8,
                       label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.grid(axis='y', alpha=0.3)
        if integer_bins:
            ax.xaxis.set_major_locator(MultipleLocator(1))

    fig.suptitle(f'{suptitle}', fontsize=14, fontweight='bold')


class MetricVisualizer(abc.ABC, Generic[T]):
    """
    Abstract base class for visualizing and comparing metrics between posterior graph samples and the powergrid graph.
    """

    def __init__(self, node_styles: List[NodeStyle]):
        self.metrics_history: List[T] = []
        self.node_styles = node_styles

    @abc.abstractmethod
    def _compute(self,
                 posterior: npt.NDArray,
                 prior: npt.NDArray,
                 samples: npt.NDArray,
                 powergrid_graph: npt.NDArray,
                 edge_index_fully_connected: npt.NDArray,
                 node_mask: npt.NDArray,
                 observation: BaseObservation) -> T:
        """
        Compute the metric of type T and return it.
        :param posterior: the posterior edge probabilities of shape [num_nodes, num_nodes]
        :param prior: the prior edge probabilities of shape [num_nodes, num_nodes]
        :param samples: samples from the posterior, (list of edge indices of shape [2, num_edges] each)
        :param powergrid_graph: powergrid graph edge indices of shape [2, num_edges]
        :param edge_index_fully_connected: fully connected graph edge indices of shape [2, num_edges]
        :param node_mask: mask indicating which nodes are part of the biggest connected component [num_nodes]
        :param observation: the observation at which the posterior was computed
        :return: the metric of type T
        """
        pass

    @abc.abstractmethod
    def _visualize(self, computation_result: T, aggregated: bool = False, show_figure: bool = False) -> Figure:
        """
        Visualizes the computed metric.
        :param computation_result: the computed metric of type T
        :param aggregated: whether or not the metric is aggregated over multiple computations or a single one
        :param show_figure: Show the figure interactively
        :return: the figure
        """
        pass

    def __call__(
            self,
            posterior: npt.NDArray,
            prior: npt.NDArray,
            samples: npt.NDArray,
            powergrid_graph: npt.NDArray,
            edge_index_fully_connected: npt.NDArray,
            node_mask: npt.NDArray,
            observation: BaseObservation,
            visualize: bool = False) -> Tuple[Figure, T]:
        """
        Computes and visualizes the respective metric or a comparison.

        :param posterior: the posterior edge probabilities [num_nodes, num_nodes]
        :param prior: the prior edge probabilities [num_nodes, num_nodes]
        :param samples: the sampled graphs edge indices from the posterior (list of [2, num_edges])
        :param powergrid_graph: the powergrid graph edge index [2, num_edges]
        :param edge_index_fully_connected: the fully connected graph edge index [2, num_edges]
        :param node_mask: boolean mask indicating which nodes are part of the biggest connected component [num_nodes]
        :param observation: the observation at which the posterior was computed
        :param visualize: whether to plot and show the figure
        :return: the matplotlib figure containing the visualization and the computed data
        """
        data = self._compute(
            posterior=posterior,
            prior=prior,
            samples=samples,
            powergrid_graph=powergrid_graph,
            edge_index_fully_connected=edge_index_fully_connected,
            node_mask=node_mask,
            observation=observation
        )
        self.metrics_history.append(data)
        fig = None
        if visualize:
            fig = self._visualize(
                computation_result=data,
                aggregated=False,
                show_figure=True
            )

        return fig, data

    def summarize(self, show_figure=False) -> Tuple[Figure, T]:
        """
        Summarizes the collected metrics over time. In order to access the history, use self.metrics_history.
        :param show_figure: whether to show the figure interactively
        :return: the matplotlib figure containing the summary visualization, and the summarized data of type T
        """
        aggregated_data = self._summarize_data()
        fig = self._visualize(computation_result=aggregated_data, aggregated=True, show_figure=show_figure)
        return fig, aggregated_data

    @abc.abstractmethod
    def _summarize_data(self) -> T:
        """
        Summarizes the collected metrics over time into a single metric of type T.
        :return: the summarized metric of type T
        """
        pass

    def save_data(self, data: T, path: Path):
        """
        Saves the summarized data to a file.
        :param data: the data to save
        :param path: the path to save the data to
        """
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, path: Path) -> T:
        """
        Loads summarized data from a file.
        :param path: the path to load the data from
        :return: the loaded data of type T
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

class DegreeDistributionVisualizer(MetricVisualizer[Dict[str, npt.NDArray]]):
    """Visualizes per-node degree distributions for latent and powergrid graphs (full and subgraph)."""
    
    def _summarize_data(self) -> Dict[str, npt.NDArray]:
        # Average degrees across all timesteps
        latent_full = np.mean([m['latent_full'] for m in self.metrics_history], axis=0)
        latent_subgraph = np.mean([m['latent_subgraph'] for m in self.metrics_history], axis=0)
        # Powergrid is a discrete graph — degrees are integers; round to nearest int
        # after averaging (the graph is static, so this is a no-op in practice)
        powergrid_full = np.round(
            np.mean([m['powergrid_full'] for m in self.metrics_history], axis=0)
        ).astype(int)

        return {
            'latent_full': latent_full,
            'latent_subgraph': latent_subgraph,
            'powergrid_full': powergrid_full,
        }

    def _compute(self, posterior, powergrid_graph, edge_index_fully_connected, node_mask, **kwargs) -> Dict[str, npt.NDArray]:
        num_nodes = len(node_mask)

        # Compute latent graph degrees (full graph - all nodes)
        latent_full_degrees = np.zeros(num_nodes, dtype=float)
        for i, distribution in enumerate(posterior):
            src, dst = edge_index_fully_connected[:, i]
            prob_exists = distribution[0:-1].sum().item()  # probability that the edge exists
            latent_full_degrees[src] += prob_exists
            latent_full_degrees[dst] += prob_exists

        # Compute latent graph degrees (subgraph - only nodes in mask)
        latent_subgraph_degrees = np.zeros(num_nodes, dtype=float)
        for i, distribution in enumerate(posterior):
            src, dst = edge_index_fully_connected[:, i]
            if node_mask[src] and node_mask[dst]:
                prob_exists = distribution[0:-1].sum().item()
                latent_subgraph_degrees[src] += prob_exists
                latent_subgraph_degrees[dst] += prob_exists
        latent_subgraph_degrees[~node_mask] = np.nan  # Set degrees of nodes outside the mask to NaN for clarity

        # Compute powergrid degrees (full graph) — discrete graph, so integer degrees
        powergrid_full_degrees = np.zeros(num_nodes, dtype=int)
        for src, dst in powergrid_graph.T:
            powergrid_full_degrees[src] += 1
            powergrid_full_degrees[dst] += 1

        return {
            'latent_full': latent_full_degrees,
            'latent_subgraph': latent_subgraph_degrees,
            'powergrid_full': powergrid_full_degrees,
        }

    def _visualize(self, computation_result: Dict[str, npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
        _plot_three_histograms(
            fig, axs,
            data_triple=(
                computation_result['powergrid_full'],
                computation_result['latent_full'],
                computation_result['latent_subgraph'],
            ),
            xlabel='Degree',
            suptitle='Degree Distribution',
            aggregated=aggregated,
        )
        plt.tight_layout()
        if show_figure:
            plt.show()
        return fig


class ClusteringCoefficientVisualizer(MetricVisualizer[Dict[str, npt.NDArray]]):
    """Visualizes per-node clustering coefficient distributions for latent and powergrid graphs (full and subgraph)."""

    def _summarize_data(self) -> Dict[str, npt.NDArray]:
        # Average clustering coefficients across all timesteps
        latent_full = np.mean([m['latent_full'] for m in self.metrics_history], axis=0)
        latent_subgraph = np.mean([m['latent_subgraph'] for m in self.metrics_history], axis=0)
        powergrid_full = np.mean([m['powergrid_full'] for m in self.metrics_history], axis=0)

        return {
            'latent_full': latent_full,
            'latent_subgraph': latent_subgraph,
            'powergrid_full': powergrid_full,
        }

    def _compute(self, posterior, powergrid_graph, edge_index_fully_connected, node_mask, samples, **kwargs) -> Dict[str, npt.NDArray]:
        """
        Compute per-node clustering coefficients for all four graph variants.
        :param posterior: the posterior distribution [E, K]
        :param powergrid_graph: the powergrid graph edge index [2, E_grid]
        :param edge_index_fully_connected: fully connected edge index [2, E_all]
        :param node_mask: mask indicating which nodes are part of the biggest connected component [num_nodes]
        :param samples: list of sampled edge indices from posterior
        :return: dict with per-node clustering coefficients for all variants
        """
        num_nodes = len(node_mask)

        # Compute average clustering coefficients for latent graph (full) from samples
        latent_full_coeffs = np.zeros(num_nodes, dtype=float)
        for sample_edges in samples:
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(sample_edges.T.astype(int))
            clustering = nx.clustering(G)
            for node_idx in range(num_nodes):
                latent_full_coeffs[node_idx] += clustering[node_idx]
        latent_full_coeffs /= max(len(samples), 1)

        # Compute average clustering coefficients for latent subgraph from samples
        latent_subgraph_coeffs = np.zeros(num_nodes, dtype=float)
        for sample_edges in samples:
            # Filter edges to only those within the mask
            subgraph_edges = []
            for src, dst in sample_edges.T:
                src, dst = int(src), int(dst)
                if node_mask[src] and node_mask[dst]:
                    subgraph_edges.append([src, dst])
            
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            if subgraph_edges:
                G.add_edges_from(subgraph_edges)
            clustering = nx.clustering(G)
            for node_idx in range(num_nodes):
                if node_mask[node_idx]:
                    latent_subgraph_coeffs[node_idx] += clustering[node_idx]
        latent_subgraph_coeffs /= max(len(samples), 1)
        latent_subgraph_coeffs[~node_mask] = np.nan  # Set coefficients of nodes outside the mask to NaN for clarity

        # Compute clustering coefficients for full powergrid graph
        G_full = nx.Graph()
        G_full.add_nodes_from(range(num_nodes))
        G_full.add_edges_from(powergrid_graph.T.astype(int))
        clustering_full = nx.clustering(G_full)
        powergrid_full_coeffs = np.array([clustering_full[i] for i in range(num_nodes)])

        return {
            'latent_full': latent_full_coeffs,
            'latent_subgraph': latent_subgraph_coeffs,
            'powergrid_full': powergrid_full_coeffs,
        }

    def _visualize(self, computation_result: Dict[str, npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
        _plot_three_histograms(
            fig, axs,
            data_triple=(
                computation_result['powergrid_full'],
                computation_result['latent_full'],
                computation_result['latent_subgraph'],
            ),
            xlabel='Clustering Coefficient',
            suptitle='Clustering Coefficient Distribution',
            aggregated=aggregated,
        )
        plt.tight_layout()
        if show_figure:
            plt.show()

        return fig


class InnerTreeNodeProbabilityVisualizer(MetricVisualizer[Dict[str, npt.NDArray]]):
    """Visualizes per-node probability of being an inner node in a minimal spanning tree."""

    def _summarize_data(self) -> Dict[str, npt.NDArray]:
        # Average probabilities across all timesteps
        latent_full = np.mean([m['latent_full'] for m in self.metrics_history], axis=0)
        latent_subgraph = np.mean([m['latent_subgraph'] for m in self.metrics_history], axis=0)
        powergrid_full = np.mean([m['powergrid_full'] for m in self.metrics_history], axis=0)

        return {
            'latent_full': latent_full,
            'latent_subgraph': latent_subgraph,
            'powergrid_full': powergrid_full,
        }

    def _compute(self, posterior, powergrid_graph, edge_index_fully_connected, node_mask, samples, **kwargs) -> Dict[str, npt.NDArray]:
        """
        Compute per-node probability of being an inner node in a minimal spanning tree.
        An inner node is a node with degree >= 2 in the spanning tree (not a leaf).
        :param posterior: the posterior distribution [E, K]
        :param powergrid_graph: the powergrid graph edge index [2, E_grid]
        :param edge_index_fully_connected: fully connected edge index [2, E_all]
        :param node_mask: mask indicating which nodes are part of the biggest connected component [num_nodes]
        :param samples: list of sampled edge indices from posterior
        :return: dict with per-node inner tree node probabilities for all variants
        """
        num_nodes = len(node_mask)

        # Compute inner tree node probabilities for latent graph (full) from samples
        latent_full_probs = np.zeros(num_nodes, dtype=float)
        for sample_edges in samples:
            if sample_edges.shape[1] == 0:
                continue
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from(sample_edges.T.astype(int))

            # Compute minimum spanning tree if graph is connected enough
            if nx.number_connected_components(G) > 0:
                try:
                    # Get all connected components
                    for component in nx.connected_components(G):
                        if len(component) > 1:
                            subgraph = G.subgraph(component)
                            mst = nx.minimum_spanning_tree(subgraph)
                            # Count degree in MST
                            degrees = dict(mst.degree())
                            for node in component:
                                # Inner node has degree >= 2 in spanning tree
                                if degrees.get(node, 0) >= 2:
                                    latent_full_probs[node] += 1
                except:
                    pass

        latent_full_probs /= max(len(samples), 1)

        # Compute inner tree node probabilities for latent subgraph from samples
        latent_subgraph_probs = np.zeros(num_nodes, dtype=float)
        for sample_edges in samples:
            # Filter edges to only those within the mask
            subgraph_edges = []
            for src, dst in sample_edges.T:
                src, dst = int(src), int(dst)
                if node_mask[src] and node_mask[dst]:
                    subgraph_edges.append([src, dst])

            if len(subgraph_edges) == 0:
                continue

            G = nx.Graph()
            G.add_nodes_from([i for i in range(num_nodes) if node_mask[i]])
            G.add_edges_from(subgraph_edges)

            if nx.number_connected_components(G) > 0:
                try:
                    for component in nx.connected_components(G):
                        if len(component) > 1:
                            subgraph = G.subgraph(component)
                            mst = nx.minimum_spanning_tree(subgraph)
                            degrees = dict(mst.degree())
                            for node in component:
                                if degrees.get(node, 0) >= 2:
                                    latent_subgraph_probs[node] += 1
                except:
                    pass

        latent_subgraph_probs /= max(len(samples), 1)

        # Compute inner tree node probabilities for full powergrid graph
        # Sample different MSTs by adding small random weights to break ties
        powergrid_full_probs = np.zeros(num_nodes, dtype=float)
        for _ in range(len(samples)):
            G_full = nx.Graph()
            G_full.add_nodes_from(range(num_nodes))
            # Add edges with small random weights to get different MSTs
            for src, dst in powergrid_graph.T:
                G_full.add_edge(int(src), int(dst), weight=np.random.uniform(0, 1e-6))

            if nx.number_connected_components(G_full) > 0:
                try:
                    for component in nx.connected_components(G_full):
                        if len(component) > 1:
                            subgraph = G_full.subgraph(component)
                            mst = nx.minimum_spanning_tree(subgraph, weight='weight')
                            degrees = dict(mst.degree())
                            for node in component:
                                if degrees.get(node, 0) >= 2:
                                    powergrid_full_probs[node] += 1.0
                except:
                    pass

        powergrid_full_probs /= max(len(samples), 1)

        return {
            'latent_full': latent_full_probs,
            'latent_subgraph': latent_subgraph_probs,
            'powergrid_full': powergrid_full_probs,
        }

    def _visualize(self, computation_result: Dict[str, npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
        _plot_three_histograms(
            fig, axs,
            data_triple=(
                computation_result['powergrid_full'],
                computation_result['latent_full'],
                computation_result['latent_subgraph'],
            ),
            xlabel='Probability of being inner node in MST',
            suptitle='Inner Tree Node Probability Distribution',
            aggregated=aggregated,
        )
        plt.tight_layout()
        if show_figure:
            plt.show()

        return fig


class BetweennessVisualizer(MetricVisualizer[Tuple[npt.NDArray, npt.NDArray]]):
    """Visualizes edge betweenness centrality of the posterior graph."""

    def _summarize_data(self) -> Dict[str, npt.NDArray]:
        latent_full = np.mean([m['betweenness_centrality_latent_full'] for m in self.metrics_history], axis=0)
        latent_subgraph = np.mean([m['betweenness_centrality_latent_sub'] for m in self.metrics_history], axis=0)
        powergrid_full = np.mean([m['betweenness_centrality_powergrid_full'] for m in self.metrics_history], axis=0)

        return {
            'betweenness_centrality_latent_full': latent_full,
            'betweenness_centrality_latent_sub': latent_subgraph,
            'betweenness_centrality_powergrid_full': powergrid_full,
        }

    def _compute(self, posterior: npt.NDArray, prior: npt.NDArray, samples: npt.NDArray, powergrid_graph: npt.NDArray,
                 edge_index_fully_connected: npt.NDArray, node_mask: npt.NDArray, observation: BaseObservation) -> T:
        # compute the betweenness centrality for nodes in the posterior graph
        num_nodes = len(node_mask)

        # Build powergrid graph (full)
        G_pg_full = nx.Graph()
        G_pg_full.add_nodes_from(range(num_nodes))
        G_pg_full.add_edges_from(powergrid_graph.T.astype(int))

        # Build powergrid subgraph (only nodes in mask)
        G_pg_sub = nx.Graph()
        G_pg_sub.add_nodes_from([i for i in range(num_nodes) if node_mask[i]])
        for src, dst in powergrid_graph.T:
            if node_mask[src] and node_mask[dst]:
                G_pg_sub.add_edge(int(src), int(dst))

        bc_pg_full = nx.betweenness_centrality(G_pg_full)
        bc_pg_full = [bc_pg_full[i] if i in bc_pg_full else 0 for i in range(num_nodes)]

        bcs_latent_sub = []
        bcs_latent_full = []
        for i, sample in enumerate(samples):
            # build latent graph (full)
            G_latent_full = nx.Graph()
            G_latent_full.add_nodes_from(range(num_nodes))
            G_latent_full.add_edges_from(sample.T.astype(int))

            G_latent_sub = nx.Graph()
            G_latent_sub.add_nodes_from([i for i in range(num_nodes) if node_mask[i]])
            for src, dst in sample.T:
                if node_mask[src] and node_mask[dst]:
                    G_latent_sub.add_edge(int(src), int(dst))

            bcs_latent_sub.append(nx.betweenness_centrality(G_latent_sub))
            bcs_latent_full.append(nx.betweenness_centrality(G_latent_full))

        bc_latent_sub = {node: np.mean([bc[node] for bc in bcs_latent_sub]) for node in bcs_latent_sub[0].keys()}
        bc_latent_full = {node: np.mean([bc[node] for bc in bcs_latent_full]) for node in bcs_latent_full[0].keys()}
        bc_latent_sub = [bc_latent_sub[i] if i in bc_latent_sub else 0 for i in range(num_nodes)]
        bc_latent_full = [bc_latent_full[i] if i in bc_latent_full else 0 for i in range(num_nodes)]

        return {
            "betweenness_centrality_latent_sub": np.array(bc_latent_sub),
            "betweenness_centrality_latent_full": np.array(bc_latent_full),
            "betweenness_centrality_powergrid_full": np.array(bc_pg_full),
        }

    def _visualize(self, computation_result: T, aggregated: bool = False, show_figure: bool = False) -> Figure:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
        _plot_three_histograms(
            fig, axs,
            data_triple=(
                computation_result['betweenness_centrality_powergrid_full'],
                computation_result['betweenness_centrality_latent_full'],
                computation_result['betweenness_centrality_latent_sub'],
            ),
            xlabel='Betweenness Centrality',
            suptitle='Betweenness Centrality Distribution',
            aggregated=aggregated,
        )
        plt.tight_layout()
        if show_figure:
            plt.show()

        return fig


class AllPairsShortestPathVisualizer(MetricVisualizer[Dict[str, npt.NDArray]]):
    """
    Visualizes path length distributions for all pairs of nodes in the latent and powergrid graphs.

    For each graph variant (latent full, latent subgraph, powergrid full) the shortest path
    lengths between every pair of reachable nodes are collected and their distribution is shown
    as a histogram.  Unreachable pairs (disconnected components) are excluded.

    Stored metric keys
    ------------------
    latent_full        : 1-D array of APSP lengths aggregated over posterior samples (full graph)
    latent_subgraph    : 1-D array of APSP lengths aggregated over posterior samples (subgraph induced by node_mask)
    powergrid_full     : 1-D array of APSP lengths for the powergrid graph (full)
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apsp_lengths(G: nx.Graph) -> npt.NDArray:
        """Return a flat array of all finite shortest-path lengths in *G*."""
        lengths = []
        for src in G.nodes:
            for dst, d in nx.single_source_shortest_path_length(G, src).items():
                if dst != src:
                    lengths.append(d)
        return np.array(lengths, dtype=float) if lengths else np.array([], dtype=float)

    def _compute(self, posterior: npt.NDArray, powergrid_graph: npt.NDArray,
                 edge_index_fully_connected: npt.NDArray, node_mask: npt.NDArray,
                 samples: npt.NDArray, **kwargs) -> Dict[str, npt.NDArray]:
        """
        Compute all-pairs shortest path length distributions.

        :param posterior: posterior edge probabilities [E, K]  (kept for API compatibility)
        :param powergrid_graph: powergrid edge index [2, E_grid]
        :param edge_index_fully_connected: fully connected edge index [2, E_all]
        :param node_mask: boolean mask for the biggest connected component [num_nodes]
        :param samples: list of sampled edge indices from the posterior (list of [2, num_edges])
        :return: dict with APSP length arrays for three graph variants
        """
        num_nodes = len(node_mask)

        # ---- powergrid full ----
        G_pg = nx.Graph()
        G_pg.add_nodes_from(range(num_nodes))
        G_pg.add_edges_from(powergrid_graph.T.astype(int))
        powergrid_full_lengths = self._apsp_lengths(G_pg)

        # ---- latent graphs – concatenate lengths across all samples ----
        masked_nodes = [i for i in range(num_nodes) if node_mask[i]]
        latent_full_lengths_all: List[npt.NDArray] = []
        latent_sub_lengths_all: List[npt.NDArray] = []

        for sample_edges in samples:
            # -- full latent graph --
            G_full = nx.Graph()
            G_full.add_nodes_from(range(num_nodes))
            G_full.add_edges_from(sample_edges.T.astype(int))
            latent_full_lengths_all.append(self._apsp_lengths(G_full))

            # -- latent subgraph (nodes within mask only) --
            sub_edges = [
                (int(s), int(d))
                for s, d in sample_edges.T
                if node_mask[int(s)] and node_mask[int(d)]
            ]
            G_sub = nx.Graph()
            G_sub.add_nodes_from(masked_nodes)
            G_sub.add_edges_from(sub_edges)
            latent_sub_lengths_all.append(self._apsp_lengths(G_sub))

        latent_full_lengths = (
            np.concatenate(latent_full_lengths_all)
            if latent_full_lengths_all else np.array([], dtype=float)
        )
        latent_sub_lengths = (
            np.concatenate(latent_sub_lengths_all)
            if latent_sub_lengths_all else np.array([], dtype=float)
        )

        return {
            'latent_full': latent_full_lengths,
            'latent_subgraph': latent_sub_lengths,
            'powergrid_full': powergrid_full_lengths,
        }

    # ------------------------------------------------------------------
    # _summarize_data
    # ------------------------------------------------------------------
    def _summarize_data(self) -> Dict[str, npt.NDArray]:
        """
        Concatenate all collected path-length arrays across timesteps.
        This gives the global distribution over the full evaluation episode.
        """
        def _concat(key: str) -> npt.NDArray:
            arrays = [m[key] for m in self.metrics_history if len(m[key]) > 0]
            return np.concatenate(arrays) if arrays else np.array([], dtype=float)

        return {
            'latent_full': _concat('latent_full'),
            'latent_subgraph': _concat('latent_subgraph'),
            'powergrid_full': _concat('powergrid_full'),
        }

    # ------------------------------------------------------------------
    # _visualize
    # ------------------------------------------------------------------
    def _visualize(self, computation_result: Dict[str, npt.NDArray],
                   aggregated: bool = False, show_figure: bool = False) -> Figure:
        """Visualize APSP distributions as three histograms side by side."""
        latent_full = computation_result['latent_full']
        latent_sub = computation_result['latent_subgraph']
        powergrid_full = computation_result['powergrid_full']

        all_arrays = [arr for arr in [latent_full, latent_sub, powergrid_full] if len(arr) > 0]
        if not all_arrays:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'No path data available', ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)
        _plot_three_histograms(
            fig, axs,
            data_triple=(powergrid_full, latent_full, latent_sub),
            xlabel='Shortest Path Length',
            suptitle='All-Pairs Shortest Path Length Distributions',
            aggregated=aggregated,
            integer_bins=True,
        )
        plt.tight_layout()
        if show_figure:
            plt.show()

        return fig


class PosteriorDistributionVisualizer(MetricVisualizer[Tuple[npt.NDArray, Optional[npt.NDArray], npt.NDArray]]):
    """Visualizes the posterior distribution histogram and latent graph structure."""

    def _summarize_data(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # Stack all posteriors and powerline edges
        all_posteriors = np.stack([m[0] for m in self.metrics_history], axis=0)
        # Just use the first powerline edges
        powerline_edges = self.metrics_history[0][1] if self.metrics_history else np.array([])
        # Compute mean posterior
        mean_posterior = np.mean(all_posteriors, axis=0)
        std_posterior = np.std(all_posteriors, axis=0)
        return mean_posterior, std_posterior, powerline_edges

    def _compute(self, posterior, powergrid_graph, **kwargs) -> Tuple[npt.NDArray, Optional[npt.NDArray], npt.NDArray]:
        """
        Store posterior and powergrid graph for visualization.
        :param posterior: the posterior distribution [E, K]
        :param powergrid_graph: the powergrid graph edge index [2, E_grid]
        :return: tuple of (posterior, powergrid_graph)
        """
        return posterior, None, powergrid_graph

    def _visualize(self, computation_result: Tuple[npt.NDArray, Optional[npt.NDArray], npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
        """
        Visualize the posterior distribution with histogram and graph structure.
        :param computation_result: tuple of (posterior, powerline_edges)
        :param aggregated: whether this is aggregated over multiple steps
        :param show_figure: whether to show the figure interactively
        :return: matplotlib figure
        """
        posterior, std_posterior, powerline_edges = computation_result
        n_cols = 1 + int(aggregated and std_posterior is not None) # show variance only if aggregated and present
        fig, axes = plt.subplots(1, n_cols, figsize=(9 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]  # Make it iterable
        import grid2op
        env = grid2op.make("l2rpn_case14_sandbox")
        powerline_edges = BusConnectivityGraphObsSpace(grid2op_observation_space=env.observation_space).to_gym(env.reset())[EDGE_INDEX]

        # Right: Graph visualization
        visualize_graph(PlottingArgs(
            num_nodes=len(self.node_styles),
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=posterior,
            skip_last_edge_type=True,
            show_legend=False
        ), ax=axes[0])
        axes[0].set_title(r'Mean RAPPO graph $\bar{q}_\phi(\mathbf{z}_{ij} = 1)$', fontsize=20, fontweight='bold')

        if aggregated and std_posterior is not None:
            latent_edge_probs = np.stack(std_posterior + 0.4)
            visualize_graph(PlottingArgs(
                num_nodes=len(self.node_styles),
                node_styles=self.node_styles,
                powerline_edge_index=powerline_edges,
                latent_edge_probs=latent_edge_probs,
                skip_last_edge_type=True
            ), ax=axes[1])
            axes[1].set_title(r'Edge-wise temporal variance over $p_\phi(\mathbf{z}_{ij} = 1 \mid s_t)$', fontsize=20, fontweight='bold')

        plt.tight_layout()

        if show_figure:
            plt.show()

        return fig


# class PathLengthVisualizer(MetricVisualizer[Dict[str, npt.NDArray]]):
#     """Visualizes distribution of shortest path lengths bridged by latent edges for full and subgraph variants."""
#
#     def _summarize_data(self) -> Dict[str, npt.NDArray]:
#         # Concatenate path lengths across all timesteps
#         latent_full = np.concatenate([m['latent_full'] for m in self.metrics_history])
#         latent_subgraph = np.concatenate([m['latent_subgraph'] for m in self.metrics_history])
#         powergrid_full = np.concatenate([m['powergrid_full'] for m in self.metrics_history])
#         powergrid_subgraph = np.concatenate([m['powergrid_subgraph'] for m in self.metrics_history])
#         node_mask = self.metrics_history[0]['node_mask']
#         powergrid_graph = self.metrics_history[0]['powergrid_graph']
#         posterior = self.metrics_history[0]['posterior']
#
#         return {
#             'latent_full': latent_full,
#             'latent_subgraph': latent_subgraph,
#             'powergrid_full': powergrid_full,
#             'powergrid_subgraph': powergrid_subgraph,
#             'node_mask': node_mask,
#             'powergrid_graph': powergrid_graph,
#             'posterior': posterior
#         }
#
#     def _compute(self, posterior, powergrid_graph, edge_index_fully_connected, node_mask, samples, **kwargs) -> Dict[str, npt.NDArray]:
#         """
#         Compute shortest path lengths for edges sampled from posterior vs powergrid baseline.
#         Analyzes both full graphs and subgraphs induced by node_mask.
#
#         :param posterior: the posterior distribution [E, K]
#         :param powergrid_graph: the powergrid graph edge index [2, E_grid]
#         :param edge_index_fully_connected: fully connected edge index [2, E_all]
#         :param node_mask: boolean mask indicating nodes in the biggest connected component
#         :param samples: list of sampled edge indices from posterior
#         :return: dict with path lengths for all variants
#         """
#         num_nodes = len(node_mask)
#
#         # Build powergrid graph (full)
#         G_pg_full = nx.Graph()
#         G_pg_full.add_nodes_from(range(num_nodes))
#         G_pg_full.add_edges_from(powergrid_graph.T.astype(int))
#
#         # Build powergrid subgraph (only nodes in mask)
#         G_pg_sub = nx.Graph()
#         G_pg_sub.add_nodes_from([i for i in range(num_nodes) if node_mask[i]])
#         for src, dst in powergrid_graph.T:
#             if node_mask[src] and node_mask[dst]:
#                 G_pg_sub.add_edge(int(src), int(dst))
#
#         # === Latent Full Graph ===
#         # Compute path lengths for sampled edges using full powergrid as baseline
#         latent_full_paths = []
#         for sample_edges in samples:
#             for src, dst in sample_edges.T:
#                 src, dst = int(src), int(dst)
#                 try:
#                     d = nx.shortest_path_length(G_pg_full, source=src, target=dst)
#                     latent_full_paths.append(d)
#                 except (nx.NetworkXNoPath, nx.NodeNotFound):
#                     pass
#
#         # === Latent Subgraph ===
#         # Compute path lengths for sampled edges within subgraph using powergrid subgraph as baseline
#         latent_subgraph_paths = []
#         for sample_edges in samples:
#             for src, dst in sample_edges.T:
#                 src, dst = int(src), int(dst)
#                 # Only consider edges within the mask
#                 if node_mask[src] and node_mask[dst]:
#                     try:
#                         d = nx.shortest_path_length(G_pg_sub, source=src, target=dst)
#                         latent_subgraph_paths.append(d)
#                     except (nx.NetworkXNoPath, nx.NodeNotFound):
#                         pass
#
#         # === Powergrid Full Graph ===
#         # Sample random node pairs and compute their distances in full powergrid
#         num_samples = min(1000, len(samples) * 10)
#         powergrid_full_paths = []
#         for _ in range(num_samples):
#             src, dst = np.random.choice(num_nodes, size=2, replace=False)
#             try:
#                 d = nx.shortest_path_length(G_pg_full, source=src, target=dst)
#                 powergrid_full_paths.append(d)
#             except (nx.NetworkXNoPath, nx.NodeNotFound):
#                 pass
#
#         # === Powergrid Subgraph ===
#         # Sample random node pairs from subgraph and compute distances
#         masked_nodes = [i for i in range(num_nodes) if node_mask[i]]
#         powergrid_subgraph_paths = []
#         if len(masked_nodes) >= 2:
#             for _ in range(num_samples):
#                 src, dst = np.random.choice(masked_nodes, size=2, replace=False)
#                 try:
#                     d = nx.shortest_path_length(G_pg_sub, source=src, target=dst)
#                     powergrid_subgraph_paths.append(d)
#                 except (nx.NetworkXNoPath, nx.NodeNotFound):
#                     pass
#
#         return {
#             'latent_full': np.array(latent_full_paths),
#             'latent_subgraph': np.array(latent_subgraph_paths),
#             'powergrid_full': np.array(powergrid_full_paths),
#             'powergrid_subgraph': np.array(powergrid_subgraph_paths),
#             'node_mask': node_mask,
#             'powergrid_graph': powergrid_graph,
#             'posterior': posterior
#         }
#
#     def _visualize(self, computation_result: Dict[str, npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
#         """
#         Visualize path length distributions in 2x3 grid:
#         Row 1: Powergrid analysis (full histogram, subgraph histogram, comparison)
#         Row 2: Latent analysis (full histogram, subgraph histogram, comparison)
#         """
#         latent_full = computation_result['latent_full']
#         latent_subgraph = computation_result['latent_subgraph']
#         powergrid_full = computation_result['powergrid_full']
#         powergrid_subgraph = computation_result['powergrid_subgraph']
#
#         # Determine global bin range
#         all_paths = []
#         if len(latent_full) > 0:
#             all_paths.append(latent_full)
#         if len(latent_subgraph) > 0:
#             all_paths.append(latent_subgraph)
#         if len(powergrid_full) > 0:
#             all_paths.append(powergrid_full)
#         if len(powergrid_subgraph) > 0:
#             all_paths.append(powergrid_subgraph)
#
#         if len(all_paths) == 0:
#             # Create empty figure
#             fig = plt.figure(figsize=(18, 12))
#             plt.text(0.5, 0.5, 'No path data available', ha='center', va='center', fontsize=16)
#             return fig
#
#         all_paths_concat = np.concatenate(all_paths)
#         max_d = int(np.ceil(all_paths_concat.max()))
#         bins = np.arange(0, max_d + 2) - 0.5  # integer-centered bins
#
#         # Create 2x3 subplot grid
#         fig, axes = plt.subplots(2, 2, figsize=(18, 12))
#
#         # === ROW 1: POWERGRID ANALYSIS ===
#
#         # Column 1: Powergrid full graph histogram
#         if len(powergrid_full) > 0:
#             axes[0, 0].hist(powergrid_full, bins=bins, color='green', edgecolor='black', alpha=0.7, density=True)
#             axes[0, 0].axvline(np.mean(powergrid_full), color='red', linestyle='--', linewidth=2,
#                               label=f'Mean: {np.mean(powergrid_full):.2f}')
#         axes[0, 0].set_xlabel('Path Length')
#         axes[0, 0].set_ylabel('Density')
#         axes[0, 0].set_title('Powergrid (Full) - Random Path Lengths')
#         axes[0, 0].legend()
#         axes[0, 0].grid(alpha=0.3)
#         axes[0, 0].xaxis.set_major_locator(MultipleLocator(1))
#
#         # Column 2: Powergrid subgraph histogram
#         if len(powergrid_subgraph) > 0:
#             axes[0, 1].hist(powergrid_subgraph, bins=bins, color='green', edgecolor='black', alpha=0.7, density=True)
#             axes[0, 1].axvline(np.mean(powergrid_subgraph), color='red', linestyle='--', linewidth=2,
#                               label=f'Mean: {np.mean(powergrid_subgraph):.2f}')
#         axes[0, 1].set_xlabel('Path Length')
#         axes[0, 1].set_ylabel('Density')
#         axes[0, 1].set_title('Powergrid (Subgraph) - Random Path Lengths')
#         axes[0, 1].legend()
#         axes[0, 1].grid(alpha=0.3)
#         axes[0, 1].xaxis.set_major_locator(MultipleLocator(1))
#
#         # === ROW 2: LATENT ANALYSIS ===
#
#         # Column 1: Latent full graph histogram
#         if len(latent_full) > 0:
#             axes[1, 0].hist(latent_full, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
#             axes[1, 0].axvline(np.mean(latent_full), color='red', linestyle='--', linewidth=2,
#                               label=f'Mean: {np.mean(latent_full):.2f}')
#         axes[1, 0].set_xlabel('Path Length')
#         axes[1, 0].set_ylabel('Density')
#         axes[1, 0].set_title('Latent (Full) - Bridged Path Lengths')
#         axes[1, 0].legend()
#         axes[1, 0].grid(alpha=0.3)
#         axes[1, 0].xaxis.set_major_locator(MultipleLocator(1))
#
#         # Column 2: Latent subgraph histogram
#         if len(latent_subgraph) > 0:
#             axes[1, 1].hist(latent_subgraph, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
#             axes[1, 1].axvline(np.mean(latent_subgraph), color='red', linestyle='--', linewidth=2,
#                               label=f'Mean: {np.mean(latent_subgraph):.2f}')
#         axes[1, 1].set_xlabel('Path Length')
#         axes[1, 1].set_ylabel('Density')
#         axes[1, 1].set_title('Latent (Subgraph) - Bridged Path Lengths')
#         axes[1, 1].legend()
#         axes[1, 1].grid(alpha=0.3)
#         axes[1, 1].xaxis.set_major_locator(MultipleLocator(1))
#
#         plt.tight_layout()
#
#         if show_figure:
#             plt.show()
#
#         return fig


class KLDivergenceVisualizer(MetricVisualizer[Tuple[npt.NDArray, npt.NDArray]]):
    """Visualizes per-edge KL divergence between posterior and prior."""

    def _summarize_data(self) -> npt.NDArray:
        # Each element of metrics_history is a 1-D array of per-edge KL values.
        # Stack across time steps and compute the per-edge mean, then flatten to 1-D.
        all_kls = np.stack(self.metrics_history, axis=0)  # shape [T, E]
        mean_kl = np.mean(all_kls, axis=0).ravel()        # shape [E]
        return mean_kl

    def _compute(self, posterior, prior, **kwargs) -> npt.NDArray:
        """
        Compute per-edge KL divergence.
        :param posterior: the posterior distribution [E, K]
        :param prior: the prior distribution [E, K]
        :return: 1-D array of per-edge KL divergence values, shape [E]
        """
        eps = 1e-7
        per_edge_kl = (posterior * (np.log(posterior + eps) - np.log(prior + eps))).sum(axis=-1)
        return per_edge_kl.ravel()

    def _visualize(self, computation_result: npt.NDArray, aggregated: bool = False, show_figure: bool = False) -> Figure:
        """
        Visualize KL divergence as a histogram consistent with _plot_three_histograms style.
        :param computation_result: 1-D array of per-edge KL divergence values
        :param aggregated: whether this is aggregated over multiple steps
        :param show_figure: whether to show the figure interactively
        :return: matplotlib figure
        """
        per_edge_kl = np.asarray(computation_result).ravel()
        per_edge_kl = per_edge_kl[np.isfinite(per_edge_kl)]

        fig, ax = plt.subplots(figsize=(6, 4))

        bins = _compute_shared_bins([per_edge_kl])
        if len(per_edge_kl) > 0:
            ax.hist(per_edge_kl, bins=bins, color=COLOR_POWERGRID, edgecolor='black', alpha=0.75, density=True)
            mean_val = float(np.mean(per_edge_kl))
            ax.axvline(mean_val, color='blue', linestyle='-.', linewidth=1.8, label=f'Mean: {mean_val:.4f}')
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('Per-Edge KL Divergence')
        ax.set_ylabel('Density')
        ax.grid(axis='y', alpha=0.3)

        ax.set_title(f'Per-Edge KL Divergence', fontsize=14)

        plt.tight_layout()

        if show_figure:
            plt.show()

        return fig


class EntropyVsKLVisualizer(MetricVisualizer[Tuple[npt.NDArray, npt.NDArray]]):
    """Visualizes the relationship between edge entropy and KL divergence."""

    def _summarize_data(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # Concatenate all entropies and KLs
        all_entropies = np.concatenate([m[0] for m in self.metrics_history])
        all_kls = np.concatenate([m[1] for m in self.metrics_history])
        return all_entropies, all_kls

    def _compute(self, posterior, prior, **kwargs) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Compute edge entropies and KL divergences.
        :param posterior: the posterior distribution [E, K]
        :param prior: the prior distribution [E, K]
        :return: tuple of (entropies, per_edge_kl)
        """
        # Compute entropy
        entropies = -np.sum(posterior * np.log(posterior + 1e-10), axis=-1)

        # Compute KL divergence
        eps = 1e-7
        per_edge_kl = (posterior * (np.log(posterior + eps) - np.log(prior + eps))).sum(axis=-1)

        return entropies, per_edge_kl

    def _visualize(self, computation_result: Tuple[npt.NDArray, npt.NDArray], aggregated: bool = False, show_figure: bool = False) -> Figure:
        """
        Visualize entropy vs KL divergence as a scatter plot.
        :param computation_result: tuple of (entropies, per_edge_kl)
        :param aggregated: whether this is aggregated over multiple steps
        :param show_figure: whether to show the figure interactively
        :return: matplotlib figure
        """
        entropies, per_edge_kl = computation_result

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(entropies, per_edge_kl, alpha=0.6, color='purple')
        plt.xlabel('Edge Entropy')
        plt.ylabel('Per-Edge KL Divergence')
        mean_entropy = np.mean(entropies)
        mean_kl = np.mean(per_edge_kl)
        plt.axvline(x=mean_entropy, color='blue', linestyle='--', linewidth=1, label=f'Mean entropy: {mean_entropy:.3f}')
        plt.axhline(y=mean_kl, color='green', linestyle='--', linewidth=1, label=f'Mean per edge KL: {mean_kl:.3f}')
        plt.legend()
        plt.title(f'Edge Entropy vs. KL Divergence')
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if show_figure:
            plt.show()

        return fig

class SymmetryMetricVisualizer(MetricVisualizer[float]):
    """Visualizes the symmetry metric of the posterior graph.

    The symmetry metric R is defined as:
    R = sum_{v!=w}(p_vw * p_wv) / sum_{v!=w}(p_vw)

    where p_vw is the probability of edge from node v to node w.
    A perfectly symmetric graph has R=1.
    """

    def _summarize_data(self) -> npt.NDArray:
        """
        Aggregate all symmetry scores into an array.
        :return: array of symmetry scores across all timesteps
        """
        return np.array(self.metrics_history)

    def _compute(self, posterior: npt.NDArray, edge_index_fully_connected: npt.NDArray, **kwargs) -> float:
        """
        Compute the symmetry metric R for the posterior.

        :param posterior: the posterior edge probabilities of shape [num_edges, num_edge_types]
        :param edge_index_fully_connected: fully connected graph edge indices of shape [2, num_edges]
        :return: the symmetry score R
        """
        # Get the number of nodes from edge_index
        num_nodes = int(edge_index_fully_connected.max()) + 1

        # Create adjacency matrix with edge probabilities
        # Sum over all edge types except "no edge" (last type) to get probability of edge existence
        adj_matrix = np.zeros((num_nodes, num_nodes))

        for i, (src, dst) in enumerate(edge_index_fully_connected.T):
            # Probability that edge exists (sum over all edge types except the last one)
            prob_exists = posterior[i, :-1].sum()
            adj_matrix[src, dst] = prob_exists

        # Compute symmetry metric: R = sum_{v!=w}(p_vw * p_wv) / sum_{v!=w}(p_vw)
        numerator = 0.0
        denominator = 0.0

        for v in range(num_nodes):
            for w in range(num_nodes):
                if v != w:
                    p_vw = adj_matrix[v, w]
                    p_wv = adj_matrix[w, v]
                    numerator += p_vw * p_wv
                    denominator += p_vw ** 2

        # Avoid division by zero
        if denominator < 1e-10:
            return 0.0

        symmetry_score = numerator / denominator
        return float(symmetry_score)

    def _visualize(self, computation_result, aggregated: bool = False, show_figure: bool = False) -> Optional[Figure]:
        """
        Visualize the symmetry metric distribution.
        Only visualizes when aggregated=True (summary mode).

        :param computation_result: either a single float (if not aggregated) or array of floats (if aggregated)
        :param aggregated: whether this is aggregated over multiple steps
        :param show_figure: whether to show the figure interactively
        :return: matplotlib figure
        """
        # Only visualize in aggregated mode
        if not aggregated:
            return None

        # In aggregated mode, computation_result is an array of symmetry scores
        symmetry_scores = computation_result

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.histplot(symmetry_scores, ax=ax, bins=30, alpha=0.7, stat='density')
        mean_symmetry = np.mean(symmetry_scores)
        ax.axvline(mean_symmetry, color='blue', linestyle='-.', linewidth=1.8, label=f'Mean: {mean_symmetry:.4f}')
        ax.legend(fontsize=9)
        ax.set_xlabel('Symmetry Score (R)')
        ax.set_ylabel('Density')
        ax.set_title(f'Per edge Posterior Symmetry Scores')

        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if show_figure:
            plt.show()

        return fig

class NodeTypes(Enum):
    GENERATOR = "generator"
    SUBSTATION = "substation"
    LOAD = "load"

class EdgeNodeTypeVisualizer(MetricVisualizer[Tuple[Dict[NodeTypes, Dict[NodeTypes, float]], Dict[NodeTypes, Dict[NodeTypes, int]], Dict[NodeTypes, Dict[NodeTypes, float]]]]):
    """Visualize the types of nodes that are connected by latent edges in the posterior."""

    @staticmethod
    def _get_node_types(observation: BaseObservation) -> List[NodeTypes]:
        sub_idx, load_idx, generator_idx, line_or_idx, line_ex_idx, _ = observation.grid_objects_types.T
        load_nodes = load_idx >= 0
        generator_nodes = generator_idx >= 0

        return [
            NodeTypes.LOAD if load_nodes[i] else
            NodeTypes.GENERATOR if generator_nodes[i] else
            NodeTypes.SUBSTATION for i in range(len(observation.grid_objects_types))
        ]


    def _compute(self,
             posterior: npt.NDArray,
             prior: npt.NDArray,
             samples: npt.NDArray,
             powergrid_graph: npt.NDArray,
             edge_index_fully_connected: npt.NDArray,
             node_mask: npt.NDArray,
             observation: BaseObservation) -> Tuple[Dict[NodeTypes, Dict[NodeTypes, float]], Dict[NodeTypes, Dict[NodeTypes, int]], Dict[NodeTypes, Dict[NodeTypes, float]]]:
        node_types = self._get_node_types(observation)
        probs_per_type_combination = {}

        for i, e in enumerate(posterior):
            prob = e[:-1].sum()
            src, dst = edge_index_fully_connected[:, i]
            src_type = node_types[src]
            dst_type = node_types[dst]

            if src_type not in probs_per_type_combination:
                probs_per_type_combination[src_type] = {}
            if dst_type not in probs_per_type_combination[src_type]:
                probs_per_type_combination[src_type][dst_type] = 0.0

            probs_per_type_combination[src_type][dst_type] += prob

        combination_possibilities = {}
        for src_type in NodeTypes:
            combination_possibilities[src_type] = {}
            for dst_type in NodeTypes:
                combination_possibilities[src_type][dst_type] = node_types.count(src_type) * node_types.count(dst_type)

        normalized_probs_per_type_comb = {}
        for src_type in probs_per_type_combination:
            normalized_probs_per_type_comb[src_type] = {}
            for dst_type in probs_per_type_combination[src_type]:
                count = combination_possibilities[src_type][dst_type]
                normalized_probs_per_type_comb[src_type][dst_type] = probs_per_type_combination[src_type][dst_type] / max(count, 1)

        return probs_per_type_combination, combination_possibilities, normalized_probs_per_type_comb

    def _visualize(self, computation_result: Tuple[Dict[NodeTypes, Dict[NodeTypes, float]], Dict[NodeTypes, Dict[NodeTypes, int]], Dict[NodeTypes, Dict[NodeTypes, float]]], aggregated: bool = False, show_figure: bool = False) -> Figure:
        """
        Visualize the node type combinations as 3 pie charts (cake diagrams).
        Each pie chart corresponds to one of the 3 dictionaries:
        1. Raw probabilities per type combination
        2. Combination possibilities (counts)
        3. Normalized probabilities per type combination
        """
        probs_per_type_combination, combination_possibilities, normalized_probs_per_type_comb = computation_result

        # Create figure with 3 subplots (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # Define unique colors for each node type combination
        # Use a distinct color palette with 9 different colors (3x3 combinations)
        combination_colors = {
            (NodeTypes.GENERATOR, NodeTypes.GENERATOR): '#E74C3C',      # Red
            (NodeTypes.GENERATOR, NodeTypes.SUBSTATION): '#F39C12',     # Orange
            (NodeTypes.GENERATOR, NodeTypes.LOAD): '#F1C40F',           # Yellow
            (NodeTypes.SUBSTATION, NodeTypes.GENERATOR): '#3498DB',     # Blue
            (NodeTypes.SUBSTATION, NodeTypes.SUBSTATION): '#2ECC71',    # Green
            (NodeTypes.SUBSTATION, NodeTypes.LOAD): '#1ABC9C',          # Turquoise
            (NodeTypes.LOAD, NodeTypes.GENERATOR): '#9B59B6',           # Purple
            (NodeTypes.LOAD, NodeTypes.SUBSTATION): '#E91E63',          # Pink
            (NodeTypes.LOAD, NodeTypes.LOAD): '#34495E',                # Dark gray
        }

        # Prepare data for each pie chart
        datasets = [
            (probs_per_type_combination, "Observed Pairs"),
            (combination_possibilities, "Combination Possibilities"),
            (normalized_probs_per_type_comb, "Normalized Probabilities")
        ]

        # Collect all unique combinations for the legend (use first dataset to determine order)
        all_combinations = []
        all_labels = []
        all_colors = []

        for src_type in NodeTypes:
            for dst_type in NodeTypes:
                combination = (src_type, dst_type)
                label = f"{src_type.value} → {dst_type.value}"
                color = combination_colors.get(combination, '#CCCCCC')
                all_combinations.append(combination)
                all_labels.append(label)
                all_colors.append(color)

        def autopct_format(pct):
            """Only show percentage if >= 5%"""
            return f'{pct:.1f}%' if pct >= 5 else ''

        for ax_idx, (data_dict, title) in enumerate(datasets):
            ax = axes[ax_idx]

            # Flatten the nested dictionary into lists for pie chart
            # Use consistent ordering based on all_combinations
            values = []
            colors = []

            for combination in all_combinations:
                src_type, dst_type = combination
                if src_type in data_dict and dst_type in data_dict[src_type]:
                    value = data_dict[src_type][dst_type]
                    if value > 0:  # Only include non-zero values
                        values.append(value)
                        colors.append(combination_colors.get(combination, '#CCCCCC'))

            # Create pie chart without labels (we'll use a legend instead)
            if values:
                wedges, texts, autotexts = ax.pie(
                    values,
                    colors=colors,
                    autopct=autopct_format,
                    startangle=90,
                    textprops={'fontsize': 10}
                )

                # Make percentage text bold and white for better visibility
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
            else:
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes)

            ax.set_title(title, fontsize=12, fontweight='bold')

        # Create a single legend for all three plots
        # Place it below the plots
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label)
                          for label, color in zip(all_labels, all_colors)]

        fig.legend(handles=legend_elements,
                  loc='lower center',
                  ncol=3,
                  bbox_to_anchor=(0.5, 0.02),
                  frameon=True,
                  fontsize=10,
                  title='Node Type Combinations',
                  title_fontsize=11)

        plt.suptitle('Edge Node Type Combinations Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])

        if show_figure:
            plt.show()

        return fig

    def _summarize_data(self) -> Tuple[Dict[NodeTypes, Dict[NodeTypes, float]], Dict[NodeTypes, Dict[NodeTypes, int]], Dict[NodeTypes, Dict[NodeTypes, float]]]:
        """
        Average the probabilities and possibilities across all timesteps.
        """
        if not self.metrics_history:
            # Return empty dicts if no history
            return {}, {}, {}

        # Initialize aggregated dictionaries
        agg_probs = {}
        agg_possibilities = {}
        agg_normalized = {}

        # Sum all values across timesteps
        for timestep_data in self.metrics_history:
            probs_per_type_combination, combination_possibilities, normalized_probs_per_type_comb = timestep_data

            # Aggregate raw probabilities
            for src_type in probs_per_type_combination:
                if src_type not in agg_probs:
                    agg_probs[src_type] = {}
                for dst_type in probs_per_type_combination[src_type]:
                    if dst_type not in agg_probs[src_type]:
                        agg_probs[src_type][dst_type] = 0.0
                    agg_probs[src_type][dst_type] += probs_per_type_combination[src_type][dst_type]

            # Aggregate possibilities (these should be constant across timesteps, so we take the last)
            for src_type in combination_possibilities:
                if src_type not in agg_possibilities:
                    agg_possibilities[src_type] = {}
                for dst_type in combination_possibilities[src_type]:
                    agg_possibilities[src_type][dst_type] = combination_possibilities[src_type][dst_type]

            # Aggregate normalized probabilities
            for src_type in normalized_probs_per_type_comb:
                if src_type not in agg_normalized:
                    agg_normalized[src_type] = {}
                for dst_type in normalized_probs_per_type_comb[src_type]:
                    if dst_type not in agg_normalized[src_type]:
                        agg_normalized[src_type][dst_type] = 0.0
                    agg_normalized[src_type][dst_type] += normalized_probs_per_type_comb[src_type][dst_type]

        # Average the accumulated values
        num_timesteps = len(self.metrics_history)
        for src_type in agg_probs:
            for dst_type in agg_probs[src_type]:
                agg_probs[src_type][dst_type] /= num_timesteps

        for src_type in agg_normalized:
            for dst_type in agg_normalized[src_type]:
                agg_normalized[src_type][dst_type] /= num_timesteps

        return agg_probs, agg_possibilities, agg_normalized

class StepVisualizer(MetricVisualizer):
    def _summarize_data(self) -> T:
        return None

    def _compute(self,
                 posterior: npt.NDArray,
                 prior: npt.NDArray,
                 samples: npt.NDArray,
                 powergrid_graph: npt.NDArray,
                 edge_index_fully_connected: npt.NDArray,
                 node_mask: npt.NDArray,
                 observation: BaseObservation) -> T:
        return {"posterior": posterior, "powergrid_graph": powergrid_graph, "rhos": observation.rho}

    def _visualize(self, computation_result: T, aggregated: bool = False, show_figure: bool = False) -> Figure:
        if aggregated:
            return None

        powerline_edges = computation_result['powergrid_graph']
        posterior = computation_result['posterior']
        rhos = computation_result['rhos']

        # Number of unique powerlines (undirected edges)
        num_powerlines = 20
        num_edges = powerline_edges.shape[1]

        # Build mapping from powerline index to edge indices in pl_edge_index
        # Powerlines connect node i (line_or) to node i+n_line (line_ex)
        # We need to find which edges in pl_edge_index correspond to powerlines
        powerline_edge_indices = []
        for pl_idx in range(num_powerlines):
            line_or_node = pl_idx
            line_ex_node = pl_idx + num_powerlines
            # Find edges connecting these nodes (bidirectional)
            for edge_idx in range(num_edges):
                src, dst = powerline_edges[:, edge_idx]
                if (src == line_or_node and dst == line_ex_node) or (src == line_ex_node and dst == line_or_node):
                    powerline_edge_indices.append(edge_idx)
                    break  # Found the edge for this powerline

        fig, axs = plt.subplots(1, 2, figsize=(24, 8), constrained_layout=True)

        # Handle case of single agent (ensure axes is 2D)
        rho_cmap = plt.colormaps['RdYlGn']  # Red (high load) -> Yellow -> Green (low load)
        neutral_gray = '#808080'  # Gray for non-powerline edges
        # Create color array for ALL edges, default to gray
        edge_colors_rho = [neutral_gray] * num_edges
        # Create width array for ALL edges, default to thin
        edge_widths_rho = [1.0] * num_edges
        # Set colors and widths for powerline edges based on rho data
        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            rho = rhos[pl_idx]
            # Normalize rho to [0, 1] for colormap
            # rho: 0.0 = no load (green), 1.0 = at limit (red)
            rho_normalized = min(1.0, max(0.0, rho))  # Clamp to [0, 1]
            color = rho_cmap(1.0 - rho_normalized)  # Invert: high rho = red (low in colormap)
            edge_colors_rho[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths_rho[edge_idx] = 3.0

        plotting_args_rho = PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            powerline_edge_colors=edge_colors_rho,
            powerline_edge_widths=edge_widths_rho,
            show_legend=False
        )

        visualize_graph(plotting_args_rho, ax=axs[0])
        axs[0].set_title(f"Line Congestion Before Failure")

        plotting_args = PlottingArgs(
            num_nodes=57,
            node_styles=self.node_styles,
            powerline_edge_index=powerline_edges,
            latent_edge_probs=posterior
        )

        visualize_graph(plotting_args, ax=axs[1])
        axs[1].set_title(f"Latent Graph Posterior")

        # Add colorbar
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Colorbar for rho (row 1) - vertical on the right
        # We use reversed colormap since we map high rho (1.0) -> red by using (1.0 - rho) with RdYlGn
        norm_rho = Normalize(vmin=0, vmax=1.5)
        sm_rho = ScalarMappable(cmap=rho_cmap.reversed(), norm=norm_rho)
        sm_rho.set_array([])
        cbar_rho = fig.colorbar(sm_rho, ax=axs[0], orientation='vertical', pad=0.15, aspect=20, fraction=0.02)
        cbar_rho.set_label('Line Congestion (ρ)', fontsize=10)
        if show_figure:
            fig.show()

        return fig
