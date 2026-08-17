"""
Simplified script to run a saved RAPPO checkpoint and display the posterior distribution.

This version shows posterior statistics and graph visualizations.
Reuses existing infrastructure from evaluate_rllib_agent.py
"""
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from tabulate import tabulate

from experiments.utils import AgentSpec, load_agent_from_spec
from analysis.analyze_latent_graphs.Metrics import DegreeDistributionVisualizer, ClusteringCoefficientVisualizer, \
    PosteriorDistributionVisualizer, KLDivergenceVisualizer, SymmetryMetricVisualizer, BetweennessVisualizer, \
    MetricVisualizer
from analysis.analyze_latent_graphs.agent_analysis_framework import PosteriorAnalyzer, LatentGraphAnalysisAgent
from grid2op_env.observation_converter import GraphObservationConverter
from rarl import fully_connected_edge_index
from rarl.prior import get_priors, get_prior_tensor
from visualization import get_node_styles
from visualization.utils import NodeStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosteriorMetrics(PosteriorAnalyzer):
    """Class to analyse and visualize RAPPO posterior distributions."""

    def __init__(self, save_dir: Path, node_styles: List[NodeStyle], prior_for_graph_edges: float = 0.9,
                 temperature: float = 0.5, num_edge_types: int = 2, num_posterior_samples: int = 100,
                 enable_stepwise_viz: bool = True):
        self.save_dir = save_dir

        self.prior_for_graph_edges = prior_for_graph_edges
        self.temperature = temperature
        self.num_edge_types = num_edge_types
        num_nodes = len(node_styles)
        self.fully_connected_edge_index = fully_connected_edge_index(num_nodes).detach().cpu().numpy()
        self.num_nodes = num_nodes
        self.num_posterior_samples = num_posterior_samples
        self.enable_stepwise_viz = enable_stepwise_viz

        self.metrics: Dict[str, MetricVisualizer] = {
            "Node Degree": DegreeDistributionVisualizer(node_styles=node_styles),
            "Clustering Coefficient": ClusteringCoefficientVisualizer(node_styles=node_styles),
            #"Inner Tree Node Probability": InnerTreeNodeProbabilityVisualizer(node_styles=node_styles),
            "Posterior Distribution": PosteriorDistributionVisualizer(node_styles=node_styles),
            "KL Divergence": KLDivergenceVisualizer(node_styles=node_styles),
            #"Entropy vs KL": EntropyVsKLVisualizer(node_styles=node_styles),
            #"Path Length": AllPairsShortestPathVisualizer(node_styles=node_styles),
            "Symmetry Analysis": SymmetryMetricVisualizer(node_styles=node_styles),
            #"Connected Node Types": EdgeNodeTypeVisualizer(node_styles=node_styles),
            #"Steps": StepVisualizer(node_styles=node_styles),
            "Betweenness Centrality": BetweennessVisualizer(node_styles=node_styles),
        }

        self.all_posteriors = []
        self.all_priors = []
        self.all_kls = []
        self.current_chronic_id = None
        self.step_count_this_episode = 0
        self.step_count_total = 0
        self.episode_count = 0

    def on_heuristic_step(self, powergrid_graph: npt.NDArray, observation: BaseObservation, environment: Environment):
        pass

    def on_rl_step(self, posterior: npt.NDArray, prior: npt.NDArray, powergrid_graph: npt.NDArray, observation: BaseObservation, _: Environment, action: BaseAction):
        self.step_count_this_episode += 1
        self.step_count_total += 1
        prior = self._get_prior(powergrid_graph, self.prior_for_graph_edges, self.temperature, self.num_edge_types)
        self.all_posteriors.append(posterior)
        self.all_priors.append(prior)

        # compute all metrics (always); only visualize and save per-step figures when enabled
        samples = self._sample_n_graphs(posterior, n=self.num_posterior_samples)
        node_mask = self._compute_mask_connected_nodes(posterior, threshold=0.5)
        for metric_name, metric in self.metrics.items():
            try:
                figure, _ = metric(
                    posterior=posterior,
                    prior=prior,
                    samples=samples,
                    powergrid_graph=powergrid_graph,
                    edge_index_fully_connected=self.fully_connected_edge_index,
                    node_mask=node_mask,
                    observation=observation,
                    visualize=self.enable_stepwise_viz,
                )
                if figure is not None:
                    save_dir = self.save_dir / "metrics" / metric_name.replace(" ", "_").lower()
                    image_save_path = save_dir / f"{self.current_chronic_id}_step_{self.step_count_total:04d}.svg"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    figure.savefig(image_save_path)
                    logger.info(f"Saved metric {metric_name} at step {self.step_count_total} to {image_save_path}")
                    plt.close(figure)

            except Exception as e:
                logger.error(f"Error computing metric {metric_name} at step {self.step_count_total}: {e}")
                traceback.print_exc()

    def on_new_episode(self, chronic_id: str):
        self.current_chronic_id = chronic_id
        self.episode_count += 1

    def on_evaluation_end(self):
        parent_dir = self.save_dir / "metrics_agg"
        parent_dir.mkdir(parents=True, exist_ok=True)

        aggregated_data_per_metric = {}
        for metric_name, metric in self.metrics.items():
            path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
            path_fig = parent_dir / f"{metric_name.replace(' ', '_').lower()}.svg"
            try:
                fig, data = metric.summarize(show_figure=True)
                metric.save_data(data, path_data)
                aggregated_data_per_metric[metric_name] = data
                logger.info(f"Saved aggregate data for {metric_name} to {path_data}")
                if fig is not None:
                    fig.savefig(path_fig)
                    plt.close(fig)
                    logger.info(f"Saved aggregate figure for {metric_name} to {path_fig}")
                else:
                    logger.warning(f"No figure returned by summarize() for {metric_name}")

            except Exception as e:
                logger.error(f"Error computing aggregate metric {metric_name}: {e}")
                traceback.print_exc()

        # Print summary table
        self.print_summary_table(aggregated_data_per_metric)

    def load_and_visualize(self):
        """
        Loads previously saved aggregated metric data from disk and generates visualizations without re-running the agent.
        """
        parent_dir = self.save_dir / "metrics_agg"
        for metric_name, metric_fn in self.metrics.items():
            path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
            path_fig = parent_dir / f"{metric_name.replace(' ', '_').lower()}.svg"
            if not path_data.exists():
                logger.warning(f"No saved data found for metric {metric_name} at {path_data}. Skipping.")
                continue
            try:
                data = metric_fn.load_data(path_data)
                figure = metric_fn._visualize(computation_result=data, aggregated=True, show_figure=True)
                if figure is not None:
                    figure.savefig(path_fig)
                    plt.close(figure)
                    logger.info(f"Visualized loaded data for {metric_name}, saved to {path_fig}")
            except Exception as e:
                logger.error(f"Error visualizing loaded metric {metric_name}: {e}")
                traceback.print_exc()

    def print_summary_table(self, aggregated_data_per_metric: Optional[Dict[str, Any]] = None):
        """
        Prints a table with one row per metric for the three graphs and columns for mean, std, min, max, median.

        If aggregated_data_per_metric is not provided, tries to load data from disk.
        """
        if aggregated_data_per_metric is None:
            parent_dir = self.save_dir / "metrics_agg"
            aggregated_data_per_metric = {}
            for metric_name, metric_fn in self.metrics.items():
                path_data = parent_dir / f"{metric_name.replace(' ', '_').lower()}.pkl"
                if path_data.exists():
                    try:
                        aggregated_data_per_metric[metric_name] = metric_fn.load_data(path_data)
                    except Exception as e:
                        logger.warning(f"Could not load data for {metric_name}: {e}")

        # ------------------------------------------------------------------ #
        # Helper: compute stats (mean, std, min, max, median) for a 1-D array
        # ------------------------------------------------------------------ #
        def _stats(arr: npt.NDArray):
            """Return (is_integer, mean, std, min, max, median) or all None if arr is empty / scalar."""
            arr = np.asarray(arr).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return False, None, None, None, None, None
            # Treat as integer if dtype is integral OR if all values are whole numbers
            # (handles legacy data saved as float that represents discrete counts)
            is_integer = np.issubdtype(arr.dtype, np.integer) or bool(np.all(arr == np.floor(arr)))
            return (is_integer,
                    float(np.mean(arr)), float(np.std(arr)),
                    float(np.min(arr)), float(np.max(arr)), float(np.median(arr)))

        def _fmt(val, digits=4, is_integer=False):
            if val is None:
                return "--"
            return f"{int(round(val))}" if is_integer else f"{val:.{digits}f}"

        def _fmt_mean_std(mean, std, digits=4, is_integer=False):
            if mean is None:
                return "--"
            if is_integer:
                # Mean of integers can be non-integer; keep one decimal for clarity
                return f"{mean:.1f} ± {std:.1f}" if std is not None else f"{mean:.1f}"
            if std is None:
                return _fmt(mean, digits)
            return f"{mean:.{digits}f} ± {std:.{digits}f}"

        # ------------------------------------------------------------------ #
        # Build rows: (metric_name, graph_label, mean±std, min, max, median)
        # ------------------------------------------------------------------ #
        rows = []

        # --- metrics that return {'latent_full', 'latent_subgraph', 'powergrid_full'} ---
        node_metrics = [
            ("Node Degree",                "Node Degree"),
            ("Clustering Coefficient",     "Clustering Coefficient"),
            ("Inner Tree Node Probability","Inner Tree Node Probability"),
            ("Path Length",                "All-Pairs Shortest Path"),
        ]
        for metric_key, display_name in node_metrics:
            data = aggregated_data_per_metric.get(metric_key)
            if data is None:
                continue
            for arr, graph_label in [
                (data.get('latent_full'),    "Latent"),
                (data.get('latent_subgraph'),"Latent (BCC)"),
                (data.get('powergrid_full'), "Power grid"),
            ]:
                if arr is None:
                    continue
                is_int, mean, std, mn, mx, med = _stats(arr)
                rows.append((display_name, graph_label,
                             _fmt_mean_std(mean, std, is_integer=is_int),
                             _fmt(mn, is_integer=is_int),
                             _fmt(mx, is_integer=is_int),
                             _fmt(med, is_integer=is_int)))
            rows.append(("", "", "", "", "", ""))  # blank separator row

        # --- Betweenness Centrality (different dict keys) ---
        data = aggregated_data_per_metric.get("Betweenness Centrality")
        if data is not None:
            for arr, graph_label in [
                (data.get('betweenness_centrality_latent_full'), "Latent"),
                (data.get('betweenness_centrality_latent_sub'),  "Latent (BCC)"),
                (data.get('betweenness_centrality_powergrid_full'), "Power grid"),
            ]:
                if arr is None:
                    continue
                is_int, mean, std, mn, mx, med = _stats(arr)
                rows.append(("Betweenness Centrality", graph_label,
                             _fmt_mean_std(mean, std, is_integer=is_int),
                             _fmt(mn, is_integer=is_int),
                             _fmt(mx, is_integer=is_int),
                             _fmt(med, is_integer=is_int)))
            rows.append(("", "", "", "", "", ""))

        # --- Posterior Distribution ---
        data = aggregated_data_per_metric.get("Posterior Distribution")
        if data is not None:
            mean_post, std_post, _ = data  # (mean_posterior [E,K], std_posterior [E,K], edges)
            # p(edge exists) = sum over all types except last
            p_exists = mean_post[:, :-1].sum(axis=1)
            p_no_edge = mean_post[:, -1]
            for arr, label in [(p_exists, "p(edge exists)"), (p_no_edge, "p(no edge)")]:
                is_int, mean, std, mn, mx, med = _stats(arr)
                rows.append(("Posterior Distribution", label,
                             _fmt_mean_std(mean, std, is_integer=is_int),
                             _fmt(mn, is_integer=is_int),
                             _fmt(mx, is_integer=is_int),
                             _fmt(med, is_integer=is_int)))
            rows.append(("", "", "", "", "", ""))

        # --- KL Divergence ---
        data = aggregated_data_per_metric.get("KL Divergence")
        if data is not None:
            is_int, mean, std, mn, mx, med = _stats(data)
            rows.append(("Per Edge KL Divergence", "--",
                         _fmt_mean_std(mean, std, is_integer=is_int),
                         _fmt(mn, is_integer=is_int),
                         _fmt(mx, is_integer=is_int),
                         _fmt(med, is_integer=is_int)))
            rows.append(("", "", "", "", "", ""))

        # --- Symmetry Analysis ---
        data = aggregated_data_per_metric.get("Symmetry Analysis")
        if data is not None:
            scores = np.asarray(data).ravel()
            mean_sym = float(np.mean(scores)) if scores.size > 0 else None
            rows.append(("Symmetry Analysis", "Latent",
                         _fmt(mean_sym), "--", "--", "--"))
            rows.append(("Symmetry Analysis", "Power grid",
                         "1.0000", "--", "--", "--"))

        # ------------------------------------------------------------------ #
        # Print readable console table with tabulate
        # ------------------------------------------------------------------ #
        headers = ["Metric", "Graph", "Mean ± Std", "Min", "Max", "Median"]
        # Filter blank separator rows for the tabulate display
        display_rows = [r for r in rows if any(c != "" for c in r)]
        print("\n" + "=" * 90)
        print("AGGREGATED GRAPH METRIC SUMMARY")
        print("=" * 90)
        print(tabulate(display_rows, headers=headers, tablefmt="rounded_outline"))
        print()

        # ------------------------------------------------------------------ #
        # Print LaTeX table ready to paste into thesis
        # ------------------------------------------------------------------ #

        # Section separators: metric names where we want a \midrule before them
        latex_midrule_before = {
            "Posterior Distribution",
            "Per Edge KL Divergence",
            "Symmetry Analysis",
        }

        latex_lines = [r"\begin{tabular}{llllll}", r"\toprule",
                       r"\textbf{Metric} & \textbf{Graph} & \textbf{Mean} $\pm$ \textbf{Std} "
                       r"& \textbf{Min} & \textbf{Max} & \textbf{Median} \\", r"\midrule"]

        prev_metric = None
        for metric, graph, mean_std_str, mn_str, mx_str, med_str in rows:
            if metric == "" and graph == "":
                continue  # skip blank separator rows (we handle separators via midrule_before)

            # Insert \midrule before new section groups
            if metric != prev_metric and metric in latex_midrule_before:
                latex_lines.append(r"\midrule")
            prev_metric = metric

            # Re-compute LaTeX-formatted mean±std from the display string (already formatted)
            # Just replace ± with $\pm$ in the string we already built
            latex_ms = mean_std_str.replace("±", r"$\pm$")

            # Escape underscores in graph label (BCC has none, but be safe)
            graph_esc = graph.replace("_", r"\_")

            latex_lines.append(
                f"{metric} & {graph_esc} & {latex_ms} & {mn_str} & {mx_str} & {med_str} \\\\"
            )

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        latex_table = "\n".join(latex_lines)
        print("\n" + "=" * 90)
        print("LATEX TABLE (copy-paste into thesis)")
        print("=" * 90)
        print(latex_table)
        print()

    def _compute_mask_connected_nodes(self, posterior: npt.NDArray, threshold: float = 0.5) -> npt.NDArray:
        """
        Graphs sampled from the posterior partition the nodes into subgraphs that have sparse connections within and subgraphs that are not connected at all.
        To analyze only the main connected component, we compute a mask over nodes that are connected via edges with existence probability above a certain threshold.
        :param posterior: The posterior distribution as a numpy array of shape (num_edges, num_edge_types)
        :param threshold: The threshold for edge existence probability to consider an edge as present
        :return: a boolean mask of shape [num_nodes] indicating which nodes are considered
        """
        edge_probs = posterior[:, 0:-1].sum(axis=1)  # Sum over all edge types except the last one (non-existence)
        edges_above_threshold = self.fully_connected_edge_index[:, edge_probs >= threshold]

        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(edges_above_threshold.T.tolist())

        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        node_mask = np.zeros(self.num_nodes, dtype=bool)
        node_mask[list(largest_cc)] = True

        return node_mask

    def _sample_n_graphs(self, posterior: npt.NDArray, n: int) -> list[npt.NDArray]:
        """
        Samples n edge_indices from the posterior distribution.
        :param posterior: the posterior distribution as a numpy array of shape (num_edges, num_edge_types)
        :param n: number of graphs to sample
        :return: list of sampled edge indices as numpy arrays
        """
        sampled_graphs = []
        E, K = posterior.shape
        for _ in range(n):
            edge_exists = np.random.random(E) < posterior[:, :-1].sum(axis=-1)
            sampled_edges = self.fully_connected_edge_index[:, edge_exists]
            sampled_graphs.append(sampled_edges)

        return sampled_graphs

    def _get_prior(self, powergrid_graph: npt.NDArray, prior_for_graph_edges: float = 0.9, temperature: float = 0.5,
                   num_edge_types: int = 2) -> npt.NDArray:
        """
        Returns the prior distribution over types for each edge as a numpy array
        :param powergrid_graph: the current powergrid graph edge index
        :param prior_for_graph_edges: the prior probability to exist for edges in the powergrid graph
        :param temperature: the fraction of additional edges that we want to infer on top of the powergrid graph
        :param num_edge_types: the number of edge types
        :return: the prior distribution as a numpy array of shape (num_edges, num_edge_types)
        """
        E = powergrid_graph.shape[1]
        prior_for_graph_edges, prior_for_non_graph_edges = get_priors(
            prob_graph_edges_exist=prior_for_graph_edges,
            num_graph_edges=E,
            num_non_graph_edges=self.fully_connected_edge_index.shape[1] - E,
            temperature=temperature
        )
        prior = get_prior_tensor(
            graph_edges=torch.from_numpy(powergrid_graph),
            all_edges=torch.from_numpy(self.fully_connected_edge_index),
            prior_for_graph_edges=prior_for_graph_edges,
            prior_for_non_graph_edges=prior_for_non_graph_edges,
            num_edge_types=num_edge_types,
        ).detach().cpu().numpy()
        return prior



def main():
    agent_spec = AgentSpec(
        name="RAPPO",
        load_path="/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48",
        checkpoint_name="checkpoint_000020",
    )
    env_name = "l2rpn_case14_sandbox_test"
    # Set to True to run the agent and compute metrics; False to load from disk and visualize only
    compute_data = False
    # Set to True to also save per-step figures (slow); False to only save aggregated figures
    enable_stepwise_viz = False
    num_episodes = 50
    max_total_duration_s = 60 * 60 * 1  # 1 hour
    save_dir = Path("results/graph_metrics_thesis")

    if compute_data:
        agent, env, gym_env = load_agent_from_spec(agent_spec=agent_spec, env_name=env_name)
        config = agent._rllib_agent.config
        analyzers_to_run = [
            PosteriorMetrics(
                save_dir=save_dir,
                node_styles=get_node_styles(env, GraphObservationConverter),
                prior_for_graph_edges=config.get("relation_awareness", {}).get("prior_prob_for_graph_edge", 0.9),
                temperature=config.get("relation_awareness", {}).get("temperature", 0.5),
                num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get(
                    "num_edge_types", 2),
                enable_stepwise_viz=enable_stepwise_viz,
            )
        ]
        analysis_agent = LatentGraphAnalysisAgent(agent, gym_env, analyzers_to_run)
        logger.info("Agent loaded! Starting episodes...\n")
        analysis_agent.analyze(num_episodes=num_episodes, max_total_duration_s=max_total_duration_s)
    else:
        # Load saved aggregated data, re-generate figures and print summary table
        agent, env, gym_env = load_agent_from_spec(agent_spec=agent_spec, env_name=env_name)
        config = agent._rllib_agent.config
        analyzer = PosteriorMetrics(
            save_dir=save_dir,
            node_styles=get_node_styles(env, GraphObservationConverter),
            prior_for_graph_edges=config.get("relation_awareness", {}).get("prior_prob_for_graph_edge", 0.9),
            temperature=config.get("relation_awareness", {}).get("temperature", 0.5),
            num_edge_types=config.get("model", {}).get("custom_model_config", {}).get("encoder", {}).get(
                "num_edge_types", 2),
        )
        analyzer.load_and_visualize()
        analyzer.print_summary_table()


if __name__ == "__main__":
    main()
