#!/usr/bin/env python3
"""
Script to visualize edge probabilities for all files with forecast in the filename.
Edge probabilities are loaded from .npy files under results/edge_probabilities/
and visualized using the graph visualization utilities.
"""
import logging
from pathlib import Path
from typing import List

import grid2op
import numpy as np
from matplotlib import pyplot as plt

from common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from visualization.utils import visualize_graph, PlottingArgs, get_node_styles, latent_edge_hist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_edge_prob_files_with_forecast(edge_probs_dir: Path) -> List[Path]:
    """
    Get all .npy files containing 'with_forecast' in the filename.

    :param edge_probs_dir: Directory containing edge probability files
    :return: List of paths to edge probability files with forecast
    """
    if not edge_probs_dir.exists():
        raise FileNotFoundError(f"Directory {edge_probs_dir} does not exist")

    files = [f for f in edge_probs_dir.glob("*.npy") if "with_forecast" in f.name]
    logger.info(f"Found {len(files)} edge probability files with forecast")
    return sorted(files)


def visualize_edge_probs_file(
    edge_probs_path: Path,
    env: grid2op.Environment.Environment,
    obs_space: BusConnectivityGraphObsSpace,
    output_dir: Path,
    skip_last_edge_type: bool = True
):
    """
    Visualize edge probabilities from a single .npy file.

    :param edge_probs_path: Path to .npy file containing edge probabilities
    :param env: Grid2Op environment
    :param obs_space: Observation space
    :param output_dir: Directory to save visualization
    :param skip_last_edge_type: Whether to skip the last edge type in visualization
    """
    logger.info(f"Processing {edge_probs_path.name}")

    # Load edge probabilities
    edge_probs = np.load(edge_probs_path)
    edge_probs = edge_probs[:, [1, 0]]
    logger.info(f"  Edge probabilities shape: {edge_probs.shape}")

    # Get observation to extract edge index
    obs = env.reset()
    obs_gym = obs_space.to_gym(obs)
    edge_index = obs_gym[EDGE_INDEX]

    # Get node styles
    node_styles = get_node_styles(env, BusConnectivityGraphObsSpace)

    # Create plotting args
    plotting_args = PlottingArgs(
        num_nodes=obs_space.num_nodes,
        node_styles=node_styles,
        powerline_edge_index=edge_index,
        latent_edge_probs=edge_probs,
        latent_edge_weight=5.0,
        skip_last_edge_type=skip_last_edge_type
    )

    # Create graph visualization
    logger.info(f"  Creating graph visualization...")
    fig_graph = visualize_graph(plotting_args)

    # Save graph visualization
    output_path_graph = output_dir / f"{edge_probs_path.stem}_graph.png"
    fig_graph.savefig(output_path_graph, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved graph visualization to {output_path_graph}")
    plt.close(fig_graph)

    # Create histogram visualization
    logger.info(f"  Creating histogram visualization...")
    fig_hist = latent_edge_hist(edge_probs, skip_last_edge_type=skip_last_edge_type)

    # Save histogram visualization
    output_path_hist = output_dir / f"{edge_probs_path.stem}_histogram.png"
    fig_hist.savefig(output_path_hist, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved histogram to {output_path_hist}")
    plt.close(fig_hist)


def compute_average_edge_probs(
    edge_prob_files: List[Path],
    edge_probs_dir: Path
) -> np.ndarray:
    """
    Compute the average edge probabilities across all files.

    :param edge_prob_files: List of paths to edge probability files
    :param edge_probs_dir: Directory containing edge probability files
    :return: Average edge probabilities as numpy array
    """
    logger.info("\nComputing average edge probabilities across all files...")

    if not edge_prob_files:
        raise ValueError("No edge probability files provided")

    # Load all edge probabilities
    all_edge_probs = []
    for edge_probs_path in edge_prob_files:
        edge_probs = np.load(edge_probs_path)
        edge_probs = edge_probs[:, [1, 0]]
        all_edge_probs.append(edge_probs)
        logger.info(f"  Loaded {edge_probs_path.name}: shape {edge_probs.shape}")

    # Stack and compute mean
    all_edge_probs_stacked = np.stack(all_edge_probs, axis=0)
    avg_edge_probs = np.mean(all_edge_probs_stacked, axis=0)

    logger.info(f"\nAverage edge probabilities computed:")
    logger.info(f"  Shape: {avg_edge_probs.shape}")
    logger.info(f"  Number of files averaged: {len(edge_prob_files)}")
    logger.info(f"  Mean across all edges and types: {np.mean(avg_edge_probs):.4f}")
    logger.info(f"  Std across all edges and types: {np.std(avg_edge_probs):.4f}")

    # Save averaged edge probabilities
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = edge_probs_dir / f"edges_averaged_with_forecast_{timestamp}.npy"
    np.save(output_path, avg_edge_probs)
    logger.info(f"\n✓ Saved average edge probabilities to {output_path}")

    return avg_edge_probs


def visualize_average_edge_probs(
    avg_edge_probs: np.ndarray,
    env: grid2op.Environment.Environment,
    obs_space: BusConnectivityGraphObsSpace,
    output_dir: Path,
    skip_last_edge_type: bool = True
):
    """
    Visualize the average edge probabilities.

    :param avg_edge_probs: Average edge probabilities
    :param env: Grid2Op environment
    :param obs_space: Observation space
    :param output_dir: Directory to save visualization
    :param skip_last_edge_type: Whether to skip the last edge type in visualization
    """
    logger.info("\nCreating visualizations for average edge probabilities...")

    # Get observation to extract edge index
    obs = env.reset()
    obs_gym = obs_space.to_gym(obs)
    edge_index = obs_gym[EDGE_INDEX]

    # Get node styles
    node_styles = get_node_styles(env, BusConnectivityGraphObsSpace)

    # Create plotting args
    plotting_args = PlottingArgs(
        num_nodes=obs_space.num_nodes,
        node_styles=node_styles,
        powerline_edge_index=edge_index,
        latent_edge_probs=avg_edge_probs,
        latent_edge_weight=5.0,
        skip_last_edge_type=skip_last_edge_type
    )

    # Create graph visualization
    logger.info("  Creating graph visualization...")
    fig_graph = visualize_graph(plotting_args)
    fig_graph.suptitle("Average Edge Probabilities (with forecast)", fontsize=14, fontweight='bold')

    # Save graph visualization
    output_path_graph = output_dir / "averaged_edge_probs_with_forecast_graph.png"
    fig_graph.savefig(output_path_graph, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved graph visualization to {output_path_graph}")
    plt.close(fig_graph)

    # Create histogram visualization
    logger.info("  Creating histogram visualization...")
    fig_hist = latent_edge_hist(avg_edge_probs, skip_last_edge_type=skip_last_edge_type)
    fig_hist.suptitle("Average Edge Probabilities Distribution (with forecast)", fontsize=14, fontweight='bold')

    # Save histogram visualization
    output_path_hist = output_dir / "averaged_edge_probs_with_forecast_histogram.png"
    fig_hist.savefig(output_path_hist, dpi=150, bbox_inches='tight')
    logger.info(f"  ✓ Saved histogram to {output_path_hist}")
    plt.close(fig_hist)


def main():
    """Main function to visualize all edge probability files with forecast."""
    # Setup paths
    project_root = Path(__file__).parent
    edge_probs_dir = project_root / "results" / "edge_probabilities"
    output_dir = edge_probs_dir / "visualizations_with_forecast"
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Edge probabilities directory: {edge_probs_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Get all edge probability files with forecast
    edge_prob_files = get_edge_prob_files_with_forecast(edge_probs_dir)

    if not edge_prob_files:
        logger.warning("No edge probability files with 'with_forecast' found!")
        return

    logger.info(f"\nFiles to process:")
    for f in edge_prob_files:
        logger.info(f"  - {f.name}")

    # Initialize environment and observation space
    logger.info("\nInitializing Grid2Op environment...")
    env = grid2op.make("l2rpn_case14_sandbox")
    obs_space = BusConnectivityGraphObsSpace(env.observation_space)
    logger.info(f"Environment initialized with {obs_space.num_nodes} nodes")

    # Process each file
    logger.info(f"\nProcessing {len(edge_prob_files)} files...")
    for i, edge_probs_path in enumerate(edge_prob_files, 1):
        logger.info(f"\n[{i}/{len(edge_prob_files)}] Processing {edge_probs_path.name}")
        try:
            visualize_edge_probs_file(
                edge_probs_path=edge_probs_path,
                env=env,
                obs_space=obs_space,
                output_dir=output_dir,
                skip_last_edge_type=True
            )
        except Exception as e:
            logger.error(f"  Error processing {edge_probs_path.name}: {e}", exc_info=True)

    # Compute and visualize average edge probabilities
    logger.info("\n" + "="*80)
    logger.info("COMPUTING AVERAGE EDGE PROBABILITIES")
    logger.info("="*80)

    try:
        avg_edge_probs = compute_average_edge_probs(edge_prob_files, edge_probs_dir)
        visualize_average_edge_probs(
            avg_edge_probs=avg_edge_probs,
            env=env,
            obs_space=obs_space,
            output_dir=output_dir,
            skip_last_edge_type=True
        )
    except Exception as e:
        logger.error(f"Error computing/visualizing average edge probabilities: {e}", exc_info=True)

    logger.info(f"\n" + "="*80)
    logger.info(f"✓ Done! All visualizations saved to {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

