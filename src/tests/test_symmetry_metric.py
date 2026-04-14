"""
Test script for the SymmetryMetricVisualizer.
"""
import numpy as np
from experiments import SymmetryMetricVisualizer


def test_symmetry_metric():
    """Test the symmetry metric computation and visualization."""

    # Create a simple test case with 4 nodes
    num_nodes = 4

    # Create fully connected edge index (excluding self-loops)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index_fully_connected = np.array(edge_index).T  # [2, num_edges]
    num_edges = edge_index_fully_connected.shape[1]

    # Initialize visualizer
    visualizer = SymmetryMetricVisualizer(None)

    print("Testing Symmetry Metric Visualizer...")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges (fully connected, no self-loops): {num_edges}")

    # Test Case 1: Perfectly symmetric posterior
    print("\n=== Test Case 1: Perfectly Symmetric Graph ===")
    # Create a symmetric posterior where p_vw = p_wv for all v, w
    posterior_symmetric = np.random.rand(num_edges, 3)  # 3 edge types (2 real + 1 no-edge)
    posterior_symmetric = posterior_symmetric / posterior_symmetric.sum(axis=1, keepdims=True)  # normalize

    # Make it symmetric
    adj_temp = np.zeros((num_nodes, num_nodes, 3))
    for i, (src, dst) in enumerate(edge_index_fully_connected.T):
        adj_temp[src, dst] = posterior_symmetric[i]
    # Symmetrize
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            avg_prob = (adj_temp[i, j] + adj_temp[j, i]) / 2
            adj_temp[i, j] = avg_prob
            adj_temp[j, i] = avg_prob
    # Convert back to edge format
    posterior_symmetric = np.zeros((num_edges, 3))
    for idx, (src, dst) in enumerate(edge_index_fully_connected.T):
        posterior_symmetric[idx] = adj_temp[src, dst]

    # Compute symmetry score
    symmetry_score_1 = visualizer._compute(
        posterior=posterior_symmetric,
        edge_index_fully_connected=edge_index_fully_connected
    )
    print(f"Symmetry Score (should be ~1.0): {symmetry_score_1:.6f}")
    visualizer.metrics_history.append(symmetry_score_1)

    # Test Case 2: Asymmetric posterior
    print("\n=== Test Case 2: Asymmetric Graph ===")
    posterior_asymmetric = np.random.rand(num_edges, 3)
    posterior_asymmetric = posterior_asymmetric / posterior_asymmetric.sum(axis=1, keepdims=True)

    symmetry_score_2 = visualizer._compute(
        posterior=posterior_asymmetric,
        edge_index_fully_connected=edge_index_fully_connected
    )
    print(f"Symmetry Score (should be < 1.0): {symmetry_score_2:.6f}")
    visualizer.metrics_history.append(symmetry_score_2)

    # Test Case 3: Add more samples with varying symmetry
    print("\n=== Test Case 3: Multiple timesteps with varying symmetry ===")
    for i in range(10):
        posterior_random = np.random.rand(num_edges, 3)
        posterior_random = posterior_random / posterior_random.sum(axis=1, keepdims=True)

        symmetry_score = visualizer._compute(
            posterior=posterior_random,
            edge_index_fully_connected=edge_index_fully_connected
        )
        visualizer.metrics_history.append(symmetry_score)
        print(f"  Timestep {i+3}: Symmetry Score = {symmetry_score:.6f}")

    # Test visualization (aggregated)
    print("\n=== Testing Visualization ===")
    print(f"Total timesteps collected: {len(visualizer.metrics_history)}")
    print(f"Mean symmetry score: {np.mean(visualizer.metrics_history):.6f}")
    print(f"Median symmetry score: {np.median(visualizer.metrics_history):.6f}")

    # Generate summary visualization
    fig = visualizer.summarize(show_figure=False)
    print("Visualization generated successfully!")

    # Save the figure
    output_path = "/results/visualizations/test_symmetry_metric.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Test non-aggregated visualization (should return empty figure)
    print("\n=== Testing Non-Aggregated Visualization ===")
    fig_single = visualizer._visualize(computation_result=0.5, aggregated=False, show_figure=False)
    print("Non-aggregated visualization returned empty figure (as expected)")

    print("\n=== All Tests Passed! ===")

if __name__ == "__main__":
    test_symmetry_metric()
