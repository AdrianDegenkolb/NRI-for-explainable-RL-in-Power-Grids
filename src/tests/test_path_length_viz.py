"""
Test the updated PathLengthVisualizer with 2x3 grid layout.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from experiments import PathLengthVisualizer
from src.visualization.utils import NodeStyle

# Create test data
num_nodes = 20
num_edges = num_nodes * (num_nodes - 1)

# Create node styles
node_styles = []
for i in range(num_nodes):
    pos = np.array([np.cos(2 * np.pi * i / num_nodes), np.sin(2 * np.pi * i / num_nodes)])
    color = 'blue' if i % 2 == 0 else 'red'
    shape = 'o'
    size = 50
    label = f'Node {i}'
    node_styles.append(NodeStyle(position=pos, color=color, shape=shape, size=size, label=label))

# Create posterior
posterior = np.random.rand(num_edges, 2)
posterior = posterior / posterior.sum(axis=1, keepdims=True)

# Create powergrid graph (ring + some connections)
powergrid_edges = []
for i in range(num_nodes - 1):
    powergrid_edges.append([i, i + 1])
powergrid_edges.append([num_nodes - 1, 0])
for i in range(0, num_nodes, 4):
    powergrid_edges.append([i, (i + 2) % num_nodes])
powergrid_graph = np.array(powergrid_edges).T

# Create fully connected edge index
edge_index_fully_connected = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            edge_index_fully_connected.append([i, j])
edge_index_fully_connected = np.array(edge_index_fully_connected).T

# Create node mask (exclude a few nodes)
node_mask = np.ones(num_nodes, dtype=bool)
node_mask[[0, 5, 10]] = False

# Create samples from posterior
samples = []
for _ in range(50):
    sample_edges = []
    for i in range(num_edges):
        if np.random.rand() < posterior[i, 0]:
            src, dst = edge_index_fully_connected[:, i]
            sample_edges.append([src, dst])
    if sample_edges:
        samples.append(np.array(sample_edges).T)
    else:
        samples.append(np.zeros((2, 0)))

print("Testing PathLengthVisualizer...")
print(f"  Nodes: {num_nodes}")
print(f"  Powergrid edges: {powergrid_graph.shape[1]}")
print(f"  Masked nodes: {node_mask.sum()}")
print(f"  Samples: {len(samples)}")

# Create visualizer
viz = PathLengthVisualizer(node_styles=node_styles)

# Test computation
try:
    fig, data = viz(
        posterior=posterior,
        prior=None,
        samples=samples,
        powergrid_graph=powergrid_graph,
        edge_index_fully_connected=edge_index_fully_connected,
        node_mask=node_mask,
        observation=None,
        show_figure=False,
    )

    print("\n✓ PathLengthVisualizer works!")
    print(f"  Data keys: {data.keys()}")
    print(f"  Latent full paths: {len(data['latent_full'])} samples")
    print(f"  Latent subgraph paths: {len(data['latent_subgraph'])} samples")
    print(f"  Powergrid full paths: {len(data['powergrid_full'])} samples")
    print(f"  Powergrid subgraph paths: {len(data['powergrid_subgraph'])} samples")

    if len(data['latent_full']) > 0:
        print(f"\n  Latent full - Mean path length: {np.mean(data['latent_full']):.2f}")
    if len(data['latent_subgraph']) > 0:
        print(f"  Latent subgraph - Mean path length: {np.mean(data['latent_subgraph']):.2f}")
    if len(data['powergrid_full']) > 0:
        print(f"  Powergrid full - Mean path length: {np.mean(data['powergrid_full']):.2f}")
    if len(data['powergrid_subgraph']) > 0:
        print(f"  Powergrid subgraph - Mean path length: {np.mean(data['powergrid_subgraph']):.2f}")

    # Save figure
    output_path = Path("/tmp/test_path_length_viz.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close(fig)

    # Check for .npy files
    npy_dir = Path("/tmp/test_path_length/npy_data")
    if npy_dir.exists():
        npy_files = list(npy_dir.glob("*.npy"))
        print(f"✓ Created {len(npy_files)} .npy files:")
        for f in npy_files:
            arr = np.load(f)
            print(f"  - {f.name}: {len(arr)} values")

    print("\n✓ Test completed successfully!")

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
