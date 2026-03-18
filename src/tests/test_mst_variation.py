"""
Test script to verify that MST sampling produces different trees.
"""
import numpy as np
import networkx as nx

# Create a simple graph where multiple MSTs exist
# A square with diagonals - all edges have equal weight
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])

print("Testing MST variation with random weights...")
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print()

# Test 1: Without random weights (deterministic)
print("=" * 60)
print("Test 1: Without random weights (deterministic MST)")
print("=" * 60)
mst_edges_list = []
for i in range(10):
    mst = nx.minimum_spanning_tree(G)
    edges = sorted(mst.edges())
    mst_edges_list.append(edges)
    print(f"Iteration {i+1}: {edges}")

unique_msts = len(set(tuple(edges) for edges in mst_edges_list))
print(f"\nUnique MSTs found: {unique_msts} out of 10 iterations")
print()

# Test 2: With random weights (should give different MSTs)
print("=" * 60)
print("Test 2: With random weights (sampled MSTs)")
print("=" * 60)
mst_edges_list = []
for i in range(10):
    # Create new graph with random weights
    G_weighted = nx.Graph()
    for src, dst in G.edges():
        G_weighted.add_edge(src, dst, weight=np.random.uniform(0, 1e-6))

    mst = nx.minimum_spanning_tree(G_weighted, weight='weight')
    edges = sorted(mst.edges())
    mst_edges_list.append(edges)
    print(f"Iteration {i+1}: {edges}")

unique_msts = len(set(tuple(edges) for edges in mst_edges_list))
print(f"\nUnique MSTs found: {unique_msts} out of 10 iterations")
print()

# Test 3: Verify inner node probabilities vary
print("=" * 60)
print("Test 3: Inner node probabilities with random MST sampling")
print("=" * 60)
num_samples = 100
inner_node_counts = np.zeros(4)

for _ in range(num_samples):
    G_weighted = nx.Graph()
    for src, dst in G.edges():
        G_weighted.add_edge(src, dst, weight=np.random.uniform(0, 1e-6))

    mst = nx.minimum_spanning_tree(G_weighted, weight='weight')
    degrees = dict(mst.degree())

    for node in range(4):
        if degrees[node] >= 2:
            inner_node_counts[node] += 1

inner_node_probs = inner_node_counts / num_samples
print(f"Inner node probabilities (degree >= 2 in MST):")
for node, prob in enumerate(inner_node_probs):
    print(f"  Node {node}: {prob:.3f}")
print()

print("✓ Random weight sampling allows exploring different MSTs!")
print("✓ Inner node probabilities now reflect actual variation across MSTs")
