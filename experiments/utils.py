from typing import List

import networkx as nx
import numpy.typing as npt
import numpy as np


def sample_graph_from_posterior(posterior: npt.NDArray, all_edges: npt.NDArray, num_samples=1) -> List[nx.Graph]:
    """
    Sample a graph from the posterior distribution.

    Args:
        posterior: A factorized distribution over interaction types [E_max, K], where E_max is the number of node pairs and K is the number of interaction types.
        all_edges: An edge index array of shape [2, E_max], where each row corresponds to an edge (source, target).
        num_samples: The number of graph samples to draw.
    Returns:
        An list of graph objects. edges in the graphs are labeled with their interaction type (0 to K-2), where K-1 corresponds to "no edge".
    """
    assert posterior.shape[0] == all_edges.shape[1], "Posterior and all_edges must have the same number of edges"
    assert all_edges.shape[0] == 2, "all_edges must have shape [2, E_max]"

    E, K = posterior.shape
    sampled_graphs = []

    for _ in range(num_samples):
        graph = nx.DiGraph()
        graph.add_nodes_from(np.unique(all_edges.flatten()))

        # Sample edge types for each edge from the posterior distribution
        sampled_types = np.array([np.random.choice(K, p=posterior[i]) for i in range(E)])

        for k in range(K - 1):
            # Get edges that belong to interaction type k
            mask = sampled_types == k
            edges_for_type = all_edges[:, mask]
            for u, v in edges_for_type.T:
                graph.add_edge(int(u), int(v), interaction_type=k)

        sampled_graphs.append(graph)

    return sampled_graphs

def edge_index_to_graph(edge_index: npt.NDArray[np.int_]) -> nx.Graph:
    """
    Converts an edge index to a networkx graph.

    Args:
        edge_index: the edge index in the format [2, E], where the first row contains source nodes and the second row contains target nodes.
    Returns:
        a networkx graph with edges corresponding to the edge index.
    """
    assert edge_index.shape[0] == 2, "Edge index must have shape [2, E]"

    graph = nx.DiGraph()
    graph.add_nodes_from(np.unique(edge_index.flatten()))
    for u, v in edge_index.T:
        graph.add_edge(int(u), int(v))
    return graph

def expected_distance(posterior: npt.NDArray, all_edges: npt.NDArray, reference_graph: nx.Graph, n: int = 10) -> float:
    """
    Computes the expected distance between the posterior distribution and a reference graph by sampling a set of graphs from the posterior
    and averaging their distances to the reference graph.

    Args:
        posterior: A factorized distribution over interaction types [E_max, K], where E_max is the number of node pairs and K is the number of interaction types.
        all_edges: An edge index array of shape [2, E_max], where each row corresponds to an edge (source, target).
        reference_graph: The graph to which the distance is computed.
        n: The number of graph samples to draw from the posterior for estimating the expected distance.
    Returns:
        The expected distance between the posterior distribution and the reference graph.
    """
    assert posterior.shape[0] == all_edges.shape[1], "Posterior and all_edges must have the same number of edges"
    assert all_edges.shape[0] == 2, "all_edges must have shape [2, E_max]"

    # Ensure reference graph has all nodes
    sampled_graphs = sample_graph_from_posterior(posterior, all_edges, num_samples=n)
    distances = []
    for sampled_graph in sampled_graphs:
        A_sample = nx.to_numpy_array(sampled_graph, dtype=int)
        A_ref = nx.to_numpy_array(reference_graph, dtype=int)
        distances.append(np.not_equal(A_ref, A_sample).sum())

    return float(np.mean(distances))
