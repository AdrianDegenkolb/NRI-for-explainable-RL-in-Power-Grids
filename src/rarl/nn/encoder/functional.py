"""Graph utility functions for the Graphormer encoder (path distances, degrees)."""

from __future__ import annotations

from typing import Tuple, Dict, List

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import to_networkx


def _bfs_paths(G: nx.Graph, source: int, cutoff=None):
    """Single-source BFS returning node paths and edge paths."""
    edges = {edge: i for i, edge in enumerate(G.edges())}
    next_level = {source: 1}
    node_paths: Dict[int, List[int]] = {source: [source]}
    edge_paths: Dict[int, List[int]] = {source: []}
    level = 0
    while next_level:
        this_level = next_level
        next_level = {}
        for v in this_level:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    next_level[w] = 1
        level += 1
        if cutoff is not None and cutoff <= level:
            break
    return node_paths, edge_paths


def all_pairs_shortest_path(
    G: nx.Graph,
) -> Tuple[Dict[int, Dict[int, List[int]]], Dict[int, Dict[int, List[int]]]]:
    paths = {n: _bfs_paths(G, n) for n in G}
    return {n: paths[n][0] for n in paths}, {n: paths[n][1] for n in paths}


def shortest_path_distance(data: Data):
    G = to_networkx(data)
    return all_pairs_shortest_path(G)


def get_in_out_degree(data: Data) -> Tuple[Tensor, Tensor]:
    n = data.num_nodes
    ei = data.edge_index
    return degree(ei[1], num_nodes=n).long(), degree(ei[0], num_nodes=n).long()


def precalculate_paths(data: Data) -> Tensor:
    """Return [N, N] tensor of shortest-path lengths (in number of nodes)."""
    node_paths, _ = shortest_path_distance(data)
    N = data.num_nodes
    lengths = torch.zeros((N, N), dtype=torch.long)
    for src, dsts in node_paths.items():
        for dst, path in dsts.items():
            lengths[src, dst] = len(path)
    return lengths
