"""Graph utility functions for the Graphormer encoder (path distances, degrees)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as scipy_shortest_path


def get_in_out_degree(data: Data) -> Tuple[Tensor, Tensor]:
    n = data.num_nodes
    ei = data.edge_index
    return degree(ei[1], num_nodes=n).long(), degree(ei[0], num_nodes=n).long()


def precalculate_paths(data: Data) -> Tensor:
    """Return [N, N] tensor of shortest-path lengths (hop count).

    Uses scipy's Dijkstra (C implementation) instead of Python-level BFS, giving
    ~100-1000× speedup for large grids (N=177 for IEEE36, N=532 for IEEE118).
    Unreachable node pairs get length 0 (treated as "no path" by SpatialEncoding).
    """
    N = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    # Undirected: add both directions so shortest_path treats the graph as undirected
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    adj = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))
    dist = scipy_shortest_path(adj, method="D", directed=False, unweighted=True)
    # isinf means unreachable → set to 0 (SpatialEncoding skips 0-length entries)
    dist = np.where(np.isinf(dist), 0, dist).astype(np.int64)
    return torch.from_numpy(dist)
