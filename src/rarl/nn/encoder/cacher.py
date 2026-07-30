"""Cache per-graph structural properties (degrees, path lengths) by edge_index hash."""

import time
from typing import Tuple, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data

from .functional import get_in_out_degree, precalculate_paths


class GraphDataCache:
    """
    Caches node degrees and pairwise shortest-path lengths keyed by graph
    structure (edge_index content).  Avoids recomputation when the same
    graph topology appears in multiple batch elements.
    """

    def __init__(self):
        self._cache: Dict[tuple, Tuple[Tensor, Tensor, Tensor]] = {}
        self._timings: dict[str, float] = {"graph_data_compute_ms": 0.0}

    def get(self, graph_data: Data) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return ``(in_degree, out_degree, path_lengths)`` for the (batched) graph.

        Processes each graph in the batch separately and concatenates results.
        """
        self._timings["graph_data_compute_ms"] = 0.0
        batch = (
            graph_data.batch
            if graph_data.batch is not None
            else torch.zeros(graph_data.num_nodes, dtype=torch.long)
        )
        edge_index = graph_data.edge_index
        edge_batch = batch[edge_index[0]]
        in_degs, out_degs, path_lens = [], [], []
        offset = 0

        for b in range(int(batch.max()) + 1):
            mask = edge_batch == b
            ei_local = edge_index[:, mask] - offset
            num_nodes = int((batch == b).sum())
            sub = Data(num_nodes=num_nodes, edge_index=ei_local)
            key = self._key(sub)

            if key not in self._cache:
                t0 = time.perf_counter()
                in_d, out_d = get_in_out_degree(sub)
                paths = precalculate_paths(sub)
                self._timings["graph_data_compute_ms"] += (time.perf_counter() - t0) * 1000
                self._cache[key] = (
                    in_d.to(edge_index.device),
                    out_d.to(edge_index.device),
                    paths.to(edge_index.device),
                )

            in_d, out_d, paths = self._cache[key]
            in_degs.append(in_d)
            out_degs.append(out_d)
            path_lens.append(paths)
            offset += num_nodes

        return (
            torch.cat(in_degs),
            torch.cat(out_degs),
            torch.stack(path_lens),
        )

    @staticmethod
    def _key(data: Data) -> tuple:
        ei = data.edge_index
        return (ei.dtype, tuple(ei.shape), ei.cpu().numpy().tobytes())
