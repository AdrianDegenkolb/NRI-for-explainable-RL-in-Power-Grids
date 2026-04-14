"""
Graph utility functions used across the RARL pipeline.
"""

from typing import Union

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse


def fully_connected_edge_index(
    num_nodes: int,
    device: Union[str, torch.device] = "cpu",
    self_loops: bool = False,
) -> Tensor:
    """
    Edge index for a fully-connected directed graph with *num_nodes* nodes.

    :param num_nodes: Number of nodes.
    :param device: Target device.
    :param self_loops: Include self-loops if True.
    :return: Edge index [2, E].
    """
    senders, receivers = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    edge_index = torch.stack([senders.flatten(), receivers.flatten()], dim=0)
    if not self_loops:
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return edge_index.to(device=device)


def fully_connected_edge_index_per_batch(
    batch: Tensor,
    device: Union[str, torch.device] = "cpu",
    self_loops: bool = False,
) -> Tensor:
    """
    Fully-connected edge index for a *batch* of graphs.

    :param batch: Graph-membership vector per node [N].
    :param device: Target device.
    :param self_loops: Include self-loops if True.
    :return: Edge index [2, sum_g E_g].
    """
    num_graphs = int(batch.max()) + 1
    edge_indices = []
    for g in range(num_graphs):
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        n = node_idx.numel()
        if n == 0:
            continue
        adj = torch.ones((n, n), dtype=torch.bool, device=device)
        if not self_loops:
            adj.fill_diagonal_(False)
        edge_index_local, _ = dense_to_sparse(adj)
        edge_indices.append(node_idx[edge_index_local])
    return torch.cat(edge_indices, dim=1)


def edge_membership_mask(
    super_edge_set: Tensor,
    sub_edge_set: Tensor,
) -> Tensor:
    """
    Boolean mask: which edges in *super_edge_set* also appear in *sub_edge_set*.

    :param super_edge_set: [2, E_super].
    :param sub_edge_set: [2, E_sub].
    :return: Bool mask [E_super].
    """
    num_nodes = super_edge_set.max() + 1
    a = super_edge_set[0] * num_nodes + super_edge_set[1]
    b = sub_edge_set[0] * num_nodes + sub_edge_set[1]
    return torch.isin(a, b)
