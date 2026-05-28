import unittest

import torch
from torch import Tensor
from torch_geometric.data import Data

from rarl.nn.encoder.cacher import GraphDataCache


class TestGraphDataCacher(unittest.TestCase):
    def setUp(self):
        self.data_retriever = GraphDataCache()
        self.edge_index = torch.Tensor([[0, 1], [1, 2], [2, 3], [1, 3]]).T.to(torch.long)
        self.graph_data = Data(num_nodes=5, edge_index=self.edge_index)

    def test_retrieve_graph_data(self):
        in_deg, out_deg, shortest_paths = self.data_retriever.get(self.graph_data)
        self.assertEqual(in_deg.shape, (5,))
        self.assertEqual(out_deg.shape, (5,))
        self.assertEqual(shortest_paths.shape, (1, 5, 5))
        torch.testing.assert_close(in_deg, Tensor([0, 1, 1, 2, 0]).to(dtype=torch.long))
        torch.testing.assert_close(out_deg, Tensor([1, 2, 1, 0, 0]).to(dtype=torch.long))
        torch.testing.assert_close(shortest_paths, Tensor([[
            [0, 1, 2, 2, 0],
            [1, 0, 1, 1, 0],
            [2, 1, 0, 1, 0],
            [2, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]]).to(dtype=torch.long))

    def test_retrieve_batched_graph_data(self):
        edge_index = torch.Tensor([[0, 1], [1, 0]]).T.to(torch.long)
        batch = torch.Tensor([0, 0, 1, 1]).long()
        graph_data = Data(num_nodes=4, edge_index=edge_index, batch=batch)
        in_deg, out_deg, shortest_paths = self.data_retriever.get(graph_data)
        self.assertEqual(in_deg.shape, (4, ))
        self.assertEqual(out_deg.shape, (4, ))
        self.assertEqual(shortest_paths.shape, (2, 2, 2))
        torch.testing.assert_close(in_deg, Tensor([1, 1, 0, 0]).to(dtype=torch.long))
        torch.testing.assert_close(out_deg, Tensor([1, 1, 0, 0]).to(dtype=torch.long))
        torch.testing.assert_close(shortest_paths, Tensor([
            [[0, 1], [1, 0]],
            [[0, 0], [0, 0]]
        ]).to(dtype=torch.long))
