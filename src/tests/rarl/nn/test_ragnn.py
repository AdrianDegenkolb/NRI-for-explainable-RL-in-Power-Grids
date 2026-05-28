import unittest
import torch
from torch_geometric.data import Data, Batch
from rarl.nn.ragnn import RAGNN


class TestRAGNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # two toy graphs for batching
        edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_index2 = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
        x1 = torch.randn(2, 4)
        x2 = torch.randn(3, 4)
        data_list = [Data(x=x1, edge_index=edge_index1),
                     Data(x=x2, edge_index=edge_index2)]
        batch = Batch.from_data_list(data_list)
        self.x = batch.x
        self.edge_index = batch.edge_index
        self.batch = batch.batch
        self.E = self.edge_index.size(1)
        self.edge_type_posterior = torch.softmax(torch.randn(self.E, 2), dim=-1)

    def test_nri_informed_gnn(self):
        model = RAGNN(
            x_dim=4,
            hidden_dim=8,
            x_out_dim=5,
            num_layers=2,
            num_edge_types=2,
            skip_last=True,
            dropout_prob=0.1,
            residual=True,
        )
        out = model(self.x, self.edge_index, self.edge_type_posterior, self.batch)
        # output should be [B, x_out_dim] = [2, 5]
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_without_residual(self):
        model = RAGNN(
            x_dim=4,
            hidden_dim=8,
            x_out_dim=5,
            num_layers=2,
            num_edge_types=2,
            skip_last=True,
            dropout_prob=0.0,
            residual=False,
        )
        out = model(self.x, self.edge_index, self.edge_type_posterior, self.batch)
        self.assertEqual(out.shape, (2, 5))

    def test_invalid_edge_weight_shape(self):
        model = RAGNN(4, 8, 5)
        bad_edge_weights = torch.randn(self.E + 1, 2)
        with self.assertRaises(AssertionError):
            _ = model(self.x, self.edge_index, bad_edge_weights, self.batch)
