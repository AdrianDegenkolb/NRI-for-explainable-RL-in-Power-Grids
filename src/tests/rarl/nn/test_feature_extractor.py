import unittest

import numpy as np
import torch

from rarl.nn.feature_extractor import RAFeatureExtractor


class TestRAFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for NRIFeatureExtractor
        self.x_dim = 5
        self.hidden_dim = 10
        self.x_out_dim = 3
        self.num_edge_types = 4
        self.dropout_prob = 0.2
        self.num_attention_heads = 2

        self.model = RAFeatureExtractor(
            x_dim=self.x_dim,
            graph_max_degree=5,
            graph_max_path_distance=5,
            hidden_dim_enc=self.hidden_dim,
            hidden_dim_gnn=self.hidden_dim,
            x_out_dim=self.x_out_dim,
            num_edge_types=self.num_edge_types,
            num_layers_gnn=3,
            num_layers_enc=2,
            num_attention_heads_enc=self.num_attention_heads,
            dropout_prob=self.dropout_prob,
        )

        # Dummy data for testing
        self.batch_size = 2
        self.num_nodes_per_batch = 3
        self.node_features = torch.rand(self.num_nodes_per_batch * self.batch_size, self.x_dim)
        self.batch = torch.tensor(np.array([[i] * self.num_nodes_per_batch for i in range(self.batch_size)]).flatten())
        self.num_fc_edges_per_batch = self.num_nodes_per_batch * (self.num_nodes_per_batch - 1)
        # Simple path-graph edges per batch element (required by the Graphormer encoder for
        # structural encodings; passing None would leave edge_index=None inside the cacher).
        # Batch 0: nodes 0-1-2,  Batch 1: nodes 3-4-5
        self.powerline_edge_index = torch.tensor(
            [[0, 1, 3, 4],
             [1, 2, 4, 5]], dtype=torch.long
        )

    def test_forward_shape(self):
        predictions, p_z_given_x = self.model(
            self.node_features, batch=self.batch,
            powerline_edge_index=self.powerline_edge_index,
        )
        # Check output shapes
        self.assertEqual(predictions.shape, (self.batch_size, self.x_out_dim))
        self.assertEqual(p_z_given_x.shape, (self.batch_size, self.num_fc_edges_per_batch, self.num_edge_types))

    def test_forward_values(self):
        predictions, p_z_given_x = self.model(
            x=self.node_features, batch=self.batch,
            powerline_edge_index=self.powerline_edge_index,
        )
        # Check that the softmax outputs are valid probabilities
        self.assertTrue(torch.all(p_z_given_x >= 0) and torch.all(p_z_given_x <= 1), "Probabilities are out of bounds")
        self.assertTrue(torch.allclose(p_z_given_x.sum(dim=-1), torch.ones(self.batch_size, p_z_given_x.size(1))), "Softmax probabilities do not sum to 1")

    def test_inference(self):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions, p_z_given_x = self.model(
                self.node_features, batch=self.batch,
                powerline_edge_index=self.powerline_edge_index,
            )
            # Check that predictions are valid after inference
            self.assertIsNotNone(predictions, "Predictions should not be None")
            self.assertIsNotNone(p_z_given_x, "Posterior probabilities should not be None")
