import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy

from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES
from rarl_rllib.ppo.rappo_policy import RAPPOTorchPolicy


class TestRAPPOTorchPolicy(unittest.TestCase):
    """Test suite for RAPPOTorchPolicy loss function with different num_edge_types."""

    def setUp(self):
        """Set up common test fixtures."""
        self.batch_size = 4
        self.num_nodes = 5
        self.num_graph_edges = 8
        self.max_edges = 10

        # Device setup
        self.device = torch.device("cpu")

    def _create_mock_policy(self, num_edge_types=2):
        """Create a mock RAPPOTorchPolicy with necessary attributes."""
        policy = Mock(spec=RAPPOTorchPolicy)
        policy.device = self.device

        # Observation space must be a real Dict so common.py can subscript it
        x_dim = 5
        policy.observation_space = spaces.Dict({
            NODES: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_nodes, x_dim), dtype=np.float32,
            ),
            EDGE_INDEX: spaces.Box(
                low=0, high=self.num_nodes - 1,
                shape=(2, self.max_edges), dtype=np.int64,
            ),
            EDGE_MASK: spaces.Box(
                low=0, high=1,
                shape=(self.max_edges,), dtype=bool,
            ),
        })

        # Mock config (matches current relation_awareness YAML structure)
        policy.config = {
            "relation_awareness": {
                "prior": {
                    "prior_prob_for_graph_edge": 0.8,
                    "temperature": 0.2,
                },
                "loss": {
                    "beta_graph_edges": 0.5,
                    "beta_non_graph_edges": 0.5,
                },
                "sampling": {
                    "tau": 1.0,
                },
                "latent_space": {
                    "num_edge_types": num_edge_types,
                },
            },
        }
        # Attributes set by init_ra_config at runtime
        policy.current_beta_graph = 0.5
        policy.current_beta_non_graph = 0.5
        policy.current_tau = 1.0
        policy.num_edge_types = num_edge_types
        policy.prior_prob_for_graph_edge = 0.8
        policy.temperature = 0.2

        return policy

    def _create_sample_batch(self):
        """Create a sample batch with proper structure."""
        # Create edge indices: [batch_size, 2, max_edges]
        edge_index_batch = torch.zeros((self.batch_size, 2, self.max_edges), dtype=torch.long)
        edge_mask = torch.zeros((self.batch_size, self.max_edges), dtype=torch.bool)

        # Fill in some valid edges for each batch
        for b in range(self.batch_size):
            # Create random edges within the node range
            num_edges_this_batch = min(self.num_graph_edges + np.random.randint(-2, 3), self.max_edges)
            num_edges_this_batch = max(1, num_edges_this_batch)  # At least 1 edge

            for e in range(num_edges_this_batch):
                # Random edges between nodes
                src = np.random.randint(0, self.num_nodes)
                dst = np.random.randint(0, self.num_nodes)
                while src == dst:  # Avoid self-loops
                    dst = np.random.randint(0, self.num_nodes)

                edge_index_batch[b, 0, e] = src
                edge_index_batch[b, 1, e] = dst
                edge_mask[b, e] = True

        # Create sample batch dict
        sample_batch = {
            "obs": {
                EDGE_INDEX: edge_index_batch,
                EDGE_MASK: edge_mask,
            }
        }

        return SampleBatch(sample_batch)

    def _create_mock_model(self, num_edge_types=2):
        """Create a mock model with get_posterior method."""
        model = Mock()
        model.tower_stats = {}
        # store_ra_tower_stats accesses model.ragnn.gnn.stats; must be iterable
        model.ragnn = Mock()
        model.ragnn.gnn = Mock()
        model.ragnn.gnn.stats = {}

        # Calculate number of edges in fully connected graph (no self-loops)
        num_fc_edges = self.num_nodes * (self.num_nodes - 1)

        # Create mock posterior: [batch_size, num_edges, num_edge_types]
        # Posteriors should be valid probability distributions (sum to 1)
        posterior_logits = torch.randn((self.batch_size, num_fc_edges, num_edge_types))
        posteriors = torch.softmax(posterior_logits, dim=-1)

        model.get_posterior = Mock(return_value=posteriors)
        model.model_gpu_towers = [model]  # For stats_fn

        return model

    def test_loss_with_2_edge_types(self):
        """Test loss computation with num_edge_types=2."""
        num_edge_types = 2

        # Create mocks
        policy = self._create_mock_policy(num_edge_types=num_edge_types)
        model = self._create_mock_model(num_edge_types=num_edge_types)
        train_batch = self._create_sample_batch()

        # Mock the parent loss - create a tensor that can be modified in-place
        base_loss = torch.tensor(1.5)

        # Use super() mock to return base_loss and bind the method
        with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
            # Call the actual loss function by binding it to the policy instance
            bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
            total_loss = bound_loss(model, None, train_batch)

        # Verify that loss was computed
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss.item() >= 0)

        # Verify model.get_posterior was called
        model.get_posterior.assert_called_once()

        # Verify tower_stats were set (keys use "ra_" prefix in common.py)
        self.assertIn("ra_kl_loss", model.tower_stats)
        self.assertIn("ra_mean_prior", model.tower_stats)
        self.assertIn("ra_mean_posterior", model.tower_stats)

        # Check shapes
        kl_loss = model.tower_stats["ra_kl_loss"]
        self.assertIsInstance(kl_loss, torch.Tensor)
        self.assertEqual(kl_loss.dim(), 0)  # Should be scalar

        # Check prior and posterior shapes: [num_edges, num_edge_types]
        mean_prior = model.tower_stats["ra_mean_prior"]
        mean_posterior = model.tower_stats["ra_mean_posterior"]
        num_fc_edges = self.num_nodes * (self.num_nodes - 1)

        self.assertEqual(mean_prior.shape, (num_fc_edges, num_edge_types))
        self.assertEqual(mean_posterior.shape, (num_fc_edges, num_edge_types))

        # Verify probabilities are valid (between 0 and 1, sum to 1)
        self.assertTrue(torch.all(mean_prior >= 0))
        self.assertTrue(torch.all(mean_prior <= 1))
        self.assertTrue(torch.all(mean_posterior >= 0))
        self.assertTrue(torch.all(mean_posterior <= 1))

        # Check that priors sum to approximately 1 for each edge
        prior_sums = mean_prior.sum(dim=-1)
        self.assertTrue(torch.allclose(prior_sums, torch.ones_like(prior_sums), atol=1e-5))

    def test_loss_with_3_edge_types(self):
        """Test loss computation with num_edge_types=3."""
        num_edge_types = 3

        # Create mocks
        policy = self._create_mock_policy(num_edge_types=num_edge_types)
        model = self._create_mock_model(num_edge_types=num_edge_types)
        train_batch = self._create_sample_batch()

        # Mock the parent loss
        base_loss = torch.tensor(2.0)

        with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
            bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
            total_loss = bound_loss(model, None, train_batch)

        # Verify that loss was computed
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss.item() >= 0)

        # Verify model.get_posterior was called
        model.get_posterior.assert_called_once()

        # Verify tower_stats were set (keys use "ra_" prefix in common.py)
        self.assertIn("ra_kl_loss", model.tower_stats)
        self.assertIn("ra_mean_prior", model.tower_stats)
        self.assertIn("ra_mean_posterior", model.tower_stats)

        # Check shapes with 3 edge types
        num_fc_edges = self.num_nodes * (self.num_nodes - 1)
        mean_prior = model.tower_stats["ra_mean_prior"]
        mean_posterior = model.tower_stats["ra_mean_posterior"]

        self.assertEqual(mean_prior.shape, (num_fc_edges, num_edge_types))
        self.assertEqual(mean_posterior.shape, (num_fc_edges, num_edge_types))

        # Verify probabilities sum to 1
        prior_sums = mean_prior.sum(dim=-1)
        self.assertTrue(torch.allclose(prior_sums, torch.ones_like(prior_sums), atol=1e-5))

    def test_kl_loss_contribution(self):
        """Test that KL loss is properly added to total loss."""
        num_edge_types = 2

        # Create mocks
        policy = self._create_mock_policy(num_edge_types=num_edge_types)
        model = self._create_mock_model(num_edge_types=num_edge_types)
        train_batch = self._create_sample_batch()

        base_loss_value = 1.0
        base_loss = torch.tensor(base_loss_value)

        with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
            bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
            total_loss = bound_loss(model, None, train_batch)

        # Total loss is base_loss + kl_loss; beta is already factored inside compute_ra_kl_loss
        kl_loss = model.tower_stats["ra_kl_loss"]
        expected_total = base_loss_value + kl_loss.item()

        self.assertAlmostEqual(total_loss.item(), expected_total, places=5)

    def test_kl_loss_non_negative(self):
        """Test that KL divergence is non-negative."""
        num_edge_types = 2

        # Create mocks
        policy = self._create_mock_policy(num_edge_types=num_edge_types)
        model = self._create_mock_model(num_edge_types=num_edge_types)
        train_batch = self._create_sample_batch()

        base_loss = torch.tensor(1.0)

        with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
            bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
            bound_loss(model, None, train_batch)

        kl_loss = model.tower_stats["ra_kl_loss"]

        # KL divergence should be non-negative
        self.assertGreaterEqual(kl_loss.item(), 0.0)

    def test_different_batch_sizes(self):
        """Test that the loss works with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            with self.subTest(batch_size=batch_size):
                self.batch_size = batch_size
                num_edge_types = 2

                policy = self._create_mock_policy(num_edge_types=num_edge_types)
                model = self._create_mock_model(num_edge_types=num_edge_types)
                train_batch = self._create_sample_batch()

                base_loss = torch.tensor(1.0)

                with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
                    bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
                    total_loss = bound_loss(model, None, train_batch)

                self.assertIsInstance(total_loss, torch.Tensor)
                self.assertTrue(torch.isfinite(total_loss))

    def test_stats_fn(self):
        """Test the stats_fn method."""
        num_edge_types = 2

        # Create mocks
        policy = self._create_mock_policy(num_edge_types=num_edge_types)
        model = self._create_mock_model(num_edge_types=num_edge_types)
        train_batch = self._create_sample_batch()

        # First call loss to populate tower_stats
        base_loss = torch.tensor(1.0)
        with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
            bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
            bound_loss(model, None, train_batch)

        # Mock parent stats_fn
        base_stats = {"policy_loss": 0.5, "vf_loss": 0.3}

        # Setup for stats_fn
        policy.model_gpu_towers = [model]

        with patch.object(PPOTorchPolicy, 'stats_fn', return_value=base_stats):
            stats = RAPPOTorchPolicy.stats_fn(policy, train_batch)

        # Verify base stats are still there
        self.assertIn("policy_loss", stats)
        self.assertIn("vf_loss", stats)

    def test_prior_posterior_shapes_consistency(self):
        """Test that prior and posterior have consistent shapes across different edge types."""
        for num_edge_types in [2, 3]:
            with self.subTest(num_edge_types=num_edge_types):
                policy = self._create_mock_policy(num_edge_types=num_edge_types)
                model = self._create_mock_model(num_edge_types=num_edge_types)
                train_batch = self._create_sample_batch()

                base_loss = torch.tensor(1.0)

                with patch.object(PPOTorchPolicy, 'loss', return_value=base_loss):
                    bound_loss = RAPPOTorchPolicy.loss.__get__(policy, RAPPOTorchPolicy)
                    bound_loss(model, None, train_batch)

                mean_prior = model.tower_stats["ra_mean_prior"]
                mean_posterior = model.tower_stats["ra_mean_posterior"]

                # Shapes should match
                self.assertEqual(mean_prior.shape, mean_posterior.shape)

                # Last dimension should be num_edge_types
                self.assertEqual(mean_prior.shape[-1], num_edge_types)
                self.assertEqual(mean_posterior.shape[-1], num_edge_types)


if __name__ == "__main__":
    unittest.main()
