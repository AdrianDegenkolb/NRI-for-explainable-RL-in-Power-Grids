import unittest
import torch
from torch_geometric.data import Data, Batch
from rarl.nn.ragnn import RAGNN, BaselineGNN


def _make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two toy graphs batched together. Returns (x, edge_index, batch)."""
    torch.manual_seed(0)
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_index2 = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(3, 4)
    batch = Batch.from_data_list([
        Data(x=x1, edge_index=edge_index1),
        Data(x=x2, edge_index=edge_index2),
    ])
    return batch.x, batch.edge_index, batch.batch


class TestRAGNN(unittest.TestCase):
    def setUp(self):
        self.x, self.edge_index, self.batch = _make_batch()
        self.E = self.edge_index.size(1)
        self.edge_type_posterior = torch.softmax(torch.randn(self.E, 2), dim=-1)

    def test_nri_informed_gnn(self):
        model = RAGNN(
            x_dim=4, hidden_dim=8, x_out_dim=5,
            num_layers=2, num_edge_types=2, skip_last=True,
            dropout_prob=0.1, residual=True,
        )
        out = model(self.x, self.edge_index, self.edge_type_posterior, self.batch)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_without_residual(self):
        model = RAGNN(
            x_dim=4, hidden_dim=8, x_out_dim=5,
            num_layers=2, num_edge_types=2, skip_last=True,
            dropout_prob=0.0, residual=False,
        )
        out = model(self.x, self.edge_index, self.edge_type_posterior, self.batch)
        self.assertEqual(out.shape, (2, 5))

    def test_invalid_edge_weight_shape(self):
        model = RAGNN(4, 8, 5)
        bad_edge_weights = torch.randn(self.E + 1, 2)
        with self.assertRaises(AssertionError):
            _ = model(self.x, self.edge_index, bad_edge_weights, self.batch)


class TestBaselineGNNComparisonOptions(unittest.TestCase):
    """GCN edge-weight positivity and the per-substation readout."""

    def setUp(self):
        self.x, self.edge_index, self.batch = _make_batch()
        self.E = self.edge_index.size(1)

    def test_gcn_edge_weights_from_attributes_are_positive(self):
        """Even edge attributes that drive the projection negative give no NaN (softplus weights)."""
        torch.manual_seed(0)
        model = BaselineGNN(x_dim=4, hidden_dim=8, x_out_dim=5, num_layers=2, edge_dim=1, conv_type="gcn")
        with torch.no_grad():
            for layer in (model.edge_proj.fc1, model.edge_proj.fc2):
                layer.weight.fill_(-5.0)
                layer.bias.fill_(-5.0)
        edge_attr = torch.full((self.E, 1), 10.0)
        weights = torch.nn.functional.softplus(model.edge_proj(edge_attr))
        self.assertTrue((weights > 0).all())
        out = model(self.x, self.edge_index, self.batch, edge_attr=edge_attr)
        self.assertFalse(torch.isnan(out).any())

    def test_substation_readout_shape(self):
        model = BaselineGNN(x_dim=4, hidden_dim=8, x_out_dim=5, num_layers=2, readout="substation", n_sub=3)
        subs = torch.tensor([[0, -1], [1, -1], [0, 2], [1, -1], [2, -1]])
        out = model(self.x, self.edge_index, self.batch, node_substations=subs, num_graphs=2)
        self.assertEqual(out.shape, (2, 3 * 5))

    def test_substation_readout_needs_n_sub(self):
        with self.assertRaises(ValueError):
            BaselineGNN(x_dim=4, hidden_dim=8, x_out_dim=5, readout="substation")


class TestBaselineGNN(unittest.TestCase):
    def setUp(self):
        self.x, self.edge_index, self.batch = _make_batch()
        self.E = self.edge_index.size(1)

    def _make_model(self, num_edge_types: int = 1, **kwargs) -> BaselineGNN:
        return BaselineGNN(x_dim=4, hidden_dim=8, x_out_dim=5,
                           num_layers=2, num_edge_types=num_edge_types, **kwargs)

    # --- output shape / numerical sanity ---

    def test_output_shape_no_edge_types(self):
        """Default call (no edge_types, no edge_weights) produces [B, x_out_dim]."""
        model = self._make_model()
        out = model(self.x, self.edge_index, self.batch)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_output_shape_with_edge_types(self):
        """Typed edges are routed to separate conv layers without error."""
        model = self._make_model(num_edge_types=3)
        edge_types = torch.randint(0, 3, (self.E,))
        out = model(self.x, self.edge_index, self.batch, edge_types=edge_types)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_output_shape_with_edge_weights(self):
        """Scalar edge weights are forwarded to GCNConv without error."""
        model = self._make_model()
        edge_weights = torch.rand(self.E)
        out = model(self.x, self.edge_index, self.batch, edge_weights=edge_weights)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())

    def test_output_shape_with_edge_types_and_weights(self):
        """Both edge_types and edge_weights provided simultaneously."""
        model = self._make_model(num_edge_types=2)
        edge_types = torch.randint(0, 2, (self.E,))
        edge_weights = torch.rand(self.E)
        out = model(self.x, self.edge_index, self.batch,
                    edge_types=edge_types, edge_weights=edge_weights)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())

    # --- gradient flow ---

    def _assert_valid_gradients(self, model: BaselineGNN) -> None:
        """All non-None grads are finite; at least one parameter received a grad."""
        grads = {name: param.grad for name, param in model.named_parameters()}
        active = {n: g for n, g in grads.items() if g is not None}
        self.assertGreater(len(active), 0, "No parameter received a gradient")
        for name, grad in active.items():
            self.assertFalse(torch.isnan(grad).any(), msg=f"NaN grad for {name}")
            self.assertFalse(torch.isinf(grad).any(), msg=f"Inf grad for {name}")

    def test_gradients_no_edge_types(self):
        """Loss.backward() propagates gradients to all active parameters."""
        model = self._make_model()
        out = model(self.x, self.edge_index, self.batch)
        out.sum().backward()
        self._assert_valid_gradients(model)

    def test_gradients_with_edge_types(self):
        """Gradients propagate correctly when edges are routed by type."""
        model = self._make_model(num_edge_types=2)
        edge_types = torch.randint(0, 2, (self.E,))
        out = model(self.x, self.edge_index, self.batch, edge_types=edge_types)
        out.sum().backward()
        self._assert_valid_gradients(model)

    def test_gradients_with_edge_weights(self):
        """Gradients flow through differentiable edge weights."""
        model = self._make_model()
        edge_weights = torch.rand(self.E, requires_grad=True)
        out = model(self.x, self.edge_index, self.batch, edge_weights=edge_weights)
        out.sum().backward()
        self.assertIsNotNone(edge_weights.grad)
        self.assertFalse(torch.isnan(edge_weights.grad).any())

    # --- edge case: all edges of one type ---

    def test_empty_edge_type_subset(self):
        """If all edges have type 0, the type-1 conv gets an empty edge_index."""
        model = self._make_model(num_edge_types=2)
        edge_types = torch.zeros(self.E, dtype=torch.long)  # all type 0
        out = model(self.x, self.edge_index, self.batch, edge_types=edge_types)
        self.assertEqual(out.shape, (2, 5))
        self.assertFalse(torch.isnan(out).any())
