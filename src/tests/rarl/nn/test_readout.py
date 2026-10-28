"""Unit tests for rarl/nn/readout.py (substation_readout)."""
import unittest

import torch

from rarl.nn.readout import substation_readout


class TestSubstationReadout(unittest.TestCase):
    def setUp(self):
        # Graph 0: nodes 0-2; graph 1: nodes 3-4. Two substations, embedding size 2.
        self.h = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [2.0, 2.0], [6.0, 6.0]])
        self.batch = torch.tensor([0, 0, 0, 1, 1])
        # Node 2 is a line node in both substations; node 4 has no second substation.
        self.subs = torch.tensor([[0, -1], [0, -1], [0, 1], [1, -1], [1, -1]])

    def test_mean_per_substation_concatenated(self):
        out = substation_readout(self.h, self.batch, self.subs, num_graphs=2, n_sub=2)
        expected = torch.tensor([
            [4.0 / 3, 4.0 / 3, 0.0, 4.0],  # sub 0: mean of nodes 0, 1, 2; sub 1: node 2
            [0.0, 0.0, 4.0, 4.0],          # sub 0: no node -> zeros; sub 1: mean of nodes 3, 4
        ])
        torch.testing.assert_close(out, expected)

    def test_gradients_reach_every_node(self):
        h = self.h.clone().requires_grad_(True)
        substation_readout(h, self.batch, self.subs, num_graphs=2, n_sub=2).sum().backward()
        self.assertTrue((h.grad.abs().sum(dim=1) > 0).all())


if __name__ == "__main__":
    unittest.main()
