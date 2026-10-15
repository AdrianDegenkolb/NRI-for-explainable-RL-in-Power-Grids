"""
Unit tests for the action choosers of imitation/rollout.py.

Tests are organised by class:
  - TestChoosers
"""
from __future__ import annotations

import unittest

import torch

from imitation.rollout import do_nothing_chooser, greedy_chooser
from imitation.tutor import DO_NOTHING_INDEX


class _FixedLogits(torch.nn.Module):
    """Stands in for a policy model: logits are passed in as the observation."""

    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs"]["logits"], state


class TestChoosers(unittest.TestCase):
    def test_greedy_chooser_takes_argmax_in_eval_mode(self):
        model = _FixedLogits()
        chooser = greedy_chooser(model, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(chooser({"logits": [0.1, 2.0, -1.0, 0.5]}), 1)

    def test_do_nothing_chooser_ignores_observation(self):
        self.assertEqual(do_nothing_chooser(None), DO_NOTHING_INDEX)
        self.assertEqual(do_nothing_chooser({"logits": [0.0, 9.0]}), DO_NOTHING_INDEX)


if __name__ == "__main__":
    unittest.main()
