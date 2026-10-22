"""
Unit tests for the action choosers of imitation/rollout.py.

Tests are organised by class:
  - TestChoosers
  - TestTopKChooser
"""
from __future__ import annotations

import unittest

import torch

from imitation.rollout import do_nothing_chooser, greedy_chooser, top_k_chooser
from imitation.tutor import DO_NOTHING_INDEX, Tutor
from tests.imitation.test_tutor import _SimulatedObs


class _FixedLogits(torch.nn.Module):
    """Stands in for a policy model: logits are passed in as the observation."""

    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs"]["logits"], state


class TestChoosers(unittest.TestCase):
    def test_greedy_chooser_takes_argmax_in_eval_mode(self):
        model = _FixedLogits()
        chooser = greedy_chooser(model, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(chooser(None, {"logits": [0.1, 2.0, -1.0, 0.5]}), 1)

    def test_do_nothing_chooser_ignores_observation(self):
        self.assertEqual(do_nothing_chooser(None, None), DO_NOTHING_INDEX)
        self.assertEqual(do_nothing_chooser(None, {"logits": [0.0, 9.0]}), DO_NOTHING_INDEX)


class TestTopKChooser(unittest.TestCase):
    """The policy ranks, simulation of its k best actions and do-nothing decides."""

    # Simulated max rho per action; the policy ranks 1 > 2 > 3, but 3 is the best action.
    MAX_RHO = [1.00, 0.99, 0.97, 0.90]
    LOGITS = [0.0, 3.0, 2.0, 1.0]

    def choose(self, k: int, max_rho: list[float] | None = None) -> int:
        actions = list(range(len(self.LOGITS)))
        tutor = Tutor(env=None, actions=actions, line_reco=False, margin=0.005)
        chooser = top_k_chooser(_FixedLogits(), torch.device("cpu"), tutor, k)
        return chooser(_SimulatedObs(max_rho or self.MAX_RHO), {"logits": self.LOGITS})

    def test_takes_best_simulated_among_top_k(self):
        self.assertEqual(self.choose(k=2), 2)

    def test_k_of_all_actions_is_the_tutor(self):
        self.assertEqual(self.choose(k=4), 3)

    def test_vetoes_top_action_not_beating_do_nothing(self):
        self.assertEqual(self.choose(k=1, max_rho=[0.95, 0.948, 0.80, 0.70]), DO_NOTHING_INDEX)

    def test_k_beyond_action_count_is_clipped(self):
        self.assertEqual(self.choose(k=100), 3)


if __name__ == "__main__":
    unittest.main()
