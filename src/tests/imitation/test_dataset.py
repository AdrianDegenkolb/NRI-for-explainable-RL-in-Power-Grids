"""
Unit tests for imitation/dataset.py.

Tests are organised by class:
  - TestSoftTargets
  - TestFromGeneralTutor
  - TestFromGreedyTutor
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from imitation.dataset import (
    GENERAL_TUTOR,
    GREEDY_TUTOR,
    HARD,
    SOFT,
    ImitationData,
    from_general_tutor,
    from_greedy_tutor,
    soft_targets,
)


class TestSoftTargets(unittest.TestCase):
    """Tests for Boltzmann targets over simulated max rho."""

    def test_rows_sum_to_one_and_argmax_is_lowest_rho(self):
        max_rho = np.array([[1.0, 0.8, 0.9], [0.7, 0.95, 0.6]], dtype=np.float32)
        target = soft_targets(max_rho, np.ones_like(max_rho, dtype=bool), temperature=0.05)
        np.testing.assert_allclose(target.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(target.argmax(axis=1), [1, 2])

    def test_invalid_actions_get_zero_probability(self):
        max_rho = np.array([[1.0, np.nan, 0.9]], dtype=np.float32)
        valid = np.array([[True, False, True]])
        self.assertEqual(soft_targets(max_rho, valid, temperature=0.1)[0, 1], 0.0)

    def test_temperature_is_in_absolute_rho_units(self):
        """The same rho gap gives the same probabilities regardless of the rho level."""
        calm = soft_targets(np.array([[0.50, 0.51]], dtype=np.float32), np.ones((1, 2), bool), 0.01)
        stressed = soft_targets(np.array([[1.20, 1.21]], dtype=np.float32), np.ones((1, 2), bool), 0.01)
        np.testing.assert_allclose(calm, stressed, rtol=1e-5)

    def test_no_valid_action_falls_back_to_do_nothing(self):
        target = soft_targets(np.full((1, 3), np.nan, np.float32), np.zeros((1, 3), bool), 0.01)
        np.testing.assert_array_equal(target, [[1.0, 0.0, 0.0]])

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            soft_targets(np.zeros((1, 2), np.float32), np.ones((1, 2), bool), 0.0)


class TestFromGeneralTutor(unittest.TestCase):
    """Tests for converting curriculumagent GeneralTutor files."""

    def test_shifts_labels_past_do_nothing_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "Scenario_a.npy", np.array([[0, 1.0, 2.0], [4, 3.0, 4.0]], dtype=np.float32))
            np.save(tmp / "Scenario_b.npy", np.array([[2, 5.0, 6.0]], dtype=np.float32))
            data = from_general_tutor(tmp, n_actions=6, action_space="k5.json")

            np.testing.assert_array_equal(data.policy_target.argmax(axis=1), [1, 5, 3])
            np.testing.assert_array_equal(data.chronic, ["Scenario_a", "Scenario_a", "Scenario_b"])
            np.testing.assert_array_equal(data.obs[2], [5.0, 6.0])
            self.assertIsNone(data.max_rho)
            self.assertEqual(data.metadata["source"], GENERAL_TUTOR)

            data.save(tmp / "out.npz")
            loaded = ImitationData.load(tmp / "out.npz")
            np.testing.assert_array_equal(loaded.policy_target, data.policy_target)
            self.assertIsNone(loaded.max_rho)
            self.assertEqual(loaded.metadata["n_actions"], 6)

    def test_rejects_labels_beyond_policy_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "Scenario_a.npy", np.array([[5, 1.0]], dtype=np.float32))
            with self.assertRaises(ValueError):
                from_general_tutor(Path(tmp), n_actions=5, action_space="k4.json")


class TestFromGreedyTutor(unittest.TestCase):
    """Tests for converting imitation.tutor files."""

    def _write_chronic(self, path: Path) -> None:
        np.savez(
            path,
            obs=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            step=np.array([10, 11], dtype=np.int32),
            max_rho=np.array([[1.0, 0.8, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32),
            valid=np.array([[True, True, False], [False, False, False]]),
            action=np.array([1, 0], dtype=np.int16),
            chronic=np.str_("Scenario_a"),
            action_space=np.str_("k2.json"),
        )

    def test_hard_and_soft_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_chronic(tmp / "Scenario_a.npz")

            hard = from_greedy_tutor(tmp, target=HARD)
            np.testing.assert_array_equal(hard.policy_target, [[0, 1, 0], [1, 0, 0]])
            self.assertEqual(hard.metadata["source"], GREEDY_TUTOR)

            soft = from_greedy_tutor(tmp, target=SOFT, temperature=0.1)
            self.assertGreater(soft.policy_target[0, 0], 0.0)
            self.assertEqual(soft.policy_target[0].argmax(), 1)

    def test_fallback_target_is_never_masked(self):
        """A state with no valid action targets do-nothing, which must stay selectable."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_chronic(Path(tmp) / "Scenario_a.npz")
            data = from_greedy_tutor(Path(tmp), target=HARD)
            self.assertTrue(data.action_mask[1, 0])
            self.assertTrue(np.all(data.action_mask[data.policy_target > 0]))
