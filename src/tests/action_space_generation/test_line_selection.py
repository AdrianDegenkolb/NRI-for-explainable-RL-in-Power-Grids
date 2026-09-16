"""
Unit tests for line_selection.py.

Tests are organised by class:
  - TestLineStressStats
  - TestSelectLinesToAttack
"""
from __future__ import annotations

import unittest

import grid2op
import numpy as np

from action_space_generation.line_selection import LineStressStats, select_lines_to_attack

ENV_NAME = "l2rpn_case14_sandbox"


class TestLineStressStats(unittest.TestCase):
    """Tests for the pure-numpy stress-tracking logic (no Grid2Op env needed)."""

    def setUp(self):
        self.stats = LineStressStats(n_line=4, near_limit_threshold=0.8)

    def test_update_accumulates_near_limit_count(self):
        """near_limit_count increments only for lines above the threshold, once per update."""
        self.stats.update(np.array([0.9, 0.5, 0.85, 0.1]))
        self.stats.update(np.array([0.9, 0.9, 0.1, 0.1]))
        np.testing.assert_array_equal(self.stats.near_limit_count, [2, 1, 1, 0])

    def test_update_tracks_running_max_rho(self):
        """max_rho keeps the highest rho seen per line across multiple updates."""
        self.stats.update(np.array([0.3, 0.9, 0.2, 0.1]))
        self.stats.update(np.array([0.5, 0.4, 0.95, 0.05]))
        np.testing.assert_array_almost_equal(self.stats.max_rho, [0.5, 0.9, 0.95, 0.1])

    def test_rank_lines_orders_by_near_limit_count_first(self):
        """A line seen near its limit more often ranks above one with a merely higher peak."""
        self.stats.update(np.array([0.99, 0.81, 0.0, 0.0]))  # line 0 and 1 both cross threshold
        self.stats.update(np.array([0.0, 0.82, 0.0, 0.0]))  # line 1 crosses again -> count 2
        ranked = self.stats.rank_lines(n_lines=2)
        self.assertEqual(ranked[0], 1)  # higher near_limit_count wins despite lower peak
        self.assertEqual(ranked[1], 0)

    def test_rank_lines_falls_back_to_max_rho_when_no_line_crosses_threshold(self):
        """With near_limit_count all zero (e.g. a short do-nothing episode), rank by max_rho."""
        self.stats.update(np.array([0.3, 0.6, 0.1, 0.5]))
        ranked = self.stats.rank_lines(n_lines=4)
        self.assertEqual(ranked, [1, 3, 0, 2])

    def test_rank_lines_caps_at_n_line(self):
        """Requesting more lines than exist returns at most n_line ids."""
        self.stats.update(np.array([0.1, 0.1, 0.1, 0.1]))
        ranked = self.stats.rank_lines(n_lines=100)
        self.assertEqual(len(ranked), 4)


class TestSelectLinesToAttack(unittest.TestCase):
    """Smoke test for the end-to-end do-nothing rollout on a real (small) grid."""

    def test_returns_requested_number_of_valid_line_ids(self):
        """select_lines_to_attack returns n_lines valid, unique line ids."""
        env = grid2op.make(ENV_NAME)
        n_line = env.n_line
        env.close()

        lines = select_lines_to_attack(env_name=ENV_NAME, n_lines=3, n_chronics=1)

        self.assertEqual(len(lines), 3)
        self.assertEqual(len(set(lines)), 3)
        for line_id in lines:
            self.assertTrue(0 <= line_id < n_line)


if __name__ == "__main__":
    unittest.main()
