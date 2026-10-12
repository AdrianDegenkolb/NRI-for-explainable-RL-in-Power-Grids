"""
Unit tests for imitation/tutor.py.

Tests are organised by class:
  - TestSelectAction
  - TestChronicSeed
  - TestLoadPolicyActions
"""
from __future__ import annotations

import unittest
from pathlib import Path

import grid2op
import numpy as np
from lightsim2grid import LightSimBackend

from imitation.tutor import DO_NOTHING_INDEX, chronic_seed, load_policy_actions, select_action

ROOT = Path(__file__).resolve().parents[3]
ACTION_SPACE = ROOT / "data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json"


class TestSelectAction(unittest.TestCase):
    """Tests for the Tutor's choice among simulated candidates."""

    def test_picks_best_topology_action_beyond_margin(self):
        max_rho = np.array([1.00, 0.90, 0.80, 0.95], dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.ones(4, dtype=bool), margin=0.005), 2)

    def test_prefers_do_nothing_on_near_tie(self):
        max_rho = np.array([0.900, 0.898, 0.950], dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.ones(3, dtype=bool), margin=0.005), DO_NOTHING_INDEX)

    def test_ignores_invalid_candidates(self):
        max_rho = np.array([1.0, np.nan, 0.9], dtype=np.float32)
        valid = np.array([True, False, True])
        self.assertEqual(select_action(max_rho, valid, margin=0.0), 2)

    def test_invalid_do_nothing_takes_any_valid_action(self):
        max_rho = np.array([np.nan, 1.5, 1.2], dtype=np.float32)
        valid = np.array([False, True, True])
        self.assertEqual(select_action(max_rho, valid, margin=0.005), 2)

    def test_nothing_valid_falls_back_to_do_nothing(self):
        max_rho = np.full(3, np.nan, dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.zeros(3, dtype=bool), margin=0.005), DO_NOTHING_INDEX)


class TestChronicSeed(unittest.TestCase):
    """Each chronic needs its own seed, or all chronics share one maintenance schedule."""

    def test_differs_between_chronics(self):
        self.assertNotEqual(chronic_seed(42, "Scenario_june_051"), chronic_seed(42, "Scenario_september_061"))

    def test_path_and_name_give_the_same_seed(self):
        self.assertEqual(chronic_seed(42, "/data/chronics/Scenario_june_051"), chronic_seed(42, "Scenario_june_051"))

    def test_depends_on_base_seed(self):
        self.assertNotEqual(chronic_seed(0, "Scenario_june_051"), chronic_seed(1, "Scenario_june_051"))

    def test_in_valid_range(self):
        self.assertTrue(0 <= chronic_seed(2**31 - 1, "Scenario_june_051") < 2**31)


class TestLoadPolicyActions(unittest.TestCase):
    """The Tutor's candidates must match the RL env's discrete action space."""

    @classmethod
    def setUpClass(cls):
        cls.env = grid2op.make("l2rpn_wcci_2020_train", backend=LightSimBackend())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_do_nothing_first_then_json_order(self):
        actions = load_policy_actions(self.env, ACTION_SPACE)
        self.assertEqual(len(actions), 209)
        self.assertEqual(actions[DO_NOTHING_INDEX], self.env.action_space({}))
        self.assertNotEqual(actions[1], self.env.action_space({}))
