"""
Unit tests for imitation/summary.py and analysis/episodes.py.

Tests are organised by class:
  - TestBestAction
  - TestSubstationTable
  - TestEpisodes
  - TestTeacherEpisodesFromLogs
  - TestRhoSlice
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import grid2op
import numpy as np
import pandas as pd
from lightsim2grid import LightSimBackend

from analysis.episodes import EPISODE_COLUMNS, load_episodes, make_episodes, save_episodes, teacher_episodes_from_logs
from imitation.summary import best_action, rho_slice, substation_table


class TestBestAction(unittest.TestCase):
    """Lowest valid max rho per state."""

    def test_skips_invalid_actions(self):
        max_rho = np.array([[0.9, 0.5, 0.7]], dtype=np.float32)
        valid = np.array([[True, False, True]])
        self.assertEqual(best_action(max_rho, valid)[0], 2)

    def test_no_valid_action_gives_minus_one(self):
        max_rho = np.full((1, 3), np.nan, dtype=np.float32)
        self.assertEqual(best_action(max_rho, np.zeros((1, 3), dtype=bool))[0], -1)


class TestSubstationTable(unittest.TestCase):
    """Per-substation aggregation of the action table."""

    def setUp(self):
        actions = pd.DataFrame({
            "index": [0, 1, 2, 3],
            "substation": [-1, 2, 2, 0],
            "tutor_picks": [5, 3, 1, 4],
            "tutor_best": [0, 2, 2, 4],
        })
        self.table = substation_table(actions, n_sub=4)

    def test_excludes_do_nothing_and_fills_empty_substations(self):
        self.assertEqual(list(self.table["substation"]), [0, 1, 2, 3])
        self.assertEqual(list(self.table["n_actions"]), [1, 0, 2, 0])
        self.assertEqual(list(self.table["tutor_picks"]), [4, 0, 4, 0])

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(self.table["tutor_pick_share"].sum(), 1.0)


class TestEpisodes(unittest.TestCase):
    """Shared episode table."""

    def test_round_trip_and_survived(self):
        df = make_episodes([
            {"agent": "tutor", "chronic": "a", "steps_survived": 10, "max_step": 10},
            {"agent": "tutor", "chronic": "b", "steps_survived": 4, "max_step": 10, "seed": 7},
        ])
        self.assertEqual(list(df.columns), EPISODE_COLUMNS)
        self.assertEqual(list(df["survived"]), [True, False])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.csv"
            save_episodes(df, path)
            loaded = load_episodes(path, path)
        self.assertEqual(len(loaded), 4)
        self.assertEqual(list(loaded["seed"][:2]), [-1, 7])

    def test_tables_without_action_counts_load_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.csv"
            path.write_text("agent,chronic,seed,steps_survived,max_step,survived,truncated,n_decisions\n"
                            "junior,a,1,10,10,True,False,3\n")
            loaded = load_episodes(path)
        self.assertEqual(list(loaded.columns), EPISODE_COLUMNS)
        self.assertEqual((loaded.loc[0, "n_decisions"], loaded.loc[0, "n_actions"], loaded.loc[0, "n_illegal"]), (3, -1, -1))


class TestTeacherEpisodesFromLogs(unittest.TestCase):
    """Survival recovered from curriculumagent Teacher logs."""

    LOG = (
        "2026-09-17 11:39:36,123 INFO [Scenario_april_000] N-1 Search started\n"
        "2026-09-17 11:39:43,182 INFO [Scenario_april_000] (246/8062) (rho=0.95): Looking for greedy action\n"
        "2026-09-17 20:10:43,559 INFO [Scenario_april_000] Search stopped at 8062/8062\n"
        "2026-09-18 01:21:35,359 INFO [Scenario_april_003] Search stopped at 979/8062\n"
        "2026-09-18 01:22:00,000 INFO [Scenario_april_004] (512/8062) (rho=0.97): Looking for greedy action\n"
    )

    def test_parses_stopped_and_truncated_chronics(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "shard_0.1.err").write_text(self.LOG)
            df = teacher_episodes_from_logs(Path(tmp), seed=42).set_index("chronic")
        self.assertEqual(df.loc["Scenario_april_000", "steps_survived"], 8062)
        self.assertTrue(df.loc["Scenario_april_000", "survived"])
        self.assertEqual(df.loc["Scenario_april_003", "steps_survived"], 979)
        self.assertFalse(df.loc["Scenario_april_003", "truncated"])
        self.assertTrue(df.loc["Scenario_april_004", "truncated"])
        self.assertEqual(df.loc["Scenario_april_004", "steps_survived"], 512)


class TestRhoSlice(unittest.TestCase):
    """rho's position in obs.to_vect()."""

    def test_matches_observation_rho(self):
        env = grid2op.make("l2rpn_wcci_2020_train", backend=LightSimBackend())
        try:
            obs = env.reset()
            np.testing.assert_allclose(obs.to_vect()[rho_slice(obs)], obs.rho, rtol=1e-6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
