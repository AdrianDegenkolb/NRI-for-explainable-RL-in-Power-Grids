"""
Unit tests for teacher_runner.py.

Requires the `curriculumagent` package (see environment_curriculum.yaml) — skipped entirely
when run in the main `L2RPN` environment, where it isn't installed.

Tests are organised by class:
  - TestBuildTeacher
  - TestRunTeacher
  - TestRunTeacherChronics
  - TestRunTeacherSingleChronic
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("curriculumagent")

from action_space_generation.teacher_runner import (
    RHO_MAX_THRESHOLD,
    RHO_N0_THRESHOLD,
    TeacherConfig,
    build_teacher,
    run_teacher,
    run_teacher_chronics,
    run_teacher_single_chronic,
)


class TestBuildTeacher(unittest.TestCase):
    """Tests for constructing a configured NMinusOneTeacher from a TeacherConfig."""

    def test_uses_config_lines_and_seed(self):
        """lines_to_attack and seed are forwarded unchanged, not the package's track1 default."""
        config = TeacherConfig(env_name="dummy_env", lines_to_attack=[1, 2, 3], save_path=Path("/tmp/x.csv"), seed=7)
        teacher = build_teacher(config)
        self.assertEqual(teacher.lines_to_attack, [1, 2, 3])
        self.assertEqual(teacher.seed, 7)

    def test_uses_this_repos_rho_thresholds(self):
        """rho_n0 must match this repo's HIGH_LEVEL_AGENT activation threshold (0.95)."""
        config = TeacherConfig(env_name="dummy_env", lines_to_attack=[1], save_path=Path("/tmp/x.csv"))
        teacher = build_teacher(config)
        self.assertEqual(teacher.rho_n0, RHO_N0_THRESHOLD)
        self.assertEqual(teacher.rho_threshold, RHO_MAX_THRESHOLD)


class TestRunTeacher(unittest.TestCase):
    """Regression test for the curriculumagent.teacher.teacher.n_minus_1_teacher() positional-arg bug.

    That wrapper passes `jobs` positionally into `number_of_years`; `run_teacher` must call
    `collect_n_minus_1_experience` directly with everything by keyword instead.
    """

    def test_calls_collect_n_minus_1_experience_with_keyword_arguments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3, 12, 27],
                save_path=Path(tmp_dir) / "shard.csv",
                seed=42,
                jobs=4,
            )
            with patch(
                    "action_space_generation.teacher_runner.NMinusOneTeacher.collect_n_minus_1_experience"
            ) as mock_collect:
                run_teacher(config, number_of_years=5)

            mock_collect.assert_called_once_with(
                save_path=config.save_path,
                env_name_path=config.env_name,
                number_of_years=5,
                jobs=4,
                save_greedy=False,
                active_search=True,
                disable_opponent=True,
            )

    def test_creates_save_path_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "nested" / "shard.csv"
            config = TeacherConfig(env_name="dummy_env", lines_to_attack=[1], save_path=save_path)
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.collect_n_minus_1_experience"):
                run_teacher(config)
            self.assertTrue(save_path.parent.exists())


class TestRunTeacherChronics(unittest.TestCase):
    """Tests for the batched SLURM-array-task entry point (regression: array-width limits).

    A one-array-task-per-chronic design hits a cluster's SLURM MaxArraySize on datasets with
    thousands of chronics (confirmed on BWUniCluster for l2rpn_wcci_2020's 2592 chronics —
    sbatch rejected the array outright). Batching chronics into one task per shard avoids this.
    """

    def test_calls_n_minus_one_agent_once_per_chronic_with_same_save_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3, 12, 27],
                save_path=Path(tmp_dir) / "shard_0.csv",
                seed=42,
            )
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.n_minus_one_agent") as mock_agent:
                run_teacher_chronics(config, chronics_ids=["0000", "0001", "0002"])

            self.assertEqual(mock_agent.call_count, 3)
            for call, chronics_id in zip(mock_agent.call_args_list, ["0000", "0001", "0002"]):
                self.assertEqual(
                    call.kwargs,
                    dict(
                        env_path=config.env_name,
                        chronics_path=None,
                        chronics_id=chronics_id,
                        save_path=config.save_path,
                        save_greedy=False,
                        active_search=True,
                        disable_opponent=True,
                    ),
                )

    def test_empty_batch_does_not_call_n_minus_one_agent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3],
                save_path=Path(tmp_dir) / "shard_0.csv",
            )
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.n_minus_one_agent") as mock_agent:
                run_teacher_chronics(config, chronics_ids=[])
            mock_agent.assert_not_called()


class TestRunTeacherSingleChronic(unittest.TestCase):
    """Tests for the single-chronic smoke-test entry point."""

    def test_calls_n_minus_one_agent_with_keyword_arguments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3, 12, 27],
                save_path=Path(tmp_dir) / "shard_0000.csv",
                seed=42,
            )
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.n_minus_one_agent") as mock_agent:
                run_teacher_single_chronic(config, chronics_id="0000")

            mock_agent.assert_called_once_with(
                env_path=config.env_name,
                chronics_path=None,
                chronics_id="0000",
                save_path=config.save_path,
                save_greedy=False,
                active_search=True,
                disable_opponent=True,
            )


if __name__ == "__main__":
    unittest.main()
