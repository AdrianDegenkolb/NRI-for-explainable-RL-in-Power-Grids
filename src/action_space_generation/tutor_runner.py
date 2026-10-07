"""
Run the Fraunhofer `curriculumagent` GeneralTutor to build the Junior's imitation dataset.

The Tutor acts greedily over the reduced action set (e.g. the N-1 Teacher's top-k): whenever
`rho.max()` reaches `do_nothing_threshold` it simulates every action and takes the one with the
lowest resulting max rho. Each such step is recorded as one row `[action_idx, *obs.to_vect()]`.
The package logs `game over at step-X` at the end of every chronic, which doubles as a survival
check for the action set.

This bypasses `curriculumagent.tutor.collect_tutor_experience.generate_tutor_experience`, which
runs every chronic in a local `multiprocessing.Pool` (`jobs=-1` -> all cores on the node, not the
SLURM allocation) and writes a single file only after *all* chronics finish — a task killed at
its walltime would lose everything. The per-chronic worker is called directly instead, and each
chronic is saved to its own `.npy` as soon as it finishes.

Must run in the `curriculum` conda environment (see `environment_curriculum.yaml`), never in
the main `L2RPN` training environment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from curriculumagent.tutor.collect_tutor_experience import (
    collect_tutor_experience_one_chronic,
    prepare_dataset,
)
from curriculumagent.tutor.tutors.general_tutor import GeneralTutor

from action_space_generation.teacher_runner import RHO_N0_THRESHOLD

# The Tutor's do-nothing threshold. curriculumagent defaults to 0.925; 0.95 matches the teacher's
# `rho_n0_threshold` and this repo's HIGH_LEVEL_AGENT activation threshold (`rho_threshold`), so
# the dataset only contains states in which our RL agent is actually asked to act.
DO_NOTHING_THRESHOLD = RHO_N0_THRESHOLD


@dataclass
class TutorConfig:
    """Configuration for one Tutor run.

    Attributes:
        env_name: Grid2Op environment name or path (e.g. "l2rpn_wcci_2020_train").
        action_space_path: `.npy` action set from `export.export_action_space_npy`.
        save_dir: Directory for the per-chronic experience files (`<chronic_name>.npy`).
        seed: Random seed for the Grid2Op environment.
    """

    env_name: str
    action_space_path: Path
    save_dir: Path
    seed: int = 42


def run_tutor_chronics(config: TutorConfig, chronics_ids: list[str]) -> None:
    """Run the GeneralTutor sequentially over one or more chronics, one output file each.

    Chronics whose output file already exists are skipped, so a re-submitted task (e.g. after a
    walltime kill) only redoes the chronic it was killed in.

    Args:
        config: Tutor configuration.
        chronics_ids: Chronic subpath names to process in order, from `list_chronic_ids`.

    Returns:
        None. Writes `config.save_dir / f"{chronic_name}.npy"` per chronic, with rows
        `[action_idx, *obs.to_vect()]` as float32.
    """
    logging.basicConfig(level=logging.INFO)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    for chronics_id in chronics_ids:
        # `list_chronic_ids` returns full chronic paths; name the file after the chronic only.
        out_path = config.save_dir / f"{Path(chronics_id).name}.npy"
        if out_path.exists():
            logging.info(f"Skipping chronic {chronics_id}: {out_path} already exists")
            continue

        records = collect_tutor_experience_one_chronic(
            action_paths=config.action_space_path,
            chronics_id=chronics_id,
            env_name_path=config.env_name,
            seed=config.seed,
            enable_logging=True,
            TutorAgent=GeneralTutor,
            tutor_kwargs={"do_nothing_threshold": DO_NOTHING_THRESHOLD},
        )
        # The worker initialises `records` with one all-zero row; its label 0 is a real action,
        # so keeping it would add a fake sample to the imitation dataset.
        records = records[1:]
        np.save(out_path, records)
        logging.info(f"Saved {len(records)} tutor samples for chronic {chronics_id} to {out_path}")


def build_tutor_dataset(experience_dir: Path, out_dir: Path, dataset_name: str, seed: int = 42) -> None:
    """Merge per-chronic Tutor files into the Junior's train/val/test split.

    Thin wrapper around `curriculumagent.tutor.collect_tutor_experience.prepare_dataset`: drops
    duplicate rows, shuffles, and splits 80/10/10 by sample.

    Args:
        experience_dir: Directory of per-chronic `.npy` files from `run_tutor_chronics`.
        out_dir: Directory for `{dataset_name}_{train,val,test}.npz`.
        dataset_name: File name prefix.
        seed: Shuffle seed.

    Returns:
        None. Writes the three `.npz` files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_dataset(traindata_path=experience_dir, target_path=out_dir, dataset_name=dataset_name, seed=seed)
