"""Load per-seed analysis data for multiple training methods."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class MethodData:
    """Aggregated analysis data for one training method across multiple seeds.

    :param name: Display name used in plot labels.
    :param action_freqs: Substation action frequencies, shape ``[n_seeds, n_sub]``.
    :param mean_rhos: Mean line loading per seed, shape ``[n_seeds, n_lines]``.
    """

    name: str
    action_freqs: npt.NDArray  # [n_seeds, n_sub]
    mean_rhos: npt.NDArray     # [n_seeds, n_lines]

    @property
    def n_seeds(self) -> int:
        return self.action_freqs.shape[0]

    @property
    def n_sub(self) -> int:
        return self.action_freqs.shape[1]

    @property
    def n_lines(self) -> int:
        return self.mean_rhos.shape[1]


def discover_analysis_dirs(experiment_dir: Path) -> List[Path]:
    """Return ``analysis/`` paths for each seed trial found in *experiment_dir*.

    A trial directory is included only when it contains both
    ``analysis/actions/sub_action_freq.npy`` and
    ``analysis/congestion/mean_rho_per_line.npy``.

    :param experiment_dir: Top-level directory containing per-seed trial subdirectories.
    :returns: Sorted list of ``<trial>/analysis`` paths.
    """
    found = []
    for trial_dir in sorted(experiment_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        actions_file = trial_dir / "analysis" / "actions" / "sub_action_freq.npy"
        congestion_file = trial_dir / "analysis" / "congestion" / "mean_rho_per_line.npy"
        if actions_file.exists() and congestion_file.exists():
            found.append(trial_dir / "analysis")
        else:
            logger.debug("Skipping %s (missing analysis data)", trial_dir.name)
    if not found:
        logger.warning("No valid analysis directories found in %s", experiment_dir)
    return found


def load_method(name: str, experiment_dir: Path) -> MethodData:
    """Load action frequency and congestion arrays for all seeds of one method.

    :param name: Display name for this method.
    :param experiment_dir: Directory containing per-seed trial subdirectories.
    :returns: :class:`MethodData` with arrays stacked along axis 0 (seeds).
    :raises ValueError: If no valid seed directories are found.
    """
    analysis_dirs = discover_analysis_dirs(experiment_dir)
    if not analysis_dirs:
        raise ValueError(
            f"No analysis data found for method '{name}' in {experiment_dir}"
        )

    action_freqs: List[npt.NDArray] = []
    mean_rhos: List[npt.NDArray] = []

    for analysis_dir in analysis_dirs:
        action_freqs.append(
            np.load(analysis_dir / "actions" / "sub_action_freq.npy")
        )
        rho_failure = analysis_dir / "congestion" / "rho_at_failure.npy"
        rho_mean = analysis_dir / "congestion" / "mean_rho_per_line.npy"
        if rho_failure.exists():
            mean_rhos.append(np.load(rho_failure))
            logger.info("  Loaded seed (rho_at_failure): %s", analysis_dir.parent.name)
        else:
            mean_rhos.append(np.load(rho_mean))
            logger.warning("  No rho_at_failure for %s, falling back to mean_rho_per_line", analysis_dir.parent.name)

    data = MethodData(
        name=name,
        action_freqs=np.stack(action_freqs, axis=0),
        mean_rhos=np.stack(mean_rhos, axis=0),
    )
    logger.info(
        "Method '%s': %d seeds, %d substations, %d lines",
        name, data.n_seeds, data.n_sub, data.n_lines,
    )
    return data
