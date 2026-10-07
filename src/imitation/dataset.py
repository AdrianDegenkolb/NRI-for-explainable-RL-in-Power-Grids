"""
Source-independent imitation dataset format.

Every Tutor's raw output is converted into the same layout, so training code never needs to know
where a dataset came from. Targets are built here, not in the Tutor, so e.g. a different
soft-label temperature only needs a rebuild, not a new rollout.

Per-sample fields (n samples, A = policy actions incl. do-nothing at index 0):
    obs            [n, obs_dim] float32  `obs.to_vect()`; rebuild with `observation_space.from_vect`.
    policy_target  [n, A] float32        Distribution over policy actions (rows sum to 1).
    action_mask    [n, A] bool           Actions known to be valid in that state (all True if unknown).
    chronic        [n] str               Chronic name, for chronic-level train/val/test splits.
    step           [n] int32             Step within the chronic (-1 if unknown).
    max_rho        [n, A] float32        Optional: simulated max rho per action (NaN if invalid),
                                         for regret evaluation. Absent if the source lacks it.
Metadata (scalars): source, target, temperature, action_space, n_actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imitation.tutor import DO_NOTHING_INDEX

GENERAL_TUTOR = "general_tutor"  # curriculumagent GeneralTutor, rows [action_idx, *obs.to_vect()]
GREEDY_TUTOR = "greedy_tutor"    # imitation.tutor, one .npz per chronic

HARD = "hard"
SOFT = "soft"


@dataclass
class ImitationData:
    """In-memory imitation dataset in the shared format (see module docstring)."""

    obs: np.ndarray
    policy_target: np.ndarray
    action_mask: np.ndarray
    chronic: np.ndarray
    step: np.ndarray
    max_rho: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.obs)

    def save(self, path: Path) -> None:
        """Write the dataset to a single `.npz`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "obs": self.obs,
            "policy_target": self.policy_target,
            "action_mask": self.action_mask,
            "chronic": self.chronic,
            "step": self.step,
        }
        if self.max_rho is not None:
            arrays["max_rho"] = self.max_rho
        arrays.update({f"meta_{k}": np.asarray(v) for k, v in self.metadata.items()})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "ImitationData":
        """Read a dataset written by `save`."""
        with np.load(path) as data:
            return cls(
                obs=data["obs"],
                policy_target=data["policy_target"],
                action_mask=data["action_mask"],
                chronic=data["chronic"],
                step=data["step"],
                max_rho=data["max_rho"] if "max_rho" in data.files else None,
                metadata={k[len("meta_"):]: data[k].item() for k in data.files if k.startswith("meta_")},
            )


def one_hot(index: np.ndarray, n_actions: int) -> np.ndarray:
    """One-hot targets [n, n_actions] from action indices [n]."""
    target = np.zeros((len(index), n_actions), dtype=np.float32)
    target[np.arange(len(index)), index] = 1.0
    return target


def soft_targets(max_rho: np.ndarray, valid: np.ndarray, temperature: float) -> np.ndarray:
    """Boltzmann targets p(a|s) ∝ exp(-(rho_a - rho_min) / T) over valid actions.

    The temperature is in rho units, not normalised per state: T = 0.01 gives actions within
    ~0.01 of the best noticeable mass in every state. The argmax equals the lowest-rho action
    for any T; invalid actions get probability 0. A state with no valid action falls back to
    do-nothing.

    Args:
        max_rho: Simulated max rho [n, A], NaN where invalid.
        valid: Validity mask [n, A].
        temperature: Softmax temperature in rho units, > 0.

    Returns:
        Targets [n, A] float32, rows summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}; use hard targets instead")
    rho = np.where(valid, max_rho, np.inf).astype(np.float64)
    rho_min = rho.min(axis=1, keepdims=True)
    no_valid = ~np.isfinite(rho_min[:, 0])
    logits = -(rho - np.where(np.isfinite(rho_min), rho_min, 0.0)) / temperature
    weights = np.where(valid, np.exp(logits), 0.0)
    weights[no_valid, DO_NOTHING_INDEX] = 1.0
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def from_general_tutor(experience_dir: Path, n_actions: int, action_space: str) -> ImitationData:
    """Convert `curriculumagent` GeneralTutor files (`results/tutor/*/experience/*.npy`).

    Its labels index the reduced action set without do-nothing, so label i becomes policy action
    i + 1. It never records do-nothing, stores no step, and simulates no alternatives, so targets
    are one-hot, all actions are marked valid, and `max_rho` is absent.

    Args:
        experience_dir: Directory of per-chronic `.npy` files.
        n_actions: Policy actions incl. do-nothing (e.g. 209 for teacher_n1_k208).
        action_space: Action-space name, stored as metadata.

    Returns:
        The dataset in the shared format.
    """
    obs, labels, chronic = [], [], []
    for path in sorted(experience_dir.glob("*.npy")):
        rows = np.load(path)
        if rows.size == 0:
            continue
        rows = rows.reshape(-1, rows.shape[-1])
        labels.append(rows[:, 0].astype(np.int64) + 1)
        obs.append(rows[:, 1:].astype(np.float32))
        chronic.append(np.full(len(rows), path.stem))
    labels = np.concatenate(labels)
    if labels.max() >= n_actions:
        raise ValueError(f"Label {labels.max()} out of range for {n_actions} policy actions")
    return ImitationData(
        obs=np.concatenate(obs),
        policy_target=one_hot(labels, n_actions),
        action_mask=np.ones((len(labels), n_actions), dtype=bool),
        chronic=np.concatenate(chronic),
        step=np.full(len(labels), -1, dtype=np.int32),
        metadata={"source": GENERAL_TUTOR, "target": HARD, "temperature": 0.0,
                  "action_space": action_space, "n_actions": n_actions},
    )


def from_greedy_tutor(experience_dir: Path, target: str, temperature: float = 0.0) -> ImitationData:
    """Convert `imitation.tutor` files (`results/imitation/*/experience/*.npz`).

    Args:
        experience_dir: Directory of per-chronic `.npz` files.
        target: `"hard"` (one-hot of the executed action) or `"soft"` (see `soft_targets`).
        temperature: Softmax temperature for `"soft"`.

    Returns:
        The dataset in the shared format.
    """
    if target not in (HARD, SOFT):
        raise ValueError(f"Unknown target '{target}', expected '{HARD}' or '{SOFT}'")
    parts: dict[str, list] = {k: [] for k in ("obs", "max_rho", "valid", "action", "chronic", "step")}
    action_space = None
    for path in sorted(experience_dir.glob("*.npz")):
        with np.load(path) as data:
            action_space = str(data["action_space"])
            if len(data["action"]) == 0:
                continue
            for key in ("obs", "max_rho", "valid", "action", "step"):
                parts[key].append(data[key])
            parts["chronic"].append(np.full(len(data["action"]), str(data["chronic"])))
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    n_actions = arrays["max_rho"].shape[1]

    if target == HARD:
        policy_target = one_hot(arrays["action"].astype(np.int64), n_actions)
    else:
        policy_target = soft_targets(arrays["max_rho"], arrays["valid"], temperature)

    # When no action is valid the Tutor falls back to do-nothing; keep that target unmasked so a
    # masked policy can still assign it probability.
    action_mask = arrays["valid"] | (policy_target > 0)

    return ImitationData(
        obs=arrays["obs"],
        policy_target=policy_target,
        action_mask=action_mask,
        chronic=arrays["chronic"],
        step=arrays["step"].astype(np.int32),
        max_rho=arrays["max_rho"],
        metadata={"source": GREEDY_TUTOR, "target": target, "temperature": float(temperature),
                  "action_space": action_space, "n_actions": n_actions},
    )
