"""Post-training analyzer measuring bus-assignment variability across evaluation episodes."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from analysis.post_training.interfaces import EpisodeAnalyzer, StepContext


class TopologyVariabilityAnalyzer(EpisodeAnalyzer):
    """Measures how much the grid topology varies during evaluation.

    Computes two complementary metrics:

    1. **Element variability** — for each element in ``topo_vect``, how often
       its bus assignment deviates from its mode.  Reported as normalised
       Bernoulli variance ``4 * p * (1-p)`` averaged over all elements
       (0 = static, 1 = maximum variability).
    2. **Topology entropy** — Shannon entropy of the frequency distribution
       over unique ``topo_vect`` states.  Normalised by ``log(N_unique)``
       so the value lies in [0, 1] (1 = uniform across all observed states).
    """

    def __init__(self) -> None:
        self._episode_topo_vects: list[npt.NDArray] = []  # each [dim_topo], int32
        self._global_topo_counter: Counter[bytes] = Counter()
        self._per_episode_unique: list[int] = []

    # ------------------------------------------------------------------
    # EpisodeAnalyzer hooks
    # ------------------------------------------------------------------

    def on_episode_start(self, episode_id: str, max_steps: int) -> None:
        self._episode_topo_vects = []

    def on_step(self, ctx: StepContext) -> None:
        self._episode_topo_vects.append(ctx.obs.topo_vect.copy())

    def on_episode_end(self, steps_survived: int, max_steps: int) -> None:
        episode_counter: Counter[bytes] = Counter()
        for tv in self._episode_topo_vects:
            episode_counter[tv.tobytes()] += 1
        self._global_topo_counter.update(episode_counter)
        self._per_episode_unique.append(len(episode_counter))
        self._episode_topo_vects = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Persist variability metrics under ``out_dir/topology_variability/``."""
        d = out_dir / "topology_variability"
        d.mkdir(parents=True, exist_ok=True)

        if not self._global_topo_counter:
            return

        # --- Reconstruct topo_vect matrix from counter ----------------------
        unique_vects = np.array([
            np.frombuffer(k, dtype=np.int32) for k in self._global_topo_counter
        ])  # [U, dim_topo]
        counts_arr = np.array(list(self._global_topo_counter.values()))  # [U]
        topo_matrix = np.repeat(unique_vects, counts_arr, axis=0)  # [T_total, dim_topo]

        # --- Metric 1: element variability --------------------------------
        # Per-element mode via weighted counting (avoids scipy version issues)
        dim_topo = unique_vects.shape[1]
        element_mode = np.empty(dim_topo, dtype=np.int32)
        for j in range(dim_topo):
            val_counts: dict[int, int] = {}
            for i, c in enumerate(counts_arr):
                v = int(unique_vects[i, j])
                val_counts[v] = val_counts.get(v, 0) + int(c)
            element_mode[j] = max(val_counts, key=val_counts.get)

        p_mode = (topo_matrix == element_mode).mean(axis=0)  # fraction at mode
        element_var = 4.0 * p_mode * (1.0 - p_mode)  # [dim_topo], in [0, 1]
        mean_element_var = float(element_var.mean())

        np.save(d / "element_variability.npy", element_var)

        # --- Metric 2: topology entropy -----------------------------------
        counts = np.array(list(self._global_topo_counter.values()), dtype=np.float64)
        probs = counts / counts.sum()
        raw_entropy = -float((probs * np.log(probs)).sum())
        n_unique = len(counts)
        max_entropy = np.log(n_unique) if n_unique > 1 else 1.0
        normalised_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        np.save(d / "unique_topology_counts.npy", counts.astype(np.int64))

        # --- Summary JSON -------------------------------------------------
        summary = {
            "mean_element_variability": mean_element_var,
            "topology_entropy_raw": raw_entropy,
            "topology_entropy_normalised": float(normalised_entropy),
            "n_unique_topologies": n_unique,
            "total_steps": int(counts.sum()),
            "per_episode_unique_mean": float(np.mean(self._per_episode_unique)),
            "per_episode_unique_std": float(np.std(self._per_episode_unique)),
        }
        with open(d / "topology_variability.json", "w") as f:
            json.dump(summary, f, indent=2)
