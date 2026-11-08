"""Edge ablation: test whether a trained GNN uses edge information.

Wraps an ObservationConverter to modify the edge_index at inference time, so the
same trained model can be evaluated under three edge conditions:

  true      — real dynamic edges (baseline for graph models, injected for edgeless)
  shuffled  — node-index permutation applied to edge_index (same graph statistics,
              wrong connectivity)
  none      — no edges (edge_mask all False; baseline for edgeless models)

For edgeless converters (EdgelessGraphObservationConverter), TRUE and SHUFFLED modes
compute edges via the parent GraphObservationConverter._get_edge_index and inject them.

Usage: wrap the gym_wrapper's observation_converter before evaluation::

    from analysis.edge_ablation import EdgeMode, EdgeAblationWrapper
    wrapper = EdgeAblationWrapper(gym_wrapper.observation_converter, EdgeMode.SHUFFLED, seed=42)
    gym_wrapper.observation_converter = wrapper
"""
from __future__ import annotations

import enum
from typing import Optional

import numpy as np
import numpy.typing as npt
from grid2op.Observation import BaseObservation

from grid2op_env.observation_converter import (
    EDGE_INDEX,
    EDGE_MASK,
    NODE_MASK,
    NODES,
    EdgelessGraphObservationConverter,
    GraphObservationConverter,
    ObservationConverter,
)


class EdgeMode(enum.Enum):
    TRUE = "true"
    SHUFFLED = "shuffled"
    NONE = "none"


class EdgeAblationWrapper:
    """Wraps an existing converter and modifies edge_index according to the chosen mode.

    Not a subclass of ObservationConverter — delegates all attribute access to the inner
    converter so it's a transparent proxy. Only ``to_gym`` is intercepted.

    :param inner: The real observation converter to wrap.
    :param mode: Edge ablation mode.
    :param seed: RNG seed for the node-index permutation (SHUFFLED mode).
    """

    def __init__(self, inner: ObservationConverter, mode: EdgeMode, seed: int = 0) -> None:
        self._inner = inner
        self._mode = mode
        self._rng = np.random.default_rng(seed)
        self._perm: Optional[npt.NDArray[np.int64]] = None
        self._is_edgeless = isinstance(inner, EdgelessGraphObservationConverter)

    def __getattr__(self, name: str):
        """Delegate attribute access to inner converter."""
        return getattr(self._inner, name)

    def _get_or_build_perm(self, active_indices: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Return shuffled active node indices, building a permutation if needed.

        :param active_indices: Sorted array of node indices where node_mask is True.
        :return: Same indices in shuffled order.
        """
        n_active = len(active_indices)
        if self._perm is None or len(self._perm) != n_active:
            self._perm = self._rng.permutation(n_active)
        return active_indices[self._perm]

    def new_episode_permutation(self) -> None:
        """Draw a new permutation — call at episode start for per-episode randomness."""
        self._perm = None

    def _inject_edges(
        self, gym_obs: dict[str, npt.NDArray], g2op_obs: BaseObservation
    ) -> dict[str, npt.NDArray]:
        """Compute real edges via the parent class and inject them into the gym obs.

        Only needed for edgeless converters in TRUE/SHUFFLED mode.
        """
        edge_index = GraphObservationConverter._get_edge_index(self._inner, g2op_obs)
        num_edges = edge_index.shape[1]
        max_edges = gym_obs[EDGE_INDEX].shape[1]
        n_to_fill = min(num_edges, max_edges)
        gym_obs[EDGE_INDEX][:, :n_to_fill] = edge_index[:, :n_to_fill]
        gym_obs[EDGE_MASK][:] = False
        gym_obs[EDGE_MASK][:n_to_fill] = True
        return gym_obs

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """Convert observation and then modify edge_index according to the mode."""
        gym_obs = self._inner.to_gym(g2op_obs)

        if self._mode is EdgeMode.NONE:
            gym_obs[EDGE_MASK][:] = False
            return gym_obs

        if self._is_edgeless and self._mode in (EdgeMode.TRUE, EdgeMode.SHUFFLED):
            gym_obs = self._inject_edges(gym_obs, g2op_obs)

        if self._mode is EdgeMode.TRUE:
            return gym_obs

        # SHUFFLED: permute active node indices in edge_index
        edge_index = gym_obs[EDGE_INDEX]
        edge_mask = gym_obs[EDGE_MASK]
        node_mask = gym_obs[NODE_MASK]
        active_indices = np.where(node_mask)[0].astype(np.int64)
        shuffled_active = self._get_or_build_perm(active_indices)

        num_nodes = gym_obs[NODES].shape[0]
        remap = np.arange(num_nodes, dtype=np.int64)
        remap[active_indices] = shuffled_active

        n_active_edges = int(edge_mask.sum())
        if n_active_edges > 0:
            edge_index[:, :n_active_edges] = remap[edge_index[:, :n_active_edges]]
        gym_obs[EDGE_INDEX] = edge_index
        return gym_obs
