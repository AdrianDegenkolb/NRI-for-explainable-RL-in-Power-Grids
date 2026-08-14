from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from grid2op.Observation import BaseObservation, ObservationSpace
from grid2op.gym_compat import GymEnv
from gymnasium.spaces import Dict, Box
from gymnasium.wrappers.normalize import RunningMeanStd

from grid2op_env.utils import get_attr_list

logger = logging.getLogger(__name__)

T = TypeVar("T")

NODES = "node_features"         # keys node features in the observation [N, x_dim]
NODE_MASK = "node_mask"         # keys boolean mask that masks which rows in the returned node-feature-tensor actually encode node features [N]
EDGES = "edge_features"         # keys edge features in the observation [E, e_dim]
EDGE_MASK = "edge_mask"         # keys boolean mask that masks which rows in the returned edge-feature-tensor actually encode edge features [E]
EDGE_INDEX = "edge_index"       # keys the edge index which notes node pairs that are connected by notes [2, E]
EDGE_TYPE = "edge_type"         # keys edge types per edge [E]
GLOBAL = "global_features"      # keys global features in the observation


_DEFAULT_NODE_FEATURES = [
    "active_power_forecast",
    "reactive_power_forecast",
    "active_power",
    "reactive_power",
    "voltage",
    "voltage_angle",
    "rho",
]


@dataclass
class _GridDimensions:
    """Static grid dimensions extracted once from the observation space."""

    n_gen: int
    n_load: int
    n_line: int
    n_storage: int

    @classmethod
    def from_obs_space(cls, obs_space: ObservationSpace) -> "_GridDimensions":
        """Extract grid dimensions from a grid2op observation space."""
        return cls(
            n_gen=obs_space.n_gen,
            n_load=obs_space.n_load,
            n_line=obs_space.n_line,
            n_storage=obs_space.n_storage,
        )

    @property
    def num_nodes(self) -> int:
        """Total number of graph nodes: line endpoints + gen + load + storage."""
        return 2 * self.n_line + self.n_gen + self.n_load + self.n_storage


class ObservationConverter(ABC, Generic[T]):
    """
    Abstract base for observation converters. Each converter:
    - Exposes a gymnasium observation space for the RL agent via `observation_space`
    - Converts a grid2op observation to that space via `to_gym`
    - Normalizes the resulting observation via `normalize`
    """

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """The gymnasium observation space this converter produces."""

    @abstractmethod
    def to_gym(self, g2op_obs: BaseObservation) -> T:
        """Convert a grid2op observation to a gym-compatible observation."""

    @abstractmethod
    def normalize(self, gym_obs: T) -> T:
        """Normalize a gym observation using a running mean/variance estimate."""

    def close(self):
        pass

    def reset_obs(self):
        pass


class GraphObservationConverter(ObservationConverter[Dict]):
    """
    Converts grid2op observations into a graph-structured Dict observation:
      - NODES:      node feature matrix       [num_nodes, x_dim]
      - EDGE_INDEX: padded adjacency list     [2, max_num_edges]
      - EDGE_MASK:  boolean mask over edges   [max_num_edges]
      - GLOBAL:     global scalar features    [6]

    Node ordering: [line_or | line_ex | gen | load | storage]

    Edges encode:
      - same-substation + same-bus connectivity (dynamic, topology-dependent)
      - line endpoint pairs (static)

    Node features are normalized per-feature using a running mean/variance
    estimate, updated only when update_normalizer=True is passed to to_gym.
    """

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        if attr_to_observe is None:
            attr_to_observe = _DEFAULT_NODE_FEATURES

        self.attr_to_observe = attr_to_observe
        self._dims = _GridDimensions.from_obs_space(g2op_obs_space)

        num_connections = g2op_obs_space.sub_info
        self._max_num_edges = int((num_connections * (num_connections - 1)).sum()) + 2 * self._dims.n_line
        x_dim = len(attr_to_observe)

        self._observation_space = Dict({
            NODES: Box(
                low=-np.inf, high=np.inf,
                shape=(self._dims.num_nodes, x_dim),
                dtype=np.float32,
            ),
            EDGE_INDEX: Box(
                low=0, high=self._dims.num_nodes - 1,
                shape=(2, self._max_num_edges),
                dtype=np.int64,
            ),
            EDGE_MASK: Box(
                low=0, high=1,
                shape=(self._max_num_edges,),
                dtype=np.bool_,
            ),
            NODE_MASK: Box(
                low=0, high=1,
                shape=(self._dims.num_nodes,),
                dtype=np.bool_,
            ),
            GLOBAL: Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
        })

        # Normalize per feature, pooling across all nodes and timesteps.
        self._normalizer = RunningMeanStd(shape=(x_dim,))
        self._timings: dict[str, float] = {}

        # Precompute static grid topology (node ordering: [line_or | line_ex | gen | load | storage])
        sub_ids = np.concatenate([
            g2op_obs_space.line_or_to_subid,
            g2op_obs_space.line_ex_to_subid,
            g2op_obs_space.gen_to_subid,
            g2op_obs_space.load_to_subid,
            g2op_obs_space.storage_to_subid,
        ])
        # Candidate topology pairs: all (i, j) with i < j in the same substation.
        # Filtering to same-bus + connected is done dynamically; this eliminates the
        # O(N²) same_sub broadcast that was recomputed on every observation step.
        N = self._dims.num_nodes
        ii, jj = np.triu_indices(N, k=1)
        same_sub_mask = sub_ids[ii] == sub_ids[jj]
        self._topo_cand_src = ii[same_sub_mask].astype(np.int64)
        self._topo_cand_dst = jj[same_sub_mask].astype(np.int64)

        line_or_nodes = np.arange(self._dims.n_line)
        line_ex_nodes = np.arange(self._dims.n_line, 2 * self._dims.n_line)
        self._line_edges = np.stack([
            np.concatenate([line_or_nodes, line_ex_nodes]),
            np.concatenate([line_ex_nodes, line_or_nodes]),
        ]).astype(np.int64)

        if verbose:
            logger.info(
                f"GraphObservationConverter: {self._dims.num_nodes} nodes, "
                f"<= {self._max_num_edges} edges, {x_dim} features per node "
                f"({', '.join(attr_to_observe)})."
            )

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        return self._dims.num_nodes

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes across all observations (= num_nodes for this converter)."""
        return self._dims.num_nodes

    @property
    def max_num_edges(self) -> int:
        return self._max_num_edges

    @property
    def x_dim(self) -> int:
        return self._observation_space[NODES].shape[1]

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a graph-structured gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        t0 = time.perf_counter()
        edge_index = self._get_edge_index(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True
        node_mask = np.ones(self._dims.num_nodes, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Args:
            gym_obs: Graph observation dict with raw node features.
        """
        node_features = gym_obs[NODES]  # (num_nodes, x_dim)
        # Update with each node as an independent sample of shape (x_dim,)
        self._normalizer.update(node_features)

        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }

    def _compute_all_node_features(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray[np.float32]]:
        """
        Compute all supported node features, each as a flat array of length
        num_nodes in order [line_or | line_ex | gen | load | storage].
        """
        dims = self._dims
        zeros_line = np.zeros(dims.n_line, dtype=np.float32)
        zeros_gen = np.zeros(dims.n_gen, dtype=np.float32)
        zeros_load = np.zeros(dims.n_load, dtype=np.float32)
        zeros_storage = np.zeros(dims.n_storage, dtype=np.float32)

        if not g2op_obs._is_done:
            t0 = time.perf_counter()
            load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
            self._timings["forecast_ms"] = (time.perf_counter() - t0) * 1000
            # Index 1 = next timestep forecast; sign convention: gen positive, load negative
            gen_p_forecast = prod_p[1].astype(np.float32)
            gen_q_forecast = prod_q[1].astype(np.float32)
            load_p_forecast = -load_p[1].astype(np.float32)
            load_q_forecast = -load_q[1].astype(np.float32)
        else:
            self._timings["forecast_ms"] = 0.0
            gen_p_forecast = zeros_gen
            gen_q_forecast = zeros_gen
            load_p_forecast = zeros_load
            load_q_forecast = zeros_load

        return {
            "active_power_forecast": np.concatenate([
                zeros_line, zeros_line, gen_p_forecast, load_p_forecast, zeros_storage,
            ]),
            "reactive_power_forecast": np.concatenate([
                zeros_line, zeros_line, gen_q_forecast, load_q_forecast, zeros_storage,
            ]),
            # Generator convention: generation positive, consumption negative
            "active_power": np.concatenate([
                g2op_obs.p_or, g2op_obs.p_ex, g2op_obs.gen_p, -g2op_obs.load_p, g2op_obs.storage_power,
            ]),
            "reactive_power": np.concatenate([
                g2op_obs.q_or, g2op_obs.q_ex, g2op_obs.gen_q, -g2op_obs.load_q, zeros_storage,
            ]),
            "voltage": np.concatenate([
                g2op_obs.v_or, g2op_obs.v_ex, g2op_obs.gen_v, g2op_obs.load_v, zeros_storage,
            ]),
            "voltage_angle": np.cos(np.concatenate([
                g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta, g2op_obs.storage_theta,
            ])),
            # rho is a line-level quantity; non-line nodes get 0
            "rho": np.concatenate([
                g2op_obs.rho, g2op_obs.rho, zeros_gen, zeros_load, zeros_storage,
            ]),
        }

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Assemble the node feature matrix for the requested attributes."""
        all_features = self._compute_all_node_features(g2op_obs)
        node_features = np.column_stack([all_features[name] for name in self.attr_to_observe])
        return node_features.astype(np.float32)

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """
        Build the edge index for the current topology.

        Two types of edges:
          1. Same-substation + same-bus pairs (dynamic, changes with topology actions)
          2. Line origin <-> extremity pairs (dynamic, excluded when the line is disconnected)

        Disconnected elements (bus == -1) are excluded from both edge types.

        Complexity: O(N + E) per call instead of O(N²), achieved by grouping
        connected nodes by (substation, bus) key and generating pairs only within
        each group. The old N×N broadcast approach allocates four boolean matrices
        of size N×N on every step, which is ~10× more memory/compute on IEEE36
        (N=177) than on IEEE14 (N=57).
        """
        bus_ids = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])

        # Filter precomputed same-substation candidate pairs by dynamic bus assignment.
        # Both endpoints must be on the same bus and connected (bus > 0).
        src = self._topo_cand_src
        dst = self._topo_cand_dst
        valid = (bus_ids[src] == bus_ids[dst]) & (bus_ids[src] > 0)
        topo_src = src[valid]
        topo_dst = dst[valid]
        topo_edges = np.stack([
            np.concatenate([topo_src, topo_dst]),
            np.concatenate([topo_dst, topo_src]),
        ])

        # Filter line edges: exclude disconnected lines (bus_ids layout is
        # [line_or | line_ex | gen | load | storage], so line_or_bus = bus_ids[:n_line]
        # and line_ex_bus = bus_ids[n_line:2*n_line]).
        n_line = self._dims.n_line
        line_connected = (bus_ids[:n_line] > 0) & (bus_ids[n_line:2 * n_line] > 0)
        # _line_edges columns: first n_line are or→ex, next n_line are ex→or
        line_mask = np.concatenate([line_connected, line_connected])
        active_line_edges = self._line_edges[:, line_mask]

        return np.concatenate([topo_edges, active_line_edges], axis=1)

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year,
            g2op_obs.month,
            g2op_obs.day,
            g2op_obs.hour_of_day,
            g2op_obs.day_of_week,
            g2op_obs.minute_of_hour,
        ], dtype=np.float32)


class HeterogeneousGraphObservationConverter(GraphObservationConverter):
    """
    Extends GraphObservationConverter with typed edges (heterogeneous graph).

    Edges are split into three types, each processed by a separate conv layer:
      0: powerline edges         (line_or <-> line_ex)
      1: same-substation, same-bus   (active physical connectivity)
      2: same-substation, diff-bus   (optional / latent connectivity)

    Adds EDGE_TYPE: [max_num_edges] int64 to the observation dict.
    Node features, normalization, and padding are inherited unchanged.
    """

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        spaces = dict(self._observation_space.spaces)
        spaces[EDGE_TYPE] = Box(
            low=0, high=2,
            shape=(self._max_num_edges,),
            dtype=np.int64,
        )
        self._observation_space = Dict(spaces)

    def _get_edge_index_with_types(self, g2op_obs: BaseObservation) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Build the edge index and per-edge type array for the current topology.

        Returns:
            edge_index: [2, E] connectivity matrix (int64)
            edge_types: [E] type index per edge: 0=powerline, 1=same-bus, 2=optional (int64)
        """
        bus_ids = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])

        src = self._topo_cand_src
        dst = self._topo_cand_dst
        both_connected = (bus_ids[src] > 0) & (bus_ids[dst] > 0)

        # Type 1: same substation, same bus (physical connectivity)
        same_bus = (bus_ids[src] == bus_ids[dst]) & both_connected
        t1_src, t1_dst = src[same_bus], dst[same_bus]

        # Type 2: same substation, different bus (optional connectivity)
        diff_bus = ~(bus_ids[src] == bus_ids[dst]) & both_connected
        t2_src, t2_dst = src[diff_bus], dst[diff_bus]

        # Make bidirectional (candidates are upper-triangle only)
        def _bidir(s: npt.NDArray, d: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
            return np.concatenate([s, d]), np.concatenate([d, s])

        t1_s, t1_d = _bidir(t1_src, t1_dst)
        t2_s, t2_d = _bidir(t2_src, t2_dst)

        # Type 0: line edges (already bidirectional in _line_edges), filter disconnected
        n_line = self._dims.n_line
        line_connected = (bus_ids[:n_line] > 0) & (bus_ids[n_line:2 * n_line] > 0)
        line_mask = np.concatenate([line_connected, line_connected])
        active_line_edges = self._line_edges[:, line_mask]
        n_line_edges = active_line_edges.shape[1]

        edge_index = np.stack([
            np.concatenate([active_line_edges[0], t1_s, t2_s]),
            np.concatenate([active_line_edges[1], t1_d, t2_d]),
        ])
        edge_types = np.concatenate([
            np.zeros(n_line_edges, dtype=np.int64),
            np.ones(len(t1_s), dtype=np.int64),
            np.full(len(t2_s), 2, dtype=np.int64),
        ])
        return edge_index, edge_types

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a heterogeneous graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        node_features = self._get_node_features(g2op_obs)

        t0 = time.perf_counter()
        edge_index, edge_types = self._get_edge_index_with_types(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True
        edge_type_padded = np.zeros(self._max_num_edges, dtype=np.int64)
        edge_type_padded[:num_edges] = edge_types
        node_mask = np.ones(self._dims.num_nodes, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            EDGE_TYPE: edge_type_padded,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """Normalize node features and pass edge types through unchanged."""
        result = super().normalize(gym_obs)
        result[EDGE_TYPE] = gym_obs[EDGE_TYPE]
        return result


# --- Substation graph observation converter ---

class SubstationGraphObservationConverter(GraphObservationConverter):
    """
    Substation-level graph: one node per active busbar, powerlines as edges.

    Each substation contributes 1–2 nodes depending on whether its elements are
    split across bus 1 and bus 2. Padded to max_nodes = 2 * n_sub with NODE_MASK.

    Node features use the same attr_to_observe as GraphObservationConverter,
    aggregated across all elements assigned to each bus:
      - power features (active/reactive): sum
      - voltage, voltage_angle: mean
      - rho: max

    Edges are one bidirectional pair per connected powerline, linking the bus
    nodes at each line's origin and extremity.
    """

    # Aggregation strategy when collapsing per-element features to per-bus nodes.
    # Features not listed here default to "sum".
    _BUS_AGGREGATION: dict[str, str] = {
        "voltage": "mean",
        "voltage_angle": "mean",
        "rho": "max",
    }

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        # super().__init__ sets: _dims, _timings, _normalizer, attr_to_observe,
        # _observation_space, _topo_cand_src/dst, _line_edges (all for element graph).
        # We override the structure-specific parts below.

        n_sub = g2op_obs_space.n_sub
        self._n_sub = n_sub
        self._max_nodes = 2 * n_sub
        self._max_num_edges = 2 * self._dims.n_line  # bidirectional line edges

        # Substation id for each element in flat ordering [line_or|line_ex|gen|load|storage]
        self._sub_ids_flat = np.concatenate([
            g2op_obs_space.line_or_to_subid,
            g2op_obs_space.line_ex_to_subid,
            g2op_obs_space.gen_to_subid,
            g2op_obs_space.load_to_subid,
            g2op_obs_space.storage_to_subid,
        ]).astype(np.int64)

        # Line endpoint substation ids for edge building
        self._line_or_subid = g2op_obs_space.line_or_to_subid.astype(np.int64)
        self._line_ex_subid = g2op_obs_space.line_ex_to_subid.astype(np.int64)

        # Precompute which feature indices use mean/max aggregation (rest default to sum)
        self._mean_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "mean"
        ]
        self._max_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "max"
        ]

        x_dim = len(self.attr_to_observe)
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(self._max_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._max_nodes - 1, shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._max_num_edges,), dtype=np.bool_),
            NODE_MASK: Box(low=0, high=1, shape=(self._max_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self._normalizer = RunningMeanStd(shape=(x_dim,))

        if verbose:
            logger.info(
                f"SubstationGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} max edges, "
                f"{x_dim} features per node ({', '.join(self.attr_to_observe)})."
            )

    @property
    def num_nodes(self) -> int:
        """Maximum number of nodes (active count varies per observation)."""
        return self._max_nodes

    @property
    def max_nodes(self) -> int:
        return self._max_nodes

    @property
    def max_num_edges(self) -> int:
        return self._max_num_edges

    def _get_nodes_and_mask(self, g2op_obs: BaseObservation) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Aggregate per-element features into per-bus node features.

        Node slot formula: 2 * sub_id + (bus - 1) for bus ∈ {1, 2}.
        Disconnected elements (bus == -1) are excluded.

        Returns:
            node_features: [max_nodes, x_dim] float32
            node_mask:     [max_nodes] bool — True for slots with ≥1 element
        """
        all_features = self._compute_all_node_features(g2op_obs)

        bus_ids_flat = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])
        connected = bus_ids_flat > 0
        active_slots = (2 * self._sub_ids_flat + (bus_ids_flat - 1))[connected].astype(np.int64)

        x_dim = len(self.attr_to_observe)
        node_features = np.zeros((self._max_nodes, x_dim), dtype=np.float64)
        node_count = np.zeros(self._max_nodes, dtype=np.int64)
        node_max = np.full((self._max_nodes, x_dim), -np.inf, dtype=np.float64)

        np.add.at(node_count, active_slots, 1)

        for f_idx, feat_name in enumerate(self.attr_to_observe):
            values = all_features[feat_name][connected]
            if self._BUS_AGGREGATION.get(feat_name) == "max":
                np.maximum.at(node_max[:, f_idx], active_slots, values)
            else:  # sum (also used as first step for mean)
                np.add.at(node_features[:, f_idx], active_slots, values)

        node_mask = node_count > 0
        for f_idx in self._mean_feat_indices:
            node_features[node_mask, f_idx] /= node_count[node_mask]
        for f_idx in self._max_feat_indices:
            node_features[node_mask, f_idx] = node_max[node_mask, f_idx]

        return node_features.astype(np.float32), node_mask

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """
        Build edge index: one bidirectional edge per connected powerline.

        Endpoints map to bus node slots: slot = 2 * sub_id + (bus - 1).
        """
        line_or_bus = g2op_obs.line_or_bus
        line_ex_bus = g2op_obs.line_ex_bus
        connected = (line_or_bus > 0) & (line_ex_bus > 0)

        or_slots = (2 * self._line_or_subid + (line_or_bus - 1))[connected]
        ex_slots = (2 * self._line_ex_subid + (line_ex_bus - 1))[connected]

        return np.stack([
            np.concatenate([or_slots, ex_slots]),
            np.concatenate([ex_slots, or_slots]),
        ]).astype(np.int64)

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a substation-level graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_index = self._get_edge_index(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Only active nodes (NODE_MASK=True) update the normalizer. Inactive
        node features are zeroed out after normalization.
        """
        node_features = gym_obs[NODES]   # (max_nodes, x_dim)
        node_mask = gym_obs[NODE_MASK]   # (max_nodes,)

        active_features = node_features[node_mask]
        if active_features.shape[0] > 0:
            self._normalizer.update(active_features)

        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        normalized[~node_mask] = 0.0

        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }


# --- Element graph observation converter ---

class ElementGraphObservationConverter(ObservationConverter[Dict]):
    """
    Element-level graph: one node per physical grid element (gen, load, line,
    storage, bus, ground). Fixed static topology with dynamic edge features.

    Node ordering: [gen | load | line | storage | (ground, bus1, bus2) × n_sub]

    Node features (x_dim=26, heterogeneous, zero-padded per type):
      Slots  0- 4: base — |p|, p, q, |v|, cos(θ)                  (all nodes)
      Slots  5-17: gen  — g_norm, g_maxup, g_maxdown, g_minuptime,
                           g_mindowntime, g_cost, g_startcost,
                           g_shutdowncost, g_type×5                 (generators)
      Slots 18-21: bus  — b_ground, b_bus1, b_bus2, b_cooldown      (buses)
      Slots 22-25: line — ρ, p_tsoverflow, p_tscooldown,
                           p_maintenance                             (powerlines)

    Edges are static (precomputed once): each element is connected to every
    busbar and ground node of its substation(s). Edge feature EDGE_ATTR[e, 0]
    is 1 if the element is currently assigned to that specific bus, 0 otherwise.
    EDGE_MASK is all-True (no padding — edge count is fixed).
    """

    # Feature layout for ElementGraphObservationConverter (x_dim = 26).
    # Slots 0-4: base features shared by all node types.
    # Slots 5-17: generator-specific features (zero for non-generators).
    # Slots 18-21: bus-specific features (zero for non-buses).
    # Slots 22-25: powerline-specific features (zero for non-powerlines).
    _ELEM_X_DIM = 26
    _ELEM_BASE_SLICE = slice(0, 5)  # |p|, p, q, |v|, cos(θ)
    _ELEM_GEN_SLICE = slice(5, 18)  # g_norm, g_maxup, ..., g_type×5
    _ELEM_BUS_SLICE = slice(18, 22)  # b_ground, b_bus1, b_bus2, b_cooldown
    _ELEM_LINE_SLICE = slice(22, 26)  # ρ, p_tsoverflow, p_tscooldown, p_maintenance

    # One-hot generator type encoding order used in grid2op
    _GEN_TYPE_ORDER = ["solar", "wind", "hydro", "thermal", "nuclear"]

    # Bus-slot offsets within each substation block: (ground=0, bus1=1, bus2=2)
    _GROUND_OFFSET = 0
    _BUS1_OFFSET = 1
    _BUS2_OFFSET = 2

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
    ):
        n_gen = g2op_obs_space.n_gen
        n_load = g2op_obs_space.n_load
        n_line = g2op_obs_space.n_line
        n_storage = g2op_obs_space.n_storage
        n_sub = g2op_obs_space.n_sub

        self._n_gen = n_gen
        self._n_load = n_load
        self._n_line = n_line
        self._n_storage = n_storage
        self._n_sub = n_sub

        # Node index offsets for each element group
        self._gen_offset = 0
        self._load_offset = n_gen
        self._line_offset = n_gen + n_load
        self._storage_offset = n_gen + n_load + n_line
        self._bus_offset = n_gen + n_load + n_line + n_storage  # start of (ground, bus1, bus2) blocks
        self._num_nodes = n_gen + n_load + n_line + n_storage + 3 * n_sub

        # Substation membership for each element group
        self._gen_subid = g2op_obs_space.gen_to_subid.astype(np.int64)
        self._load_subid = g2op_obs_space.load_to_subid.astype(np.int64)
        self._line_or_subid = g2op_obs_space.line_or_to_subid.astype(np.int64)
        self._line_ex_subid = g2op_obs_space.line_ex_to_subid.astype(np.int64)
        self._storage_subid = g2op_obs_space.storage_to_subid.astype(np.int64)

        # Static generator features (don't change per step)
        self._static_gen_features = self._build_static_gen_features(g2op_obs_space)

        # Precompute static edge_index and lookup tables for dynamic edge_attr
        self._edge_index, self._edge_element_idx, self._edge_bus_idx, \
            self._fwd_bus_selector = self._build_static_edges()
        self._num_edges = self._edge_index.shape[1]

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._num_nodes, self._ELEM_X_DIM), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._num_nodes - 1,
                            shape=(2, self._num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1,
                           shape=(self._num_edges,), dtype=np.bool_),
            EDGES: Box(low=0, high=1,
                       shape=(self._num_edges, 1), dtype=np.float32),
            NODE_MASK: Box(low=0, high=1,
                           shape=(self._num_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._normalizer = RunningMeanStd(shape=(self._ELEM_X_DIM,))
        self._timings: dict[str, float] = {}

        if verbose:
            logger.info(
                f"ElementGraphObservationConverter: {self._num_nodes} nodes "
                f"({n_gen} gen, {n_load} load, {n_line} line, {n_storage} storage, "
                f"{3 * n_sub} bus/ground), {self._num_edges} edges, "
                f"x_dim={self._ELEM_X_DIM}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def max_nodes(self) -> int:
        return self._num_nodes

    @property
    def x_dim(self) -> int:
        return self._ELEM_X_DIM

    @property
    def num_edges(self) -> int:
        return self._num_edges

    # ------------------------------------------------------------------
    # Static precomputation helpers
    # ------------------------------------------------------------------

    def _bus_node(self, sub_id: int, bus: int) -> int:
        """
        Node index for a busbar slot.

        Args:
            sub_id: Substation index.
            bus:    0=ground, 1=bus1, 2=bus2.
        Returns:
            Absolute node index.
        """
        return self._bus_offset + 3 * sub_id + bus

    def _build_static_gen_features(self, obs_space: ObservationSpace) -> npt.NDArray[np.float32]:
        """
        Precompute the time-invariant generator feature columns (slots 5-17).

        Returns:
            [n_gen, 13] float32 array of static generator features.
        """
        n_gen = self._n_gen
        feats = np.zeros((n_gen, 13), dtype=np.float32)

        # g_norm placeholder (slot 0 of gen block) — computed dynamically per step
        # g_maxup … g_shutdowncost (slots 1-8)
        feats[:, 1] = obs_space.gen_max_ramp_up
        feats[:, 2] = obs_space.gen_max_ramp_down
        feats[:, 3] = obs_space.gen_min_uptime
        feats[:, 4] = obs_space.gen_min_downtime
        feats[:, 5] = obs_space.gen_cost_per_MW
        feats[:, 6] = obs_space.gen_startup_cost
        feats[:, 7] = obs_space.gen_shutdown_cost

        # g_type one-hot (slots 8-12)
        for i, gen_type in enumerate(obs_space.gen_type):
            t = gen_type.lower()
            if t in self._GEN_TYPE_ORDER:
                feats[i, 8 + self._GEN_TYPE_ORDER.index(t)] = 1.0

        return feats

    def _build_static_edges(self) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """
        Build the static (time-invariant) bidirectional edge index.

        Each element is connected to every busbar slot (ground=0, bus1=1, bus2=2)
        at its substation. Lines connect at both endpoints (origin + extremity).
        Loads do NOT connect to ground (disconnecting a load ends the episode).

        Also builds fwd_bus_selector: an index into the per-timestep array
        [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus] for each
        forward edge. Line or-side and ex-side edges point to different slices,
        so _get_edge_attr uses the correct bus assignment for each endpoint.

        Returns:
            edge_index:        [2, E] int64 — (src, dst) pairs
            edge_element_idx:  [E//2] int64 — element node index for each forward edge
            edge_bus_idx:      [E//2] int64 — bus node index for each forward edge
            fwd_bus_selector:  [E//2] int64 — index into concat bus array per forward edge
        """
        srcs, dsts = [], []
        elem_idxs, bus_idxs, selectors = [], [], []

        # Offsets into [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus]
        sel_gen_base = 0
        sel_load_base = self._n_gen
        sel_line_or_base = self._n_gen + self._n_load
        sel_line_ex_base = self._n_gen + self._n_load + self._n_line
        sel_storage_base = self._n_gen + self._n_load + 2 * self._n_line

        def _add(elem_node: int, bus_node: int, selector: int) -> None:
            srcs.extend([elem_node, bus_node])
            dsts.extend([bus_node, elem_node])
            elem_idxs.append(elem_node)
            bus_idxs.append(bus_node)
            selectors.append(selector)

        # Generators: connect to ground, bus1, bus2 at their substation
        for i in range(self._n_gen):
            s = int(self._gen_subid[i])
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._gen_offset + i, self._bus_node(s, b), sel_gen_base + i)

        # Loads: connect to bus1, bus2 only (no ground)
        for i in range(self._n_load):
            s = int(self._load_subid[i])
            for b in (self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._load_offset + i, self._bus_node(s, b), sel_load_base + i)

        # Lines: connect to ground, bus1, bus2 at both origin and extremity substations.
        # Or-side edges use line_or_bus; ex-side edges use line_ex_bus.
        for i in range(self._n_line):
            line_node = self._line_offset + i
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(line_node, self._bus_node(int(self._line_or_subid[i]), b),
                     sel_line_or_base + i)
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(line_node, self._bus_node(int(self._line_ex_subid[i]), b),
                     sel_line_ex_base + i)

        # Storage: connect to ground, bus1, bus2 at their substation
        for i in range(self._n_storage):
            s = int(self._storage_subid[i])
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._storage_offset + i, self._bus_node(s, b), sel_storage_base + i)

        edge_index = np.array([srcs, dsts], dtype=np.int64)
        edge_element_idx = np.array(elem_idxs, dtype=np.int64)
        edge_bus_idx = np.array(bus_idxs, dtype=np.int64)
        fwd_bus_selector = np.array(selectors, dtype=np.int64)
        return edge_index, edge_element_idx, edge_bus_idx, fwd_bus_selector

    # ------------------------------------------------------------------
    # Per-step feature computation
    # ------------------------------------------------------------------

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the [num_nodes, 26] node feature matrix for the current observation.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Node feature matrix [num_nodes, 26] float32.
        """
        X = np.zeros((self._num_nodes, self._ELEM_X_DIM), dtype=np.float32)

        # --- Generators ---
        g_p = g2op_obs.gen_p.astype(np.float32)
        g_q = g2op_obs.gen_q.astype(np.float32)
        g_v = g2op_obs.gen_v.astype(np.float32)
        g_theta = g2op_obs.gen_theta.astype(np.float32)
        g_idx = slice(self._gen_offset, self._gen_offset + self._n_gen)
        X[g_idx, 0] = np.abs(g_p)
        X[g_idx, 1] = g_p
        X[g_idx, 2] = g_q
        X[g_idx, 3] = np.abs(g_v)
        X[g_idx, 4] = np.cos(np.deg2rad(g_theta))
        # Static gen features (slots 5-17), g_norm is dynamic (slot 5)
        X[g_idx, self._ELEM_GEN_SLICE] = self._static_gen_features
        p_min = np.abs(g2op_obs.gen_pmin).astype(np.float32)
        p_max = np.abs(g2op_obs.gen_pmax).astype(np.float32)
        denom = p_max - p_min
        g_norm = np.where(denom > 0, (np.abs(g_p) - p_min) / denom, 0.0)
        X[g_idx, 5] = g_norm  # overwrite slot 5 (g_norm)

        # --- Loads ---
        l_p = g2op_obs.load_p.astype(np.float32)
        l_q = g2op_obs.load_q.astype(np.float32)
        l_v = g2op_obs.load_v.astype(np.float32)
        l_theta = g2op_obs.load_theta.astype(np.float32)
        l_idx = slice(self._load_offset, self._load_offset + self._n_load)
        X[l_idx, 0] = np.abs(l_p)
        X[l_idx, 1] = l_p
        X[l_idx, 2] = l_q
        X[l_idx, 3] = np.abs(l_v)
        X[l_idx, 4] = np.cos(np.deg2rad(l_theta))

        # --- Lines (single node per line, use origin-side electrical quantities) ---
        ln_p = g2op_obs.p_or.astype(np.float32)
        ln_q = g2op_obs.q_or.astype(np.float32)
        ln_v = g2op_obs.v_or.astype(np.float32)
        ln_theta = g2op_obs.theta_or.astype(np.float32)
        ln_idx = slice(self._line_offset, self._line_offset + self._n_line)
        X[ln_idx, 0] = np.abs(ln_p)
        X[ln_idx, 1] = ln_p
        X[ln_idx, 2] = ln_q
        X[ln_idx, 3] = np.abs(ln_v)
        X[ln_idx, 4] = np.cos(np.deg2rad(ln_theta))
        X[ln_idx, 22] = g2op_obs.rho.astype(np.float32)
        X[ln_idx, 23] = g2op_obs.timestep_overflow.astype(np.float32)
        X[ln_idx, 24] = g2op_obs.time_before_cooldown_line.astype(np.float32)
        X[ln_idx, 25] = g2op_obs.duration_next_maintenance.astype(np.float32)

        # --- Storage ---
        if self._n_storage > 0:
            st_p = g2op_obs.storage_power.astype(np.float32)
            st_idx = slice(self._storage_offset, self._storage_offset + self._n_storage)
            # grid2op doesn't expose per-storage voltage/theta directly; use zeros
            X[st_idx, 0] = np.abs(st_p)
            X[st_idx, 1] = st_p

        # --- Bus / Ground nodes ---
        cooldown = g2op_obs.time_before_cooldown_sub.astype(np.float32)  # [n_sub]
        for s in range(self._n_sub):
            ground_node = self._bus_node(s, self._GROUND_OFFSET)
            bus1_node = self._bus_node(s, self._BUS1_OFFSET)
            bus2_node = self._bus_node(s, self._BUS2_OFFSET)
            cd = cooldown[s]
            # one-hot b_type: [b_ground, b_bus1, b_bus2]
            X[ground_node, 18] = 1.0
            X[ground_node, 21] = cd
            X[bus1_node, 19] = 1.0
            X[bus1_node, 21] = cd
            X[bus2_node, 20] = 1.0
            X[bus2_node, 21] = cd

        return X

    def _get_edge_attr(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the dynamic edge attribute array: 1 if the element is currently
        connected to the specific bus node on that edge, else 0.

        Uses _fwd_bus_selector to index into a concatenated bus array
        [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus], so
        or-side and ex-side line edges get the correct endpoint bus assignment.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Edge attribute array [num_edges, 1] float32.
        """
        def _clamp(arr: npt.NDArray) -> npt.NDArray[np.int64]:
            """Map bus -1 (disconnected) → 0 (ground slot)."""
            return np.where(arr > 0, arr, 0).astype(np.int64)

        storage_bus = (
            _clamp(g2op_obs.storage_bus) if self._n_storage > 0
            else np.zeros(0, dtype=np.int64)
        )
        # Layout: [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus]
        all_buses = np.concatenate([
            _clamp(g2op_obs.gen_bus),
            _clamp(g2op_obs.load_bus),
            _clamp(g2op_obs.line_or_bus),
            _clamp(g2op_obs.line_ex_bus),
            storage_bus,
        ])

        bus_slot = (self._edge_bus_idx - self._bus_offset) % 3  # 0=ground, 1=bus1, 2=bus2
        active = (all_buses[self._fwd_bus_selector] == bus_slot).astype(np.float32)

        # edge_index interleaves (elem→bus, bus→elem) per edge pair, so forward
        # edges land at even positions and reverse edges at odd positions.
        attr_full = np.zeros(self._num_edges, dtype=np.float32)
        attr_full[0::2] = active
        attr_full[1::2] = active

        return attr_full[:, np.newaxis]

    # ------------------------------------------------------------------
    # ObservationConverter interface
    # ------------------------------------------------------------------

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to an element-level graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_attr = self._get_edge_attr(g2op_obs)
        self._timings["edge_attr_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)
        node_mask = np.ones(self._num_nodes, dtype=np.bool_)
        edge_mask = np.ones(self._num_edges, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: self._edge_index,
            EDGE_MASK: edge_mask,
            EDGES: edge_attr,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Args:
            gym_obs: Graph observation dict with raw node features.
        """
        node_features = gym_obs[NODES]  # [num_nodes, 26]
        self._normalizer.update(node_features)
        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            EDGES: gym_obs[EDGES],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year, g2op_obs.month, g2op_obs.day,
            g2op_obs.hour_of_day, g2op_obs.day_of_week, g2op_obs.minute_of_hour,
        ], dtype=np.float32)


# --- Flat observation converter ---

class FlatObservationConverter(ObservationConverter[gym.spaces.Dict]):
    """
    Converts grid2op observations into a flat Dict observation where each key
    corresponds to a named g2op attribute (e.g. rho, p_or, load_p).

    Features are normalized online using a running mean and variance estimate
    over the concatenated flat observation vector.
    """

    def __init__(
        self,
        gym_env: GymEnv,
        attr_to_observe: list[str],
        verbose: bool = False,
    ):
        self.attr_to_observe = attr_to_observe

        # Use the gym env's obs space only to derive attribute shapes and raw extraction
        self._raw_obs_space = gym_env.observation_space.keep_only_attr(attr_to_observe)

        # Build observation space: unbounded boxes since normalization is applied at runtime
        self._observation_space = gym.spaces.Dict({
            attr: Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32)
            for attr, space in self._raw_obs_space.spaces.items()
        })

        # Single running normalizer over the full concatenated flat vector
        total_dim = sum(int(np.prod(space.shape)) for space in self._raw_obs_space.spaces.values())
        self.normalizer = RunningMeanStd(shape=(total_dim,))

        if verbose:
            logger.info(
                f"FlatObservationConverter: {len(attr_to_observe)} attributes, "
                f"{total_dim} total features ({', '.join(attr_to_observe)})."
            )

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def to_gym(self, g2op_obs: BaseObservation) -> dict:
        raw = dict(self._raw_obs_space.to_gym(g2op_obs))
        return self.normalize(raw)

    def normalize(self, gym_obs: dict) -> dict:
        flat = np.concatenate([
            np.asarray(gym_obs[attr], dtype=np.float64).ravel()
            for attr in self.attr_to_observe
        ])
        self.normalizer.update(flat[np.newaxis])
        normalized = (flat - self.normalizer.mean) / np.sqrt(self.normalizer.var + 1e-8)

        result = {}
        offset = 0
        for attr in self.attr_to_observe:
            shape = self._observation_space[attr].shape
            size = int(np.prod(shape))
            result[attr] = normalized[offset:offset + size].reshape(shape).astype(np.float32)
            offset += size
        return result


# --- Factory ---
def make_observation_converter(gym_env: GymEnv, env_config: dict) -> ObservationConverter:
    """Construct the appropriate ObservationConverter from env_config."""
    mode = env_config.get("observation_space", "FlatSpace")
    if mode == "GraphObsSpace" or mode == "BusConnectivityGraphObsSpace":
        return GraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "HeterogeneousGraphObsSpace":
        return HeterogeneousGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "SubstationGraphObsSpace":
        return SubstationGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "ElementGraphObsSpace":
        return ElementGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            verbose=env_config.get("verbose", False),
        )
    elif mode == "FlatObsSpace":
        return FlatObservationConverter(
            gym_env=gym_env,
            attr_to_observe=get_attr_list(env_config.get("g2op_input", ["p_i", "p_l", "q_i", "q_l", "a", "v_i", "v_l", "theta_i", "theta_l","r", "t"])),
            verbose=env_config.get("verbose", False),
        )
    else:
        raise ValueError(f"Unknown observation space type: {mode}")

if __name__ == "__main__":
    import grid2op
    env = grid2op.make("l2rpn_wcci_2022")
    graph = GraphObservationConverter(env.observation_space)
    print(graph.num_nodes)
