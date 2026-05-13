from __future__ import annotations

import logging
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

NODES = "node_features"
EDGES = "edge_features"
EDGE_INDEX = "edge_index"
EDGE_MASK = "edge_mask"
GLOBAL = "global_features"

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
      - NODES:      node feature matrix      [num_nodes, x_dim]
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
            GLOBAL: Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
        })

        # Normalize per feature, pooling across all nodes and timesteps.
        self._normalizer = RunningMeanStd(shape=(x_dim,))

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
        node_features = self._get_node_features(g2op_obs)
        edge_index = self._get_edge_index(g2op_obs)
        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True

        return self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            GLOBAL: global_features,
        })

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
            GLOBAL: gym_obs[GLOBAL],
        }

    def _compute_all_node_features(
        self, g2op_obs: BaseObservation
    ) -> dict[str, npt.NDArray[np.float32]]:
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
            load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
            # Index 1 = next timestep forecast; sign convention: gen positive, load negative
            gen_p_forecast = prod_p[1].astype(np.float32)
            gen_q_forecast = prod_q[1].astype(np.float32)
            load_p_forecast = -load_p[1].astype(np.float32)
            load_q_forecast = -load_q[1].astype(np.float32)
        else:
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
            "voltage_angle": np.concatenate([
                g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta, g2op_obs.storage_theta,
            ]),
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
          2. Line origin <-> extremity pairs (static)

        Disconnected elements (bus == -1) are excluded from type-1 edges.
        """
        dims = self._dims

        # Node ordering: [line_or | line_ex | gen | load | storage]
        sub_ids = np.concatenate([
            g2op_obs.line_or_to_subid, g2op_obs.line_ex_to_subid,
            g2op_obs.gen_to_subid, g2op_obs.load_to_subid,
            g2op_obs.storage_to_subid,
        ])
        bus_ids = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])

        # Vectorized same-substation + same-bus edges, excluding disconnected nodes
        connected = bus_ids > 0  # bus == -1 means disconnected
        same_sub = sub_ids[:, None] == sub_ids[None, :]  # (N, N)
        same_bus = bus_ids[:, None] == bus_ids[None, :]  # (N, N)
        valid = connected[:, None] & connected[None, :]  # both endpoints connected
        upper = np.triu(np.ones((dims.num_nodes, dims.num_nodes), dtype=bool), k=1)
        topo_mask = same_sub & same_bus & valid & upper
        topo_src, topo_dst = np.where(topo_mask)
        # Make bidirectional
        topo_edges = np.stack([
            np.concatenate([topo_src, topo_dst]),
            np.concatenate([topo_dst, topo_src]),
        ])  # (2, 2*n_topo_edges)

        # Static line endpoint edges: line_or node i <-> line_ex node i+n_line
        line_or_nodes = np.arange(dims.n_line)
        line_ex_nodes = np.arange(dims.n_line, 2 * dims.n_line)
        line_edges = np.stack([
            np.concatenate([line_or_nodes, line_ex_nodes]),
            np.concatenate([line_ex_nodes, line_or_nodes]),
        ])  # (2, 2*n_line)

        return np.concatenate([topo_edges, line_edges], axis=1).astype(np.int64)

    @staticmethod
    def _get_global_features(
        g2op_obs: BaseObservation,
    ) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year,
            g2op_obs.month,
            g2op_obs.day,
            g2op_obs.hour_of_day,
            g2op_obs.day_of_week,
            g2op_obs.minute_of_hour,
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
    if mode == "GraphObsSpace":
        return GraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
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
