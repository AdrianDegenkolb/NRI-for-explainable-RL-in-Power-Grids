"""
Converters that transform grid2op observations into gymnasium Dict observations for RL agents.
Each converter defines its own observation space and handles the g2op → gym conversion.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from grid2op.Observation import BaseObservation, ObservationSpace
from grid2op.gym_compat import GymEnv
from gymnasium.wrappers.normalize import RunningMeanStd

from src.grid2op_env.utils import get_attr_list

logger = logging.getLogger(__name__)

T = TypeVar("T")

NODES = "node_features"
EDGES = "edge_features"
EDGE_INDEX = "edge_index"
EDGE_MASK = "edge_mask"
GLOBAL = "global_features"


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


# --- Graph observation converter ---

_DEFAULT_NODE_FEATURES = [
    "active_power_forecast",
    "reactive_power_forecast",
    "active_power",
    "reactive_power",
    "voltage",
    "voltage_angle",
    "current",
    "rho",
]


class GraphObservationConverter(ObservationConverter[Dict]):
    """
    Converts grid2op observations into a graph-structured Dict observation:
      - NODES:      node feature matrix [num_nodes, x_dim]
      - EDGE_INDEX: padded adjacency list [2, max_num_edges]
      - EDGE_MASK:  boolean mask over EDGE_INDEX [max_num_edges]
      - GLOBAL:     global scalar features [6]

    Nodes represent loads, generators, and powerline-bus-connections.
    Edges encode bus-connectivity within substations and powerline connections.
    Node features are normalized using a running mean and variance estimate.
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

        n_gen = g2op_obs_space.n_gen
        n_load = g2op_obs_space.n_load
        n_line = g2op_obs_space.n_line
        self._num_nodes = n_gen + n_load + 2 * n_line
        num_connections = g2op_obs_space.sub_info
        self._max_num_edges = int((num_connections * (num_connections - 1)).sum()) + 2 * n_line
        x_dim = len(attr_to_observe)

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(self._num_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._num_nodes, shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._max_num_edges,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self.normalizer = RunningMeanStd(shape=(self._num_nodes, x_dim))

        if verbose:
            logger.info(
                f"GraphObservationConverter: {self._num_nodes} nodes, "
                f"<= {self._max_num_edges} edges, {x_dim} features per node "
                f"({', '.join(attr_to_observe)})."
            )

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def max_num_edges(self) -> int:
        return self._max_num_edges

    @property
    def x_dim(self) -> int:
        return self._observation_space[NODES].shape[1]

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
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

    def normalize(self, gym_obs: Dict) -> Dict:
        self.normalizer.update(gym_obs[NODES])
        mean, var = self.normalizer.mean, self.normalizer.var
        return {
            NODES: (gym_obs[NODES] - mean) / np.sqrt(var + 1e-8),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }

    @staticmethod
    def _compute_all_node_features(g2op_obs: BaseObservation) -> dict[str, np.ndarray]:
        """Compute all available node features keyed by name."""
        n_line, n_gen, n_load = g2op_obs.n_line, g2op_obs.n_gen, g2op_obs.n_load

        if not g2op_obs._is_done:
            P_MW = np.concatenate([g2op_obs.gen_p, g2op_obs.load_p])
            Q_MVar = np.concatenate([g2op_obs.gen_q, g2op_obs.load_q])
            V_kV = np.concatenate([g2op_obs.gen_v, g2op_obs.load_v])
            theta_deg = np.concatenate([g2op_obs.gen_theta, g2op_obs.load_theta])

            S = (P_MW + 1j * Q_MVar) * 1e6
            V_mag = V_kV * 1e3
            theta_rad = np.deg2rad(theta_deg)
            V_phasor = V_mag * (np.cos(theta_rad) + 1j * np.sin(theta_rad))
            I_phasor = np.conj(S) / (np.sqrt(3) * V_phasor)
            I_mag = np.abs(I_phasor)

            load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
            active_power_forecast = np.concatenate([np.zeros((2 * n_line,)), prod_p[1], -load_p[1]])
            reactive_power_forecast = np.concatenate([np.zeros((2 * n_line,)), prod_q[1], -load_q[1]])
        else:
            I_mag = np.zeros(n_gen + n_load)
            active_power_forecast = np.zeros(2 * n_line + n_gen + n_load)
            reactive_power_forecast = np.zeros(2 * n_line + n_gen + n_load)

        return {
            "active_power_forecast": active_power_forecast,
            "reactive_power_forecast": reactive_power_forecast,
            "active_power": np.concatenate([g2op_obs.p_or, g2op_obs.p_ex, g2op_obs.gen_p, -g2op_obs.load_p]),
            "reactive_power": np.concatenate([g2op_obs.q_or, g2op_obs.q_ex, g2op_obs.gen_q, -g2op_obs.load_q]),
            "voltage": np.concatenate([g2op_obs.v_or, g2op_obs.v_ex, g2op_obs.gen_v, g2op_obs.load_v]),
            "voltage_angle": np.concatenate([g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta]),
            "current": np.concatenate([g2op_obs.a_or, g2op_obs.a_ex, I_mag]),
            "rho": np.concatenate([g2op_obs.rho, g2op_obs.rho, np.zeros(n_gen + n_load)]),
        }

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        all_features = self._compute_all_node_features(g2op_obs)
        node_features = np.column_stack([all_features[name] for name in self.attr_to_observe])
        return node_features.astype(np.float32)

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int32]:
        connected_to_sub = np.concatenate([
            g2op_obs.line_or_to_subid, g2op_obs.line_ex_to_subid,
            g2op_obs.gen_to_subid, g2op_obs.load_to_subid,
        ])
        connected_to_bus = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
        ])

        edge_index = []
        for i in range(self._num_nodes):
            for j in range(i + 1, self._num_nodes):
                if connected_to_bus[i] == connected_to_bus[j] and connected_to_sub[i] == connected_to_sub[j]:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        for i in range(g2op_obs.n_line):
            edge_index.append([i, i + g2op_obs.n_line])
            edge_index.append([i + g2op_obs.n_line, i])

        return np.array(edge_index).transpose().astype(np.int32)

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
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
