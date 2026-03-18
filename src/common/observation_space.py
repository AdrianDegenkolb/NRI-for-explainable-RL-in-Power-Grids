"""
This script contains observation space classes that transform the observation from grid2op into gymnasium Dict-like
observations. The dict structures the data into node-data, edge-data, global data, edge index and edge mask.
The edge index is an adjacency list of shape [2, MAX_NUM_EDGES].
The edge mask is a boolean mask of shape [MAX_NUM_EDGES,]
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from grid2op.Observation import BaseObservation, ObservationSpace
from gymnasium.spaces import Dict, Box
from torch_geometric.data import Data

NODES = "node_features"
EDGES = "edge_features"
EDGE_INDEX = "edge_index"
EDGE_MASK = "edge_mask"
GLOBAL = "global_features"

logger = logging.getLogger(__name__)


class GraphObservationSpace(ABC, Dict):
    """
    Superclass for any Graph observation space. Implements a dict observation space with the following spaces:
    - NODES node features shaped [num_nodes, x_dim]
    - EDGE_INDEX node adjacency with a fix amount of edges shaped [2, max_n_edge]
    - EDGE_MASK encodes the edges within EDGE_INDEX that are non-padding shaped [num_edges]
    - EDGES (optional) edge features shaped [max_n_edge, e_dim]
    - GLOBAL (optional) global features [...]
    """
    @abstractmethod
    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Transforms the grid2op observation instance into a dict observation compatible with the gymnasium API.
        :param g2op_obs: The grid2op observation.
        :return: a gym-like dict observation.
        """

    def close(self):
        pass  # this is just dictated by grid2op - of course without interface...

    @property
    def num_nodes(self) -> int:
        """
        @return: the number of nodes in the observation space
        """
        return self.spaces[NODES].shape[0]

    @property
    def max_num_edges(self) -> int:
        """
        @return: the maximum number of edges in the observation space
        """
        return self.spaces[EDGE_INDEX].shape[1]

    @property
    def x_dim(self) -> int:
        """
        @return: the node feature dimensionality
        """
        return self.spaces[NODES].shape[1]

    @property
    def e_dim(self) -> Optional[int]:
        """
        @return: edge feature dimensionality if this observation space contains the EDGE space else None.
        """
        return self.spaces[EDGES].shape[1] if EDGES in self.spaces.keys() else None

    @property
    @abstractmethod
    def node_feature_names(self):
        pass


class BusConnectivityGraphObsSpace(GraphObservationSpace):
    """
    This observation space outputs a graph structured as follows:
    - loads, generators and powerline-bus-connections are modelled as nodes
    - these nodes are adjacent based on their connectivity to the buses
    Node features include:
    - active/reactive power,
    - voltage, voltage angle
    - current
    equivalent to https://beta-grid2op.readthedocs.io/en/latest/grid_graph.html#graph3-the-connectivity-graph
    """
    def __init__(self, grid2op_observation_space: ObservationSpace, normalization_boundaries: Optional[dict] = None, verbose: bool = False):
        obs_space = grid2op_observation_space
        num_node = obs_space.n_gen + obs_space.n_load + 2 * obs_space.n_line
        num_connections = obs_space.sub_info
        num_line = obs_space.n_line
        max_n_edge = (num_connections * (num_connections - 1)).sum() + 2 * num_line
        x_dim = len(self.node_feature_names)
        global_dim = 6
        if normalization_boundaries is not None:
            self.normalization_min = np.array([normalization_boundaries[name][0] for name in self.node_feature_names])
            self.normalization_max = np.array([normalization_boundaries[name][1] for name in self.node_feature_names])
        else:
            self.normalization_min = None
            self.normalization_max = None

        super().__init__({
            NODES: Box(low=-np.inf, high=np.inf, shape=(num_node, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=1, shape=(2, max_n_edge), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(max_n_edge,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(global_dim, )),
        })

        if verbose:
            logger.info(f"Using graph observation space with {self.num_nodes} nodes, ≤ {self.max_num_edges} edges and {self.x_dim} features per node ({', '.join(self.node_feature_names)}).")

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        # get data
        node_features = self.get_node_features(g2op_obs)
        edge_index = self.get_edge_index(g2op_obs)
        global_features = self.get_global_features(g2op_obs)
        # pad edge index
        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self.max_num_edges), dtype=int)
        edge_index_padded[:, :num_edges] = edge_index
        # add edge_mask
        edge_mask = np.zeros(self.max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True

        return {
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            GLOBAL: global_features
        }

    def get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int32]:
        """
        Compute the [2, E_max] shaped padded edge index. As a padding 0s are added.
        :param g2op_obs: The g2op observation
        :return: The padded edge index
        """
        connected_to_sub = np.concatenate([g2op_obs.line_or_to_subid, g2op_obs.line_ex_to_subid, g2op_obs.gen_to_subid, g2op_obs.load_to_subid])
        connected_to_bus = np.concatenate([g2op_obs.line_or_bus, g2op_obs.line_ex_bus, g2op_obs.gen_bus, g2op_obs.load_bus])

        edge_index = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if connected_to_bus[i] == connected_to_bus[j] and connected_to_sub[i] == connected_to_sub[j]:
                    # nodes share substation and bus -> edge
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        for i in range(g2op_obs.n_line):
            # node i and i + n_line are endpoints of the same powerline and should be connected
            edge_index.append([i, i + g2op_obs.n_line])
            edge_index.append([i + g2op_obs.n_line, i])

        return np.array(edge_index).transpose().astype(np.int32)

    def get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Compute [N, X_dim]-shaped node features from a grid2op observation.
        :param g2op_obs: The g2op observation
        :return: The node features
        """
        # Compute currents for generators and loads
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

            # Concatenate features for all nodes: line ends + generators + loads
            load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
            active_power_forecast = np.concatenate([np.zeros((2 * g2op_obs.n_line,)), prod_p[1], -load_p[1]])
            reactive_power_forecast = np.concatenate([np.zeros((2 * g2op_obs.n_line,)), prod_q[1], -load_q[1]])

        else:
            I_mag = np.zeros(shape=(g2op_obs.n_gen + g2op_obs.n_load, ))
            active_power_forecast = np.zeros((2 * g2op_obs.n_line + g2op_obs.n_gen + g2op_obs.n_load,))
            reactive_power_forecast = np.zeros((2 * g2op_obs.n_line + g2op_obs.n_gen + g2op_obs.n_load,))

        active_power = np.concatenate([g2op_obs.p_or, g2op_obs.p_ex, g2op_obs.gen_p, -g2op_obs.load_p])
        reactive_power = np.concatenate([g2op_obs.q_or, g2op_obs.q_ex, g2op_obs.gen_q, -g2op_obs.load_q])
        voltage = np.concatenate([g2op_obs.v_or, g2op_obs.v_ex, g2op_obs.gen_v, g2op_obs.load_v])
        voltage_angle = np.concatenate([g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta])
        current = np.concatenate([g2op_obs.a_or, g2op_obs.a_ex, I_mag])
        rho = np.concatenate([g2op_obs.rho, g2op_obs.rho, np.zeros((g2op_obs.n_gen + g2op_obs.n_load,))])
        bus_indices = np.concatenate([g2op_obs.line_or_bus, g2op_obs.line_ex_bus, g2op_obs.gen_bus, g2op_obs.load_bus])
        element_indices = np.concatenate([g2op_obs.line_or_pos_topo_vect, g2op_obs.line_or_pos_topo_vect, g2op_obs.gen_pos_topo_vect, g2op_obs.load_pos_topo_vect])
        substation_indices = np.concatenate([g2op_obs.line_or_to_subid, g2op_obs.line_ex_to_subid, g2op_obs.gen_to_subid, g2op_obs.load_to_subid])

        features = [
            active_power_forecast,
            reactive_power_forecast,
            active_power,
            reactive_power,
            voltage,
            voltage_angle,
            current,
            rho,
            #bus_indices,
            #element_indices,
            #substation_indices
        ]

        node_features = np.array(features).transpose()
        return self.normalize(node_features).astype(np.float32)

    def normalize(self, node_features: np.ndarray) -> np.ndarray:
        if self.normalization_max is None:
            return node_features
        else:
            return (node_features - self.normalization_min) / (self.normalization_max - self.normalization_min)



    @staticmethod
    def get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Returns a numpy representation of global features in the observation. The returned numpy array is of shape
        (n, ) where n is the number of global features.
        :param g2op_obs: The observation
        :return: the numpy representation of the global features
        """
        return np.array([
            g2op_obs.year,
            g2op_obs.month,
            g2op_obs.day,
            g2op_obs.hour_of_day,
            g2op_obs.day_of_week,
            g2op_obs.minute_of_hour
        ])

    @property
    def node_feature_names(self) -> List[str]:
        feature_names = [
            "active_power_forecast",
            "reactive_power_forecast",
            "active_power",
            "reactive_power",
            "voltage",
            "voltage_angle",
            "current",
            "rho",
            #"bus_indices",
            #"element_indices",
            #"substation_indices"
        ]

        return feature_names

def gym2pytorch_geometric_data(observation: dict[str, np.ndarray]) -> Data:
    """
    Transforms an UNBATCHED gym-like dict observation into a pytorch geometric data object.
    :param observation: gym-like dict
    :return: A pytorch geometric data object
    """
    node_features = observation[NODES]
    edge_features = observation.get(EDGES)
    global_features = observation.get(GLOBAL)
    edge_index = observation[EDGE_INDEX][:, observation[EDGE_MASK]]
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, global_attr=global_features)
