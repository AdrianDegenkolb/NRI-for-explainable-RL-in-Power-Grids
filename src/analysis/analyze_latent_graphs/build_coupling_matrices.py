import sys
from pathlib import Path

import numpy.typing as npt
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

project_root = Path.cwd().parent.parent  # Adjusts for notebook being in src/visualization/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rarl.graph import fully_connected_edge_index


import numpy as np

def ptdf_coupling_matrix(
    PTDF: np.ndarray,          # (n_branches, n_buses)
    node_to_bus: np.ndarray,   # (n_nodes,), -1 allowed for disconnected
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Given the PTDF compute C_{ij}^{PTDF} by the cosine similarity between the vectors s_i=PTDF_{:,b(i)} and s_j=PTDF_{:,b(j)} for all nodes i,j
    if node i is not connected to a bus b(i) is undefined and we set s_i = [0,...0]

    :param PTDF: The PTDF in shape [n_lines, n_bus]
    :param node_to_bus: the mapping b(i), if node i is not connected to any bus b(i) < 0
    :param eps: for numerical stability
    :return: C_{ij}^{PTDF}
    """
    n_br, n_bus = PTDF.shape
    n_nodes = node_to_bus.shape[0]

    # Build S with zeros for disconnected nodes
    S = np.zeros((n_br, n_nodes), dtype=PTDF.dtype) # S has shape [n_line, n_nodes]
    connected = node_to_bus >= 0
    if connected.any():
        idx = node_to_bus[connected].astype(np.int64)
        if idx.min() < 0 or idx.max() >= n_bus:
            raise ValueError(f"node_to_bus out of PTDF range: [{idx.min()}, {idx.max()}] vs n_bus={n_bus}")
        S[:, connected] = PTDF[:, idx]

    # now we have S and want to compute C by the cosine similarity -> S^T * S / normalize
    col_norm = np.linalg.norm(S, axis=0, keepdims=True)  # (1, n_nodes)
    S_unit = S / (col_norm + eps)
    C = S_unit.T @ S_unit  # (n_nodes, n_nodes)

    # Define diagonal: 1 for connected nonzero columns, else 0
    nonzero = (col_norm.reshape(-1) > eps)
    np.fill_diagonal(C, nonzero.astype(C.dtype))
    return C

def build_node_to_bus_mapping(obs: BaseObservation) -> np.ndarray:
    """
    Returns node_to_bus with shape (n_nodes,), in the node order:
      [line_or(0..n_line-1),
       line_ex(0..n_line-1),
       gen(0..n_gen-1),
       load(0..n_load-1)]

    bus_id convention (PTDF column index):
        bus_id = 2*sub_id + (busbar-1), busbar in {1,2}
    If an element is disconnected (topo_vect == -1), bus_id is set to -1.
    """
    def bus_id_from(_sub_id: int, _busbar: int) -> int:
        # busbar: 1 or 2
        return (_busbar - 1) * obs.n_sub + _sub_id

    topo_vect = obs.topo_vect  # (dim_topo,)
    # Lines
    line_or_to_sub = obs.line_or_to_subid
    line_ex_to_sub = obs.line_ex_to_subid
    line_or_pos    = obs.line_or_pos_topo_vect
    line_ex_pos    = obs.line_ex_pos_topo_vect

    # Gens / loads
    gen_to_sub  = obs.gen_to_subid
    load_to_sub = obs.load_to_subid
    gen_pos     = obs.gen_pos_topo_vect
    load_pos    = obs.load_pos_topo_vect

    n_line = len(line_or_to_sub)
    n_gen  = len(gen_to_sub)
    n_load = len(load_to_sub)

    n_nodes = 2 * n_line + n_gen + n_load
    node_to_bus = np.full(n_nodes, -1, dtype=np.int64)

    # --- fill line origins ---
    for l in range(n_line):
        sub = line_or_to_sub[l]
        busbar = topo_vect[line_or_pos[l]]
        if busbar != -1:
            node_to_bus[l] = bus_id_from(sub, busbar)

    # --- fill line ends ---
    offset = n_line
    for l in range(n_line):
        sub = line_ex_to_sub[l]
        busbar = topo_vect[line_ex_pos[l]]
        if busbar != -1:
            node_to_bus[offset + l] = bus_id_from(sub, busbar)

    # --- fill generators ---
    offset = 2 * n_line
    for g in range(n_gen):
        sub = gen_to_sub[g]
        busbar = topo_vect[gen_pos[g]]
        if busbar != -1:
            node_to_bus[offset + g] = bus_id_from(sub, busbar)

    # --- fill loads ---
    offset = 2 * n_line + n_gen
    for d in range(n_load):
        sub = load_to_sub[d]
        busbar = topo_vect[load_pos[d]]
        if busbar != -1:
            node_to_bus[offset + d] = bus_id_from(sub, busbar)

    # Optional sanity check: if your PTDF is [n_branches, 2*n_sub]
    # then all non-negative ids must be < 2*n_sub
    valid = node_to_bus[node_to_bus >= 0]
    if valid.size > 0 and (valid.max() >= 2 * obs.n_sub):
        raise ValueError(
            f"Computed bus_id out of range: max={valid.max()}, expected < {2*obs.n_sub}. "
            f"Check bus indexing / backend bus count."
        )

    return node_to_bus

def get_ptdf_from_env(env: Environment) -> np.ndarray:
    """
    Retrieves the PTDF from the environment
    :param env: the grid2op environment
    :return: the PTDF
    """
    assert isinstance(env.backend, LightSimBackend)
    grid = env.backend._grid
    # run dc powerflow with trivial starting values. PTDF only depends on the network topology not on the operating point
    Vinit = np.ones(grid.total_bus(), dtype=complex)
    _ = grid.dc_pf(Vinit, 10, 1e-8)
    return grid.get_ptdf()

def get_PTDF_coupling_matrix(env: Environment) -> npt.NDArray:
    """
    The PTDF describes the sensitivity of line flows to injections at buses. Each entry PTDF[i, j] indicates how much the flow on line i changes when there is an injection at bus j (and a corresponding withdrawal at the slack bus). This is crucial for understanding how actions that change injections at buses will affect the power flows on the lines, which is essential for making informed decisions in power grid management and control.
    We can use the PTDF to derive a coupling matrix between nodes in the grid, which can be used to analyze how changes at one node (e.g., a generator or load) might affect other nodes (e.g., lines or other generators/loads) in terms of power flow. This coupling matrix can then be compared to learned couplings in an RL agent to see if the agent has implicitly learned the physical relationships in the grid.
    :param env: the grid2op env
    :return C_{ij}^{PTDF} [n_nodes, n_nodes]
    """
    PTDF = get_ptdf_from_env(env)
    assert PTDF.shape[1] == 2 * env.n_sub
    node_to_bus = build_node_to_bus_mapping(env.current_obs)
    C_nodes = ptdf_coupling_matrix(
        PTDF=PTDF,
        node_to_bus=node_to_bus,
    )
    return C_nodes


def get_PTDF_based_coupling_index(env: Environment) -> npt.NDArray:
    """
    Similar to get_coupling_matrix this function returns C_{ij}^{PTDF}. However, it returns it as flat vector with shape [E] where the i-th entry
    corresponds to the i-th fully connected node pair.
    :param env: the grid2op environment
    :return: C_{ij}^{PTDF} [E]
    """
    C_nodes = get_PTDF_coupling_matrix(env)
    edge_index_fully_connected = fully_connected_edge_index(num_nodes=C_nodes.shape[0])
    src = edge_index_fully_connected[0].cpu().numpy()
    dst = edge_index_fully_connected[1].cpu().numpy()
    c_vec = C_nodes[src, dst] # shape [n_edges]
    return c_vec

def compute_node_risk_vector(obs: BaseObservation, PTDF: np.ndarray) -> np.ndarray:
    """
    Computes the risk vector r(s_t) of shape (n_nodes,) for the current observation.

    Node ordering:
      [line_or(0..n_line-1), line_ex(0..n_line-1), gen(0..n_gen-1), load(0..n_load-1)]

    For line endpoint nodes i (origin or extremity of line l(i)):
        r_i = rho_{l(i)}

    For generator/load nodes i connected to bus b(i):
        r_i = sum_l  PTDF[l, b(i)] * rho_l

    For disconnected nodes (topo_vect == -1): r_i = 0.0

    :param obs: current grid2op observation
    :param PTDF: Power Transfer Distribution Factor matrix, shape [n_branches, 2*n_sub]
    :return: risk vector r, shape (n_nodes,)
    """
    n_line = obs.n_line
    n_gen  = obs.n_gen
    n_load = obs.n_load
    n_nodes = 2 * n_line + n_gen + n_load

    rho = obs.rho  # (n_line,) relative line load

    r = np.zeros(n_nodes, dtype=np.float64)

    # --- Line origin nodes ---
    for l in range(n_line):
        if obs.topo_vect[obs.line_or_pos_topo_vect[l]] != -1:
            r[l] = rho[l]

    # --- Line extremity nodes ---
    for l in range(n_line):
        if obs.topo_vect[obs.line_ex_pos_topo_vect[l]] != -1:
            r[n_line + l] = rho[l]

    # --- Generator and load nodes (PTDF-weighted risk) ---
    node_to_bus = build_node_to_bus_mapping(obs)  # (n_nodes,)

    for g in range(n_gen):
        idx = 2 * n_line + g
        b = node_to_bus[idx]
        if b >= 0:
            r[idx] = float(PTDF[:, b] @ rho)

    for d in range(n_load):
        idx = 2 * n_line + n_gen + d
        b = node_to_bus[idx]
        if b >= 0:
            r[idx] = float(PTDF[:, b] @ rho)

    return r


def get_risk_vector(env: Environment) -> npt.NDArray:
    """
    Returns the per-node risk vector r(s_t) of shape [n_nodes] for the current timestep.

    The risk of node i in state s_t is:
        r_i(s_t) = rho_{l(i)}                        if i is a line endpoint of line l(i)
        r_i(s_t) = sum_l  PTDF[l, b(i)] * rho_l      if i is a generator- or load-bus node

    C_{ij}^{risk} = corr_t(r_i(s_t), r_j(s_t)) is then computed in the verifier by
    accumulating these vectors over time and computing pairwise Pearson correlations.

    :param env: the grid2op environment (must have a current observation)
    :return: r(s_t) of shape [n_nodes]
    """
    PTDF = get_ptdf_from_env(env)
    obs  = env.current_obs
    return compute_node_risk_vector(obs, PTDF)
