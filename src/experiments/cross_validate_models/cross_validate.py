import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from tqdm import tqdm

from src.common.observation_space import BusConnectivityGraphObsSpace, EDGE_INDEX
from src.experiments.analyze_latent_graphs.hypo3_action_effect_coupling import get_reconfigured_nodes
from src.experiments.utils import AgentSpec, load_agent_from_spec
from src.visualization import visualize_graph, PlottingArgs, get_node_styles, GridPlottingArgs, visualize_grid

logger = logging.getLogger(__name__)


class CrossValidateResult:
    """
    Represents the result of querying a backup model in steps where model one fails.
    Attributes:
        failing_agent (AgentSpec): The first model specification.
        backup_agent (AgentSpec): The second model specification.
        additional_timesteps (Dict[str, int]): A dictionary mapping chronic ids to the number of timesteps that the backup model managed to survive after the initial model failed.
        failing_agent_completed (Dict[str, bool]): A dictionary mapping chronic ids to whether the failing agent completed the episode.
        backup_agent_completed (Dict[str, Optional[bool]]): A dictionary mapping chronic ids to whether the backup agent completed the episode (None if the failing agent was not used).
        connected_lines_before_failure (Dict[str, List[int]]): A dictionary mapping chronic ids to the list of disconnected lines before failure.
        rhos_before_failure (Dict[str, List[int]]): A dictionary mapping chronic ids to the rho values per line.
    """

    def __init__(self, failing_agent: AgentSpec, backup_agent: AgentSpec, additional_timesteps: Dict[str, Optional[int]] = None, failing_agent_completed: Dict[str, bool] = None, backup_agent_completed: Dict[str, Optional[bool]] = None):
        self.failing_agent = failing_agent
        self.backup_agent = backup_agent
        self.additional_timesteps = additional_timesteps or {}
        self.failing_agent_completed = failing_agent_completed or {}
        self.backup_agent_completed = backup_agent_completed or {}
        self.connected_lines_before_failure: Dict[str, List[int]] = {}
        self.rhos_before_failure: Dict[str, List[int]] = {}
        # Per-node reconfiguration counts accumulated over all evaluated steps [N], set lazily
        self._node_action_counts: Optional[npt.NDArray] = None
        self._total_steps: int = 0
        # Per-substation action counts and acting-step counter for the spatial distribution
        self._sub_action_counts: Optional[npt.NDArray] = None
        self._action_steps: int = 0


def run_until_failure(agent: BaseAgent, g2op_env: Environment, backup_env: Environment, result: CrossValidateResult) -> bool:
    """
    Runs the specified agent on the next chronic until it fails.
    The backup environment is stepped alongside to keep them in sync. After this method the backup environment is at
    the timestep where the main environment failed.
    :param agent: The agent
    :param g2op_env: The environment
    :param backup_env: The backup environment to step alongside the main environment
    :param result: The cross-validation result object to document the results
    :return: whether the agent was able to complete the episode
    """
    obs = g2op_env.reset()
    backup_env.reset()

    chronic_id = g2op_env.chronics_handler.get_name()
    max_chronic_iter = g2op_env.max_episode_duration()

    done = False
    reward = 0
    num_steps = 0
    while True:
        action = agent.act(obs, reward=reward, done=done)

        # --- Track reconfiguration frequency ---
        reconfigured = get_reconfigured_nodes(action, obs)
        if reconfigured:
            N = 2 * obs.n_line + obs.n_gen + obs.n_load
            n_sub = obs.n_sub
            if result._node_action_counts is None:
                result._node_action_counts = np.zeros(N, dtype=np.int64)
            if result._sub_action_counts is None:
                result._sub_action_counts = np.zeros(n_sub, dtype=np.int64)
            for node_idx in reconfigured:
                if 0 <= node_idx < N:
                    result._node_action_counts[node_idx] += 1
            # Map reconfigured nodes → substations and count each substation at most once per step
            touched_subs: set = set()
            n_line = obs.n_line
            n_gen  = obs.n_gen
            for node_idx in reconfigured:
                if node_idx < n_line:                        # line_or
                    touched_subs.add(int(obs.line_or_to_subid[node_idx]))
                elif node_idx < 2 * n_line:                 # line_ex
                    touched_subs.add(int(obs.line_ex_to_subid[node_idx - n_line]))
                elif node_idx < 2 * n_line + n_gen:         # gen
                    touched_subs.add(int(obs.gen_to_subid[node_idx - 2 * n_line]))
                else:                                        # load
                    touched_subs.add(int(obs.load_to_subid[node_idx - 2 * n_line - n_gen]))
            for s in touched_subs:
                result._sub_action_counts[s] += 1
            result._action_steps += 1
        result._total_steps += 1

        new_obs, reward, done, info = g2op_env.step(action)
        num_steps += 1
        if done:
            if num_steps == max_chronic_iter:
                result.failing_agent_completed[chronic_id] = True
                result.connected_lines_before_failure[chronic_id] = None
                result.rhos_before_failure[chronic_id] = None
                result.additional_timesteps[chronic_id] = None
                result.backup_agent_completed[chronic_id] = None
                return True
            else:
                result.failing_agent_completed[chronic_id] = False
                result.connected_lines_before_failure[chronic_id] = obs.line_status.tolist()
                result.rhos_before_failure[chronic_id] = obs.rho.tolist()
                return False

        obs = new_obs
        # Step the backup environment as well to keep them in sync
        backup_env.step(action)


def continue_env_with_backup_agent(backup_g2op_env: Environment, backup_agent: BaseAgent, result: CrossValidateResult):
    """
    Given a backup environment that is already one step before a failing state, continue it with the backup agent until done.
    :param backup_g2op_env: the backup environment
    :param backup_agent: the backup agent
    :param result: a result object to document the results
    """
    chronic_id = backup_g2op_env.chronics_handler.get_name()
    max_chronic_iter = backup_g2op_env.max_episode_duration()

    obs = backup_g2op_env.current_obs
    reward = 0
    done = False
    num_steps = 0
    while True:
        action = backup_agent.act(obs, reward=reward, done=done)
        obs, reward, done, info = backup_g2op_env.step(action)
        num_steps += 1
        if done:
            result.additional_timesteps[chronic_id] = num_steps
            if backup_g2op_env.nb_time_step == max_chronic_iter:
                result.backup_agent_completed[chronic_id] = True
            else:
                result.backup_agent_completed[chronic_id] = False

            return


def cross_validate(failing_agent_spec: AgentSpec, backup_agent_spec: AgentSpec, num_chronics = 50) -> CrossValidateResult:
    """
    Cross-validate two models by evaluating model2 on the timesteps where model1 fails.
    :param failing_agent_spec: the first model specification
    :param backup_agent_spec: the second model specification
    :param num_chronics: how many chronics to evaluate on (randomly sampled)
    :return: a cross-validation result object
    """
    # load models and envs
    failing_agent_load, backup_agent_load = [load_agent_from_spec(agent_spec, "l2rpn_case14_sandbox_test") for agent_spec in [failing_agent_spec, backup_agent_spec]]
    failing_agent, g2op_env, _ = failing_agent_load
    backup_agent, backup_env, _ = backup_agent_load

    # run on evaluation chronics
    result = CrossValidateResult(
        failing_agent=failing_agent_spec,
        backup_agent=backup_agent_spec,
    )
    for _ in tqdm(range(num_chronics), desc=f"Cross-validating {failing_agent_spec.name} with {backup_agent_spec.name}", unit="chronic"):

        try:
            completed = run_until_failure(failing_agent, g2op_env, backup_env, result)
            try:
                if not completed:
                    # continue env from failing state with backup agent
                    continue_env_with_backup_agent(backup_env, backup_agent, result)
            except Exception as e:
                logger.error(
                    f"Error during backup agent evaluation of {backup_agent_spec.name} on chronic {backup_env.chronics_handler.get_name()}, skip this chronic")
                logger.exception(e)
                result.backup_agent_completed[backup_env.chronics_handler.get_name()] = None
                result.additional_timesteps[backup_env.chronics_handler.get_name()] = None

        except Exception as e:
            logger.error(f"Error during rollout of {failing_agent_spec.name} and on chronic {g2op_env.chronics_handler.get_name()}, skip this chronic:")
            logger.exception(e)
            result.failing_agent_completed[g2op_env.chronics_handler.get_name()] = None
            result.backup_agent_completed[g2op_env.chronics_handler.get_name()] = None
            result.additional_timesteps[g2op_env.chronics_handler.get_name()] = None

    return result


def save_cross_validate_results(results: List[CrossValidateResult], save_path: Path):
    """
    Save the cross-validation results to a specified path as a JSON file.
    :param results: list of cross-validation results
    :param save_path: path to save the results
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        import json
        json_results = [
            {
                "failing_agent": result.failing_agent.name,
                "backup_agent": result.backup_agent.name,
                "additional_timesteps": result.additional_timesteps,
                "failing_agent_completed": result.failing_agent_completed,
                "backup_agent_completed": result.backup_agent_completed,
                "lines_before_failure": result.connected_lines_before_failure,
                "rhos_before_failure": result.rhos_before_failure,
            }
            for result in results
        ]
        json.dump(json_results, f, indent=4)


# ---------------------------------------------------------------------------
# Data-computation helpers
# ---------------------------------------------------------------------------

def compute_cross_validation_data(results: List[CrossValidateResult], save_dir: Path) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, List[str], List[str]]:
    """
    Compute the cross-validation heatmap data from *results* and persist it.

    Saved files
    -----------
    cv_map.npy            – 2-D float array (failing_agents × backup_agents), NaN where no pair exists
    cv_std.npy            – 2-D float array with per-cell std of additional timesteps
    cv_n_failures.npy     – 2-D float array with number of failure states per (failing, backup) pair
    cv_rescue_frac.npy    – 2-D float array: fraction of failure states where backup agent took over
                            (rescued = backup_agent_completed is not None)
    cv_failing_agents.npy – 1-D array of failing agent names
    cv_backup_agents.npy  – 1-D array of backup agent names

    :param results: list of cross-validation results
    :param save_dir: directory in which to save the .npy files
    :return: ``(cv_map, cv_std, cv_n_failures, cv_rescue_frac, all_failing_agents, all_backup_agents)``
    """
    all_failing_agents = sorted({r.failing_agent.name for r in results})
    all_backup_agents = sorted({r.backup_agent.name for r in results})

    cv_map         = np.full((len(all_failing_agents), len(all_backup_agents)), np.nan, dtype=float)
    cv_std         = np.full((len(all_failing_agents), len(all_backup_agents)), np.nan, dtype=float)
    cv_n_failures  = np.full((len(all_failing_agents), len(all_backup_agents)), np.nan, dtype=float)
    cv_rescue_frac = np.full((len(all_failing_agents), len(all_backup_agents)), np.nan, dtype=float)

    for result in results:
        i = all_failing_agents.index(result.failing_agent.name)
        j = all_backup_agents.index(result.backup_agent.name)

        # --- additional-timestep statistics (existing) ---
        valid = [s for s in result.additional_timesteps.values() if s is not None]
        cv_map[i, j] = np.mean(valid) if valid else np.nan
        cv_std[i, j] = np.std(valid) if valid else np.nan

        # --- failure count and rescue fraction (new) ---
        # A "failure state" is an episode where the failing agent did NOT complete (completed == False).
        # A failure state is "rescued" when the backup agent managed to take over (backup_agent_completed
        # is True or False – i.e. not None, meaning it actually ran at least one step).
        n_failures = sum(1 for v in result.failing_agent_completed.values() if v is False)
        n_rescued  = sum(
            1 for chronic_id, completed in result.failing_agent_completed.items()
            if completed is False and result.backup_agent_completed.get(chronic_id) is not None
        )
        cv_n_failures[i, j]  = float(n_failures)
        cv_rescue_frac[i, j] = (n_rescued / n_failures) if n_failures > 0 else np.nan

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "cv_map.npy",         cv_map)
    np.save(save_dir / "cv_std.npy",         cv_std)
    np.save(save_dir / "cv_n_failures.npy",  cv_n_failures)
    np.save(save_dir / "cv_rescue_frac.npy", cv_rescue_frac)
    np.save(save_dir / "cv_failing_agents.npy", np.array(all_failing_agents))
    np.save(save_dir / "cv_backup_agents.npy",  np.array(all_backup_agents))
    logger.info("Saved cross-validation data to %s", save_dir)

    return cv_map, cv_std, cv_n_failures, cv_rescue_frac, all_failing_agents, all_backup_agents


def compute_failing_edges_data(results: List[CrossValidateResult], save_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute per-agent mean connection rates and rho values from *results* and persist them.

    Saved files (one pair per agent, named after the agent)
    --------------------------------------------------------
    failing_edges_agent_names.npy          – 1-D array of agent names (ordering)
    failing_edges_connection_<agent>.npy   – mean connection rate per powerline  (shape: n_line)
    failing_edges_rho_<agent>.npy          – mean rho per powerline              (shape: n_line)
    failing_edges_pl_edge_index.npy        – powerline edge-index array          (shape: 2 × n_edges)
    failing_edges_powerline_edge_indices.npy – mapping powerline → edge index    (shape: n_line)

    :param results: list of cross-validation results
    :param save_dir: directory in which to save the .npy files
    :return: ``(lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices)``
    """
    import grid2op

    results_with_failures = [r for r in results if r.connected_lines_before_failure]
    if not results_with_failures:
        logger.warning("No failures found – cannot compute failing-edge data.")
        return {}, {}, np.array([]), np.array([])

    lines_connected_before: Dict[str, np.ndarray] = {}
    lines_connected_std: Dict[str, np.ndarray] = {}
    lines_connected_min: Dict[str, np.ndarray] = {}
    lines_connected_max: Dict[str, np.ndarray] = {}
    # Global (time × edge) statistics for connectivity
    conn_global_mean: Dict[str, float] = {}
    conn_global_std:  Dict[str, float] = {}
    conn_global_min:  Dict[str, float] = {}
    rhos_before_failure: Dict[str, np.ndarray] = {}
    rhos_std: Dict[str, np.ndarray] = {}
    rhos_min: Dict[str, np.ndarray] = {}
    rhos_max: Dict[str, np.ndarray] = {}
    # Global (time × edge) statistics for rho
    rho_global_mean: Dict[str, float] = {}
    rho_global_std:  Dict[str, float] = {}
    rho_global_max:  Dict[str, float] = {}

    for result in results_with_failures:
        name = result.failing_agent.name
        conn_arr = np.array([v for v in result.connected_lines_before_failure.values() if v is not None])
        rho_arr = np.array([v for v in result.rhos_before_failure.values() if v is not None])
        lines_connected_before[name] = conn_arr.mean(axis=0)
        lines_connected_std[name] = conn_arr.std(axis=0)
        lines_connected_min[name] = conn_arr.min(axis=0)
        lines_connected_max[name] = conn_arr.max(axis=0)
        # Global stats: flatten over both time and edge dimensions simultaneously
        conn_flat = conn_arr.ravel()
        conn_global_mean[name] = float(conn_flat.mean())
        conn_global_std[name]  = float(conn_flat.std())
        conn_global_min[name]  = float(conn_flat.min())
        rhos_before_failure[name] = rho_arr.mean(axis=0)
        rhos_std[name] = rho_arr.std(axis=0)
        rhos_min[name] = rho_arr.min(axis=0)
        rhos_max[name] = rho_arr.max(axis=0)
        rho_flat = rho_arr.ravel()
        rho_global_mean[name] = float(rho_flat.mean())
        rho_global_std[name]  = float(rho_flat.std())
        rho_global_max[name]  = float(rho_flat.max())

    env = grid2op.make("l2rpn_case14_sandbox_val")
    obs_space = BusConnectivityGraphObsSpace(env.observation_space)
    obs = env.reset()
    gym_obs = obs_space.to_gym(obs)
    pl_edge_index = gym_obs[EDGE_INDEX]
    edge_mask = gym_obs['edge_mask']

    num_powerlines = env.n_line
    num_edges = int(edge_mask.sum())

    powerline_edge_indices: List[int] = []
    for pl_idx in range(num_powerlines):
        line_or_node = pl_idx
        line_ex_node = pl_idx + num_powerlines
        for edge_idx in range(num_edges):
            src, dst = pl_edge_index[:, edge_idx]
            if (src == line_or_node and dst == line_ex_node) or (src == line_ex_node and dst == line_or_node):
                powerline_edge_indices.append(edge_idx)
                break

    powerline_edge_indices_arr = np.array(powerline_edge_indices)

    save_dir.mkdir(parents=True, exist_ok=True)
    agent_names = list(lines_connected_before.keys())
    np.save(save_dir / "failing_edges_agent_names.npy", np.array(agent_names))
    np.save(save_dir / "failing_edges_pl_edge_index.npy", pl_edge_index)
    np.save(save_dir / "failing_edges_powerline_edge_indices.npy", powerline_edge_indices_arr)
    for agent_name in agent_names:
        safe = agent_name.replace(" ", "_")
        np.save(save_dir / f"failing_edges_connection_{safe}.npy", lines_connected_before[agent_name])
        np.save(save_dir / f"failing_edges_connection_std_{safe}.npy", lines_connected_std[agent_name])
        np.save(save_dir / f"failing_edges_connection_min_{safe}.npy", lines_connected_min[agent_name])
        np.save(save_dir / f"failing_edges_connection_max_{safe}.npy", lines_connected_max[agent_name])
        # Global (time × edge) connectivity scalars
        np.save(save_dir / f"failing_edges_connection_global_mean_{safe}.npy", np.array(conn_global_mean[agent_name]))
        np.save(save_dir / f"failing_edges_connection_global_std_{safe}.npy",  np.array(conn_global_std[agent_name]))
        np.save(save_dir / f"failing_edges_connection_global_min_{safe}.npy",  np.array(conn_global_min[agent_name]))
        np.save(save_dir / f"failing_edges_rho_{safe}.npy", rhos_before_failure[agent_name])
        np.save(save_dir / f"failing_edges_rho_std_{safe}.npy", rhos_std[agent_name])
        np.save(save_dir / f"failing_edges_rho_min_{safe}.npy", rhos_min[agent_name])
        np.save(save_dir / f"failing_edges_rho_max_{safe}.npy", rhos_max[agent_name])
        # Global (time × edge) rho scalars
        np.save(save_dir / f"failing_edges_rho_global_mean_{safe}.npy", np.array(rho_global_mean[agent_name]))
        np.save(save_dir / f"failing_edges_rho_global_std_{safe}.npy",  np.array(rho_global_std[agent_name]))
        np.save(save_dir / f"failing_edges_rho_global_max_{safe}.npy",  np.array(rho_global_max[agent_name]))
    logger.info("Saved failing-edges data to %s", save_dir)

    return lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices_arr


# ---------------------------------------------------------------------------
# Reconfiguration frequency
# ---------------------------------------------------------------------------

def compute_reconfiguration_frequency_data(
    results: List[CrossValidateResult],
    save_dir: Path,
) -> Dict[str, npt.NDArray]:
    """
    Collect per-substation action counts from each *failing* agent's rollout and
    normalise them into an empirical probability distribution over substations.

    For each timestep where a non-do-nothing action was taken, the acting substation
    is incremented once.  Dividing by the total number of acting steps gives
    P(action at substation s), which sums to 1 across substations.

    Saved files
    -----------
    reconfig_freq_agent_names.npy           – 1-D array of agent names
    reconfig_freq_<agent>.npy               – probability distribution [n_sub]
    reconfig_sub_counts_<agent>.npy         – raw integer substation count array [n_sub]
    reconfig_action_steps_<agent>.npy       – scalar: total acting steps
    reconfig_counts_<agent>.npy             – raw node-level count array [N] (legacy)
    reconfig_total_steps_<agent>.npy        – scalar: total steps evaluated (legacy)

    :param results: list of cross-validation results (one per agent pair)
    :param save_dir: directory in which to save the .npy files
    :return: mapping agent_name → probability distribution array [n_sub]
    """
    # Aggregate per *failing* agent
    agent_sub_counts: Dict[str, npt.NDArray] = {}
    agent_action_steps: Dict[str, int] = {}
    # Legacy node-level data kept for backward compat
    agent_node_counts: Dict[str, npt.NDArray] = {}
    agent_total_steps: Dict[str, int] = {}

    for result in results:
        name = result.failing_agent.name

        # --- substation-level (new) ---
        if result._sub_action_counts is not None:
            if name in agent_sub_counts:
                agent_sub_counts[name] = agent_sub_counts[name] + result._sub_action_counts
                agent_action_steps[name] = agent_action_steps[name] + result._action_steps
            else:
                agent_sub_counts[name] = result._sub_action_counts.copy()
                agent_action_steps[name] = result._action_steps

        # --- node-level (legacy) ---
        if result._node_action_counts is not None:
            if name in agent_node_counts:
                agent_node_counts[name] = agent_node_counts[name] + result._node_action_counts
                agent_total_steps[name] = agent_total_steps[name] + result._total_steps
            else:
                agent_node_counts[name] = result._node_action_counts.copy()
                agent_total_steps[name] = result._total_steps

    freq: Dict[str, npt.NDArray] = {}
    for name, sub_counts in agent_sub_counts.items():
        acting = max(agent_action_steps[name], 1)
        freq[name] = sub_counts.astype(np.float64) / acting

    save_dir.mkdir(parents=True, exist_ok=True)
    agent_names = list(freq.keys())
    np.save(save_dir / "reconfig_freq_agent_names.npy", np.array(agent_names))
    for name in agent_names:
        safe = name.replace(" ", "_")
        np.save(save_dir / f"reconfig_freq_{safe}.npy", freq[name])
        np.save(save_dir / f"reconfig_sub_counts_{safe}.npy", agent_sub_counts[name])
        np.save(save_dir / f"reconfig_action_steps_{safe}.npy", np.array(agent_action_steps[name]))
        # legacy files
        if name in agent_node_counts:
            np.save(save_dir / f"reconfig_counts_{safe}.npy", agent_node_counts[name])
            np.save(save_dir / f"reconfig_total_steps_{safe}.npy", np.array(agent_total_steps[name]))
    logger.info("Saved reconfiguration frequency data to %s", save_dir)

    return freq


def _shrink_axis_box(ax, left: float = 0.0, right: float = 0.0,
                     bottom: float = 0.0, top: float = 0.0):
    """Shrink an axis inside its allocated cell by fractions of its size."""
    pos = ax.get_position()
    new_x0 = pos.x0 + pos.width * left
    new_y0 = pos.y0 + pos.height * bottom
    new_w = pos.width * (1.0 - left - right)
    new_h = pos.height * (1.0 - bottom - top)
    ax.set_position([new_x0, new_y0, new_w, new_h])


def _paint_single_agent_reconfig(
    agent_name: str,
    freq: npt.NDArray,   # [N] normalised frequency (counts / total_steps)
    pl_edge_index: npt.NDArray,
    save_path: Path,
    show: bool = False,
):
    """
    Render a histogram figure for one agent's reconfiguration frequency.

    The graph visualisation is shown in the combined failing-edges figure
    (column 2); this standalone figure contains only the bar chart and
    its colorbar.
    """
    N = freq.shape[0]
    counts_pct = freq * 100.0          # convert to percent of timesteps

    cmap = mpl.colormaps["YlOrRd"]
    zero_color = (0.85, 0.85, 0.85, 1.0)
    vmax = 0.12
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(7, 4), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.0, 0.05],
        wspace=0.12,
    )
    ax_hist = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    st = fig.suptitle(
        f"Node reconfiguration frequency (%) — {agent_name}",
        x=0.5, y=0.98, ha="center",
    )


    _shrink_axis_box(ax_hist, left=0.03, right=0.03, bottom=0.06, top=0.06)
    _shrink_axis_box(cax, left=0.0, right=0.0, bottom=0.06, top=0.06)

    hist_pos = ax_hist.get_position()
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, hist_pos.y0, cax_pos.width, hist_pos.height])

    x = np.arange(N)
    bar_colors = [zero_color if c <= 0.0 else cmap(norm(c)) for c in counts_pct]
    ax_hist.bar(x, counts_pct, color=bar_colors, edgecolor="none")
    ax_hist.set_xlabel("Node index")
    ax_hist.set_ylabel("Reconfiguration frequency (%)")
    ax_hist.set_xlim(-0.5, N - 0.5)
    ax_hist.margins(x=0.02, y=0.05)
    ax_hist.set_ylim(0.0, vmax)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Reconfiguration\nfrequency (%)")

    fig.subplots_adjust(top=0.86)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05, bbox_extra_artists=[st])
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05,
                bbox_extra_artists=[st])
    if show:
        plt.show()
    plt.close(fig)


def paint_reconfiguration_frequency(
    freq: Dict[str, npt.NDArray],
    pl_edge_index: npt.NDArray,
    save_dir: Path,
    show: bool = False,
):
    """
    Paint per-agent reconfiguration-frequency histogram figures.

    The graph visualisation is embedded in the combined failing-edges figure.
    One ``reconfig_freq_<agent>.png/.svg`` file (histogram only) is created per agent.

    :param freq: mapping agent_name → normalised frequency array [N]
    :param pl_edge_index: powerline edge-index (shape 2 × n_edges)
    :param save_dir: directory in which to save the figures
    :param show: whether to display figures interactively
    """
    for agent_name, f in freq.items():
        safe = agent_name.replace(" ", "_")
        _paint_single_agent_reconfig(
            agent_name=agent_name,
            freq=f,
            pl_edge_index=pl_edge_index,
            save_path=save_dir / f"reconfig_freq_{safe}.svg",
            show=show,
        )


def repaint_reconfiguration_frequency(
    data_dir: Path,
    save_dir: Path,
    show: bool = False,
):
    """
    Load saved reconfiguration-frequency data from *data_dir* and repaint the figures.

    :param data_dir: directory containing the ``reconfig_freq_*.npy`` files
    :param save_dir: directory in which to save the figures
    :param show: whether to display figures interactively
    """
    agent_names: List[str] = np.load(
        data_dir / "reconfig_freq_agent_names.npy", allow_pickle=True
    ).tolist()
    pl_edge_index = np.load(data_dir / "failing_edges_pl_edge_index.npy")

    freq: Dict[str, npt.NDArray] = {}
    for name in agent_names:
        safe = name.replace(" ", "_")
        p = data_dir / f"reconfig_freq_{safe}.npy"
        if p.exists():
            freq[name] = np.load(p)

    paint_reconfiguration_frequency(freq, pl_edge_index, save_dir, show=show)

def paint_cross_validation_results(
    cv_map: np.ndarray,
    cv_std: Optional[np.ndarray],
    all_failing_agents: List[str],
    all_backup_agents: List[str],
    save_path: Path,
    cv_n_failures: Optional[np.ndarray] = None,
    cv_rescue_frac: Optional[np.ndarray] = None,
    show: bool = False,
):
    """
    Render the cross-validation heatmaps from pre-computed data and save them.

    Two subplots are always drawn side by side:
      Left  – average additional timesteps gained when the backup takes over
              (cell annotation: mean ± std)
      Right – rescue fraction: number of failure states successfully continued by
              the backup agent / number of failure states from the failing agent
              (cell annotation: fraction, number of failures shown in parentheses)

    :param cv_map: 2-D float array (failing_agents × backup_agents)
    :param cv_std: 2-D float array with std of additional timesteps (can be None)
    :param all_failing_agents: ordered list of failing-agent names (row labels)
    :param all_backup_agents: ordered list of backup-agent names (column labels)
    :param save_path: where to save the figure
    :param cv_n_failures: 2-D float array with number of failure states per cell (can be None)
    :param cv_rescue_frac: 2-D float array with rescue fraction per cell (can be None)
    :param show: whether to display the figure interactively
    """

    has_rescue = cv_rescue_frac is not None
    ncols = 2 if has_rescue else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 1, 4))
    if ncols == 1:
        axes = [axes]

    # ------------------------------------------------------------------ #
    # Left subplot – average additional timesteps                         #
    # ------------------------------------------------------------------ #
    ax = axes[0]
    im = ax.imshow(cv_map, cmap="viridis")

    ax.set_xticks(np.arange(len(all_backup_agents)))
    ax.set_yticks(np.arange(len(all_failing_agents)))
    ax.set_xticklabels(all_backup_agents)
    ax.set_yticklabels(all_failing_agents)
    ax.set_xlabel("Backup Model")
    ax.set_ylabel("Failing Model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(cv_map.shape[0]):
        for j in range(cv_map.shape[1]):
            if not np.isnan(cv_map[i, j]):
                mean_val = cv_map[i, j]
                if cv_std is not None and not np.isnan(cv_std[i, j]):
                    label = f"{mean_val:.1f}\n±{cv_std[i, j]:.1f}"
                else:
                    label = f"{mean_val:.1f}"
                ax.text(
                    j, i, label,
                    ha="center", va="center",
                    color="white" if mean_val < np.nanmean(cv_map) else "black",
                )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg. Additional Timesteps")
    ax.set_title("Avg. Additional Timesteps\n(Failing → Backup)")

    # ------------------------------------------------------------------ #
    # Right subplot – rescue fraction                                     #
    # ------------------------------------------------------------------ #
    if has_rescue:
        ax2 = axes[1]
        im2 = ax2.imshow(cv_rescue_frac, cmap="RdYlGn", vmin=0.0, vmax=1.0)

        ax2.set_xticks(np.arange(len(all_backup_agents)))
        ax2.set_yticks(np.arange(len(all_failing_agents)))
        ax2.set_xticklabels(all_backup_agents)
        ax2.set_yticklabels(all_failing_agents)
        ax2.set_xlabel("Backup Model")
        ax2.set_ylabel("Failing Model")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        for i in range(cv_rescue_frac.shape[0]):
            for j in range(cv_rescue_frac.shape[1]):
                frac = cv_rescue_frac[i, j]
                if not np.isnan(frac):
                    n_fail = int(cv_n_failures[i, j]) if cv_n_failures is not None and not np.isnan(cv_n_failures[i, j]) else None
                    n_rescued = int(round(frac * n_fail)) if n_fail is not None else None
                    if n_fail is not None and n_rescued is not None:
                        label = f"{frac:.0%}\n({n_rescued}/{n_fail})"
                    else:
                        label = f"{frac:.0%}"
                    # Use dark text on bright cells (high fraction), light text on dark
                    text_color = "black" if frac > 0.5 else "white"
                    ax2.text(j, i, label, ha="center", va="center", color=text_color)

        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label("Rescue Fraction")
        ax2.set_title("Rescue Fraction\n(rescued failures / total failures)")

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def paint_failing_edges_connectivity(
    lines_connected_before: Dict[str, np.ndarray],
    pl_edge_index: np.ndarray,
    powerline_edge_indices: np.ndarray,
    save_path: Path,
    show: bool = False,
):
    """
    Render one figure showing the **line connection rate** before failure for each agent.

    :param lines_connected_before: mapping agent_name → mean connection-rate array (shape: n_line)
    :param pl_edge_index: powerline edge-index array (shape: 2 × n_edges)
    :param powerline_edge_indices: mapping powerline index → edge index in pl_edge_index
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    import grid2op
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    env = grid2op.make("l2rpn_case14_sandbox_val")
    num_edges = pl_edge_index.shape[1]
    connection_cmap = plt.colormaps['RdYlGn']
    neutral_gray = '#808080'

    num_agents = len(lines_connected_before)
    fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 4), constrained_layout=True)
    if num_agents == 1:
        axes = np.array([axes])

    for idx, (agent_name, connection_rates) in enumerate(lines_connected_before.items()):
        edge_colors = [neutral_gray] * num_edges
        edge_widths = [1.0] * num_edges

        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            connection_rate = connection_rates[pl_idx]
            color = connection_cmap((connection_rate - 0.8) / 0.2)
            edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths[edge_idx] = 3.0

        plotting_args = GridPlottingArgs(
            env = env,
            node_size=900,
            font_size=18,
            line_colors=np.array(edge_colors)[powerline_edge_indices],
            line_widths=np.array(edge_widths)[powerline_edge_indices],
            show_legend=False,
        )
        visualize_grid(plotting_args, ax=axes[idx])
        axes[idx].set_title(f"{agent_name}", fontsize=30)

    norm = Normalize(vmin=0.9, vmax=1.0)
    sm = ScalarMappable(cmap=connection_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', pad=0.02, aspect=30, fraction=0.02)
    cbar.set_label('Mean Connection Rate\nBefore Failure', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def paint_failing_edges_rho(
    rhos_before_failure: Dict[str, np.ndarray],
    pl_edge_index: np.ndarray,
    powerline_edge_indices: np.ndarray,
    save_path: Path,
    show: bool = False,
):
    """
    Render one figure showing the **line congestion (rho)** before failure for each agent.

    :param rhos_before_failure: mapping agent_name → mean rho array (shape: n_line)
    :param pl_edge_index: powerline edge-index array (shape: 2 × n_edges)
    :param powerline_edge_indices: mapping powerline index → edge index in pl_edge_index
    :param save_path: where to save the figure
    :param show: whether to display the figure interactively
    """
    import grid2op
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    env = grid2op.make("l2rpn_case14_sandbox_val")
    num_edges = pl_edge_index.shape[1]
    rho_cmap = plt.colormaps['RdYlGn']
    neutral_gray = '#808080'

    num_agents = len(rhos_before_failure)
    fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 4), constrained_layout=True)
    if num_agents == 1:
        axes = np.array([axes])

    for idx, (agent_name, rhos) in enumerate(rhos_before_failure.items()):
        edge_colors = [neutral_gray] * num_edges
        edge_widths = [4.0] * num_edges

        for pl_idx, edge_idx in enumerate(powerline_edge_indices):
            rho = rhos[pl_idx]
            rho_normalized = max(0.0, rho)
            color = rho_cmap(1.0 - rho_normalized)
            edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
            edge_widths[edge_idx] = 3.0

        plotting_args = GridPlottingArgs(
            env=env,
            node_size=900,
            font_size=18,
            node_color='white',
            line_colors=np.asarray(edge_colors)[powerline_edge_indices],
            line_widths=np.asarray(edge_widths)[powerline_edge_indices],
            show_legend=False,
        )
        visualize_grid(plotting_args, ax=axes[idx])
        axes[idx].set_title(f"{agent_name}", fontsize=30)

    norm = Normalize(vmin=0, vmax=1.2)
    sm = ScalarMappable(cmap=rho_cmap.reversed(), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', pad=0.02, aspect=30, fraction=0.02)
    cbar.set_label('Mean Line Congestion (ρ)\nBefore Failure', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# Row labels shown on the left side of the combined figure (one per agent, in order)
_FAILING_EDGES_ROW_LABELS = ["RAPPO", "GNN baseline", "MLP baseline"]


def paint_failing_edges_combined(
    lines_connected_before: Dict[str, np.ndarray],
    rhos_before_failure: Dict[str, np.ndarray],
    pl_edge_index: np.ndarray,
    powerline_edge_indices: np.ndarray,
    save_path: Path,
    freq: Optional[Dict[str, npt.NDArray]] = None,
    show: bool = False,
    rho_global_stats: Optional[Dict[str, Dict[str, float]]] = None,
    conn_global_stats: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Render a **combined** figure with agents as rows and metrics as columns.

    Layout (3 rows × 2-3 columns):
      - Column 0: mean connectivity in failure states
      - Column 1: mean congestion profile in failure states
      - Column 2 (optional): spatial action distribution graph (reconfiguration frequency)
      - Row labels (rotated 90°) on the left: one per agent
      - Column headers on top
      - One horizontal colorbar per column at the bottom (columns 0 and 1 only)

    :param lines_connected_before: mapping agent_name → mean connection-rate array (shape: n_line)
    :param rhos_before_failure: mapping agent_name → mean rho array (shape: n_line)
    :param pl_edge_index: powerline edge-index array (shape: 2 × n_edges)
    :param powerline_edge_indices: mapping powerline index → edge index in pl_edge_index
    :param save_path: where to save the figure
    :param freq: optional mapping agent_name → normalised reconfiguration frequency [N]; when
        provided a third column showing the spatial action distribution graph is added.
    :param show: whether to display the figure interactively
    """
    import grid2op
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize


    env = grid2op.make("l2rpn_case14_sandbox_val")
    num_edges = pl_edge_index.shape[1]
    neutral_gray = '#808080'

    connection_cmap = plt.colormaps['RdYlGn']
    rho_cmap = plt.colormaps['RdYlGn']
    reconfig_cmap = mpl.colormaps["YlOrRd"]
    reconfig_zero_color = (0.85, 0.85, 0.85, 1.0)

    has_freq = freq is not None and len(freq) > 0

    # Agent order: prefer the canonical row-label order, fall back to dict order
    agent_names_conn = list(lines_connected_before.keys())
    # Build ordered list aligned to _FAILING_EDGES_ROW_LABELS where possible
    ordered_agents = []
    for label in _FAILING_EDGES_ROW_LABELS:
        # case-insensitive prefix match
        match = next((a for a in agent_names_conn if a.lower().startswith(label.split()[0].lower())), None)
        if match is not None and match not in ordered_agents:
            ordered_agents.append(match)
    # append any remaining agents not matched by the canonical labels
    for a in agent_names_conn:
        if a not in ordered_agents:
            ordered_agents.append(a)

    # Dynamic connectivity range: vmin = global minimum across all agents
    all_conn_values = np.concatenate([lines_connected_before[a] for a in ordered_agents if a in lines_connected_before])
    conn_vmin = float(np.min(all_conn_values))
    conn_vmax = 1.0

    # Dynamic reconfig range: vmax = global maximum substation probability across all agents.
    # Handles both new [n_sub] format and legacy [N] node-level format.
    if has_freq:
        _n_sub  = env.n_sub
        _n_line = env.n_line
        _n_gen  = env.n_gen
        _n_load = env.n_load
        reconfig_vmax = 0.0
        for _f in freq.values():
            if _f.shape[0] == _n_sub:
                _sub_max = float(_f.max())
            else:
                _sc = np.zeros(_n_sub, dtype=np.float64)
                for i in range(_n_line):
                    _sc[env.line_or_to_subid[i]] += _f[i]
                for i in range(_n_line):
                    _sc[env.line_ex_to_subid[i]] += _f[_n_line + i]
                for i in range(_n_gen):
                    _sc[env.gen_to_subid[i]] += _f[2 * _n_line + i]
                for i in range(_n_load):
                    _sc[env.load_to_subid[i]] += _f[2 * _n_line + _n_gen + i]
                _tot = _sc.sum()
                _sc = _sc / _tot if _tot > 0 else _sc
                _sub_max = float(_sc.max())
            reconfig_vmax = max(reconfig_vmax, _sub_max)
        if reconfig_vmax == 0.0:
            reconfig_vmax = 1.0
    else:
        reconfig_vmax = 1.0

    # Dynamic rho range: vmax = global maximum rho across all agents and powerlines
    all_rho_values = np.concatenate([rhos_before_failure[a] for a in ordered_agents if a in rhos_before_failure])
    rho_vmax = float(np.max(all_rho_values))
    if rho_vmax == 0.0:
        rho_vmax = 1.0

    # Font sizes
    LABEL_FS = 22    # column / row labels
    CBAR_LABEL_FS = 20  # colorbar axis label
    CBAR_TICK_FS  = 18  # colorbar tick numbers

    num_agents = len(ordered_agents)
    col_labels = [
        "Mean connectivity\nin failure states",
        "Mean congestion profile\nin failure states",
    ]
    if has_freq:
        col_labels.append("Spatial action distribution")
    num_cols = len(col_labels)
    # all columns get a bottom colorbar
    num_cbar_cols = num_cols

    cell_w, cell_h = 6, 4
    row_label_w = 1.2   # extra width reserved for row labels
    cbar_h = 0.55       # height reserved for each colorbar strip at the bottom
    cbar_label_pad = 0.65  # extra figure-inches below colorbars for tick + label text

    fig_w = row_label_w + num_cols * cell_w
    fig_h = num_agents * cell_h + cbar_h + cbar_label_pad

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Fractions of total figure height
    cbar_bottom_frac  = cbar_label_pad / fig_h          # space below cbar axes for label
    cbar_top_frac     = (cbar_label_pad + cbar_h) / fig_h  # top of cbar axes

    # GridSpec: num_agents graph rows (colorbars live in a separate gs_cbar below)
    gs = gridspec.GridSpec(
        num_agents, num_cols,
        figure=fig,
        left=row_label_w / fig_w,
        right=0.98,
        top=0.93,
        bottom=cbar_top_frac + 0.01,
        hspace=0.08,
        wspace=0.05,
    )
    # Colorbar GridSpec spans all columns
    gs_cbar = gridspec.GridSpec(
        1, num_cbar_cols,
        figure=fig,
        left=row_label_w / fig_w,
        right=0.98,
        top=cbar_top_frac - 0.01,
        bottom=cbar_bottom_frac,
        hspace=0.0,
        wspace=0.25,
    )

    axes = np.empty((num_agents, num_cols), dtype=object)
    for row in range(num_agents):
        for col in range(num_cols):
            axes[row, col] = fig.add_subplot(gs[row, col])

    cbar_axes = [fig.add_subplot(gs_cbar[0, col]) for col in range(num_cbar_cols)]

    # ------------------------------------------------------------------ #
    # Fill cells  (collect data for combined summary tables below)        #
    # ------------------------------------------------------------------ #
    conn_all: Dict[str, np.ndarray] = {}      # agent → connection_rates [n_line]
    rho_all:  Dict[str, np.ndarray] = {}      # agent → rhos [n_line]
    sub_freq_all: Dict[str, np.ndarray] = {}  # agent → sub_freq [n_sub]

    for row_idx, agent_name in enumerate(ordered_agents):
        # --- column 0: connectivity ---
        if agent_name in lines_connected_before:
            connection_rates = lines_connected_before[agent_name]
            conn_all[agent_name] = connection_rates

            edge_colors = [neutral_gray] * num_edges
            edge_widths = [1.0] * num_edges
            conn_range = conn_vmax - conn_vmin if conn_vmax > conn_vmin else 1e-6
            for pl_idx, edge_idx in enumerate(powerline_edge_indices):
                cr = connection_rates[pl_idx]
                color = connection_cmap((cr - conn_vmin) / conn_range)
                edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
                edge_widths[edge_idx] = 3.0
            plotting_args = GridPlottingArgs(
                env=env,
                node_size=700,
                font_size=14,
                line_colors=np.array(edge_colors)[powerline_edge_indices],
                line_widths=np.array(edge_widths)[powerline_edge_indices],
                show_legend=False,
            )
            visualize_grid(plotting_args, ax=axes[row_idx, 0])

        # --- column 1: rho ---
        if agent_name in rhos_before_failure:
            rhos = rhos_before_failure[agent_name]
            rho_all[agent_name] = rhos

            edge_colors = [neutral_gray] * num_edges
            edge_widths = [4.0] * num_edges
            for pl_idx, edge_idx in enumerate(powerline_edge_indices):
                rho = rhos[pl_idx]
                rho_normalized = max(0.0, rho) / rho_vmax if rho_vmax > 0 else 0.0
                color = rho_cmap(1.0 - rho_normalized)
                edge_colors[edge_idx] = plt.matplotlib.colors.rgb2hex(color[:3])
                edge_widths[edge_idx] = 3.0
            plotting_args = GridPlottingArgs(
                env=env,
                node_size=700,
                font_size=14,
                node_color='white',
                line_colors=np.asarray(edge_colors)[powerline_edge_indices],
                line_widths=np.asarray(edge_widths)[powerline_edge_indices],
                show_legend=False,
            )
            visualize_grid(plotting_args, ax=axes[row_idx, 1])

        # --- column 2: spatial action distribution (per-substation probability) ---
        if has_freq and agent_name in freq:
            n_sub  = env.n_sub
            n_line = env.n_line
            n_gen  = env.n_gen
            n_load = env.n_load
            raw = freq[agent_name]

            if raw.shape[0] == n_sub:
                sub_freq = raw
            else:
                sub_counts = np.zeros(n_sub, dtype=np.float64)
                for i in range(n_line):
                    sub_counts[env.line_or_to_subid[i]] += raw[i]
                for i in range(n_line):
                    sub_counts[env.line_ex_to_subid[i]] += raw[n_line + i]
                for i in range(n_gen):
                    sub_counts[env.gen_to_subid[i]] += raw[2 * n_line + i]
                for i in range(n_load):
                    sub_counts[env.load_to_subid[i]] += raw[2 * n_line + n_gen + i]
                total = sub_counts.sum()
                sub_freq = sub_counts / total if total > 0 else sub_counts

            sub_freq_all[agent_name] = sub_freq

            reconfig_norm = mpl.colors.Normalize(vmin=0.0, vmax=reconfig_vmax)
            sub_colors = [reconfig_cmap(reconfig_norm(sub_freq[s])) for s in range(n_sub)]
            plotting_args = GridPlottingArgs(
                env=env,
                node_size=700,
                font_size=14,
                node_colors=sub_colors,
                font_color='black',
                show_legend=False,
            )
            visualize_grid(plotting_args, ax=axes[row_idx, 2])

    # ------------------------------------------------------------------ #
    # Combined summary tables                                             #
    # ------------------------------------------------------------------ #
    _agents = [a for a in ordered_agents]
    _COL_W  = 10   # width of each agent / numeric column
    _ROW_W  = 5    # width of the index column (line / sub)
    _STAT_W = 9    # width of Mean / Std / Min / Max columns

    def _sep(n_agent_cols: int, extra: int = 3) -> str:
        return "  " + "-" * (_ROW_W + _COL_W * n_agent_cols + _STAT_W * extra)

    # --- Connectivity ---
    if conn_all:
        n_cols = len([a for a in _agents if a in conn_all])
        print(f"\n{'='*70}")
        print("  MEAN CONNECTIVITY IN FAILURE STATES  (1.0 = always connected)")
        _has_cg = conn_global_stats is not None
        if _has_cg:
            print(f"  {'Agent':<20}  {'Mean(lines)':>{_STAT_W}}  {'Std(lines)':>{_STAT_W}}  {'Min(lines)':>{_STAT_W}}  {'Mean(t×e)':>{_STAT_W}}  {'Std(t×e)':>{_STAT_W}}  {'Min(t×e)':>{_STAT_W}}")
            print(f"  {'-'*20}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}")
        else:
            print(f"  {'Agent':<20}  {'Mean':>{_STAT_W}}  {'Std':>{_STAT_W}}  {'Min':>{_STAT_W}}")
            print(f"  {'-'*20}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}")
        for a in _agents:
            if a not in conn_all:
                continue
            v = conn_all[a]
            if _has_cg:
                cg = conn_global_stats.get(a, {})
                gm = cg.get('mean'); gs = cg.get('std'); gn = cg.get('min')
                fmt = lambda x: f"{x:>{_STAT_W}.3f}" if x is not None else f"{'N/A':>{_STAT_W}}"
                print(f"  {a:<20}  {v.mean():>{_STAT_W}.3f}  {v.std():>{_STAT_W}.3f}  {v.min():>{_STAT_W}.3f}  {fmt(gm)}  {fmt(gs)}  {fmt(gn)}")
            else:
                print(f"  {a:<20}  {v.mean():>{_STAT_W}.3f}  {v.std():>{_STAT_W}.3f}  {v.min():>{_STAT_W}.3f}")
        print()
        header = f"  {'Line':>{_ROW_W}}" + "".join(f"{a:>{_COL_W}}" for a in _agents if a in conn_all)
        print(header)
        print(_sep(n_cols, 0))
        n_lines = len(next(iter(conn_all.values())))
        for pl_idx in range(n_lines):
            row = f"  {pl_idx:>{_ROW_W}}"
            for a in _agents:
                if a in conn_all:
                    row += f"{conn_all[a][pl_idx]:>{_COL_W}.3f}"
            print(row)

    # --- Rho ---
    if rho_all:
        n_cols = len([a for a in _agents if a in rho_all])
        print(f"\n{'='*70}")
        print("  MEAN CONGESTION (ρ) IN FAILURE STATES")
        _has_rg = rho_global_stats is not None
        if _has_rg:
            print(f"  {'Agent':<20}  {'Mean(lines)':>{_STAT_W}}  {'Std(lines)':>{_STAT_W}}  {'Max(lines)':>{_STAT_W}}  {'Mean(t×e)':>{_STAT_W}}  {'Std(t×e)':>{_STAT_W}}  {'Max(t×e)':>{_STAT_W}}")
            print(f"  {'-'*20}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}")
        else:
            print(f"  {'Agent':<20}  {'Mean':>{_STAT_W}}  {'Std':>{_STAT_W}}  {'Max':>{_STAT_W}}")
            print(f"  {'-'*20}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}")
        for a in _agents:
            if a not in rho_all:
                continue
            v = rho_all[a]
            if _has_rg:
                rg = rho_global_stats.get(a, {})
                gm = rg.get('mean'); gs = rg.get('std'); gx = rg.get('max')
                fmt = lambda x: f"{x:>{_STAT_W}.3f}" if x is not None else f"{'N/A':>{_STAT_W}}"
                print(f"  {a:<20}  {v.mean():>{_STAT_W}.3f}  {v.std():>{_STAT_W}.3f}  {v.max():>{_STAT_W}.3f}  {fmt(gm)}  {fmt(gs)}  {fmt(gx)}")
            else:
                print(f"  {a:<20}  {v.mean():>{_STAT_W}.3f}  {v.std():>{_STAT_W}.3f}  {v.max():>{_STAT_W}.3f}")
        print()
        header = f"  {'Line':>{_ROW_W}}" + "".join(f"{a:>{_COL_W}}" for a in _agents if a in rho_all)
        print(header)
        print(_sep(n_cols, 0))
        n_lines = len(next(iter(rho_all.values())))
        for pl_idx in range(n_lines):
            row = f"  {pl_idx:>{_ROW_W}}"
            for a in _agents:
                if a in rho_all:
                    row += f"{rho_all[a][pl_idx]:>{_COL_W}.3f}"
            print(row)

    # --- Action distribution ---
    if sub_freq_all:
        n_cols   = len([a for a in _agents if a in sub_freq_all])
        n_sub_pr = len(next(iter(sub_freq_all.values())))
        print(f"\n{'='*70}")
        print("  SPATIAL ACTION DISTRIBUTION  P(action at substation s)  [sums to 1]")
        print(f"  {'Agent':<20}  {'Mean*':>{_STAT_W}}  {'Std':>{_STAT_W}}  {'Max':>{_STAT_W}}  {'Sum':>{_STAT_W}}")
        print(f"  {'(* over acted substations only)':<20}")
        print(f"  {'-'*20}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}  {'-'*_STAT_W}")
        for a in _agents:
            if a not in sub_freq_all:
                continue
            v = sub_freq_all[a]
            acted = v[v > 0.0]
            mean_acted = acted.mean() if len(acted) > 0 else 0.0
            print(f"  {a:<20}  {mean_acted:>{_STAT_W}.3f}  {v.std():>{_STAT_W}.3f}  {v.max():>{_STAT_W}.3f}  {v.sum():>{_STAT_W}.3f}")
        print()
        header = f"  {'Sub':>{_ROW_W}}" + "".join(f"{a:>{_COL_W}}" for a in _agents if a in sub_freq_all)
        print(header)
        print(_sep(n_cols, 0))
        for s in range(n_sub_pr):
            row = f"  {s:>{_ROW_W}}"
            for a in _agents:
                if a in sub_freq_all:
                    row += f"{sub_freq_all[a][s]:>{_COL_W}.3f}"
            print(row)

    # ------------------------------------------------------------------ #
    # Column headers (top of first row)                                   #
    # ------------------------------------------------------------------ #
    for col_idx, col_label in enumerate(col_labels):
        axes[0, col_idx].set_title(col_label, fontsize=LABEL_FS, fontweight='bold', pad=10)

    # ------------------------------------------------------------------ #
    # Row labels (rotated 90°, left of each row)                          #
    # ------------------------------------------------------------------ #
    row_label_texts = _FAILING_EDGES_ROW_LABELS[:num_agents]
    for row_idx in range(num_agents):
        label = row_label_texts[row_idx] if row_idx < len(row_label_texts) else ordered_agents[row_idx]
        ax = axes[row_idx, 0]
        ax.annotate(
            label,
            xy=(0, 0.5),
            xycoords='axes fraction',
            xytext=(-0.18, 0.5),
            textcoords='axes fraction',
            fontsize=LABEL_FS,
            fontweight='bold',
            va='center',
            ha='center',
            rotation=90,
            annotation_clip=False,
        )

    # ------------------------------------------------------------------ #
    # Colorbars at the bottom (horizontal, one per column)                #
    # ------------------------------------------------------------------ #
    tick_fmt = ticker.FormatStrFormatter('%.1f')

    # Connectivity colorbar — 3 ticks with 1 decimal digit
    norm_conn = Normalize(vmin=conn_vmin, vmax=conn_vmax)
    sm_conn = ScalarMappable(cmap=connection_cmap, norm=norm_conn)
    sm_conn.set_array([])
    cbar_conn = fig.colorbar(sm_conn, cax=cbar_axes[0], orientation='horizontal')
    cbar_conn.set_label('Mean Connection Rate Before Failure', fontsize=CBAR_LABEL_FS)
    cbar_conn.ax.tick_params(labelsize=CBAR_TICK_FS)
    cbar_conn.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune=None))
    cbar_conn.ax.xaxis.set_major_formatter(tick_fmt)

    # Rho colorbar
    norm_rho = Normalize(vmin=0, vmax=rho_vmax)
    sm_rho = ScalarMappable(cmap=rho_cmap.reversed(), norm=norm_rho)
    sm_rho.set_array([])
    cbar_rho = fig.colorbar(sm_rho, cax=cbar_axes[1], orientation='horizontal')
    cbar_rho.set_label('Mean Line Congestion (ρ) Before Failure', fontsize=CBAR_LABEL_FS)
    cbar_rho.ax.tick_params(labelsize=CBAR_TICK_FS)
    cbar_rho.ax.xaxis.set_major_formatter(tick_fmt)

    # Spatial action distribution colorbar
    if has_freq and len(cbar_axes) > 2:
        norm_reconfig = Normalize(vmin=0.0, vmax=reconfig_vmax)
        sm_reconfig = ScalarMappable(cmap=reconfig_cmap, norm=norm_reconfig)
        sm_reconfig.set_array([])
        cbar_reconfig = fig.colorbar(sm_reconfig, cax=cbar_axes[2], orientation='horizontal')
        cbar_reconfig.set_label('Action Frequency per Substation', fontsize=CBAR_LABEL_FS)
        cbar_reconfig.ax.tick_params(labelsize=CBAR_TICK_FS)
        cbar_reconfig.ax.xaxis.set_major_formatter(tick_fmt)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.15)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level wrappers (kept for convenience / backward-compatibility)
# ---------------------------------------------------------------------------

def visualize_cross_validation_results(results: List[CrossValidateResult], save_path: Path, show: bool = False):
    """Compute cross-validation data and paint the heatmap in one step."""
    data_dir = save_path.parent
    cv_map, cv_std, cv_n_failures, cv_rescue_frac, failing_agents, backup_agents = compute_cross_validation_data(results, data_dir)
    paint_cross_validation_results(cv_map, cv_std, failing_agents, backup_agents, save_path,
                                   cv_n_failures=cv_n_failures, cv_rescue_frac=cv_rescue_frac, show=show)


def visualize_failing_edges(
    results: List[CrossValidateResult],
    save_path_connectivity: Path,
    save_path_rho: Path,
    save_path_combined: Optional[Path] = None,
    show: bool = False,
):
    """Compute failing-edge data and paint the combined figure (and optionally the legacy single figures)."""
    data_dir = save_path_connectivity.parent
    lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices = (
        compute_failing_edges_data(results, data_dir)
    )
    if not lines_connected_before:
        return
    freq = compute_reconfiguration_frequency_data(results, data_dir)
    # Load the freshly-saved global stats to pass to the combined figure
    _loaded = _load_failing_edges_data(data_dir)
    _agent_names = _loaded[0]
    _conn_gm, _conn_gs, _conn_gn = _loaded[11], _loaded[12], _loaded[13]
    _rho_gm,  _rho_gs,  _rho_gx  = _loaded[14], _loaded[15], _loaded[16]
    rho_global_stats: Optional[Dict[str, Dict[str, float]]] = None
    conn_global_stats: Optional[Dict[str, Dict[str, float]]] = None
    if any(v is not None for v in _rho_gm.values()):
        rho_global_stats = {a: {'mean': _rho_gm[a], 'std': _rho_gs[a], 'max': _rho_gx[a]}
                            for a in _agent_names}
    if any(v is not None for v in _conn_gm.values()):
        conn_global_stats = {a: {'mean': _conn_gm[a], 'std': _conn_gs[a], 'min': _conn_gn[a]}
                             for a in _agent_names}
    # Combined figure (new default)
    _combined = save_path_combined or save_path_connectivity.with_name("failing_edges_combined" + save_path_connectivity.suffix)
    paint_failing_edges_combined(
        lines_connected_before, rhos_before_failure, pl_edge_index, powerline_edge_indices,
        _combined, freq=freq or None, show=show,
        rho_global_stats=rho_global_stats,
        conn_global_stats=conn_global_stats,
    )
    # Keep legacy figures as well
    paint_failing_edges_connectivity(lines_connected_before, pl_edge_index, powerline_edge_indices, save_path_connectivity, show=False)
    paint_failing_edges_rho(rhos_before_failure, pl_edge_index, powerline_edge_indices, save_path_rho, show=False)


# ---------------------------------------------------------------------------
# Repaint helpers – load saved .npy files and recreate figures
# ---------------------------------------------------------------------------

def repaint_cross_validation_results(data_dir: Path, save_path: Path, show: bool = False):
    """
    Load pre-computed cross-validation data from *data_dir* and repaint the heatmap.
    """
    cv_map = np.load(data_dir / "cv_map.npy")
    cv_std_path = data_dir / "cv_std.npy"
    cv_std = np.load(cv_std_path) if cv_std_path.exists() else None
    all_failing_agents = np.load(data_dir / "cv_failing_agents.npy", allow_pickle=True).tolist()
    all_backup_agents  = np.load(data_dir / "cv_backup_agents.npy",  allow_pickle=True).tolist()

    cv_n_failures_path  = data_dir / "cv_n_failures.npy"
    cv_rescue_frac_path = data_dir / "cv_rescue_frac.npy"
    cv_n_failures  = np.load(cv_n_failures_path)  if cv_n_failures_path.exists()  else None
    cv_rescue_frac = np.load(cv_rescue_frac_path) if cv_rescue_frac_path.exists() else None

    paint_cross_validation_results(cv_map, cv_std, all_failing_agents, all_backup_agents, save_path,
                                   cv_n_failures=cv_n_failures, cv_rescue_frac=cv_rescue_frac, show=show)


def _load_failing_edges_data(data_dir: Path):
    agent_names: List[str] = np.load(
        data_dir / "failing_edges_agent_names.npy", allow_pickle=True
    ).tolist()
    pl_edge_index          = np.load(data_dir / "failing_edges_pl_edge_index.npy")
    powerline_edge_indices = np.load(data_dir / "failing_edges_powerline_edge_indices.npy")

    lines_connected_before: Dict[str, np.ndarray] = {}
    lines_connected_std:    Dict[str, np.ndarray] = {}
    lines_connected_min:    Dict[str, np.ndarray] = {}
    lines_connected_max:    Dict[str, np.ndarray] = {}
    # Global (time × edge) scalars – optional, may not exist in older data
    conn_global_mean: Dict[str, Optional[float]] = {}
    conn_global_std:  Dict[str, Optional[float]] = {}
    conn_global_min:  Dict[str, Optional[float]] = {}
    rhos_before_failure:    Dict[str, np.ndarray] = {}
    rhos_std:               Dict[str, np.ndarray] = {}
    rhos_min:               Dict[str, np.ndarray] = {}
    rhos_max:               Dict[str, np.ndarray] = {}
    rho_global_mean: Dict[str, Optional[float]] = {}
    rho_global_std:  Dict[str, Optional[float]] = {}
    rho_global_max:  Dict[str, Optional[float]] = {}

    for agent_name in agent_names:
        safe = agent_name.replace(" ", "_")
        lines_connected_before[agent_name] = np.load(data_dir / f"failing_edges_connection_{safe}.npy")
        rhos_before_failure[agent_name]    = np.load(data_dir / f"failing_edges_rho_{safe}.npy")
        for dst, key in [
            (lines_connected_std, f"failing_edges_connection_std_{safe}.npy"),
            (lines_connected_min, f"failing_edges_connection_min_{safe}.npy"),
            (lines_connected_max, f"failing_edges_connection_max_{safe}.npy"),
            (rhos_std,            f"failing_edges_rho_std_{safe}.npy"),
            (rhos_min,            f"failing_edges_rho_min_{safe}.npy"),
            (rhos_max,            f"failing_edges_rho_max_{safe}.npy"),
        ]:
            p = data_dir / key
            dst[agent_name] = np.load(p) if p.exists() else None
        # Global scalars (present only in data produced with the updated code)
        for dst, key in [
            (conn_global_mean, f"failing_edges_connection_global_mean_{safe}.npy"),
            (conn_global_std,  f"failing_edges_connection_global_std_{safe}.npy"),
            (conn_global_min,  f"failing_edges_connection_global_min_{safe}.npy"),
            (rho_global_mean,  f"failing_edges_rho_global_mean_{safe}.npy"),
            (rho_global_std,   f"failing_edges_rho_global_std_{safe}.npy"),
            (rho_global_max,   f"failing_edges_rho_global_max_{safe}.npy"),
        ]:
            p = data_dir / key
            dst[agent_name] = float(np.load(p)) if p.exists() else None

    return (
        agent_names, pl_edge_index, powerline_edge_indices,
        lines_connected_before, lines_connected_std, lines_connected_min, lines_connected_max,
        rhos_before_failure, rhos_std, rhos_min, rhos_max,
        conn_global_mean, conn_global_std, conn_global_min,
        rho_global_mean, rho_global_std, rho_global_max,
    )


def repaint_failing_edges(
    data_dir: Path,
    save_path_connectivity: Path,
    save_path_rho: Path,
    save_path_combined: Optional[Path] = None,
    show: bool = False,
):
    """
    Load pre-computed failing-edge data from *data_dir* and repaint the combined figure
    (and optionally the legacy individual figures).
    """
    loaded = _load_failing_edges_data(data_dir)
    agent_names, pl_edge_index, powerline_edge_indices = loaded[0], loaded[1], loaded[2]
    lines_connected_before = loaded[3]
    rhos_before_failure = loaded[7]
    # Global (time × edge) stats – present only in data produced with updated code
    _conn_gm, _conn_gs, _conn_gn = loaded[11], loaded[12], loaded[13]
    _rho_gm,  _rho_gs,  _rho_gx  = loaded[14], loaded[15], loaded[16]
    # Build per-agent dicts for paint_failing_edges_combined
    rho_global_stats: Optional[Dict[str, Dict[str, float]]] = None
    conn_global_stats: Optional[Dict[str, Dict[str, float]]] = None
    if any(v is not None for v in _rho_gm.values()):
        rho_global_stats = {a: {'mean': _rho_gm[a], 'std': _rho_gs[a], 'max': _rho_gx[a]}
                            for a in agent_names}
    if any(v is not None for v in _conn_gm.values()):
        conn_global_stats = {a: {'mean': _conn_gm[a], 'std': _conn_gs[a], 'min': _conn_gn[a]}
                             for a in agent_names}

    # Load reconfiguration frequency data if available
    freq: Optional[Dict[str, npt.NDArray]] = None
    reconfig_names_path = data_dir / "reconfig_freq_agent_names.npy"
    if reconfig_names_path.exists():
        reconfig_names: List[str] = np.load(reconfig_names_path, allow_pickle=True).tolist()
        freq = {}
        for name in reconfig_names:
            safe = name.replace(" ", "_")
            p = data_dir / f"reconfig_freq_{safe}.npy"
            if p.exists():
                freq[name] = np.load(p)
        if not freq:
            freq = None

    _combined = save_path_combined or save_path_connectivity.with_name(
        "failing_edges_combined" + save_path_connectivity.suffix
    )
    paint_failing_edges_combined(
        lines_connected_before, rhos_before_failure,
        pl_edge_index, powerline_edge_indices,
        _combined, freq=freq, show=show,
        rho_global_stats=rho_global_stats,
        conn_global_stats=conn_global_stats,
    )
    paint_failing_edges_connectivity(lines_connected_before, pl_edge_index, powerline_edge_indices, save_path_connectivity, show=False)
    paint_failing_edges_rho(rhos_before_failure, pl_edge_index, powerline_edge_indices, save_path_rho, show=False)


def print_table_rho(data_dir: Path):
    from tabulate import tabulate as _tabulate
    loaded = _load_failing_edges_data(data_dir)
    agent_names, _pl, powerline_edge_indices = loaded[0], loaded[1], loaded[2]
    rhos, rhos_std, rhos_min, rhos_max = loaded[7], loaded[8], loaded[9], loaded[10]
    n_lines = len(powerline_edge_indices)
    for agent_name in agent_names:
        mean = rhos[agent_name]
        std  = rhos_std[agent_name]
        mn   = rhos_min[agent_name]
        mx   = rhos_max[agent_name]
        rows = []
        for pl_idx in range(n_lines):
            rows.append([pl_idx,
                         f"{mean[pl_idx]:.4f}",
                         f"{std[pl_idx]:.4f}"  if std  is not None else "N/A",
                         f"{mn[pl_idx]:.4f}"   if mn   is not None else "N/A",
                         f"{mx[pl_idx]:.4f}"   if mx   is not None else "N/A"])
        headers = ["Line", "Mean ρ", "Std ρ", "Min ρ", "Max ρ"]
        print(f"\n=== Rho before failure — agent: {agent_name} ===")
        print(_tabulate(rows, headers=headers, tablefmt="github"))


def print_table_connectivity(data_dir: Path):
    from tabulate import tabulate as _tabulate
    loaded = _load_failing_edges_data(data_dir)
    agent_names, _pl, powerline_edge_indices = loaded[0], loaded[1], loaded[2]
    conn, conn_std, conn_min, conn_max = loaded[3], loaded[4], loaded[5], loaded[6]
    n_lines = len(powerline_edge_indices)
    for agent_name in agent_names:
        mean = conn[agent_name]
        std  = conn_std[agent_name]
        mn   = conn_min[agent_name]
        mx   = conn_max[agent_name]
        rows = []
        for pl_idx in range(n_lines):
            rows.append([pl_idx,
                         f"{mean[pl_idx]:.4f}",
                         f"{std[pl_idx]:.4f}"  if std  is not None else "N/A",
                         f"{mn[pl_idx]:.4f}"   if mn   is not None else "N/A",
                         f"{mx[pl_idx]:.4f}"   if mx   is not None else "N/A"])
        headers = ["Line", "Mean conn.", "Std conn.", "Min conn.", "Max conn."]
        print(f"\n=== Connectivity before failure — agent: {agent_name} ===")
        print(_tabulate(rows, headers=headers, tablefmt="github"))


def print_table_agent_summary(data_dir: Path):
    from tabulate import tabulate as _tabulate
    loaded = _load_failing_edges_data(data_dir)
    (agent_names, _pl, _pwl,
     conn, _conn_std, _conn_min, _conn_max,
     rhos, _rhos_std, _rhos_min, _rhos_max,
     conn_global_mean, conn_global_std, conn_global_min,
     rho_global_mean, rho_global_std, rho_global_max) = loaded
    rows = []
    for agent_name in agent_names:
        rho_mean_per_line  = rhos[agent_name]
        conn_mean_per_line = conn[agent_name]
        # Global simultaneous stats (may be None for older saved data)
        rg_mean = rho_global_mean.get(agent_name)
        rg_std  = rho_global_std.get(agent_name)
        rg_max  = rho_global_max.get(agent_name)
        cg_mean = conn_global_mean.get(agent_name)
        cg_std  = conn_global_std.get(agent_name)
        cg_min  = conn_global_min.get(agent_name)
        fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
        rows.append([agent_name,
                     # per-line-averaged (two-step)
                     f"{rho_mean_per_line.mean():.4f}",
                     f"{rho_mean_per_line.std():.4f}",
                     f"{rho_mean_per_line.max():.4f}",
                     # simultaneous time×edge
                     fmt(rg_mean), fmt(rg_std), fmt(rg_max),
                     # per-line-averaged connectivity
                     f"{conn_mean_per_line.mean():.4f}",
                     f"{conn_mean_per_line.std():.4f}",
                     f"{conn_mean_per_line.min():.4f}",
                     # simultaneous time×edge connectivity
                     fmt(cg_mean), fmt(cg_std), fmt(cg_min)])
    headers = ["Agent",
               "ρ Mean(lines)", "ρ Std(lines)", "ρ Max(lines)",
               "ρ Mean(t×e)",   "ρ Std(t×e)",   "ρ Max(t×e)",
               "conn Mean(lines)", "conn Std(lines)", "conn Min(lines)",
               "conn Mean(t×e)",   "conn Std(t×e)",   "conn Min(t×e)"]
    print("\n=== Agent summary ===")
    print(_tabulate(rows, headers=headers, tablefmt="github"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from itertools import product

    cwd = os.getcwd()

    model1 = AgentSpec(name="RAPPO", load_path=Path(cwd, "results/agents/CustomPPO_0_426b7_2026-01-19_10-28-48"), checkpoint_name="checkpoint_000020")
    model2 = AgentSpec(name="MLP", load_path=Path(cwd, "/home/adrian/Dev/NRI-for-explainable-RL-in-Power-Grids/results/agents/CustomPPO_0_48ac9_2026-01-19_14-39-31_MLP"), checkpoint_name="checkpoint_000020")
    model3 = AgentSpec(name="GNN", load_path=Path(cwd, "/home/adrian/Dev/NRI-for-explainable-RL-in-Power-Grids/results/agents/CustomPPO_0_4cbd2_2026-01-19_14-39-38_GNN"), checkpoint_name="checkpoint_000023")

    num_episodes = 50

    results_dir = Path("results/cross_validation")
    save_results_to = results_dir / "cross_validate_models.json"
    save_heatmap_to = results_dir / "cross_validate_models.svg"
    save_connectivity_to = results_dir / "failing_edges_connectivity.svg"
    save_rho_to = results_dir / "failing_edges_rho.svg"

    compute_data = True

    models = [model1, model2, model3]
    pairs = [(m1, m2) for m1, m2 in product(models, models) if m1.name != m2.name]

    if compute_data:
        results: List[CrossValidateResult] = []
        with ProcessPoolExecutor(max_workers=1) as ex:
            futures = [ex.submit(cross_validate, m1, m2, num_episodes) for (m1, m2) in pairs]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                print(f"Cross-validation between {result.failing_agent.name} and {result.backup_agent.name}: {result.additional_timesteps}")

        save_cross_validate_results(results, save_path=save_results_to)
        compute_reconfiguration_frequency_data(results, save_dir=results_dir)
        compute_cross_validation_data(results, save_dir=results_dir)
        compute_failing_edges_data(results, save_dir=results_dir)

    repaint_failing_edges(results_dir, save_path_connectivity=save_connectivity_to, save_path_rho=save_rho_to, show=True)
    repaint_failing_edges(results_dir, save_path_connectivity=save_connectivity_to.with_suffix(".png"), save_path_rho=save_rho_to.with_suffix(".png"), show=False)
    repaint_reconfiguration_frequency(results_dir, save_dir=results_dir, show=True)
    repaint_reconfiguration_frequency(results_dir, save_dir=results_dir, show=False)
    repaint_cross_validation_results(results_dir, save_path=save_heatmap_to.with_suffix(".png"), show=False)
    repaint_cross_validation_results(results_dir, save_path=save_heatmap_to.with_suffix(".svg"), show=True)


if __name__ == "__main__":
    import logging
    logging.getLogger("src.ra_agents.RAFeatureExtractor").setLevel(logging.ERROR)
    logging.getLogger("pandapower.convert_format").setLevel(logging.WARNING)
    main()
