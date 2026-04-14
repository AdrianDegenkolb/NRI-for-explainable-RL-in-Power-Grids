"""
Shared heuristic action implementations used by both the environment-side heuristics and agents.

Functions here operate on Grid2Op observations and action spaces to modify actions according to rules:
- reconnection_rule
- revert_to_reference_topo
- disconnection_rule

Each function is pure with respect to input arguments and returns a possibly updated action.
"""
from typing import List

import numpy as np
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation


def reconnection_rule(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace) -> BaseAction:
    """
    Reconnect all disconnected lines.

    :param observation: The current observation.
    :param current_action: The action (so far).
    :param action_space: The action space.
    :return: The updated action including line reconnections.
    """
    line_stat_s = observation.line_status
    cooldown = observation.time_before_cooldown_line
    can_be_reco = ~line_stat_s & (cooldown == 0)
    if can_be_reco.any():
        for id_ in can_be_reco.nonzero()[0]:
            current_action += action_space({"set_line_status": [(int(id_), +1)]})

    return current_action


def revert_to_reference_topo(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace, reset_topo: float) -> BaseAction:
    """
    Revert substations to reference topology when below a rho threshold and simulation indicates improvement.

    :param observation: The current observation.
    :param current_action: The action (so far).
    :param action_space: The action space.
    :param reset_topo: The threshold for which to consider resetting the topology (if max_rho is smaller).
    :return: The updated action including resetting the topology.
    """
    rho_max = (observation.rho.max() if observation.rho.max() > 0 else 2)
    if (rho_max < reset_topo) and (observation.current_step < observation.max_step - 1):
        subs_changed = np.unique(observation._topo_vect_to_sub[observation.topo_vect != 1])
        if len(subs_changed):
            sim_obs, _, _, _ = observation.simulate(current_action)
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            action_options: List[BaseAction] = []
            max_rhos = np.zeros(len(subs_changed))
            rewards = np.zeros(len(subs_changed))
            for i, sub in enumerate(subs_changed):
                action = action_space({
                    "set_bus": {
                        "substations_id": [
                            (int(sub), np.ones(observation.sub_info[int(sub)], dtype=int))
                        ]
                    }
                })
                action_options.append(action)
                sim_obs, rw, tmp_done, tmp_info = observation.simulate(current_action + action)
                max_rhos[i] = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                rewards[i] = rw
            if len(rewards) and max_rhos[int(np.argmax(rewards))] < cur_max_rho:
                current_action += action_options[int(np.argmax(rewards))]
    return current_action


def disconnection_rule(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace) -> BaseAction:
    """Manually disconnect a line during sustained overflow if simulation indicates improvement."""
    if np.any(observation.timestep_overflow > 1):
        sim_obs, _, _, _ = observation.simulate(current_action)
        cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
        id_ = int(observation.timestep_overflow.argmax())
        action = current_action + action_space({"set_line_status": [(id_, -1)]})
        sim_obs, _, _, _ = observation.simulate(action)
        if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
            current_action = action
    return current_action

