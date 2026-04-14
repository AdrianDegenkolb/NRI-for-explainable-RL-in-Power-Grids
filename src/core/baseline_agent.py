"""
This script implements a baseline agent. The agent is a greedy agent meaning it will simulate several actions
returned by its _get_tested_action method and execute the one with the highest simulated reward.
Besides the heuristic actions candidates like reconnecting powerlines and doing nothing the agent receives action candidates
from topology policies. These policies may implement a RL-component to predict topological actions. The RL-component evaluates
the actions the topology policy returns the k best actions to the agent to simulate.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Runner import Runner
from grid2op.Runner.runner import runner_returned_type
from stable_baselines3.common.policies import BasePolicy

from src.core.constants import SEED
from src.core.heuristic_actions import reconnection_rule, revert_to_reference_topo, disconnection_rule

logger = logging.getLogger(__name__)


class HeuristicsAgent(BaseAgent):
    """
    This agent executes heuristic rules based on the rule_config given.

    rule_config can contain:
    -   rho_threshold: float value. Activation threshold of the lower level agent.
    -   line_reco: Boolean value. If True: attempt to reconnect all disconnected power lines if
        this is beneficial for the max rho value.
    -   line_disc: Boolean value. If True: manually disconnect a line during sustained periods of
        overflow in order to avoid permanent damage. Reconnect the line back soon after the
        cooldown period ends.
    -   reset_topo: float value. Revert Threshold. If the max load rho < reset_topo, the agent
        will execute actions to revert to the reference topology.
    """

    def __init__(
            self,
            action_space: ActionSpace,
            rule_config: dict,
    ):
        BaseAgent.__init__(self, action_space)
        self.activation_thresh = rule_config.get("activation_threshold", 0.95)
        self.line_reco = rule_config.get("line_reco", True)
        self.line_disc = rule_config.get("line_disc", False)
        self.reset_topo = rule_config.get("reset_topo", 0.5)
        self.simulate = rule_config.get("simulate", True)

    def activate_agent(self, observation: BaseObservation):
        max_rho = (observation.rho.max() if observation.rho.max() > 0 else 2)
        return max_rho > self.activation_thresh

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        current_action = self.action_space({})
        if self.line_reco:
            current_action = reconnection_rule(observation, current_action, self.action_space)
        if self.reset_topo:
            current_action = revert_to_reference_topo(observation, current_action, self.action_space, self.reset_topo)
        if self.line_disc:
            current_action = disconnection_rule(observation, current_action, self.action_space)
        return current_action

    def simulate_combinations(self,
                              observation: BaseObservation,
                              topo_actions: List[BaseAction],
                              rb_action: BaseAction) -> BaseAction:
        if self.simulate:
            action_candidates = [rb_action] + [rb_action + ta for ta in topo_actions] + topo_actions
            best_action = rb_action
            best_rho = 2
            for action in action_candidates:
                sim_obs, _, _, _ = observation.simulate(action)
                sim_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                if sim_rho < best_rho:
                    best_rho = sim_rho
                    best_action = action
            action = best_action
        else:
            action = rb_action + topo_actions[0]
        return action


class BaselineAgent(HeuristicsAgent):
    """
    This particular greedy baseline will simulate the following actions:

    - Do nothing
    - Reconnections of disconnected powerlines
    - k Topology actions proposed the TopologyPolicy (if rho gets too high)
    """

    def __init__(
            self,
            g2op_action_space: ActionSpace,
            rule_config: dict,
            rl_policy: BasePolicy
    ):
        """
        :param g2op_action_space: The action space
        :param rl_policy: A policy proposing topology related actions from the same action space
        :param rule_config: contains heuristic rule descriptions
        """
        super().__init__(g2op_action_space, rule_config)
        self.rl_policy = rl_policy

    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """

        # First do rule based part of the agent, line reconnections, disconnections and revert topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)

        if HeuristicsAgent.activate_agent(self, observation):
            # Get action from trained RL-agent when in danger.
            gym_obs = self.rl_policy.observation_space.to_gym(observation)
            topo_action, _ = self.rl_policy.predict(gym_obs, deterministic=True)
            topo_action_grid2op = self.rl_policy.action_space.from_gym(topo_action)

            action = HeuristicsAgent.simulate_combinations(self, observation, [topo_action_grid2op], rb_action)
        else:
            action = rb_action

        return action


def evaluate_agent(agent: BaseAgent, env: Environment, path_results: Path, num_episodes: int,
                   max_episode_length: Optional[int] = None, verbose=True) -> List[runner_returned_type]:
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param num_episodes: the number of episodes to run
    :param path_results: where to store the results
    :param max_episode_length: the maximum number of steps to take per episode
    :param verbose: print extra explanatory or diagnostic information
    :return: the evaluation results from the runner
    """
    logging.getLogger("grid2op.Environment.baseEnv.grid2op_Runner").disabled = True
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    path_results.mkdir(exist_ok=True, parents=True)
    res = runner.run(
        nb_episode=num_episodes,
        max_iter=max_episode_length,
        path_save=path_results,
        add_detailed_output=True,
        pbar=verbose,
        env_seeds=[SEED] * num_episodes,
    )

    if verbose:
        # print results
        _print_runner_results(res)

    # Compute and store summary metrics across episodes
    _store_summary_metrics(res=res, path_results=path_results, verbose=verbose)

    if verbose:
        logger.info(f"Evaluation results are stored in: {path_results}")

    return res


def _print_runner_results(res: List[runner_returned_type]):
    for _, chron_id, cum_reward, nb_time_step, max_ts, data in res:
        logger.info(f"Chronics: '{chron_id}', Return: {cum_reward:.2f}, "
                    f"Survival Duration: {nb_time_step:.0f} / {max_ts:.0f}, "
                    f"Per-step-reward: {np.nan_to_num(data.rewards).mean():.2f} ± {np.nan_to_num(data.rewards).std():.2f}")


def _store_summary_metrics(res: List[runner_returned_type], path_results: Path, verbose: bool = True) -> None:
    """
    Compute averaged Completed Episodes % and Survived Steps % across episodes and store them in a summary JSON.

    Completed Episodes %: fraction of episodes that reached their max allowed steps (nb_time_step == max_ts) * 100.
    Survived Steps %: average over episodes of (nb_time_step / max_ts) * 100.

    :param res: list of runner returns
    :param path_results: the folder in which to store the summary_metrics.json
    :param verbose: print extra explanatory or diagnostic information
    """
    # only include episodes with positive max_timesteps
    res = [ep_info for ep_info in res if ep_info[4] > 0]

    if not len(res):
        return

    completed_flags = []
    survived_ratios = []

    for _, _, _, nb_time_step, max_ts, _ in res:
        completed_flags.append(1.0 if nb_time_step >= max_ts else 0.0)
        survived_ratios.append(float(nb_time_step) / float(max_ts))

    completed_episodes_frac = (sum(completed_flags) / float(len(completed_flags)))
    survived_steps_frac = float(np.mean(survived_ratios))

    summary = {
        "episodes": len(res),
        "completed_episodes_pct": round(completed_episodes_frac * 100, 4),
        "survived_steps_pct": round(survived_steps_frac * 100, 4)
    }

    # Persist a summary file in the root result folder
    path_results.mkdir(parents=True, exist_ok=True)
    summary_path = Path(path_results, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Log a brief summary
    if verbose:
        logger.info(f"Summary metrics: Completed Episodes: {completed_episodes_frac * 100:.2f}%, Survived Steps: {survived_steps_frac * 100:.2f}%")
