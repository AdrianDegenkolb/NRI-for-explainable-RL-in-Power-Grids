from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from grid2op.Environment import Environment
from tqdm import tqdm

from agents import RllibAgent
from analysis.post_training.interfaces import EpisodeAnalyzer, StepContext
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES
from rarl.graph import fully_connected_edge_index
from rarl.prior import get_priors, get_prior_tensor
from rarl_rllib.model import RARLModel

logger = logging.getLogger(__name__)


class PostTrainingRunner:
    def __init__(
        self,
        agent: RllibAgent,
        env: Environment,
        analyzers: List[EpisodeAnalyzer],
    ) -> None:
        self.agent = agent
        self.env = env
        self.analyzers = analyzers
        self._has_encoder: bool = isinstance(
            getattr(agent._rllib_agent, "model", None), RARLModel
        )
        logger.info(
            "PostTrainingRunner: encoder=%s, analyzers=%s",
            self._has_encoder,
            [type(a).__name__ for a in analyzers],
        )

    def run(self, num_episodes: Optional[int] = None) -> None:
        n_available = len(self.env.chronics_handler.available_chronics())
        n_episodes = n_available if num_episodes is None else min(num_episodes, n_available)
        logger.info("Running %d episodes.", n_episodes)

        for ep_idx in range(n_episodes):
            obs = self.env.reset()
            episode_id = self.env.chronics_handler.get_name()
            max_steps = self.env.max_episode_duration()

            for a in self.analyzers:
                a.on_episode_start(episode_id, max_steps)

            done = False
            step_idx = 0
            reward = 0.0

            pbar = tqdm(
                total=max_steps,
                desc=f"[{ep_idx+1}/{n_episodes}] {episode_id}",
                unit="step",
                leave=False,
            )
            while not done:
                is_rl_step = self.agent.activate_agent(obs)
                action = self.agent.act(obs, reward, done)

                posterior, prior = None, None
                if self._has_encoder and is_rl_step:
                    posterior, prior = self._extract_posterior_prior(obs)

                obs_next, reward, done, info = self.env.step(action)

                ctx = StepContext(
                    obs=obs,
                    action=action,
                    obs_next=obs_next,
                    reward=reward,
                    done=done,
                    info=info,
                    is_rl_step=is_rl_step,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env=self.env,
                    posterior=posterior,
                    prior=prior,
                )
                for a in self.analyzers:
                    a.on_step(ctx)

                obs = obs_next
                step_idx += 1
                pbar.update(1)

            pbar.close()
            steps_survived = self.env.nb_time_step
            for a in self.analyzers:
                a.on_episode_end(steps_survived, max_steps)

            status = "OK" if steps_survived >= max_steps else f"FAIL@{steps_survived}"
            logger.info("Episode %s: %s/%s (%s)", episode_id, steps_survived, max_steps, status)

    def save_all(self, out_dir: Path) -> None:
        for a in self.analyzers:
            a.save(out_dir)

    def _extract_posterior_prior(self, obs) -> tuple[np.ndarray, np.ndarray]:
        model = self.agent._rllib_agent.model
        posterior_t: torch.Tensor = model.get_posterior()
        if posterior_t.dim() == 3:
            posterior_t = posterior_t[0]
        posterior = posterior_t.cpu().detach().numpy()

        try:
            prior = self._compute_prior(obs, posterior.shape[0])
        except Exception:
            logger.debug("Prior computation failed; using uniform.", exc_info=True)
            prior = np.full_like(posterior, 1.0 / posterior.shape[-1])

        return posterior, prior

    def _compute_prior(self, obs, E: int) -> np.ndarray:
        ra_config = self.agent._rllib_agent.config["relation_awareness"]
        self.agent.gym_wrapper.update_obs(obs)
        cur = self.agent.gym_wrapper.cur_gym_obs
        edge_index = torch.from_numpy(cur[EDGE_INDEX])
        edge_mask = cur[EDGE_MASK]
        masked_edge_index = edge_index[:, edge_mask]
        n_graph_edges = int(edge_mask.sum())
        N = cur[NODES].shape[0]
        all_edges = fully_connected_edge_index(N)

        prior_graph, prior_non_graph = get_priors(
            prob_graph_edges_exist=ra_config["prior"]["prior_prob_for_graph_edge"],
            num_graph_edges=n_graph_edges,
            num_non_graph_edges=all_edges.shape[1] - n_graph_edges,
            temperature=ra_config["prior"]["temperature"],
        )
        prior_tensor, _ = get_prior_tensor(
            graph_edges=masked_edge_index,
            all_edges=all_edges,
            prior_for_graph_edges=prior_graph,
            prior_for_non_graph_edges=prior_non_graph,
            num_edge_types=ra_config["latent_space"]["num_edge_types"],
            return_mask=True,
        )
        return prior_tensor.cpu().numpy()
