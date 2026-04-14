from pathlib import Path

import grid2op
import hydra
from grid2op.Agent import RecoPowerlineAgent, DoNothingAgent
from lightsim2grid import LightSimBackend
from omegaconf import DictConfig

from src.core.rewards import HRL2023Reward
from src.core.baseline_agent import evaluate_agent
from src.core.constants import EVAL_PATH, SEED


def evaluate(cfg: DictConfig):
    """
    Evaluate heuristic agents on our datasets.

    :param cfg: the hydra config
    """
    for dataset in ["train", "test", "val"]:
        env = grid2op.make(f"{cfg.env.name}_{dataset}", backend=LightSimBackend(), reward_class=HRL2023Reward)
        env.seed(SEED)
        for agent, name in zip([RecoPowerlineAgent(env.action_space), DoNothingAgent(env.action_space)], ["reco_powerline_agent", "do_nothing_agent"]):
            evaluate_agent(
                agent=agent,
                env=env,
                num_episodes=cfg.rl.eval.final.nb_episodes,
                path_results=Path(EVAL_PATH, "heuristic_agents", name, dataset)
            )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()