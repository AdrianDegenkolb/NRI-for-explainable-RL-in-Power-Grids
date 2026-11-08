"""
Re-run post-training analysis on an existing trial directory.

PPO mode (default): loads the latest RLlib checkpoint from a training run.
Junior mode (--junior): loads policy_weights.pt + config.yaml from an IL run.

Both modes run TopologyVariabilityAnalyzer and write results to
<trial_dir>/analysis/topology_variability/.

Examples:
    # PPO checkpoint
    PYTHONPATH=$(pwd)/src python experiments/rerun_analysis.py \
        --trial-dir results/.../CustomPPO_GNN_xxxxx

    # Junior IL weights
    PYTHONPATH=$(pwd)/src python experiments/rerun_analysis.py --junior \
        --trial-dir results/junior/.../no_edges \
        --env-split test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.post_training.analyzers import TopologyVariabilityAnalyzer
from analysis.post_training.runner import PostTrainingRunner
from core.train import get_num_available_episodes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PPO helpers
# ---------------------------------------------------------------------------

def find_checkpoint(trial_dir: Path) -> str:
    """Return the name of the latest checkpoint_* directory in *trial_dir*.

    :param trial_dir: Path to the trial directory.
    :return: Name of the checkpoint directory (e.g. ``checkpoint_000002``).
    """
    # Restrict to numeric dirs (e.g. checkpoint_000002) to avoid matching
    # checkpoint_results.json that Ray Tune writes alongside them.
    candidates = sorted(d for d in trial_dir.glob("checkpoint_[0-9]*") if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_[0-9]* directories found in {trial_dir}")
    return candidates[-1].name


def _load_ppo_agent(args: argparse.Namespace, trial_dir: Path):
    """Load an RLlib agent from a PPO trial directory.

    :return: (agent, env, env_name, num_episodes)
    """
    from core.constants import RL_POLICY
    from core.loading import load_config, load_rllib_agent, preprocess_config

    checkpoint_name = args.checkpoint or find_checkpoint(trial_dir)
    params = load_config(trial_dir)
    params = preprocess_config(params)
    env_config = params["env_config"]

    env_name = args.env_name or env_config["env_name"].removesuffix("_train") + "_test"
    num_episodes = (
        get_num_available_episodes(env_name)
        if args.num_episodes == "all"
        else int(args.num_episodes)
    )
    logger.info("PPO trial:  %s", trial_dir)
    logger.info("Checkpoint: %s", checkpoint_name)
    logger.info("Env:        %s (%d episodes)", env_name, num_episodes)

    agent, env, _wrapper = load_rllib_agent(
        checkpoint_path=trial_dir,
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config,
    )
    return agent, env, num_episodes


# ---------------------------------------------------------------------------
# Junior helpers
# ---------------------------------------------------------------------------

class _JuniorAgentAdapter:
    """Wraps a Junior greedy policy in the interface PostTrainingRunner expects.

    :param chooser: ActionChooser from imitation.rollout (obs, gym_obs) -> action index.
    :param actions: List of Grid2Op BaseAction objects.
    :param converter: ObservationConverter for the Junior's obs space.
    :param rho_threshold: Act only when max rho exceeds this.
    :param line_reco: Whether to apply reconnection_rule to every action.
    :param action_space: Grid2Op action space (for reconnection_rule).
    """

    def __init__(self, chooser, actions, converter, rho_threshold, line_reco, action_space):
        self._chooser = chooser
        self._actions = actions
        self._converter = converter
        self._rho_threshold = rho_threshold
        self._line_reco = line_reco
        self._action_space = action_space
        self._rllib_agent = None   # signals PostTrainingRunner: no encoder
        self.gym_wrapper = None

    def activate_agent(self, obs) -> bool:
        """Always delegate to the Junior policy."""
        return True

    def act(self, obs, reward, done):
        """Return the greedy action, applying rho threshold and line reconnection."""
        from imitation.tutor import DO_NOTHING_INDEX
        from core.heuristic_actions import reconnection_rule

        gym_obs = self._converter.to_gym(obs)
        idx = DO_NOTHING_INDEX
        if obs.rho.max() > self._rho_threshold:
            idx = self._chooser(obs, gym_obs)
        action = self._actions[idx]
        if self._line_reco:
            action = reconnection_rule(obs, action, self._action_space)
        return action


def _load_junior_agent(args: argparse.Namespace, trial_dir: Path):
    """Load a Junior IL agent from policy_weights.pt + config.yaml.

    :return: (agent, env, num_episodes)
    """
    import torch
    from grid2op.gym_compat import GymEnv
    from omegaconf import OmegaConf

    from core.rllib_config import build_env_config, register_custom_models
    from grid2op_env.observation_converter import make_observation_converter
    from grid2op_env.utils import make_g2op_env
    from imitation.pretrain_run import CONFIG_FILE, WEIGHTS_FILE, build_model
    from imitation.rollout import greedy_chooser
    from imitation.tutor import load_policy_actions

    cfg = OmegaConf.load(trial_dir / CONFIG_FILE)
    register_custom_models()

    env_split = args.env_split or "test"
    env_config = build_env_config(cfg, env_split)
    env = make_g2op_env(env_config)

    env_name = env.env_name
    num_episodes = (
        get_num_available_episodes(env_name)
        if args.num_episodes == "all"
        else int(args.num_episodes)
    )
    logger.info("Junior trial: %s", trial_dir)
    logger.info("Env:          %s / %s (%d episodes)", env_name, env_split, num_episodes)

    actions = load_policy_actions(
        env, ROOT / f"data/action_spaces/{env.env_name}/{env_config['action_space']}.json"
    )
    converter = make_observation_converter(
        GymEnv(env, with_forecast=True), env_config, backend=env.backend
    )

    device = torch.device("cpu")
    model = build_model(cfg, converter.observation_space, len(actions))
    model.load_state_dict(torch.load(trial_dir / WEIGHTS_FILE, map_location=device))

    chooser = greedy_chooser(model, device)
    agent = _JuniorAgentAdapter(
        chooser=chooser,
        actions=actions,
        converter=converter,
        rho_threshold=env_config["rho_threshold"],
        line_reco=env_config.get("line_reco", True),
        action_space=env.action_space,
    )
    return agent, env, num_episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Load agent and run the topology variability analyzer."""
    trial_dir = Path(args.trial_dir).resolve()

    if args.junior:
        agent, env, num_episodes = _load_junior_agent(args, trial_dir)
    else:
        agent, env, num_episodes = _load_ppo_agent(args, trial_dir)

    out_dir = trial_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = PostTrainingRunner(agent, env, [TopologyVariabilityAnalyzer()])
    runner.run(num_episodes=num_episodes)
    runner.save_all(out_dir)

    logger.info("Done. Results in: %s", out_dir / "topology_variability")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trial-dir", required=True,
        help="PPO trial directory (checkpoint_* + params.json) or Junior run dir (policy_weights.pt + config.yaml).",
    )
    parser.add_argument(
        "--junior", action="store_true",
        help="Load a Junior IL policy (policy_weights.pt) instead of an RLlib checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="[PPO] Checkpoint folder name (e.g. checkpoint_000002). Default: latest.",
    )
    parser.add_argument(
        "--env-name", default=None,
        help="[PPO] Grid2Op environment name. Default: derive _test from params.",
    )
    parser.add_argument(
        "--env-split", default=None, choices=["val", "test"],
        help="[Junior] Grid2Op env split to evaluate on. Default: test.",
    )
    parser.add_argument(
        "--num-episodes", default="all",
        help="Number of evaluation episodes, or 'all'. Default: all.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_parser().parse_args())
