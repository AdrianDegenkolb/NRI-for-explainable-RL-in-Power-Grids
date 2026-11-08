"""
Re-run post-training analysis on an existing trial directory.

Loads the checkpoint from a completed training run and runs only the
TopologyVariabilityAnalyzer (or all analyzers with ``--all``).  Useful for
adding new analysis to old experiments without retraining.

Example:
    PYTHONPATH=$(pwd)/src python experiments/rerun_analysis.py \
        --trial-dir results/.../CustomPPO_GNN_xxxxx \
        --env-name l2rpn_case14_sandbox_test
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
from core.constants import RL_POLICY
from core.loading import load_config, load_rllib_agent, preprocess_config
from core.train import get_num_available_episodes

logger = logging.getLogger(__name__)


def find_checkpoint(trial_dir: Path) -> str:
    """Return the name of the latest checkpoint_* directory in *trial_dir*.

    :param trial_dir: Path to the trial directory.
    :return: Name of the checkpoint directory (e.g. ``checkpoint_000002``).
    """
    # Restrict to numeric checkpoint dirs (e.g. checkpoint_000002) to avoid
    # matching checkpoint_results.json that Ray Tune writes alongside them.
    candidates = sorted(d for d in trial_dir.glob("checkpoint_[0-9]*") if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_[0-9]* directories found in {trial_dir}")
    return candidates[-1].name


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint and run the topology variability analyzer."""
    trial_dir = Path(args.trial_dir).resolve()
    checkpoint_name = args.checkpoint or find_checkpoint(trial_dir)

    # --- Load config --------------------------------------------------------
    params = load_config(trial_dir)
    params = preprocess_config(params)
    env_config = params["env_config"]

    env_name = args.env_name
    if env_name is None:
        env_name = env_config["env_name"].removesuffix("_train") + "_test"

    num_episodes = (
        get_num_available_episodes(env_name)
        if args.num_episodes == "all"
        else int(args.num_episodes)
    )

    logger.info("Trial:      %s", trial_dir)
    logger.info("Checkpoint: %s", checkpoint_name)
    logger.info("Env:        %s (%d episodes)", env_name, num_episodes)

    # --- Load agent ---------------------------------------------------------
    agent, env, _wrapper = load_rllib_agent(
        checkpoint_path=trial_dir,
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=env_name,
        env_config=env_config,
    )

    # --- Run analysis -------------------------------------------------------
    out_dir = trial_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzers = [TopologyVariabilityAnalyzer()]
    runner = PostTrainingRunner(agent, env, analyzers)
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
        help="Path to the trial directory (contains checkpoint_* and params.json).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint folder name (e.g. checkpoint_000002). Default: latest.",
    )
    parser.add_argument(
        "--env-name", default=None,
        help="Grid2Op environment name for evaluation. Default: derive _test from params.",
    )
    parser.add_argument(
        "--num-episodes", default="all",
        help="Number of evaluation episodes, or 'all'. Default: all.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_parser().parse_args())
