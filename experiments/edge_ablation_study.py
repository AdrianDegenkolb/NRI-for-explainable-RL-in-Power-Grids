"""Edge ablation study: does a trained GNN use edge information?

Loads a PPO checkpoint and evaluates it under three edge conditions:
  true      — real dynamic edges (baseline)
  shuffled  — random node-index permutation of edge_index
  none      — all edges removed (edge_mask = False)

Writes per-mode summary_metrics.json and a combined edge_ablation_results.json.

Usage:
    PYTHONPATH=$(pwd)/src python experiments/edge_ablation_study.py \
        --trial-dir results/.../CustomPPO_GNN_xxxxx \
        --num-episodes all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.edge_ablation import EdgeAblationWrapper, EdgeMode
from core.constants import RL_POLICY
from core.evaluate import evaluate_agent
from core.loading import load_config, load_rllib_agent, preprocess_config
from core.train import get_num_available_episodes

logger = logging.getLogger(__name__)


def find_checkpoint(trial_dir: Path) -> str:
    """Return the name of the latest checkpoint directory."""
    candidates = sorted(d for d in trial_dir.glob("checkpoint_[0-9]*") if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_[0-9]* directories found in {trial_dir}")
    return candidates[-1].name


def run_edge_ablation(
    trial_dir: Path,
    checkpoint_name: str | None,
    env_name: str | None,
    num_episodes: int | str,
    modes: list[EdgeMode],
    seed: int,
) -> dict:
    """Run evaluation under each edge mode and return combined results.

    :param trial_dir: Path to the PPO trial directory.
    :param checkpoint_name: Checkpoint folder name, or None for latest.
    :param env_name: Grid2Op env name override, or None to derive from params.
    :param num_episodes: Number of episodes or 'all'.
    :param modes: Edge modes to evaluate.
    :param seed: RNG seed for shuffled mode and evaluation.
    :return: Dict mapping mode name to summary metrics.
    """
    checkpoint_name = checkpoint_name or find_checkpoint(trial_dir)
    params = load_config(trial_dir)
    params = preprocess_config(params)
    env_config = params["env_config"]

    if env_name is None:
        env_name = env_config["env_name"].removesuffix("_train") + "_test"

    n_episodes = (
        get_num_available_episodes(env_name)
        if num_episodes == "all"
        else int(num_episodes)
    )

    logger.info("Trial:      %s", trial_dir)
    logger.info("Checkpoint: %s", checkpoint_name)
    logger.info("Env:        %s (%d episodes)", env_name, n_episodes)
    logger.info("Modes:      %s", [m.value for m in modes])

    out_dir = trial_dir / "analysis" / "edge_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for mode in modes:
        logger.info("--- Evaluating edge mode: %s ---", mode.value)

        agent, g2op_env, gym_wrapper = load_rllib_agent(
            checkpoint_path=trial_dir,
            policy_name=RL_POLICY,
            checkpoint_name=checkpoint_name,
            env_name=env_name,
            env_config=env_config.copy(),
        )

        real_converter = gym_wrapper.observation_converter
        wrapper = EdgeAblationWrapper(real_converter, mode, seed=seed)
        gym_wrapper.observation_converter = wrapper

        mode_dir = out_dir / mode.value
        evaluate_agent(
            agent=agent,
            env=g2op_env,
            path_results=mode_dir,
            num_episodes=n_episodes,
            seed=seed,
            verbose=True,
        )

        summary_path = mode_dir / "summary_metrics.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[mode.value] = json.load(f)
            logger.info(
                "  %s: completed=%.1f%%, survived=%.1f%%",
                mode.value,
                results[mode.value]["completed_episodes_pct"],
                results[mode.value]["survived_steps_pct"],
            )

    combined_path = out_dir / "edge_ablation_results.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Combined results saved to %s", combined_path)

    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trial-dir", required=True,
        help="PPO trial directory (checkpoint_* + params.json).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint folder name. Default: latest.",
    )
    parser.add_argument(
        "--env-name", default=None,
        help="Grid2Op environment name. Default: derive _test from params.",
    )
    parser.add_argument(
        "--num-episodes", default="all",
        help="Number of evaluation episodes, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["true", "shuffled", "none"],
        choices=["true", "shuffled", "none"],
        help="Edge modes to evaluate. Default: all three.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for shuffled permutation and evaluation. Default: 42.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()

    modes = [EdgeMode(m) for m in args.modes]
    run_edge_ablation(
        trial_dir=Path(args.trial_dir).resolve(),
        checkpoint_name=args.checkpoint,
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        modes=modes,
        seed=args.seed,
    )
