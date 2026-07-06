"""
Post-training analysis: run all test episodes and save data + plots.

Usage:
    PYTHONPATH=$(pwd)/src python experiments/post_training_analysis.py \
        --checkpoint-path results/experiments/2026_07_06_IEEE36/rappo/trial_xxx \
        --env-name l2rpn_ieee_neurips_2020_track1_small_test \
        --out-dir results/experiments/2026_07_06_IEEE36/rappo/trial_xxx/analysis
"""
import argparse
import logging
import shutil
from pathlib import Path

from core.loading import load_config, preprocess_config, load_rllib_agent
from core.constants import RL_POLICY

from analysis.post_training.runner import PostTrainingRunner
from analysis.post_training.analyzers import SurvivalAnalyzer, CongestionProfileAnalyzer, TopologyActionAnalyzer
from analysis.post_training.analyzers.h1_coupling import H1ElectricalCouplingAnalyzer
from analysis.post_training.analyzers.h2_coupling import H2RiskCouplingAnalyzer
from analysis.post_training.analyzers.h3_coupling import H3ActionEffectAnalyzer
from rarl_rllib.model import RARLModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _detect_encoder(agent) -> bool:
    return isinstance(getattr(agent._rllib_agent, "model", None), RARLModel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Checkpoint folder name. Defaults to latest found.")
    parser.add_argument("--env-name", type=str, required=True,
                        help="Grid2Op test environment name.")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Number of test episodes (default: all).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <checkpoint-path>/analysis).")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    checkpoint_name = args.checkpoint_name

    # Accept either the trial dir or the checkpoint dir itself as --checkpoint-path.
    # If the path is named checkpoint_* (or contains policies/), treat parent as trial dir.
    if checkpoint_name is None:
        if checkpoint_path.name.startswith("checkpoint_") or (checkpoint_path / "policies").exists():
            checkpoint_name = checkpoint_path.name
            checkpoint_path = checkpoint_path.parent
            logger.info("Detected checkpoint dir; trial dir: %s, checkpoint: %s", checkpoint_path, checkpoint_name)
        else:
            candidates = sorted(checkpoint_path.glob("checkpoint_*"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint_* in {checkpoint_path}")
            checkpoint_name = candidates[-1].name
            logger.info("Auto-selected checkpoint: %s", checkpoint_name)

    params = load_config(checkpoint_path)
    params = preprocess_config(params)
    env_config = params["env_config"]

    agent, env, _ = load_rllib_agent(
        checkpoint_path=checkpoint_path,
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=args.env_name,
        env_config=env_config,
    )

    out_dir = args.out_dir or (checkpoint_path / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    has_encoder = _detect_encoder(agent)
    logger.info("Encoder model: %s", has_encoder)

    analyzers = [
        SurvivalAnalyzer(),
        CongestionProfileAnalyzer(env_name=args.env_name),
        TopologyActionAnalyzer(env_name=args.env_name),
    ]
    if has_encoder:
        analyzers += [
            H1ElectricalCouplingAnalyzer(out_dir=out_dir / "h1"),
            H2RiskCouplingAnalyzer(out_dir=out_dir / "h2"),
            H3ActionEffectAnalyzer(out_dir=out_dir / "h3"),
        ]

    runner = PostTrainingRunner(agent, env, analyzers)
    runner.run(num_episodes=args.num_episodes)
    runner.save_all(out_dir)

    # Copy TensorBoard event files so the analysis folder is self-contained
    for tf_file in checkpoint_path.glob("events.out.tfevents.*"):
        shutil.copy2(tf_file, out_dir / tf_file.name)
    logger.info("Analysis complete. Results in: %s", out_dir)


if __name__ == "__main__":
    main()
