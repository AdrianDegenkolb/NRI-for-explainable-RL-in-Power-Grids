"""
Export the RL_POLICY network of an RLlib checkpoint in the format of experiments/pretrain_policy.py
(`policy_weights.pt` + `config.yaml` in one run dir), so an RL agent can be evaluated by
experiments/imitation_rollout.py exactly like a pretrained ("junior") policy.

The weights are read from the policy state pickle (`policies/<RL_POLICY>/policy_state.pkl`, key
`weights`: the model's state_dict as numpy arrays), so neither Ray nor the policy class is needed.
The config is the Hydra config train.py saved for the run (`<storage_path>/<name>/.hydra/config.yaml`),
saved resolved like pretrain_policy.py's: `${hydra:...}` interpolations are filled from the run's own
`.hydra/hydra.yaml`, since they cannot be resolved outside a Hydra app.

Example:
    PYTHONPATH=$(pwd)/src python experiments/export_policy_weights.py \\
        --experiment-dir results/2026_10_14_IEEE36_imitation/senior_s0 \\
        --out results/2026_10_14_IEEE36_imitation/eval/senior_s0
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.constants import RL_POLICY


def latest_checkpoint(experiment_dir: Path) -> Path:
    """The newest `checkpoint_*` of the experiment's only trial (Tune keeps the final one)."""
    # Directories only: the trial dir also holds checkpoint_results.json.
    checkpoints = sorted(c for c in experiment_dir.glob("*/checkpoint_*") if c.is_dir())
    trials = {c.parent for c in checkpoints}
    if len(trials) != 1:
        raise ValueError(f"Expected one trial with checkpoints in {experiment_dir}, found {len(trials)}")
    return checkpoints[-1]


def load_policy_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """The RL_POLICY model's state_dict stored in an RLlib algorithm checkpoint."""
    policy_dir = checkpoint / "policies" / RL_POLICY
    info = json.loads((policy_dir / "rllib_checkpoint.json").read_text())
    with open(policy_dir / info["state_file"], "rb") as f:
        state = pickle.load(f)
    return {name: torch.as_tensor(value) for name, value in state["weights"].items()}


def resolved_run_config(hydra_dir: Path):
    """The run's Hydra config with all interpolations resolved, `${hydra:...}` from `hydra.yaml`."""
    hydra_cfg = OmegaConf.load(hydra_dir / "hydra.yaml").hydra
    OmegaConf.register_new_resolver("hydra", lambda key: OmegaConf.select(hydra_cfg, key), replace=True)
    cfg = OmegaConf.load(hydra_dir / "config.yaml")
    OmegaConf.resolve(cfg)
    return cfg


def main(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(experiment_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(load_policy_weights(checkpoint), out / "policy_weights.pt")
    OmegaConf.save(resolved_run_config(experiment_dir / ".hydra"), out / "config.yaml")
    (out / "source_checkpoint.txt").write_text(f"{checkpoint.resolve()}\n")
    print(f"Exported {checkpoint} to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True,
                        help="train.py output dir: <experiment.storage_path>/<experiment.name>")
    parser.add_argument("--checkpoint", default=None, help="Default: the trial's newest checkpoint")
    parser.add_argument("--out", required=True, help="Run dir for experiments/imitation_rollout.py")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
