"""
Imitation pretraining of the RL policy network on Tutor data (src/imitation/pretrain.py).

Uses the same Hydra config as experiments/train.py, so `model`, `obs_space` and `env` define the
same network and observation converter that RL training will use. The `imitation` group sets the
data and optimisation. Imitation data has no forecasts, so `obs_space.use_forecasts=false`.

Writes to `imitation.out_dir`:
    policy_weights.pt   state_dict of the RL_POLICY model -> `imitation.weights_path` for RL
    history.csv         per-epoch train loss and validation metrics
    test_metrics.json   metrics of the best model on the test split
    config.yaml         the resolved Hydra config (used by experiments/imitation_rollout.py)

Example:
    PYTHONPATH=$(pwd)/src python experiments/pretrain_policy.py env=case36 model=gnn \\
        obs_space=default obs_space.use_forecasts=false relation_awareness=disabled \\
        imitation=default imitation.dataset=data/imitation/case36_k208_greedy_soft_t0.01.npz \\
        experiment.name=pretrain_gnn_graph_soft
"""
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from grid2op.gym_compat import GymEnv
from omegaconf import DictConfig, OmegaConf

from core.constants import set_seed
from core.rllib_config import PPO_CUSTOM_MODELS, build_env_config, build_model_config, register_custom_models
from grid2op_env.observation_converter import make_observation_converter
from grid2op_env.utils import make_g2op_env
from imitation.dataset import ImitationData, load_split, split_indices
from imitation.pretrain import PretrainConfig, build_policy_model, evaluate, train
from imitation.torch_dataset import ImitationTorchDataset, load_or_convert

logger = logging.getLogger(__name__)


def policy_model_config(cfg: DictConfig) -> dict:
    """RLlib model dict of the PPO RL_POLICY, as `_build_policies` in experiments/train.py sets it."""
    model_config = build_model_config(cfg)
    model_config["custom_model"] = PPO_CUSTOM_MODELS.get(cfg.model.custom_model, model_config["custom_model"])
    return model_config


def converter_observation_space(env_config: dict):
    """The RL agent's observation space, i.e. the converter's."""
    g2op_env = make_g2op_env(env_config)
    try:
        return make_observation_converter(GymEnv(g2op_env, with_forecast=True), env_config, backend=g2op_env.backend).observation_space
    finally:
        g2op_env.close()


@hydra.main(version_base=None, config_path="../configs/rllib", config_name="config")
def main(cfg: DictConfig) -> None:
    im = cfg.imitation
    if "dataset" not in im:
        raise SystemExit("Select the pretraining settings with imitation=default")
    env_config = build_env_config(cfg, "train")
    if env_config["use_forecasts"] and env_config["observation_space"] != "FlatObsSpace":
        raise SystemExit("Imitation data has no forecasts: set obs_space.use_forecasts=false")
    set_seed(cfg.experiment.seed)
    register_custom_models()
    out_dir = Path(im.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    split = load_split(Path(im.split))
    data = ImitationData.load(Path(im.dataset))
    eval_path = Path(im.eval_dataset or im.dataset)
    eval_data = data if eval_path == Path(im.dataset) else ImitationData.load(eval_path)
    if data.policy_target.shape[1] != eval_data.policy_target.shape[1]:
        raise SystemExit("dataset and eval_dataset have different action spaces")

    obs = load_or_convert(Path(im.dataset), data, env_config, Path(im.cache_dir), im.warmup_states, cfg.experiment.seed)
    eval_obs = obs if eval_data is data else load_or_convert(
        eval_path, eval_data, env_config, Path(im.cache_dir), im.warmup_states, cfg.experiment.seed)

    train_idx = split_indices(data.chronic, split, "train")
    if im.max_train_samples is not None and len(train_idx) > im.max_train_samples:
        rng = np.random.default_rng(cfg.experiment.seed)
        train_idx = np.sort(rng.choice(train_idx, size=im.max_train_samples, replace=False))
    train_set = ImitationTorchDataset(obs, data, train_idx)
    val_set = ImitationTorchDataset(eval_obs, eval_data, split_indices(eval_data.chronic, split, "val"))
    test_set = ImitationTorchDataset(eval_obs, eval_data, split_indices(eval_data.chronic, split, "test"))
    logger.info(f"Samples: train {len(train_set)} ({im.dataset}), val {len(val_set)}, test {len(test_set)} ({eval_path})")

    model = build_policy_model(converter_observation_space(env_config), data.policy_target.shape[1], policy_model_config(cfg))
    config = PretrainConfig(epochs=im.epochs, batch_size=im.batch_size, lr=im.lr, weight_decay=im.weight_decay,
                            patience=im.patience, device=im.device, seed=cfg.experiment.seed)
    best_state, history = train(model, train_set, val_set, config)

    torch.save(best_state, out_dir / "policy_weights.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    test_metrics = evaluate(model, test_set, config.batch_size, torch.device(config.device))
    (out_dir / "test_metrics.json").write_text(json.dumps({"n_train": len(train_set), **test_metrics}, indent=1))
    logger.info(f"Test: {test_metrics}. Wrote {out_dir}/policy_weights.pt")


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
