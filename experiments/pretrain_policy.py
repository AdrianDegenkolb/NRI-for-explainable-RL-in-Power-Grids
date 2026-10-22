"""
Imitation pretraining of the RL policy network on Tutor data (src/imitation/pretrain.py).

Uses the same Hydra config as experiments/train.py, so `model`, `obs_space` and `env` define the
same network and observation converter that RL training will use. The `imitation` group sets the
data and optimisation. Imitation data has no forecasts, so `obs_space.use_forecasts=false`.

Writes a run directory (policy_weights.pt, history.csv, val_predictions.npz, test_metrics.json,
config.yaml; see src/imitation/pretrain_run.py) to `imitation.out_dir`.

Example:
    PYTHONPATH=$(pwd)/src python experiments/pretrain_policy.py env=case36 model=gnn \\
        obs_space=default obs_space.use_forecasts=false relation_awareness=disabled \\
        imitation=default imitation.dataset=data/imitation/case36_k208_greedy_soft_t0.01.npz \\
        experiment.name=pretrain_gnn_graph_soft
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from core.constants import set_seed
from core.rllib_config import build_env_config, register_custom_models
from imitation.pretrain_run import build_datasets, check_pretraining_config, converter_observation_space, run_pretraining


@hydra.main(version_base=None, config_path="../configs/rllib", config_name="config")
def main(cfg: DictConfig) -> None:
    env_config = build_env_config(cfg, "train")
    check_pretraining_config(cfg, env_config)
    set_seed(cfg.experiment.seed)
    register_custom_models()
    sets = build_datasets(cfg, env_config)
    run_pretraining(cfg, sets, converter_observation_space(env_config), Path(cfg.imitation.out_dir))


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
