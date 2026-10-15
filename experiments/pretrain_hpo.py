"""
Optuna search over imitation pretraining settings (src/imitation/hpo.py), one study per dataset.

Config: configs/rllib/pretrain_hpo.yaml (GNN + GAT on IEEE-36, search space in `hpo`). Each worker
joins the study `experiment.name` in `imitation.out_dir/study.log` and runs trials until the study
has `hpo.n_trials`. Run once with `hpo.convert_only=true` first, so the workers find the converted
observations in the cache instead of all converting them at once.

Writes to `imitation.out_dir` (results/pretraining/<study>/):
    study.log           Optuna journal
    config.yaml         base config of the study (env config for do-nothing rollouts)
    trial_<n>/          run directory of every trial (src/imitation/pretrain_run.py)
Summarize with experiments/summarize_hpo.py.

Example:
    PYTHONPATH=$(pwd)/src python experiments/pretrain_hpo.py experiment.name=junior_hard \\
        imitation.dataset=data/imitation/case36_k208_greedy_hard.npz imitation.device=cuda hpo.worker_id=0
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from core.constants import set_seed
from core.rllib_config import build_env_config, register_custom_models
from imitation.hpo import load_study, make_objective, run_study
from imitation.pretrain_run import CONFIG_FILE, build_datasets, check_pretraining_config, converter_observation_space

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/rllib", config_name="pretrain_hpo")
def main(cfg: DictConfig) -> None:
    env_config = build_env_config(cfg, "train")
    check_pretraining_config(cfg, env_config)
    set_seed(cfg.experiment.seed)
    register_custom_models()
    study_dir = Path(cfg.imitation.out_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, study_dir / CONFIG_FILE, resolve=True)

    sets = build_datasets(cfg, env_config)
    if cfg.hpo.convert_only:
        logger.info("Observations converted and cached (hpo.convert_only=true)")
        return
    study = load_study(study_dir, cfg.experiment.name, sampler_seed=cfg.hpo.sampler_seed + cfg.hpo.worker_id,
                       pruner=OmegaConf.to_container(cfg.hpo.pruner))
    study.set_user_attr("dataset", str(cfg.imitation.dataset))
    run_study(study, make_objective(cfg, sets, converter_observation_space(env_config), study_dir), cfg.hpo.n_trials)


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
