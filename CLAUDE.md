# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements Neural Relational Inference (NRI) adapted for power grid systems, combining latent edge discovery with Reinforcement Learning (RL) for explainable control policies. It detects latent relationships between power grid components (using a discrete graph encoder) and uses those to inform a downstream RL agent trained with Ray RLlib.

## Commands

### Environment Setup
```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py  # Download and split Grid2Op scenario data (14/36/118-bus)
```

Action-space generation for the 36/118-bus grids (`src/action_space_generation/`) requires a
**separate** `curriculum` environment (`environment_curriculum.yaml`) — its pinned
`curriculumagent` dependency conflicts with `L2RPN`'s `ray`/`numpy` versions. Never install
`curriculumagent` into `L2RPN`.

### Training
```bash
# Default RAGNN + PPO (from project root, PYTHONPATH must include .)
PYTHONPATH=$(pwd) python experiments/train.py

# Key config group overrides
python experiments/train.py training=ppo model=ragnn obs_space=graph
python experiments/train.py training=sac model=mlp relation_awareness=disabled
python experiments/train.py experiment=test_minimal  # Quick smoke test

# CLI overrides
python experiments/train.py experiment.nb_timesteps=500000 training.lr=3e-4
```

### Tests
```bash
python -m pytest src/tests/
python -m pytest src/tests/nri/test_nri_module.py::TestNRI -v  # Single test
```

## Architecture

The training pipeline is configured entirely via **Hydra** (configs/rllib/) and orchestrated through `experiments/train.py`.

### Config Groups (configs/rllib/)
- **experiment/**: trial-level settings (timesteps, seed, checkpoint freq)
- **training/**: algorithm params — `ppo.yaml`, `sac.yaml`, `ppo_test.yaml`, `sac_test.yaml`
- **model/**: network architecture — `ragnn.yaml` (full model), `gnn.yaml` (no encoder), `mlp.yaml` (baseline)
- **obs_space/**: `graph.yaml` (nodes/edge_index/edge_mask) or `flat.yaml`
- **env/**: Grid2Op case definition (case14)
- **reward/**: reward function class (`rho`, `scaled_l2rpn`, `constant`, `alpha_zero`)
- **relation_awareness/**: `default.yaml` (enable encoder + KL annealing) or `disabled.yaml`

### Data Flow
```
Hydra Config
    │
    ├─► CustomizedGrid2OpEnvironment (MultiAgentEnv)
    │       Three agents: RL_AGENT, HIGH_LEVEL_AGENT, DO_NOTHING_AGENT
    │       ObservationConverter: raw Grid2Op obs → {NODES, EDGE_INDEX, EDGE_MASK}
    │
    └─► RARLModel (TorchModelV2, src/rarl_rllib/)
            RAFeatureExtractor (src/rarl/nn/)
                Encoder → infer p(z|x) discrete latent edges
                RAGNN   → embed graph using inferred edges
                FCNet   → policy logits + value head
```

### Key Source Modules
- **src/rarl/nn/**: Pure PyTorch — `encoder/` (discrete graph encoder), `ragnn.py` (relation-aware GNN), `feature_extractor.py`, `sampling.py` (Gumbel-Softmax)
- **src/rarl_rllib/**: RLlib integration — `RARLModel` (TorchModelV2 wrapper), annealing callbacks (`callback.py`), policy registration
- **src/grid2op_env/**: Grid2Op wrappers — multi-agent env, observation/action converters, multi-agent policy implementations
- **src/core/**: `constants.py` (global paths/names), `train.py` (training setup helpers), `loading.py` (checkpoint loading), `evaluate.py`
- **src/algorithms/**: `CustomPPO`, `CustomSAC`, Optuna integration
- **src/analysis/**: Post-training analysis — latent graph visualization, MetricAnalyzer
- **src/action_space_generation/**: Builds reduced topology action spaces for IEEE-36/118 via
  the Fraunhofer `curriculumagent` N-1 Teacher (`data/action_spaces/l2rpn_wcci_2020|2022/`).
  Runs only in the `curriculum` conda env — see `experiments/build_action_space.py` and
  `experiments/benchmark_teacher_cost.py`.
- **src/imitation/**: Greedy Tutor for imitation data in the policy's own action space
  (do-nothing + action-space JSON), recording the simulated max rho of every candidate per state.
  `dataset.py` converts any Tutor's raw output into one shared format (`obs`, `policy_target`,
  `action_mask`, `chronic`, `step`, optional `max_rho`); targets are built there, not in the Tutor.
  Runs in `L2RPN` — see `experiments/imitation_data.py` and `slurm/unicluster/imitation_tutor_array.sh`.
  Pretraining: `experiments/pretrain_policy.py` (same Hydra config as `train.py`, plus `imitation=default`
  and `obs_space.use_forecasts=false`) trains the RL_POLICY network on that data and writes
  `policy_weights.pt`; `train.py imitation=default imitation.weights_path=...` starts PPO from it with a
  critic warm-up (`ImitationInitCallback`). `experiments/imitation_rollout.py` measures a policy's
  survival per chronic. Pretraining needs a GPU (`slurm/unicluster/pretrain_policy.sh`).
  Shared run setup (datasets, model, run directory) is in `src/imitation/pretrain_run.py`; model selection
  uses the validation KL (CE − target entropy). Junior HPO: `experiments/pretrain_hpo.py` (config
  `configs/rllib/pretrain_hpo.yaml`, Optuna in `src/imitation/hpo.py`, one study per dataset, journal file
  storage shared by parallel workers), `experiments/summarize_hpo.py`, SLURM chain
  `slurm/unicluster/pretrain_hpo.sh`, rollouts `slurm/unicluster/imitation_rollout_array.sh`
  (`--do-nothing` for the reference), plots `experiments/junior_results.ipynb`.
  Training curves and per-state val analysis of the best trials: `slurm/unicluster/junior_retrain.sh` retrains
  them into `results/pretraining/junior_training/<study>/` (the only input of `experiments/junior_training.ipynb`).

### Constants (src/core/constants.py)
Defines global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`, `EDGE_PROBS_PATH`), agent name strings (`RL_AGENT`, `HIGH_LEVEL_AGENT`, `DO_NOTHING_AGENT`), and policy names used consistently across the codebase.

### Callbacks
- `AnnealingCallback`: decays `beta` (KL weight) and `tau` (Gumbel-Softmax temperature) during training
- `TuneCallback`: logs custom metrics used by Ray Tune stoppers/schedulers

### Multi-Agent Setup
RLlib's MultiAgentEnv with three policies: `DO_NOTHING_POLICY`, `RL_POLICY` (trainable RAGNN), `HIGH_LEVEL_POLICY` (selector). Policy mapping is handled in `src/grid2op_env/multi_agent_policies/`.

## Dependencies
Python 3.10–3.12. Key packages: `torch`, `torch-geometric`, `ray[rllib,tune]`, `grid2op`, `gymnasium`, `hydra-core`, `optuna`, `stable-baselines3`. See `environment.yaml` for the full conda spec.
