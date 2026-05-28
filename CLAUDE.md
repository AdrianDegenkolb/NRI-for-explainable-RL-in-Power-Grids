# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements Neural Relational Inference (NRI) adapted for power grid systems, combining latent edge discovery with Reinforcement Learning (RL) for explainable control policies. It detects latent relationships between power grid components (using a discrete graph encoder) and uses those to inform a downstream RL agent trained with Ray RLlib.

## Commands

### Environment Setup
```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py  # Download and split Grid2Op scenario data
```

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

### Constants (src/core/constants.py)
Defines global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`, `EDGE_PROBS_PATH`), agent name strings (`RL_AGENT`, `HIGH_LEVEL_AGENT`, `DO_NOTHING_AGENT`), and policy names used consistently across the codebase.

### Callbacks
- `AnnealingCallback`: decays `beta` (KL weight) and `tau` (Gumbel-Softmax temperature) during training
- `TuneCallback`: logs custom metrics used by Ray Tune stoppers/schedulers

### Multi-Agent Setup
RLlib's MultiAgentEnv with three policies: `DO_NOTHING_POLICY`, `RL_POLICY` (trainable RAGNN), `HIGH_LEVEL_POLICY` (selector). Policy mapping is handled in `src/grid2op_env/multi_agent_policies/`.

## Dependencies
Python 3.10–3.12. Key packages: `torch`, `torch-geometric`, `ray[rllib,tune]`, `grid2op`, `gymnasium`, `hydra-core`, `optuna`, `stable-baselines3`. See `environment.yaml` for the full conda spec.
