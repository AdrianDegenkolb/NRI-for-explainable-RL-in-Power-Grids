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
- **model/**: network architecture — `ragnn.yaml` (full model), `gnn.yaml` (no encoder), `mlp.yaml` (baseline;
  RLlib's default, which builds one MLP per attribute of a Dict observation), `flat_mlp.yaml` (one MLP on the
  concatenated flat observation, Fraunhofer Junior size; `FlatMLPModel` in `src/rarl_rllib/ppo/mlp_ppo_model.py`),
  `gcn_substation_readout.yaml` (GCN; node embeddings averaged per substation and concatenated, `src/rarl/nn/readout.py`;
  needs `node_substations: true`)
- **obs_space/**: `default.yaml` (graph: nodes/edge_index/edge_mask), `complete_graph.yaml` (graph plus
  line/substation state and node-type one-hot, no voltage angles; used for imitation), `flat.yaml` or
  `flat_vect.yaml` (flat, every attribute of `obs.to_vect()`), `graph_comparison/` (the 12 representations of the
  graph comparison, all with the Complete node features via `node_features: complete`: bus graphs merge them per
  bus by `_BUS_AGGREGATION`, `no_edges` is the structure control, `flat` the MLP input; run with
  `slurm/unicluster/compare_graph_obs_spaces_multiseed.sh`, GPU by default, `DEVICE=cpu` for CPU nodes)
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
  uses `imitation.monitor` (default: validation KL = CE − target entropy). Junior HPO: `experiments/pretrain_hpo.py`
  (config `configs/rllib/pretrain_hpo.yaml`: `CompleteGraphObsSpace`, objective `regret_top20` = regret of
  top-20 simulation, percentile pruner; Optuna in `src/imitation/hpo.py`, one study per dataset, journal file
  storage shared by parallel workers, sampler seed hashed from study name + worker), `experiments/summarize_hpo.py`, SLURM chain
  `slurm/unicluster/pretrain_hpo.sh` (`CONFIG_NAME=pretrain_hpo_mlp` for the flat-MLP study), rollouts
  `slurm/unicluster/imitation_rollout_array.sh` (`--do-nothing` for the reference; episode tables of
  `src/analysis/episodes.py` count decisions, executed and illegal topology actions). Training curves and per-state val analysis of the best trials:
  `slurm/unicluster/junior_retrain.sh` retrains them into `results/junior/junior_training/<study>/`
  (`OVERRIDES` + `OUT_SUFFIX` for ablations, e.g. `<study>_complete_obs`).
  `experiments/junior_results.ipynb` plots those runs (validation split only) and the HPO studies.
  Senior (PPO from the Junior): `slurm/unicluster/imitation_vs_rl36.sh <experiment> <junior_run_dir>` trains
  `senior_warmup`, `senior_nowarmup` and `baseline` (random init) per seed, all with the Junior's network and
  observation space (`src/imitation/senior.py` → `experiments/architecture_overrides.py`, dropout set to 0),
  exports each final policy and rolls every agent out on the val and test envs (argmax + top-k), plus the
  do-nothing and Tutor references. Its RL runs set `reward.normalize_by_episode_length=true` (reward / max_step
  in `CustomizedGrid2OpEnvironment`): unscaled, the RL agent collects every grid step's reward until its next
  decision, value targets reach hundreds, and RLlib's `vf_clip_param` (a cap on each sample's squared error)
  zeroes the critic's gradient — explained variance stayed ≈ 0 in all earlier PPO runs. During the critic warm-up `set_policy_trainable` (`src/rarl_rllib/callback.py`)
  freezes actor and GNN, and keeps their BatchNorm layers in eval mode so the sampling policy can't drift.
  The Senior runs keep BatchNorm in eval mode for the whole run (`imitation.freeze_batch_norm`: RLlib samples in
  eval mode but computes the PPO loss in train mode) and set `training.kl_coeff=0.0` (RLlib's old API stack
  ignores `use_kl_loss: false` and applies an adaptive KL penalty).
  36-bus graph comparison with RL: `slurm/unicluster/graph_comparison_36bus_senior.sh` retrains the best Junior of
  each `graph36_<obs>` HPO study (all 12) with new seeds (GPU; `JUNIOR_EXPERIMENT` copies existing Juniors of an
  earlier run instead), fine-tunes each with `senior_warmup` and rolls Junior and Senior out (argmax, val + test)
  into `results/senior/<experiment>/<obs>/`; `experiments/graph_comparison_36bus_results.ipynb` tabulates them.
  Each Senior runs on CPU and GPU in parallel (`RL_DEVICES`; the PPO updates, not sampling, dominate a 36-bus
  Senior's run time): the first to finish claims `eval/senior_s<seed>.claim`, exports and cancels the other;
  `eval/senior_s<seed>/trained_on.txt` records which one. It exports the kept checkpoint with the highest val
  survival (`export_policy_weights.py --select best_val`, `imitation.senior.select_best_val_checkpoint`;
  `checkpoint_selection.json`), so Senior val survival is slightly optimistic and test is the unbiased split.
  The rollouts run inside the Junior job and inside the winning Senior's job after its export
  (`slurm/unicluster/parallel_rollouts.sh`, `ROLLOUT_PROCS` processes per split, default 2 like the reused
  Juniors' rollouts: the converter's running normalizer makes results depend slightly on the chronic split), so
  there are no rollout jobs; a failed Junior rollout does not block the Seniors.
  Training and rollouts read the chronics from a copy on the node-local SSD (`env.chronics_dir=$TMPDIR/data_grid2op`,
  `imitation_rollout.py --chronics-dir`; a missing path makes grid2op silently fall back to `~/data_grid2op`),
  Juniors write their conversion cache there (`imitation.cache_dir`; one cache file per seed and representation,
  up to 9 GB). `export_policy_weights.py` writes `env.chronics_dir: null`, since that copy is gone after the job. The RL model's base configs are `model=gcn_substation_readout obs_space=graph_comparison/<obs>`
  (`model=mlp` for flat), not `model=gnn obs_space=default` (their extra keys would change the network).
  RL chronic order (`CustomizedGrid2OpEnvironment._setup_chronics`, helpers in `src/grid2op_env/utils.py`): every
  env copy gets the seed `worker_seed(experiment.seed, worker_index, vector_index)` from RLlib's `EnvContext`;
  training envs shuffle their chronics with it (`env.shuffle_scenarios`, default true; before 2026-11-11 every
  worker played the sorted chronics in lockstep, i.e. month by month). Evaluation worker i plays val chronic
  i, i + n, ... (n = `evaluation_num_workers`), each seeded with `chronic_seed(evaluation.seed, chronic)` as in
  the imitation rollouts, so every evaluation covers all val chronics with fixed maintenance (before: 18
  chronics per evaluation, the next 18 the next time). `checkpoint_results.json` records each kept
  checkpoint's `training_iteration`; RLlib repeats the last evaluation on iterations without one.
  The critic warm-up recognizes the value head of RLlib's default Dict model (`value_layer`) as well as
  `_value_branch` (`rarl_rllib.callback.is_value_parameter`).
  `experiments/senior_results.ipynb` plots TensorBoard curves (`src/analysis/training_curves.py`) and survival
  tables (`analysis.episodes.survival_metrics`, shared with the Junior notebook).
  Results per pipeline step: `results/teacher/`, `results/tutor/` (Tutor runs, `<run>/experience` + `summary`),
  `results/junior/` (pretraining, HPO studies, rollouts), `results/senior/` (PPO fine-tuning + val/test rollouts).
  Notebooks: `experiments/{teacher,tutor,junior,senior}_results.ipynb`.

### Constants (src/core/constants.py)
Defines global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`, `EDGE_PROBS_PATH`), agent name strings (`RL_AGENT`, `HIGH_LEVEL_AGENT`, `DO_NOTHING_AGENT`), and policy names used consistently across the codebase.

### Callbacks
- `AnnealingCallback`: decays `beta` (KL weight) and `tau` (Gumbel-Softmax temperature) during training
- `TuneCallback`: logs custom metrics used by Ray Tune stoppers/schedulers

### Multi-Agent Setup
RLlib's MultiAgentEnv with three policies: `DO_NOTHING_POLICY`, `RL_POLICY` (trainable RAGNN), `HIGH_LEVEL_POLICY` (selector). Policy mapping is handled in `src/grid2op_env/multi_agent_policies/`.

## Dependencies
Python 3.10–3.12. Key packages: `torch`, `torch-geometric`, `ray[rllib,tune]`, `grid2op`, `gymnasium`, `hydra-core`, `optuna`, `stable-baselines3`. See `environment.yaml` for the full conda spec.
