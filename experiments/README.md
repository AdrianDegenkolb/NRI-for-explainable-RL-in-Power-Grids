# Experiments

All scripts must be launched from the **project root**. Two directories must be on the
Python path — `src/` (library code) and the project root itself (`experiments/` is
imported as a package by some analysis modules):

```bash
cd /path/to/NRI-for-explainable-RL-in-Power-Grids
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/<script>.py
```

---

## `slurm/` — Cluster job submission (bwUniCluster)

The fastest way to run training. Each script submits one SLURM job per random seed via
`sbatch`, sets up the conda environment, and calls `train.py` with the right Hydra
overrides. Run from anywhere in the repo — the scripts resolve the project root
automatically.

```bash
bash experiments/slurm/cpu/rappo.sh        # submit RAPPO on IEEE 14-bus (5 seeds)
bash experiments/slurm/cpu/rappo36.sh      # submit RAPPO on IEEE 36-bus (seed 0)
bash experiments/slurm/highmem/rappo118.sh # submit RAPPO on IEEE 118-bus (seed 0, highmem)
```

### `slurm/unicluster/`

Used for IEEE 14-bus and IEEE 36-bus training. Jobs run on the `cpu_il` / `cpu` partitions
(64 CPUs, up to 249 GB RAM per job).

| Script | Model | Grid | Seeds | Walltime |
|---|---|---|---|---|
| `rappo.sh` | RAPPO (encoder + RAGNN + PPO) | IEEE 14 | 0–4 | 20 h |
| `radqn.sh` | RADQN (encoder + RAGNN + Rainbow DQN) | IEEE 14 | 0 | 72 h |
| `rasac.sh` | RASAC (encoder + RAGNN + SAC) | IEEE 14 | 0 | 20 h |
| `ppo_mlp.sh` | PPO + MLP baseline | IEEE 14 | 0–4 | 20 h |
| `ppo_gnn.sh` | PPO + GNN baseline | IEEE 14 | 0–4 | 20 h |
| `dqn_mlp.sh` | Rainbow DQN + MLP baseline | IEEE 14 | 0 | 72 h |
| `dqn_gnn.sh` | Rainbow DQN + GNN baseline | IEEE 14 | 0 | 72 h |
| `sac_mlp.sh` | SAC + MLP baseline | IEEE 14 | 0 | 20 h |
| `sac_gnn.sh` | SAC + GNN baseline | IEEE 14 | 0 | 20 h |
| `rappo36.sh` | RAPPO | IEEE 36 | 0 | 72 h |
| `radqn36.sh` | RADQN | IEEE 36 | 0 | 72 h |
| `ppo_mlp36.sh` | PPO + MLP baseline | IEEE 36 | 0 | 72 h |
| `ppo_gnn36.sh` | PPO + GNN baseline | IEEE 36 | 0 | 72 h |
| `dqn_mlp36.sh` | Rainbow DQN + MLP baseline | IEEE 36 | 0 | 72 h |
| `dqn_gnn36.sh` | Rainbow DQN + GNN baseline | IEEE 36 | 0 | 72 h |

### Logs and outputs

Each script creates `results/<date>_<grid>/<variant>/out/` before submitting.
SLURM stdout and stderr are written there as `<variant>_s<seed>.<jobid>.log` and
`error_<variant>_s<seed>.<jobid>.log`. Checkpoints land in
`results/experiments/<date>_<grid>/<variant>/`.

---

## `train.py` — Train an RL policy (local)

The same training entry point used by the SLURM scripts, useful for local smoke tests.

```bash
# Quick smoke test (10 timesteps, no GPU, single worker)
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/train.py experiment=test_minimal training=ppo_test

# MLP baseline (flat obs, no graph)
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/train.py model=mlp obs_space=flat relation_awareness=disabled

# Ad-hoc override
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/train.py experiment.nb_timesteps=500000 env=case36
```

See `configs/README.md` for the full list of config groups and override options.

---

## `compare_models.py` — Compare agent behaviour

Cross-validates trained models to characterise how they differ behaviourally. For each pair
of models (A as primary, B as backup), it runs episodes where A acts until it fails, then
lets B take over — measuring how many additional timesteps B can recover. It also extracts
failure-state topology and congestion profiles and plots spatial action distributions.

**Before running:** update the `load_path` and `checkpoint_name` fields near the top of
the script to point at the checkpoints you want to compare.

```bash
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/compare_models.py
```

Results (SVG/PNG figures + a JSON data file) are saved to `results/cross_validation/`.

#### Output figures

**`agent_failure_states.png`** — Cross-validation heatmaps: how well each backup model
recovers the failure states of each primary model.

![Agent failure states](.images/agent_failure_states.png)

**`agent_behavior_comparison.png`** — Per-model visualisations on the IEEE 14-bus grid:
mean connectivity in failure states (left), mean line congestion profile in failure states
(centre), and spatial action distribution across substations (right).

![Agent behaviour comparison](.images/agent_behavior_comparison.png)

---

## `analyze_latent_graphs.py` — Analyse latent graph hypotheses

Runs a trained RAPPO agent on test episodes and records the inferred posterior edge
probabilities at every step. Three hypotheses are then tested (thesis §5.4.3):

| Hypothesis | Question |
|---|---|
| **H1 — Electrical coupling** | Do high-probability latent edges correlate with power flow sensitivity (PTDF)? |
| **H2 — Risk coupling** | Do latent edges connect components that tend to fail together (cascading risk)? |
| **H3 — Action-effect coupling** | Do latent edges link components whose behaviour is jointly affected by the agent's actions? |

**Before running:** update `load_path` / `checkpoint_name` near the top of the script to
point at the RAPPO checkpoint to analyse, and set `env_name` to the desired test
environment.

```bash
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/analyze_latent_graphs.py
```

Set `compute_data = False` to skip the episode rollout and re-plot from previously saved
results.

---

## Other files

| File | Purpose |
|---|---|
| `utils.py` | Shared helpers: `AgentSpec` dataclass, `load_agent_from_spec` |
| `action_spaces.py` | Utility script for inspecting / generating action space JSON files |
| `evaluate_heuristic_agents.py` | Evaluates rule-based heuristic agents (do-nothing, greedy reconnect) as additional baselines |
