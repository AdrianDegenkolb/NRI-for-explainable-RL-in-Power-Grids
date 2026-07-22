# Project Summary for Qiong

**Branch:** `main` | **Last updated:** 2026-07-20

---

## Data Flow

```
Grid2Op environment
      │  raw observation
      ▼
GraphObservationConverter
      │  graph dict: {NODES [N,F], EDGE_INDEX, EDGE_MASK}
      ▼
RAFeatureExtractor
      ├─► GraphormerNRIEncoder   → predicts posterior p(z|x) over edge types
      ├─► Gumbel-Softmax sampler → differentiable discrete edge sample
      └─► RAGNN                  → relation-aware message passing → 64-dim embedding
      ▼
RL policy head (PPO / DQN)
      ▼
Grid2Op topology action
```

---

## Model Variants Compared

| Name | Encoder | Graph | RL algorithm |
|------|---------|-------|--------------|
| **RAPPO** | NRI (learned) | Inferred | PPO |
| **RADQN** | NRI (learned) | Inferred | Rainbow DQN |
| **GNN-PPO** | None | Fixed (powerline graph) | PPO |
| **GNN-DQN** | None | Fixed (powerline graph) | DQN |
| **MLP-PPO** | None | None (flat obs) | PPO |
| **MLP-DQN** | None | None (flat obs) | DQN |

---

## Grids Tested

- **IEEE 14-bus** (N=57 nodes in the graph representation, 3,192 fully-connected edges)
- **IEEE 36-bus** (N=177 nodes, 31,152 fully-connected edges)
- **IEEE 118-bus** (larger, still early-stage)

Training runs on **bwUniCluster 3.0** (HPC). Each run is a Slurm job.

---

## Experimental Results

### IEEE 14-bus — single seed, 2026-05-22 batch

| Run | Status | Episodes survived (test) | Steps survived (test) |
|-----|--------|--------------------------|----------------------|
| MLP-PPO | ✅ | 98% | 99.75% |
| GNN-PPO | ✅ | 98% | 98.70% |
| RAPPO | ✅ | 96% | 97.97% |
| MLP-DQN | ✅ | 10% | 30.92% |
| GNN-DQN | ✅ | 6% | 27.79% |
| RADQN | ✅ | 52% | 70.50% |

### IEEE 36-bus — 5-seed multi-seed, 2026-07-18/20 batch (all fixes applied)

144 test episodes per seed. Values are mean ± std across 5 seeds.

| Run | Status | Episodes survived (test) | Steps survived (test) |
|-----|--------|--------------------------|----------------------|
| RAPPO | ✅ | 26.1 ± 2.7% | 43.2 ± 1.8% |
| PPO-GNN | ✅ | 26.0 ± 2.8% | 43.1 ± 1.8% |
| PPO-MLP | ✅ | 26.1 ± 2.7% | 43.1 ± 1.8% |
| MLP-DQN | ✅ (single seed, old batch) | 8.33% | 32.90% |
| RADQN | ❌ crashed (old batch) | — | — |
| GNN-DQN | ❌ crashed (old batch) | — | — |

### IEEE 118-bus — single seed, 2026-05-22 batch

| Run | Status | Episodes survived (test) | Steps survived (test) |
|-----|--------|--------------------------|----------------------|
| MLP-PPO | ✅ | 3.61% | 22.52% |
| GNN-PPO | ✅ | 2.41% | 18.13% |

**Key observations:**
- On 14-bus, all PPO variants reach ~96–99% step survival regardless of architecture.
- On 36-bus (new multi-seed results with all fixes), all three PPO variants converge to nearly **identical** performance (~26% completed episodes, ~43% steps survived). The differences between RAPPO, PPO-GNN, and PPO-MLP are within the seed variance (±1.8 pp on steps%). The NRI encoder provides no measurable benefit over a fixed-topology GNN or flat MLP on this grid.
- The improvement over the old single-seed 36-bus results (32–33% → 43% steps) is attributable to the `grad_clip: 40.0` fix and the corrected KL annealing schedule.

---

## Key Findings and Fixes

### Finding 1 — Graph collapse explains identical curves

The NRI encoder builds a dense latent graph during early training, then **collapses to a
near-empty graph** once KL annealing begins. After collapse, RAPPO/RADQN is behaviourally
equivalent to MLP — which explains why all training curves look the same.

- **IEEE14 RAPPO:** Dense at step 71k → near-empty by step 83k (annealing starts at 60k)
- **IEEE14 RADQN:** Collapses between steps 45M–93M
- Collapse was driven by excessive KL pressure: the old config applied ~12× more pressure than needed to trigger collapse

**Fix applied:** Beta ceiling lowered from 5.0 → 1.0, warm-start at 0.1 instead of 0.0.
Collapse was happening at beta ≈ 0.4; the new schedule reaches that value much later and smoother in training.

---

### Finding 2 — Encoder gradients die in DQN

The encoder receives no training signal in DQN variants. Grad norms decay to exactly zero
(on 36-bus already after ~900k steps — before annealing even starts).

**Root cause:** Two interacting mechanisms:
1. Once the encoder converges to the prior, KL(q‖p) → 0 and its gradient vanishes.
2. Gumbel-Softmax Jacobian collapses to zero as the distribution becomes near-one-hot,
   cutting the TD loss gradient path back to the encoder.

**Fix applied:** Straight-through Gumbel-Softmax (`hard=True` during training). The
gradient is copied directly to the encoder logits, bypassing the collapsed Jacobian.

---

### Finding 3 — GNN gradient explosion scales with grid size for PPO

GNN-PPO and RAPPO showed exploding gradients on large grids. Root cause: PPO had no
`grad_clip` configured at all (DQN had 40.0 by default).

| Grid | PPO-GNN grad norm |
|------|------------------|
| 14-bus | 10–25 (volatile) |
| 36-bus | spikes to ~110 |
| 118-bus | reaches 1000–1500 |

**Fix applied:** `grad_clip: 40.0` added to the PPO config.

---

### Finding 4 — 36-bus graph runs crashed (O(N²) bottlenecks)

Three 36-bus runs (RAPPO, RADQN, GNN-DQN) failed with Ray GCS heartbeat timeouts.
Root cause: CPU-bound graph computation on rollout workers blocked the Ray event loop
for 49–64 seconds per iteration, starving the GCS of keepalive heartbeats.

Three bottlenecks identified and fixed:

| Bottleneck | Before | After | Speedup |
|-----------|--------|-------|---------|
| Observation converter (`_get_edge_index`) | O(N²) per env step — rebuilt 31,329-element matrix every step | Precomputed 491 substation-pair candidates; 1D filter at runtime | 6.1× on IEEE36 |
| Fully-connected edge index (`fully_connected_edge_index`) | Rebuilt 31,152-edge tensor every training step | Module-level cache; free after first call | ~∞ after warmup |
| Latent graph visualization (`visualize_graph`) | O(E) Python loop — up to 31,152 `G.add_edge` calls | `np.where` prefilter before entering loop | 652× at low edge density |

These performance improvements also **reduced memory and compute requirements** substantially,
allowing the 36-bus runs to be submitted with significantly lower resource allocations than before.

---

## Annealing Bug (introduced in post-thesis refactoring, fixed 2026-05-18)

> **Note:** This bug did not exist during the thesis. It was introduced when the training
> callbacks were refactored into the current Hydra/RLlib configuration system.

After refactoring, `current_beta` and `current_tau` never moved from their initial values —
KL annealing was silently disabled for all runs. Two bugs in `callback.py`:

1. `hasattr(policy, "current_beta")` — wrong attribute name (should be `current_beta_graph`),
   so the annealing state was never initialised and the callback returned immediately every iteration.
2. Config lookup used wrong key paths (`beta_start` instead of `beta_graph_edges_start` nested
   under `relation_awareness.loss`).

All results from the refactored codebase prior to the May 2026 fix are therefore invalid for
evaluating KL annealing effects.

---

## Current Status (2026-07-20)

Multi-seed IEEE36 runs (5 seeds each) completed for RAPPO, PPO-GNN, and PPO-MLP with all fixes applied.
Full per-seed breakdown and training curves are in `experiments/compare_survival.ipynb`.

---

## Codebase Structure

```
src/
├── rarl/               Pure PyTorch model (encoder, RAGNN, KL loss, annealing)
│   └── nn/
│       ├── encoder/    GraphormerNRIEncoder
│       ├── ragnn.py    Relation-Aware GNN
│       ├── feature_extractor.py  Full pipeline
│       └── sampling.py Gumbel-Softmax
├── rarl_rllib/         RLlib integration (RAPPO, RADQN, RASAC policies + callbacks)
├── grid2op_env/        Grid2Op ↔ RLlib bridge (env, obs converter, action converter)
├── algorithms/         CustomPPO/SAC/DQN (checkpoint fix + batch sizing fix)
├── core/               Training orchestration, evaluation, constants
├── analysis/           Post-training: latent graph analysis, cross-validation
└── visualization/      Plotting utilities

experiments/
├── train.py            Main entry point
└── slurm/              Slurm job scripts for cluster runs

configs/rllib/          Hydra config tree (experiment, training, model, obs_space, ...)
```

### Setup

```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py   # downloads Grid2Op scenario data
```

### Run RAPPO on IEEE36

```bash
PYTHONPATH=$(pwd)/src python experiments/train.py \
    training=ppo model=ragnn obs_space=graph relation_awareness=default \
    experiment.seed=0 experiment.name=rappo36
```

---

## Still Open

- **DQN variants on 36-bus:** RADQN and GNN-DQN still crashed in the old batch; not yet re-run with fixes.
- **DQN encoder gradient death:** Straight-through fix applied but not yet verified on new runs.
- **GNN gradient explosion on 118-bus:** Grad clip applied; whether it restores performance unknown.
- **Latent graph analysis hypotheses** (H1: electrical coupling, H2: risk coupling, H3: action-effect coupling): framework implemented, per-seed analysis runs exist under `analysis/` in each trial directory.
