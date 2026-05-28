# Source Package Overview

All Python source code lives here. Add this directory to `PYTHONPATH` when running scripts
from the project root:

```bash
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/train.py
```

---

## Data Flow

```
Grid2Op environment
      │  raw observation (BaseObservation)
      ▼
grid2op_env/observation_converter.py
      │  graph dict: {NODES [N,F], EDGE_INDEX, EDGE_MASK}
      ▼
rarl/nn/feature_extractor.py   (RAFeatureExtractor)
      ├─► rarl/nn/encoder/        GraphormerNRIEncoder
      │         infers p(z|x): posterior over edge types
      ├─► rarl/nn/sampling.py     Gumbel-Softmax sample
      └─► rarl/nn/ragnn.py        RAGNN message passing → 64-dim embedding
      │
      ▼
rarl_rllib/                    RLlib policy wrapper
      │  logits / Q-values / action distribution
      ▼
grid2op_env/action_converters.py
      │  Grid2Op action
      ▼
Grid2Op environment  (step)
```

---

## Packages

### `rarl/` — Core model (pure PyTorch)

The neural network stack, independent of RLlib. All training-framework-agnostic model code
lives here.

| Module | Description |
|---|---|
| `nn/encoder/` | `GraphormerNRIEncoder` — global self-attention over nodes, predicts K edge-type logits for every pair in the fully-connected graph |
| `nn/ragnn.py` | Relation-Aware GNN — K−1 separate GCNConv passes per layer, weighted by the encoder's posterior; residual connections, batch norm, ELU |
| `nn/feature_extractor.py` | `RAFeatureExtractor` — wires encoder → sampler → RAGNN → MLP into a single `forward` call |
| `nn/sampling.py` | Gumbel-Softmax with annealed temperature τ; supports `hard=True` (straight-through) for DQN |
| `nn/mlp.py` | Plain MLP used as the policy head |
| `loss.py` | KL divergence loss between posterior and prior, with separate β weights for graph and non-graph edges |
| `prior.py` | Constructs the prior distribution tensor over all edges given the known powerline graph |
| `graph.py` | Graph utilities: fully-connected edge index, subgraph extraction |
| `annealing.py` | Schedule functions for β (KL weight) and τ (temperature) |

---

### `rarl_rllib/` — RLlib integration

Wraps the `rarl/` model in RLlib's `TorchModelV2` interface and defines custom policies
and training callbacks.

| Module | Description |
|---|---|
| `ppo/rappo_model.py` + `rappo_policy.py` | RAPPO — PPO policy with KL loss added to the PPO surrogate |
| `sac/rasac_model.py` + `rasac_policy.py` | RASAC — SAC with a dedicated encoder optimiser (separate from actor/critic) |
| `dqn/radqn_model.py` + `radqn_policy.py` | RADQN — Rainbow DQN (dueling, double-Q, C51, PER, n-step) with KL loss |
| `ppo/gnn_ppo_model.py` | GNN-PPO baseline — fixed precomputed edge probs, no encoder |
| `sac/gnn_sac_model.py` | GNN-SAC baseline |
| `dqn/gnn_dqn_model.py` + `gnn_dqn_policy.py` | GNN-DQN baseline |
| `dqn/mlp_dqn_policy.py` | MLP-DQN baseline (flat observation) |
| `callback.py` | `AnnealingCallback` (decays β and τ on schedule) and `TuneCallback` (logs custom metrics for Ray Tune) |
| `model.py` | Shared base model utilities |
| `common.py` | Shared helpers across RA policy variants |

---

### `grid2op_env/` — Environment wrappers

Bridges Grid2Op and RLlib's `MultiAgentEnv` interface.

| Module | Description |
|---|---|
| `env.py` | `CustomizedGrid2OpEnvironment` — the RLlib `MultiAgentEnv`; routes steps to three agents |
| `observation_converter.py` | `GraphObservationConverter` (graph dict) and `FlatObservationConverter` (flat Box); both normalise online with a running mean/variance |
| `action_converters.py` | Maps discrete RLlib actions to Grid2Op topology actions |
| `rewards.py` | Custom reward functions (`rho`, `scaled_l2rpn`, `constant`, `alpha_zero`) |
| `multi_agent_policies/do_nothing_policy.py` | DO_NOTHING agent — always returns the no-op action |
| `multi_agent_policies/select_agent_policy.py` | HIGH_LEVEL agent — decides whether the RL agent or do-nothing agent acts based on ρ threshold |

---

### `algorithms/` — Custom RLlib algorithm classes

Thin subclasses of RLlib's `PPO`, `SAC`, and `DQN`. The KL loss is **not** added here —
it lives inside the custom policies in `rarl_rllib/`. These subclasses exist for three
reasons: (1) fix an RLlib bug where `load_checkpoint` ignores `policy_ids` and crashes
when only a subset of policies were checkpointed; (2) store `my_log_level` and curriculum
training state from the config; (3) `CustomPPO` additionally overrides
`training_step` with a corrected `custom_synchronous_parallel_sample` that counts steps
per trainable policy only, giving an accurate `train_batch_size` in the multi-agent setup.

| Module | Description |
|---|---|
| `custom_ppo.py` | `CustomPPO` — accurate batch sizing + checkpoint bugfix |
| `custom_sac.py` | `CustomSAC` — checkpoint bugfix |
| `custom_dqn.py` | `CustomDQN` — checkpoint bugfix |
| `optuna_search.py` | Optuna integration for hyperparameter search via Ray Tune |

---

### `core/` — Training infrastructure

| Module | Description |
|---|---|
| `constants.py` | Global output paths (`EVAL_PATH`, `EDGE_PROBS_PATH`), agent name strings, policy name constants |
| `train.py` | `run_training` — builds the Ray Tune `Tuner`, runs it, prints checkpoint summary, triggers post-training evaluation |
| `evaluate.py` | `evaluate_rllib_checkpoint` — loads a checkpoint and runs evaluation episodes |
| `loading.py` | Checkpoint loading helpers (resolves paths, restores RLlib algorithm state) |
| `utils.py` | Miscellaneous utilities |
| `heuristic_actions.py` | Heuristic action selection logic used by some baseline agents |

---

### `agents/` — Agent wrappers

High-level agent classes used by analysis and evaluation scripts (not by RLlib training
directly).

| Module | Description |
|---|---|
| `rllib_agent.py` | `RllibAgent` — wraps a loaded RLlib policy for step-by-step interaction with a Grid2Op environment |
| `rho_greedy_agent.py` | Greedy agent that always picks the action minimising maximum ρ |
| `heuristic_agent.py` | Rule-based agent for baseline comparison |

---

### `analysis/` — Post-training analysis

| Package | Description |
|---|---|
| `analyze_latent_graphs/` | Runs RAPPO on test episodes and tests three hypotheses about what the inferred latent edges represent (H1: electrical coupling, H2: risk coupling, H3: action-effect coupling). Entry point: `experiments/analyze_latent_graphs.py`. |
| `cross_validate_models/` | Cross-validates pairs of trained agents (one fails, the other rescues) to characterise behavioural differences. Entry point: `experiments/compare_models.py`. |

---

### `visualization/` — Plotting utilities

Standalone scripts for visualising training results and edge probabilities. Not part of
the main training loop; run interactively or via Jupyter.

---

### `tests/`

Unit and integration tests, run with `pytest`:

```bash
python -m pytest src/tests/                         # all tests
python -m pytest src/tests/ra_agents/               # model tests only
python -m pytest src/tests/nri/test_nri_module.py   # single file
```

| Subfolder | Coverage |
|---|---|
| `common/` | Environment, observation space, reward, MLP |
| `nri/` | Encoder, ELBO loss, sampling, NRI module end-to-end |
| `ra_agents/` | Feature extractor, RAGNN, RAPPO, Graphormer encoder, KL loss |
| `visualization/` | Visualisation utilities |
