"""
Synthesizes a case36 RAPPO training batch and calls learn_on_batch once for profiling.

Run from project root:
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python experiments/profile_learn_on_batch.py

Or with cProfile directly:
    conda run -n L2RPN PYTHONPATH=$(pwd)/src python -m cProfile -s cumtime \
        experiments/profile_learn_on_batch.py 2>&1 | head -60
"""

import os
from pathlib import Path

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.models import ModelCatalog

from core.constants import RL_AGENT
from core.loading import load_config, preprocess_config
from grid2op_env import CustomizedGrid2OpEnvironment
from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK

PARAMS_DIR = Path(
    "results/2026_05_26_IEEE36/rappo/CustomPPO_RARL_4808025_62cfb_2026-05-26_00-30-51"
)
# Use sgd_minibatch_size (16), not train_batch_size (1024).
# learn_on_batch runs the model on the FULL batch before the SGD loop (for GAE).
# With B=1024 and N=177 nodes, the FC edge_index has 31.9M edges → 32 GB allocation
# in GCNConv alone. B=16 → 498k edges → fits in RAM.
# Multiply measured time by (train_batch_size/sgd_minibatch_size * num_sgd_iter) = 320
# to estimate total learn_time per training iteration.
TRAIN_BATCH_SIZE = 16


def _make_obs(obs_space, batch_size: int) -> dict:
    nodes_shape = obs_space[NODES].shape        # (177, 7)
    ei_shape = obs_space[EDGE_INDEX].shape       # (2, 1100)
    mask_shape = obs_space[EDGE_MASK].shape      # (1100,)
    n_nodes = nodes_shape[0]
    e_max = mask_shape[0]

    # ~118 active edges (59 lines × 2 directions), rest zero-padded
    n_active = 118
    src = np.arange(n_active // 2) % n_nodes
    dst = (src + 1) % n_nodes
    ei_active = np.stack([np.concatenate([src, dst]),
                          np.concatenate([dst, src])], axis=0).astype(np.int32)

    edge_index = np.zeros((batch_size, *ei_shape), dtype=np.int32)
    edge_index[:, :, :n_active] = ei_active[np.newaxis]

    edge_mask = np.zeros((batch_size, e_max), dtype=np.float32)
    edge_mask[:, :n_active] = 1.0

    return {
        NODES: np.random.randn(batch_size, *nodes_shape).astype(np.float32),
        EDGE_INDEX: edge_index,
        EDGE_MASK: edge_mask,
    }


def _make_batch(obs_space, action_space, batch_size: int) -> SampleBatch:
    obs = _make_obs(obs_space, batch_size)
    n_act = action_space.n
    return SampleBatch({
        SampleBatch.OBS:               obs,
        SampleBatch.NEXT_OBS:          obs,
        SampleBatch.ACTIONS:           np.random.randint(0, n_act, size=(batch_size,)),
        SampleBatch.REWARDS:           np.zeros(batch_size, dtype=np.float32),
        SampleBatch.TERMINATEDS:       np.zeros(batch_size, dtype=bool),
        SampleBatch.TRUNCATEDS:        np.zeros(batch_size, dtype=bool),
        SampleBatch.INFOS:             [{} for _ in range(batch_size)],
        SampleBatch.ACTION_LOGP:       np.full(batch_size, -5.0, dtype=np.float32),
        SampleBatch.ACTION_DIST_INPUTS: np.zeros((batch_size, n_act), dtype=np.float32),
        SampleBatch.VF_PREDS:          np.zeros(batch_size, dtype=np.float32),
        "advantages":                  np.ones(batch_size, dtype=np.float32),
        "value_targets":               np.zeros(batch_size, dtype=np.float32),
    })


def main():
    import time

    print("Loading config ...")
    params = load_config(PARAMS_DIR)
    params = preprocess_config(params)

    print("Creating env (for obs/action spaces) ...")
    env = CustomizedGrid2OpEnvironment(params["env_config"])
    obs_space = env.observation_space[RL_AGENT]
    act_space = env.action_space[RL_AGENT]
    print(f"  NODES {obs_space[NODES].shape}  "
          f"EDGE_INDEX {obs_space[EDGE_INDEX].shape}  "
          f"actions {act_space.n}")

    print("Registering custom models ...")
    from rarl_rllib import RAActorCriticModel
    from rarl_rllib.ppo.rappo_policy import RAPPOTorchPolicy
    ModelCatalog.register_custom_model("ra_actor_critic_model", RAActorCriticModel)
    ModelCatalog.register_custom_model("ragnn_model", RAActorCriticModel)

    print("Creating RAPPOTorchPolicy ...")
    params["model"].setdefault("max_seq_len", 20)
    policy = RAPPOTorchPolicy(obs_space, act_space, params)
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    policy.callbacks = DefaultCallbacks()

    print(f"Synthesizing batch (size={TRAIN_BATCH_SIZE}) ...")
    batch = _make_batch(obs_space, act_space, TRAIN_BATCH_SIZE)

    # Warm-up (JIT / first-call overhead)
    print("Warm-up pass ...")
    policy.learn_on_batch(_make_batch(obs_space, act_space, 16))

    # ── profile target ────────────────────────────────────────────────────────
    print(f"\nRunning learn_on_batch(batch_size={TRAIN_BATCH_SIZE}) — profile this call.")
    t0 = time.perf_counter()
    policy.learn_on_batch(batch)
    print(f"learn_on_batch wall time: {time.perf_counter() - t0:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
