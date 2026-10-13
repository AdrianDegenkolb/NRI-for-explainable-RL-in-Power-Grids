"""
Unit tests for imitation/pretrain.py, imitation/torch_dataset.py and the critic warm-up of
rarl_rllib/callback.py.

Tests are organised by class:
  - TestSoftCrossEntropy
  - TestEvaluate
  - TestTrimEdges
  - TestCriticWarmup
  - TestPPOCompatibility
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from core.rllib_config import register_custom_models
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, GLOBAL, NODE_MASK, NODES
from imitation.pretrain import build_policy_model, evaluate, policy_logits, soft_cross_entropy
from imitation.torch_dataset import _trim_edges
from rarl_rllib.callback import is_value_parameter, set_policy_trainable

N_NODES, N_EDGES, N_ACTIONS = 6, 10, 4


def graph_obs_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        NODES: gym.spaces.Box(-np.inf, np.inf, shape=(N_NODES, 3), dtype=np.float32),
        EDGE_INDEX: gym.spaces.Box(0, N_NODES - 1, shape=(2, N_EDGES), dtype=np.int64),
        EDGE_MASK: gym.spaces.Box(0, 1, shape=(N_EDGES,), dtype=np.bool_),
        NODE_MASK: gym.spaces.Box(0, 1, shape=(N_NODES,), dtype=np.bool_),
        GLOBAL: gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
    })


def graph_batch(batch_size: int) -> dict[str, torch.Tensor]:
    edge_index = torch.randint(0, N_NODES, (batch_size, 2, N_EDGES))
    edge_mask = torch.zeros(batch_size, N_EDGES, dtype=torch.bool)
    edge_mask[:, :7] = True
    return {
        NODES: torch.randn(batch_size, N_NODES, 3),
        EDGE_INDEX: edge_index,
        EDGE_MASK: edge_mask,
        NODE_MASK: torch.ones(batch_size, N_NODES, dtype=torch.bool),
        GLOBAL: torch.randn(batch_size, 6),
    }


GNN_MODEL_CONFIG = {
    "custom_model": "gnn_model",
    "fcnet_hiddens": [16],
    "fcnet_activation": "relu",
    "post_fcnet_hiddens": [],
    "custom_model_config": {"gnn": {"hidden_dim": 8, "out_dim": 8, "num_layers": 2, "conv_type": "gcn"}},
}


def gnn_model():
    register_custom_models()
    return build_policy_model(graph_obs_space(), N_ACTIONS, GNN_MODEL_CONFIG)


class TestSoftCrossEntropy(unittest.TestCase):
    def test_equals_cross_entropy_for_one_hot_targets(self):
        logits = torch.randn(5, N_ACTIONS)
        labels = torch.tensor([0, 1, 2, 3, 1])
        expected = F.cross_entropy(logits, labels)
        actual = soft_cross_entropy(logits, F.one_hot(labels, N_ACTIONS).float())
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)


class _FixedLogits(torch.nn.Module):
    """Stands in for a policy model: logits are passed in as the observation."""

    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs"]["logits"], state


class TestEvaluate(unittest.TestCase):
    def test_regret_and_good_action_rate(self):
        # Sample 0 picks action 1 (best). Sample 1 picks action 2: valid, 0.05 worse than best.
        # Sample 2 picks invalid action 3.
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
        target = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        max_rho = torch.tensor([[0.9, 0.8, 0.85, 0.9], [0.9, 0.80, 0.85, 0.9], [0.9, 0.9, 0.9, float("nan")]])
        valid = torch.tensor([[True] * 4, [True] * 4, [True, True, True, False]])
        dataset = [{"obs": {"logits": logits[i]}, "target": target[i], "max_rho": max_rho[i], "valid": valid[i]}
                   for i in range(3)]
        metrics = evaluate(_FixedLogits(), dataset, batch_size=2, device=torch.device("cpu"))
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["good_action_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["invalid_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["regret"], 0.025, places=5)
        self.assertEqual(metrics["do_nothing_rate"], 0.0)


class TestTrimEdges(unittest.TestCase):
    def test_trims_edge_axes_to_largest_edge_count(self):
        mask = np.zeros((2, 6), dtype=bool)
        mask[0, :3] = True
        mask[1, :2] = True
        arrays = {EDGE_MASK: mask, EDGE_INDEX: np.zeros((2, 2, 6), dtype=np.int32), NODES: np.zeros((2, 6, 3))}
        trimmed = _trim_edges(arrays)
        self.assertEqual(trimmed[EDGE_MASK].shape, (2, 3))
        self.assertEqual(trimmed[EDGE_INDEX].shape, (2, 2, 3))
        self.assertEqual(trimmed[NODES].shape, (2, 6, 3))


class TestCriticWarmup(unittest.TestCase):
    """During the warm-up only the value branch may change."""

    def test_frozen_policy_keeps_logits_while_value_learns(self):
        torch.manual_seed(0)
        model = gnn_model()
        self.assertTrue(any(is_value_parameter(n) for n, _ in model.named_parameters()))
        set_policy_trainable(SimpleNamespace(model=model), False)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        obs = graph_batch(8)
        logits = policy_logits(model, obs)
        loss = model.value_function().pow(2).mean() + logits.sum() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            changed = not torch.equal(before[name], param.detach())
            self.assertEqual(changed, is_value_parameter(name), name)

        set_policy_trainable(SimpleNamespace(model=model), True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class TestPPOCompatibility(unittest.TestCase):
    """Pretrained weights must load strictly into the model PPO builds (ImitationInitCallback)."""

    def test_state_dict_loads_into_ppo_policy_model(self):
        pretrained = gnn_model()
        config = PPOConfig().framework("torch").training(model=GNN_MODEL_CONFIG).to_dict()
        config["_disable_preprocessor_api"] = True  # as in experiments/train.py
        policy = PPOTorchPolicy(graph_obs_space(), gym.spaces.Discrete(N_ACTIONS), config)
        policy.model.load_state_dict(pretrained.state_dict())


if __name__ == "__main__":
    unittest.main()
