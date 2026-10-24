"""
Unit tests for rarl_rllib/ppo/mlp_ppo_model.py (FlatMLPModel).

Tests are organised by class:
  - TestFlatMLPModel
  - TestFlatMLPTraining
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from core.rllib_config import register_custom_models
from imitation.pretrain import build_policy_model, policy_logits
from rarl_rllib.callback import is_value_parameter, set_policy_trainable

N_ACTIONS = 5
MODEL_CONFIG = {
    "custom_model": "flat_mlp_model",
    "fcnet_hiddens": [16, 8],
    "fcnet_activation": "relu",
    "post_fcnet_hiddens": [],
    "custom_model_config": {"mlp": {"dropout_prob": 0.3}},
}


def flat_obs_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "rho": gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        "topo_vect": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        "year": gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),  # scalar, as the date fields of FlatObsSpace
    })


def flat_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {key: torch.randn(batch_size, *space.shape) for key, space in flat_obs_space().spaces.items()}


def flat_mlp():
    register_custom_models()
    return build_policy_model(flat_obs_space(), N_ACTIONS, MODEL_CONFIG)


class TestFlatMLPModel(unittest.TestCase):
    def test_one_mlp_on_the_concatenated_observation(self):
        model = flat_mlp()
        first = model.policy_net[0]
        self.assertEqual(first.in_features, 4 + 6 + 1)
        self.assertEqual([m.out_features for m in model.policy_net if isinstance(m, torch.nn.Linear)], [16, 8, N_ACTIONS])
        self.assertEqual(sum(isinstance(m, torch.nn.Dropout) for m in model.policy_net), 2)

    def test_logits_and_values_have_batch_shape(self):
        model = flat_mlp()
        logits = policy_logits(model, flat_batch(3))
        self.assertEqual(tuple(logits.shape), (3, N_ACTIONS))
        self.assertEqual(tuple(model.value_function().shape), (3,))

    def test_weights_are_orthogonal(self):
        weight = flat_mlp().policy_net[0].weight.detach()  # [16, 11]: orthonormal columns
        torch.testing.assert_close(weight.T @ weight, torch.eye(weight.shape[1]), atol=1e-5, rtol=0)

    def test_eval_mode_is_deterministic(self):
        model = flat_mlp()
        model.eval()
        obs = flat_batch(4)
        torch.testing.assert_close(policy_logits(model, obs), policy_logits(model, obs))


class TestFlatMLPTraining(unittest.TestCase):
    def test_critic_warmup_trains_only_the_value_branch(self):
        torch.manual_seed(0)
        model = flat_mlp()
        set_policy_trainable(SimpleNamespace(model=model), False)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        logits = policy_logits(model, flat_batch(8))
        loss = model.value_function().pow(2).mean() + logits.sum() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, param in model.named_parameters():
            self.assertEqual(not torch.equal(before[name], param.detach()), is_value_parameter(name), name)

    def test_state_dict_loads_into_ppo_policy_model(self):
        pretrained = flat_mlp()
        config = PPOConfig().framework("torch").training(model=MODEL_CONFIG).to_dict()
        config["_disable_preprocessor_api"] = True  # as in experiments/train.py
        policy = PPOTorchPolicy(flat_obs_space(), gym.spaces.Discrete(N_ACTIONS), config)
        policy.model.load_state_dict(pretrained.state_dict())


if __name__ == "__main__":
    unittest.main()
