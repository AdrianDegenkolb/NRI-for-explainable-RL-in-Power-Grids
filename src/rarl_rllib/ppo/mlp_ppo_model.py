"""
Plain MLP policy on flat observations, as the Junior/Senior network of Lehna et al. 2023
(Fraunhofer CurriculumAgent): all observation attributes concatenated into one vector, then one
fully connected network with ReLU, dropout and orthogonal initialisation.

RLlib's default model is not this network for Dict observations (e.g. FlatObsSpace): its
ComplexInputNetwork builds a separate MLP per attribute and only joins them in the output layer.

Registered as "flat_mlp_model" (`core.rllib_config.register_custom_models`), configured by
`configs/rllib/model/flat_mlp.yaml`.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Dict, Discrete
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from core.utils import getl

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def flat_obs_dim(obs_space: Dict) -> int:
    """Length of the concatenated observation vector of a Dict of Box spaces."""
    return int(sum(np.prod(space.shape) for space in obs_space.spaces.values()))


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str, dropout_prob: float) -> nn.Sequential:
    """Fully connected network: (Linear, activation, Dropout) per hidden layer, then a linear output layer.

    Args:
        in_dim: Input features.
        hidden: Neurons per hidden layer.
        out_dim: Output features.
        activation: Key of `_ACTIVATIONS`.
        dropout_prob: Dropout after every hidden layer (0 = none).

    Returns:
        The network, with orthogonal weights and zero biases.
    """
    layers: list[nn.Module] = []
    for size in hidden:
        layers += [nn.Linear(in_dim, size), _ACTIVATIONS[activation]()]
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        in_dim = size
    layers.append(nn.Linear(in_dim, out_dim))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class FlatMLPModel(TorchModelV2, nn.Module):
    """Actor-critic MLP on the concatenation of all observation attributes.

    Policy and value function are separate networks of the same hidden sizes (`fcnet_hiddens`).
    The value network's parameter names contain `_value_branch`, so the critic warm-up
    (`rarl_rllib.callback.set_policy_trainable`) trains it alone.

    custom_model_config:
        mlp.dropout_prob: Dropout after every hidden layer of the policy network (default 0).
    """

    def __init__(self, obs_space: Dict, action_space: Discrete, num_outputs: int, model_config: ModelConfigDict,
                 name: str, **kwargs):
        """Args: see TorchModelV2; `kwargs` is RLlib's custom_model_config."""
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        obs_space = getattr(obs_space, "original_space", obs_space)
        self._keys = list(obs_space.spaces)
        in_dim = flat_obs_dim(obs_space)
        hidden = list(model_config["fcnet_hiddens"])
        activation = model_config.get("fcnet_activation", "relu")
        dropout_prob = float(getl(kwargs.get("mlp") or {}, "dropout_prob", 0.0))
        self.policy_net = mlp(in_dim, hidden, num_outputs, activation, dropout_prob)
        self._value_branch = mlp(in_dim, hidden, 1, activation, 0.0)
        self._features: torch.Tensor | None = None

    def flatten(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the observation attributes in observation-space order: [B, flat_obs_dim].

        Scalar attributes (shape (), e.g. the date fields of FlatObsSpace) arrive as [B] tensors.
        """
        return torch.cat([obs[key].float().reshape(obs[key].shape[0], -1) for key in self._keys], dim=1)

    def forward(self, input_dict: dict[str, TensorType], state: list[TensorType], seq_lens: TensorType
                ) -> tuple[TensorType, list[TensorType]]:
        """Action logits [B, num_outputs] from `input_dict["obs"]` (dict of [B, ...] tensors)."""
        self._features = self.flatten(input_dict["obs"])
        return self.policy_net(self._features), state

    def value_function(self) -> TensorType:
        """State values [B] of the last forward pass."""
        return self._value_branch(self._features).squeeze(-1)
