"""
rarl-rllib – RLlib integration for Relation-Aware RL.

Key exports
-----------
RARLModel          – TorchModelV2 for PPO: encoder + RAGNN → policy/value heads.
RASACTorchModel    – SACTorchModel subclass: encoder + RAGNN → embedding for SAC heads.
RAPPOTorchPolicy   – PPOTorchPolicy augmented with RAGNN KL regularization.
RASACTorchPolicy   – SACTorchPolicy equivalent built with RAGNN KL regularization.
AnnealingCallback  – RLlib callback that anneals beta and tau.
"""

from .model import RAActorCriticModel, RASACTorchModel, RADQNTorchModel
from .policies.rasac import RASACTorchPolicy
from .policies.rappo import RAPPOTorchPolicy
from .policies.radqn import RADQNTorchPolicy
from .callback import AnnealingCallback, CustomMetricsCallback, TuneCallback

__all__ = [
    "RAActorCriticModel",
    "RASACTorchModel",
    "RADQNTorchModel",
    "RAPPOTorchPolicy",
    "RASACTorchPolicy",
    "RADQNTorchPolicy",
    "AnnealingCallback",
]
