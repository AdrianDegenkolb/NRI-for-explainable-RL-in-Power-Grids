"""
rarl-rllib – RLlib integration for Relation-Aware RL.

Key exports
-----------
RARLModel          – TorchModelV2 for PPO: encoder + RAGNN → policy/value heads.
RAPPOTorchPolicy   – PPOTorchPolicy augmented with RAGNN KL regularization.
AnnealingCallback  – RLlib callback that anneals beta and tau.
"""

from .ppo import *
from .callback import AnnealingCallback, CustomMetricsCallback, TuneCallback, PretrainingCallback

__all__ = [
    "RAActorCriticModel",
    "RAPPOTorchPolicy",
    "AnnealingCallback",
    "CustomMetricsCallback",
    "TuneCallback",
    "PretrainingCallback",
]
