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

from .model import RARLModel, RASACTorchModel
from .policy import RAPPOTorchPolicy, RASACTorchPolicy
from .callback import AnnealingCallback, CustomMetricsCallback, TuneCallback

__all__ = [
    "RARLModel",
    "RASACTorchModel",
    "RAPPOTorchPolicy",
    "RASACTorchPolicy",
    "AnnealingCallback",
]
