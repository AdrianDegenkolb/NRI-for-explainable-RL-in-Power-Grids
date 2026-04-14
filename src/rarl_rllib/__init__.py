"""
rarl-rllib – RLlib integration for Relation-Aware RL.

Key exports
-----------
RARLModel          – TorchModelV2 that wraps RAFeatureExtractor.
make_rarl_policy   – Factory that makes any TorchPolicy relation-aware.
AnnealingCallback  – RLlib callback that anneals beta and tau.
"""

from .model import RARLModel
from .policy import make_rarl_policy
from .callback import AnnealingCallback

__all__ = ["RARLModel", "make_rarl_policy", "AnnealingCallback"]
