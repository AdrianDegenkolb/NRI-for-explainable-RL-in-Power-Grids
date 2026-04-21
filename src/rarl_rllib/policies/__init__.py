"""
rarl_rllib.policies – RA-augmented RLlib policy implementations.

Modules
-------
common   Shared helpers for RA policy initialization, KL loss computation,
         prior/mask construction, and TensorBoard metric aggregation.
rappo    RAPPOTorchPolicy — PPO policy with RAGNN KL regularization.
rasac    RASACTorchPolicy — SAC policy with RAGNN KL regularization and
         a dedicated encoder optimizer.
"""

from .rappo import RAPPOTorchPolicy
from .rasac import RASACTorchPolicy

__all__ = ["RAPPOTorchPolicy", "RASACTorchPolicy"]
