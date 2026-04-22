"""RAPPOTorchPolicy — PPO policy augmented with RAGNN KL regularization.

Extends RLlib's PPOTorchPolicy (TorchPolicyV2, on-policy) by:
  1. Initializing RA annealing attributes (beta, tau) from the
     ``relation_awareness`` config block.
  2. Appending a KL regularization term to the PPO loss that penalizes
     divergence between the encoder's discrete latent edge posterior and
     a structured prior.
  3. Reporting RA-specific metrics (KL components, edge fractions,
     annealing state) alongside standard PPO stats.
"""

from typing import Dict

from ray.rllib import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.models import ModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType

from rarl_rllib.common import init_ra_config, apply_ra_kl_loss, build_ra_stats_dict


class RAPPOTorchPolicy(PPOTorchPolicy):
    """PPO policy augmented with RAGNN KL regularization and RA metrics."""

    def __init__(self, observation_space, action_space, config):
        init_ra_config(self, config)
        super().__init__(observation_space, action_space, config)

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class,
        train_batch: SampleBatch,
    ) -> TensorType:
        base_loss = super().loss(model, dist_class, train_batch)
        return apply_ra_kl_loss(self, model, train_batch, base_loss)

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().stats_fn(train_batch)
        stats.update(build_ra_stats_dict(self.model_gpu_towers))
        return stats
