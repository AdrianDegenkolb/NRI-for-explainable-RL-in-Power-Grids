"""
Custom DQN extending RLLib's DQN with fixes shared across all custom algorithms:
  - Correct per-policy checkpoint restore (policy_ids bugfix)
  - Log level and curriculum learning state initialization
"""

import logging
import os
from typing import Callable, Optional

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.checkpoints import get_checkpoint_info

logger = logging.getLogger(__name__)


class CustomDQN(DQN):
    def __init__(
        self,
        config: Optional[dict] = None,
        env=None,
        logger_creator: Optional[Callable] = None,
        **kwargs,
    ):
        logger.info(f"my_log_level: {config['my_log_level']}")
        self.my_log_level = config["my_log_level"]
        self.curriculum_training = config.get("env_config", {}).get("curriculum_training", False)
        self.curriculum_threshold = config.get("env_config", {}).get("curriculum_thresholds", [])
        if self.curriculum_training:
            print("Curriculum training is enabled.")
            print("Curriculum thresholds: ", self.curriculum_threshold)
        super().__init__(config, env, logger_creator, **kwargs)

    @override(Algorithm)
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        checkpoint_info = get_checkpoint_info(checkpoint_dir)
        checkpoint_data = Algorithm._checkpoint_info_to_algorithm_state(
            checkpoint_info, policy_ids=checkpoint_info["policy_ids"]
        )
        self.__setstate__(checkpoint_data)
        if self.config._enable_new_api_stack:
            learner_state_dir = os.path.join(checkpoint_dir, "learner")
            self.learner_group.load_state(learner_state_dir)
        self.callbacks.on_checkpoint_loaded(algorithm=self)
