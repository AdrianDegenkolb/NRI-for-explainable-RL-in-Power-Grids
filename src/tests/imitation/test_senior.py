"""
Unit tests for imitation/senior.py (RL overrides from a Junior run's config).

Tests are organised by class:
  - TestFormatValue
  - TestArchitectureOverrides
"""
from __future__ import annotations

import shlex
import unittest

from omegaconf import DictConfig, OmegaConf

from imitation.senior import SKIPPED_KEYS, architecture_overrides, flatten, format_value, shell_overrides
from tests.hydra_config import compose_config

# As in slurm/unicluster/imitation_vs_rl36.sh, before the architecture overrides.
RL_OVERRIDES = ["training=ppo", "env=case36", "model=gnn", "obs_space=default", "relation_awareness=disabled",
                "imitation=default"]
# A Junior HPO trial that differs from the gnn defaults in every searched model setting.
JUNIOR_OVERRIDES = ["model.gnn.hidden_dim=512", "model.gnn.num_layers=4", "model.gnn.dropout_prob=0.27",
                    "model.fcnet_hiddens=[256,256]"]


def section_leaves(cfg: DictConfig, section: str) -> dict:
    """`{dotted.key: value}` of one config section, unresolved, without machine-specific keys."""
    flat = flatten(OmegaConf.to_container(cfg[section], resolve=False), section)
    return {k: v for k, v in flat.items() if k not in SKIPPED_KEYS}


class TestFormatValue(unittest.TestCase):
    def test_hydra_grammar(self):
        self.assertEqual(format_value([256, 256]), "[256,256]")
        self.assertEqual(format_value(["r", "t"]), "[r,t]")
        self.assertEqual(format_value(None), "null")
        self.assertEqual(format_value(False), "false")
        self.assertEqual(format_value(0.25), "0.25")


class TestArchitectureOverrides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.junior = compose_config("pretrain_hpo", JUNIOR_OVERRIDES)

    def test_rl_config_gets_junior_network_and_observations(self):
        rl = compose_config("config", RL_OVERRIDES + architecture_overrides(self.junior, dropout_prob=None))
        for section in ("model", "obs_space"):
            self.assertEqual(section_leaves(rl, section), section_leaves(self.junior, section), section)
        self.assertEqual(rl.obs_space.observation_space, "CompleteGraphObsSpace")
        self.assertEqual(list(rl.model.fcnet_hiddens), [256, 256])

    def test_dropout_is_replaced(self):
        rl = compose_config("config", RL_OVERRIDES + architecture_overrides(self.junior))
        self.assertEqual(rl.model.gnn.dropout_prob, 0.0)
        self.assertEqual(rl.model.gnn.hidden_dim, 512)

    def test_machine_specific_paths_are_skipped(self):
        self.assertFalse(any("edge_probs_path" in o for o in architecture_overrides(self.junior)))

    def test_shell_line_splits_back_into_the_overrides(self):
        overrides = architecture_overrides(self.junior)
        self.assertEqual(shlex.split(shell_overrides(overrides)), overrides)


if __name__ == "__main__":
    unittest.main()
