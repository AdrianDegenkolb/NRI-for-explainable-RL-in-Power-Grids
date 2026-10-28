"""
Unit tests for rarl_rllib/ppo/gnn_ppo_model.py (GNNBaselineModel) with the per-substation readout,
on observations of the graph-comparison converters.
"""
from __future__ import annotations

import unittest
import warnings

import grid2op
import numpy as np
import torch
from grid2op.gym_compat import GymEnv
from lightsim2grid import LightSimBackend

from core.rllib_config import register_custom_models
from grid2op_env.observation_converter import make_observation_converter
from imitation.pretrain import build_policy_model, policy_logits

warnings.filterwarnings("ignore")
N_ACTIONS = 4
MODEL_CONFIG = {
    "custom_model": "gnn_model",
    "fcnet_hiddens": [16],
    "fcnet_activation": "relu",
    "post_fcnet_hiddens": [],
    "custom_model_config": {"gnn": {"hidden_dim": 8, "out_dim": 3, "num_layers": 2, "conv_type": "gcn",
                                    "readout": "substation"}},
}


class TestGNNBaselineModelSubstationReadout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register_custom_models()
        cls.env = grid2op.make("l2rpn_case14_sandbox", backend=LightSimBackend())
        cls.gym_env = GymEnv(cls.env)
        cls.obs = cls.env.reset(seed=0, options={"time serie id": 0})

    def batch(self, converter, size: int) -> dict[str, torch.Tensor]:
        """The converted observation repeated `size` times, as tensors."""
        g = converter.to_gym(self.obs)
        return {key: torch.as_tensor(np.stack([value] * size)) for key, value in g.items()}

    def model_and_batch(self, mode: str):
        config = {"observation_space": mode, "node_features": "complete", "node_substations": True}
        converter = make_observation_converter(self.gym_env, config, backend=self.env.backend)
        return build_policy_model(converter.observation_space, N_ACTIONS, MODEL_CONFIG), self.batch(converter, 2)

    def test_head_input_is_substations_times_out_dim(self):
        for mode in ["GraphObsSpace", "SubstationGraphObsSpace", "LODFGraphObsSpace", "ElementLODFGraphObsSpace"]:
            with self.subTest(mode=mode):
                model, _ = self.model_and_batch(mode)
                first_hidden = model.mlp._hidden_layers[0]._model[0]
                self.assertEqual(first_hidden.in_features, self.env.n_sub * 3)

    def test_logits_and_values(self):
        for mode in ["GraphObsSpace", "PTDFGraphObsSpace", "ElementGraphObsSpace"]:
            with self.subTest(mode=mode):
                model, batch = self.model_and_batch(mode)
                logits = policy_logits(model, batch)
                self.assertEqual(logits.shape, (2, N_ACTIONS))
                self.assertEqual(model.value_function().shape, (2,))
                self.assertFalse(torch.isnan(logits).any())

    def test_missing_node_substations_raises(self):
        converter = make_observation_converter(self.gym_env, {"observation_space": "GraphObsSpace"},
                                               backend=self.env.backend)
        with self.assertRaises(ValueError):
            build_policy_model(converter.observation_space, N_ACTIONS, MODEL_CONFIG)


if __name__ == "__main__":
    unittest.main()
