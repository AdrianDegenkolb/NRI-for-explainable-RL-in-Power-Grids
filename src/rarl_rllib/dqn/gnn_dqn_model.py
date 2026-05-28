import typing
from typing import List, Tuple

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from ray.rllib import SampleBatch
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK
from rarl import BaselineGNN
from rarl_rllib.common import assert_graph_obs_space_and_get_x_dim


class GNNBaselineDQNModel(DQNTorchModel):
    """GNN baseline model compatible with DQN's Q-value / dueling heads."""

    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        action_space,
        num_outputs: int | None,
        model_config: ModelConfigDict,
        name: str,
        *,
        q_hiddens=(256,),
        dueling: bool = False,
        dueling_activation: str = "relu",
        num_atoms: int = 1,
        use_noisy: bool = False,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma0: float = 0.5,
        add_layer_norm: bool = False,
        **kwargs,
    ):
        gnn_cfg = kwargs["gnn"]
        gnn_out_dim = gnn_cfg["out_dim"]

        embedding_space = Box(-np.inf, np.inf, shape=(gnn_out_dim,), dtype=np.float32)
        super().__init__(
            embedding_space, action_space, gnn_out_dim, model_config, name,
            q_hiddens=q_hiddens,
            dueling=dueling,
            dueling_activation=dueling_activation,
            num_atoms=num_atoms,
            use_noisy=use_noisy,
            v_min=v_min,
            v_max=v_max,
            sigma0=sigma0,
            add_layer_norm=add_layer_norm,
        )
        self.obs_space = obs_space
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(shift=0, space=obs_space)

        self.gnn = BaselineGNN(
            x_dim=assert_graph_obs_space_and_get_x_dim(obs_space),
            hidden_dim=gnn_cfg["hidden_dim"],
            x_out_dim=gnn_out_dim,
            num_layers=gnn_cfg["num_layers"],
            dropout_prob=gnn_cfg.get("dropout_prob", 0.0),
            residual=gnn_cfg.get("residual", True),
        )

    def forward(
        self,
        input_dict: typing.Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"]
        node_features_batch = obs[NODES]
        edge_index_batch = obs[EDGE_INDEX]
        edge_mask = obs[EDGE_MASK]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        valid_edges = edge_mask.bool()
        edge_index_batch = edge_index_batch.permute(1, 0, 2)
        edge_index_batch = edge_index_batch[:, valid_edges]

        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        edge_index_batch += offsets.unsqueeze(0)

        embedding = self.gnn(x=x, batch=batch, edge_index=edge_index_batch.to(dtype=torch.long))
        return embedding, state
