import typing
from typing import List, Tuple

import numpy as np
import torch
from gymnasium.spaces import Dict, Discrete, Box
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn, Tensor

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK
from rarl import BaselineGNN
from rarl_rllib.common import assert_graph_obs_space_and_get_x_dim


class GNNBaselineModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: Dict,
                 action_space: Discrete,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.gnn: BaselineGNN = BaselineGNN(
            x_dim=assert_graph_obs_space_and_get_x_dim(obs_space),
            hidden_dim=kwargs['gnn']['hidden_dim'],
            x_out_dim=kwargs['gnn']['out_dim'],
            num_layers=kwargs['gnn']['num_layers'],
            dropout_prob=kwargs['gnn'].get('dropout_prob', 0.0),
            residual=kwargs['gnn'].get('residual', True),
        )
        # Build downstream MLP head(s)
        # Create a Box space for the GNN output to pass to FCN
        gnn_output_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(kwargs['gnn']['out_dim'],),
            dtype=np.float32
        )
        self.mlp = FullyConnectedNetwork(
            obs_space=gnn_output_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name + "_fully_connected_network",
        )

    def forward(self, input_dict: typing.Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        node_features_batch = input_dict["obs"][NODES]  # [B, N, node_in_dim]
        edge_index_batch = input_dict["obs"][EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = input_dict["obs"][EDGE_MASK]  # [B, E_max]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Flatten nodes
        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Mask edges
        valid_edges = edge_mask.bool()
        edge_index_batch = edge_index_batch.permute(1, 0, 2)  # [2, B, E_max]
        edge_index_batch = edge_index_batch[:, valid_edges]  # [2, total_E]

        # Add per-graph node offsets
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        edge_index_batch += offsets.unsqueeze(0)

        # GNN to produce graph-level representation [B, gnn_out_dim]
        gnn_out: Tensor = self.gnn(x=x, batch=batch, edge_index=edge_index_batch.to(dtype=torch.long))

        # Pass GNN output through FCN (which expects input_dict format)
        mlp_input_dict = {"obs": gnn_out}
        logits, _ = self.mlp(mlp_input_dict, state, seq_lens)
        return logits, []

    def value_function(self) -> Tensor:
        # RLlib expects shape [B]
        return self.mlp.value_function()
