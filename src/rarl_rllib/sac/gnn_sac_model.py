import typing
from typing import Optional

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from ray.rllib import SampleBatch
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK, EDGE_TYPE, EDGES
from rarl import BaselineGNN
from core.utils import getl
from rarl_rllib.common import assert_graph_obs_space_and_get_x_dim


class GNNBaselineSACModel(SACTorchModel):
    """
    SAC model that uses RAFeatureExtractor (encoder + RAGNN) as the shared
    backbone for both the actor and Q-networks.

    Architecture
    ------------
    Graph obs → RAFeatureExtractor (encoder → RAGNN) → embedding [B, gnn_out_dim]
    embedding → action_model (inherited) → action distribution params
    embedding → q_net      (inherited) → Q-values

    The encoder runs once in forward(), returning an embedding that replaces
    the raw observation for all downstream SAC heads. The posterior p(z|x) is
    cached and retrieved via get_posterior() for KL loss computation.

    The SAC actor/Q heads (action_model, q_net, twin_q_net) are built by the
    parent SACTorchModel on the embedding space (Box [gnn_out_dim]), so they
    receive correctly-sized input from forward().

    Configuration (under custom_model_config)
    -----------------------------------------
    Same encoder/gnn/sampling schema as RARLModel.
    """

    def __init__(
            self,
            obs_space: gymnasium.spaces.Dict,
            action_space,
            num_outputs: Optional[int],
            model_config: ModelConfigDict,
            name: str,
            policy_model_config: Optional[dict] = None,
            q_model_config: Optional[dict] = None,
            twin_q: bool = False,
            initial_alpha: float = 1.0,
            target_entropy: Optional[float] = None,
            **kwargs,
    ):
        cfg = kwargs  # encoder, gnn, sampling from custom_model_config
        gnn_cfg = cfg.get("gnn", {})
        gnn_out_dim = getl(gnn_cfg, "out_dim", 64)

        # Build SAC actor/Q heads sized for the GNN embedding, not the raw graph obs.
        embedding_space = Box(-np.inf, np.inf, shape=(gnn_out_dim,), dtype=np.float32)
        super().__init__(
            embedding_space, action_space, num_outputs, model_config, name,
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
        )
        self.obs_space = obs_space
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(shift=0, space=obs_space)

        self.gnn = BaselineGNN(
            x_dim=assert_graph_obs_space_and_get_x_dim(obs_space),
            hidden_dim=getl(gnn_cfg, "hidden_dim", 64),
            x_out_dim=gnn_out_dim,
            num_layers=getl(gnn_cfg, "num_layers", 3),
            dropout_prob=getl(gnn_cfg, "dropout_prob", 0.0),
            residual=getl(gnn_cfg, "residual", True),
            num_edge_types=int(obs_space[EDGE_TYPE].high.flat[0]) + 1 if EDGE_TYPE in obs_space.spaces else 1,
            edge_dim=obs_space[EDGES].shape[-1] if EDGES in obs_space.spaces else None,
            conv_type=getl(gnn_cfg, "conv_type", "gcn"),
            num_heads=getl(gnn_cfg, "num_heads", 4),
        )

    def forward(
        self,
        input_dict: typing.Dict[str, TensorType],
        state: typing.List[TensorType],
        seq_lens: TensorType,
    ) -> typing.Tuple[TensorType, typing.List[TensorType]]:
        obs = input_dict["obs"]
        node_features_batch = obs[NODES]
        edge_index_batch = obs[EDGE_INDEX]
        edge_mask = obs[EDGE_MASK]
        edge_attr = obs.get(EDGES)       # [B, E_max, edge_dim] or None
        edge_types = obs.get(EDGE_TYPE)  # [B, E_max] or None

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        valid_edges = edge_mask.bool()
        edge_index_batch = edge_index_batch.permute(1, 0, 2)
        edge_index_batch = edge_index_batch[:, valid_edges]

        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        edge_index_batch += offsets.unsqueeze(0)

        if edge_attr is not None:
            edge_attr = edge_attr[valid_edges]   # [total_E, edge_dim]
        if edge_types is not None:
            edge_types = edge_types[valid_edges]  # [total_E]

        embedding = self.gnn(
            x=x,
            batch=batch,
            edge_index=edge_index_batch.to(dtype=torch.long),
            edge_attr=edge_attr,
            edge_types=edge_types,
        )
        return embedding, state
