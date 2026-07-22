import typing
from typing import Optional, List, Tuple

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box
from ray.rllib import SampleBatch
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import Tensor

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK
from rarl import RAFeatureExtractor
from rarl_rllib.model import RARLModel
from rarl_rllib.common import assert_graph_obs_space_and_get_x_dim


class RADQNTorchModel(DQNTorchModel, RARLModel):
    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        action_space,
        num_outputs: Optional[int],
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
        cfg = kwargs  # encoder, gnn, sampling from custom_model_config
        enc_cfg = cfg["encoder"]
        gnn_cfg = cfg["gnn"]
        samp_cfg = cfg.get("sampling", {})
        gnn_out_dim = gnn_cfg["out_dim"]

        # Build DQN Q-heads sized for the GNN embedding, not the raw graph obs.
        # DQNTorchModel uses num_outputs as the input dim for advantage/value
        # modules. Since forward() returns [B, gnn_out_dim], we must pass
        # gnn_out_dim here so the Q-head layers have the correct input size.
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
        # Restore the real obs space and view requirement.
        # super().__init__ stored embedding_space in both self.obs_space and
        # self.view_requirements[OBS], which would cause the policy to build
        # dummy batches with flat [B, gnn_out_dim] tensors instead of graph dicts.
        self.obs_space = obs_space
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(shift=0, space=obs_space)

        x_dim = assert_graph_obs_space_and_get_x_dim(obs_space)
        self.ragnn = RAFeatureExtractor(
            x_dim=x_dim,
            graph_max_degree=enc_cfg["max_degree"],
            graph_max_path_distance=enc_cfg["max_path_distance"],
            hidden_dim_enc=enc_cfg["hidden_dim"],
            num_layers_enc=enc_cfg["num_layers"],
            num_attention_heads_enc=enc_cfg.get("num_attention_heads", 2),
            num_edge_types=enc_cfg.get("num_edge_types", 2),
            hidden_dim_gnn=gnn_cfg["hidden_dim"],
            num_layers_gnn=gnn_cfg["num_layers"],
            x_out_dim=gnn_out_dim,
            dropout_prob=gnn_cfg.get("dropout_prob", 0.0),
            residual=gnn_cfg.get("residual", True),
            tau=samp_cfg.get("tau_end", samp_cfg.get("tau", 1.0)),
            conv_type=gnn_cfg.get("conv_type", "gcn"),
        )
        self.batched_p_z_given_x: Optional[Tensor] = None

    def forward(
        self,
        input_dict: typing.Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Run the RA encoder on graph obs and return the graph embedding.

        Returns embedding of shape [B, gnn_out_dim] as model_out, which is
        then consumed by the inherited get_action_model_outputs() and
        get_q_values() methods.
        """
        obs = input_dict["obs"]
        node_features_batch = obs[NODES]    # [B, N, node_in_dim]
        edge_index_batch = obs[EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = obs[EDGE_MASK]          # [B, E_max]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        valid_edges = edge_mask.bool()
        edge_index_batch = edge_index_batch.permute(1, 0, 2)  # [2, B, E_max]
        edge_index_batch = edge_index_batch[:, valid_edges]    # [2, total_E]

        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        edge_index_batch += offsets.unsqueeze(0)

        embedding, self.batched_p_z_given_x = self.ragnn(
            x=x,
            batch=batch,
            powerline_edge_index=edge_index_batch.to(dtype=torch.long),
        )
        return embedding, state  # [B, gnn_out_dim]

    def get_posterior(self) -> Tensor:
        assert self.batched_p_z_given_x is not None, "Posterior not computed yet."
        return self.batched_p_z_given_x

    def set_tau(self, tau: float) -> None:
        """Update the Gumbel-Softmax temperature in the encoder."""
        self.ragnn.set_tau(tau)

    def policy_variables(self):
        """Return actor-head-only parameters for the actor optimizer.

        The encoder (ragnn) is intentionally excluded here; it is updated by
        a dedicated encoder optimizer via encoder_variables().
        """
        return list(self.action_model.parameters())

    def encoder_variables(self):
        """Return encoder parameters for the dedicated encoder optimizer."""
        return list(self.ragnn.parameters())
