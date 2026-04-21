"""
RARLModel: TorchModelV2 wrapper around RAFeatureExtractor.

This model reads graph-structured observations (node features + edge index +
edge mask) and produces action logits + value estimates via:
  RAFeatureExtractor → graph embedding → RLlib FullyConnectedNetwork heads.

Observation space must expose three keys (configurable):
  nodes_key  : FloatTensor [B, N, node_dim]
  edge_index_key : LongTensor [B, 2, E_max]  (or [B, E_max, 2])
  edge_mask_key  : BoolTensor  [B, E_max]
"""
import typing
from typing import List, Optional, Tuple

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete, Dict
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import Tensor, nn

from src.grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK
from src.rarl import BaselineGNN
from src.rarl.nn import RAFeatureExtractor


class RARLModel(TorchModelV2, nn.Module):
    """
    RLlib TorchModelV2 that uses an encoder + RAGNN as the feature backbone.

    Architecture
    ------------
    RAFeatureExtractor (encoder → RAGNN) → RLlib FullyConnectedNetwork (policy logits + value heads).

    The encoder infers a discrete latent graph structure over the
    observation, and the RAGNN uses that structure to produce a graph-level
    embedding.  The posterior ``p(z | x)`` from the last forward pass is
    accessible via :meth:`get_posterior`.

    Observation space requirements
    -------------------------------
    ``obs_space`` must be a ``gymnasium.spaces.Dict`` with at minimum the keys
    ``NODES``, ``EDGE_INDEX``, and ``EDGE_MASK`` (imported from
    ``src.grid2op_env.observation_converter``).  It must also expose an
    ``x_dim`` attribute giving the per-node feature dimensionality.

    Configuration
    -------------
    All architecture hyperparameters are passed through
    ``model_config["custom_model_config"]`` (RLlib unpacks this into
    ``**kwargs``).  Schema::

        custom_model_config:
          encoder:
            hidden_dim: 64          # encoder hidden size
            num_layers: 2           # encoder depth
            num_edge_types: 2       # number of discrete edge-type categories
            num_attention_heads: 2  # multi-head attention heads in encoder
            max_degree: 8           # max node degree (Graphormer positional enc)
            max_path_distance: 55   # max shortest-path distance (positional enc)
          gnn:
            hidden_dim: 256         # RAGNN hidden size
            out_dim: 64             # graph-level embedding dimension
            num_layers: 3           # RAGNN depth
            dropout_prob: 0.0       # dropout (default 0)
            residual: true          # residual connections (default true)
          sampling:
            tau: 1.0                # initial Gumbel-Softmax temperature;
                                    # may also be keyed as tau_end for
                                    # compatibility with AnnealingCallback

    Public methods (beyond the TorchModelV2 interface)
    ---------------------------------------------------
    get_posterior() -> Tensor
        Returns ``p(z | x)`` of shape ``[B, E, num_edge_types]`` from the
        most recent forward pass.
    set_tau(tau: float)
        Updates the Gumbel-Softmax temperature in the underlying sampler.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Dict,
        action_space: Discrete,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = kwargs  # custom_model_config is unpacked into kwargs by RLlib

        enc_cfg = cfg["encoder"]
        gnn_cfg = cfg["gnn"]
        samp_cfg = cfg.get("sampling", {})

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
            x_out_dim=gnn_cfg["out_dim"],
            dropout_prob=gnn_cfg.get("dropout_prob", 0.0),
            residual=gnn_cfg.get("residual", True),
            tau=samp_cfg.get("tau_end", samp_cfg.get("tau", 1.0)),
        )

        gnn_out_space = Box(-np.inf, np.inf, shape=(gnn_cfg["out_dim"],), dtype=np.float32)
        self.mlp = FullyConnectedNetwork(
            obs_space=gnn_out_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name + "_fcn",
        )
        self.batched_p_z_given_x: Optional[Tensor] = None

    def forward(self, input_dict: typing.Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[
        TensorType, List[TensorType]]:
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

        # RAGNN to produce graph-level representation [B, gnn_out_dim]
        gnn_out, self.batched_p_z_given_x = self.ragnn(
            x=x,
            batch=batch,
            powerline_edge_index=edge_index_batch.to(dtype=torch.long)
        )

        # Pass GNN output through FCN (which expects input_dict format)
        mlp_input_dict = {"obs": gnn_out}
        logits, _ = self.mlp(mlp_input_dict, state, seq_lens)
        return logits, []

    def get_posterior(self) -> Tensor:
        """
        Returns the posterior distribution for the most recent forward pass.
        Note that a forward call has to be performed first before this method can return anything and thus that calling
        this method does not cause an extra forward pass through the network.
        :return: Posterior distribution tensor of shape [BATCH, NUM_EDGES, NUM_EDGE_TYPES].
        """
        assert self.batched_p_z_given_x is not None, "Posterior not computed yet."
        return self.batched_p_z_given_x

    def set_tau(self, tau: float):
        """Update the temperature parameter of the Gumbel-Softmax sampler."""
        self.ragnn.set_tau(tau)

    def value_function(self) -> Tensor:
        # RLlib expects shape [B]
        return self.mlp.value_function()


class RASACTorchModel(SACTorchModel):
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
        enc_cfg = cfg["encoder"]
        gnn_cfg = cfg["gnn"]
        samp_cfg = cfg.get("sampling", {})
        gnn_out_dim = gnn_cfg["out_dim"]

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
        """Returns p(z|x) from the most recent forward pass. Shape: [B, E, K]."""
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


def assert_graph_obs_space_and_get_x_dim(obs_space: spaces.Dict) -> int:
    """
    Checks that the given dict space is a graph obs space and returns the node feature dimension.
    :param obs_space: the observation space to check,
    :return: the node feature dimension (x_dim) if the checks pass
    :raise AssertionError: if the obs_space does not contain node features, edge index and edge mask subspaces
    """

    assert NODES in obs_space.spaces, f"obs_space must contain '{NODES}' key for node features"
    assert EDGE_INDEX in obs_space.spaces, f"obs_space must contain '{EDGE_INDEX}' key for edge indices"
    assert EDGE_MASK in obs_space.spaces, f"obs_space must contain '{EDGE_MASK}' key for edge masks"

    _, x_dim = obs_space[NODES].shape
    return x_dim


ModelCatalog.register_custom_model("rarl_model", RARLModel)
ModelCatalog.register_custom_model("gnn_model", GNNBaselineModel)
