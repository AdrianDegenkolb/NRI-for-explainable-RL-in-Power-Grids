import logging
import time
import typing
from typing import Optional, List, Tuple

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn, Tensor

logger = logging.getLogger(__name__)

from grid2op_env.observation_converter import NODES, EDGE_INDEX, EDGE_MASK
from rarl import RAFeatureExtractor
from rarl_rllib.model import RARLModel
from rarl_rllib.common import assert_graph_obs_space_and_get_x_dim


class RAActorCriticModel(TorchModelV2, RARLModel):
    """
    Default Actor Critic model with RA encoder sampling and message passing

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

        # Compute top-K budget (0 = disabled)
        sparse_cfg = cfg.get("sparsification", {})
        top_k_mult = sparse_cfg.get("top_k_multiplier", 0)
        top_k_budget = 0
        if top_k_mult > 0:
            n_powerlines = sparse_cfg.get("n_powerlines_directed", 0)
            if n_powerlines > 0:
                temperature = sparse_cfg.get("temperature", 0.5)
                top_k_budget = int(top_k_mult * (1 + temperature) * n_powerlines)
                logger.info(
                    "Top-K sparsification enabled: multiplier=%s temperature=%s "
                    "n_powerlines_directed=%s → K_budget=%d",
                    top_k_mult, temperature, n_powerlines, top_k_budget,
                )
            else:
                logger.warning(
                    "top_k_multiplier=%s but n_powerlines_directed not set; "
                    "sparsification disabled. Add n_lines to env config.",
                    top_k_mult,
                )

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
            top_k_budget=top_k_budget,
            conv_type=gnn_cfg.get("conv_type", "gcn"),
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
        self._timings: dict[str, float] = {}

    def forward(self, input_dict: typing.Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[
        TensorType, List[TensorType]]:
        node_features_batch = input_dict["obs"][NODES]  # [B, N, node_in_dim]
        edge_index_batch = input_dict["obs"][EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = input_dict["obs"][EDGE_MASK]  # [B, E_max]

        # --- Edge preprocessing ---
        t0 = time.perf_counter()
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
        self._timings["edge_prep_ms"] = (time.perf_counter() - t0) * 1000

        # RAGNN to produce graph-level representation [B, gnn_out_dim]
        gnn_out, self.batched_p_z_given_x = self.ragnn(
            x=x,
            batch=batch,
            powerline_edge_index=edge_index_batch.to(dtype=torch.long)
        )

        # Pass GNN output through FCN (which expects input_dict format)
        t0 = time.perf_counter()
        mlp_input_dict = {"obs": gnn_out}
        logits, _ = self.mlp(mlp_input_dict, state, seq_lens)
        self._timings["fcn_ms"] = (time.perf_counter() - t0) * 1000

        # Collect all sub-module timings
        self._timings.update(self.ragnn._timings)
        return logits, []

    def get_posterior(self) -> Tensor:
        assert self.batched_p_z_given_x is not None, "Posterior not computed yet."
        return self.batched_p_z_given_x

    def set_tau(self, tau: float):
        self.ragnn.set_tau(tau)

    def value_function(self) -> Tensor:
        # RLlib expects shape [B]
        return self.mlp.value_function()
