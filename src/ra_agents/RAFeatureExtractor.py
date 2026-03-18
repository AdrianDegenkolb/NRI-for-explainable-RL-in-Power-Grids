"""
This script contains the relations aware FeatureExtractor (RAFeatureExtractor) and Wrappers that integrate the normal
GNN model and RA-GNN model into RLlib's model catalog.
"""
import logging
from os import PathLike
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as f
from gymnasium.spaces import Discrete, Box
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn, Tensor
from torch_geometric.utils import to_dense_batch

from src.common.observation_space import GraphObservationSpace, NODES, EDGE_INDEX, EDGE_MASK
from src.nri.Sampling import GumbelSoftmax
from src.nri.utils import fully_connected_edge_index_per_batch, fully_connected_edge_index
from .Encoder import Encoder
from .RAGNN import RAGNN, BaselineGNN
from .graphormer.GraphormerEncoder import GraphormerNRIEncoder

logger = logging.getLogger(__name__)

class RAFeatureExtractor(nn.Module):
    """
    Combines an encoder with a relation-aware GNN (RA-GNN).

    The encoder predicts edge-type logits for each edge. These logits are used both
    to compute soft edge-type probabilities (for monitoring) and to sample discrete
    edge-type assignments via the Gumbel–Softmax trick. The sampled edge types are
    then used by the RAGNN to perform conditioned message passing.

    Pipeline:
        1. Encode graph structure to edge-type logits.
        2. Compute posterior probabilities p(z|x) via softmax.
        3. Sample edge-type assignments using Gumbel–Softmax.
        4. Pass node features and sampled edges into the GNN.

    :param x_dim (int): Input node feature dimension.
    :param hidden_dim (int): Hidden dimension shared by encoder and GNN.
    :param num_edge_types (int): Number of edge types (K).
    :param num_gnn_layers (int): Number of layers in both encoder and GNN.
    :param num_attention_heads_enc (int): Number of attention heads in the Graphormer encoder (ignored if use_graphormer=False).
    :param x_out_dim (int): Output node feature dimension.
    :param dropout_prob (float): Dropout probability used in both encoder and GNN.
    :param tau (float): Initial temperature for Gumbel-Softmax sampling.
    :param use_graphormer (bool): Whether to use Graphormer-based encoder. If False, uses NRI-based encoder.
    :param max_degree (Optional[int]): Maximum node degree for Graphormer encoder.
    :param max_path_distance (Optional[int]): Maximum path distance for Graphormer encoder.
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim_enc: int,
        num_layers_enc: int,
        num_attention_heads_enc: int,
        num_edge_types: int,
        hidden_dim_gnn: int,
        num_layers_gnn: int,
        x_out_dim: int,
        dropout_prob: float,
        tau: int = 1.0,
        residual: bool = False,
        use_graphormer: bool = False,
        max_degree: Optional[int] = None,
        max_path_distance: Optional[int] = None,
    ):
        super().__init__()
        if use_graphormer:
            assert max_degree is not None and max_path_distance is not None
            self.encoder = GraphormerNRIEncoder(
                x_dim=x_dim,
                hidden_dim=hidden_dim_enc,
                num_edge_types=num_edge_types,
                num_layers=num_layers_enc,
                num_attention_heads=num_attention_heads_enc,
                max_degree=max_degree,
                max_path_distance=max_path_distance,
            )
        else:
            self.encoder: Encoder = Encoder(
                x_dim=x_dim,
                hidden_dim=hidden_dim_enc,
                num_edge_types=num_edge_types,
                dropout_prob=dropout_prob,
            )
        self.gumbel_softmax = GumbelSoftmax(tau=tau)
        self.gnn: RAGNN = RAGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim_gnn,
            x_out_dim=x_out_dim,
            num_layers=num_layers_gnn,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
            residual=residual,
            skip_last=True,
        )
        self.x_out_dim = x_out_dim

    def set_tau(self, tau: float):
        """Update the temperature parameter of the Gumbel-Softmax sampler."""
        self.gumbel_softmax.tau = tau

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, powerline_edge_index: Optional[Tensor] = None, edge_set: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x (Tensor): Node features of shape [B*N, x_dim].
            powerline_edge_index (Tensor):  Graph connectivity in COO format [2, B*E'].
            batch (Tensor): Batch vector mapping each node to its graph [B*N]. (defaults to every node in the same batch)
            edge_set (Tensor): Which edges to infer probabilities for, Graph connectivity in COO format [2, B*E]. (defaults to fully connected per batch) THIS NEEDS TO HAVE A CONSISTENT NUMBER OF EDGES PER BATCH!

        Returns:
            tuple[Tensor, Tensor]:
                predictions (Tensor): Graph-level features [B, x_out_dim].
                p_z_given_x (Tensor): Soft edge-type posterior probabilities [B, E, K].
        """
        BxN, x_dim = x.shape[-2:]
        batch = batch if batch is not None else torch.zeros(BxN).to(x.device)
        edge_set = edge_set if edge_set is not None else fully_connected_edge_index_per_batch(batch, x.device)

        # get posterior
        encoder_logits: Tensor = self.encoder.forward(x=x, batch=batch, edge_set=edge_set, powerline_edge_index=powerline_edge_index) # [B*E, K]
        p_z_given_x: Tensor = f.softmax(encoder_logits, dim=-1) # [B*E, K]
        p_one_hot: Tensor = self.gumbel_softmax.forward(x=encoder_logits) # [B*E, K]

        # condition gnn on posterior and push x
        predictions: Tensor = self.gnn.forward(x=x, batch=batch, edge_index=edge_set, edge_type_posterior=p_one_hot)

        # transform posterior into batched format.
        edge_batch = batch[edge_set[0]] # edge is in the same batch as incident nodes
        batched_p_z_given_x, mask = to_dense_batch(p_z_given_x, edge_batch)
        assert torch.all(mask), "Different number of edges across batches is not allowed."

        return predictions, batched_p_z_given_x

class NRIBasedGNN(nn.Module):
    def __init__(
        self,
        edge_probs_path: PathLike,
        x_dim: int,
        hidden_dim_gnn: int,
        num_layers_gnn: int,
        x_out_dim: int,
        dropout_prob: float,
        residual: bool = False,
    ):
        super().__init__()

        N = 57  # TODO fix hack
        self.edge_index = fully_connected_edge_index(N)  # [2, E], dummy edge index
        self.edge_probs = Tensor(np.load(edge_probs_path))  # [E, K]

        self.gnn: RAGNN = RAGNN(
            x_dim=x_dim,
            hidden_dim=hidden_dim_gnn,
            x_out_dim=x_out_dim,
            num_layers=num_layers_gnn,
            dropout_prob=dropout_prob,
            residual=residual,
        )
        self.x_out_dim = x_out_dim

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Node features of shape [B*N, x_dim].
            batch (Tensor): Batch vector mapping each node to its graph [B*N]. (defaults to every node in the same batch)

        Returns:
            Tensor:
                predictions (Tensor): Graph-level features [B, x_out_dim].
        """
        BxN, x_dim = x.shape[-2:]
        batch = batch if batch is not None else torch.zeros(BxN).to(x.device)

        # get posterior
        B = batch.max().item() + 1
        edge_index = self.edge_index.unsqueeze(-1).repeat(1,1,B).view(2, -1)  # [2, B*E]
        K = self.edge_probs.size(-1)
        edge_probs = self.edge_probs.unsqueeze(-1).repeat(1, 1, B).view(-1, K)  # [2, B*E]

        # condition gnn on posterior and push x
        predictions: Tensor = self.gnn.forward(x=x, batch=batch, edge_index=edge_index, edge_type_posterior=edge_probs)

        return predictions


class RAFeatureExtractorSB3(BaseFeaturesExtractor):
    """
    Wraps the RAFeatureExtractor to be compatible with the sb3 API.
    """

    def __init__(
            self,
            observation_space: GraphObservationSpace,
            hidden_dim: int,
            out_dim: int,
            num_layers: int,
            num_edge_types: int,
            dropout_prob: float = 0.0,
            use_graphormer: bool = True,
            max_degree: Optional[int] = None,
            max_path_distance: Optional[int] = None,
    ):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim=out_dim)
        self.gnn_feature_extractor = RAFeatureExtractor(
            x_dim=observation_space.x_dim if hasattr(observation_space, 'x_dim') else 8, # TODO fix hack
            hidden_dim_gnn=hidden_dim,
            hidden_dim_enc=hidden_dim,
            x_out_dim=out_dim,
            num_layers_gnn=num_layers,
            num_layers_enc=num_layers,
            num_edge_types=num_edge_types,
            dropout_prob=dropout_prob,
            max_degree=max_degree,
            max_path_distance=max_path_distance,
            use_graphormer=use_graphormer,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]
        powerline_edge_index_batch = observations[EDGE_INDEX]
        edge_mask = observations[EDGE_MASK]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Flatten nodes
        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Mask edges
        valid_edges = edge_mask.bool()
        powerline_edge_index_batch = powerline_edge_index_batch.permute(1, 0, 2)  # [2, B, E_max]
        powerline_edge_index_batch = powerline_edge_index_batch[:, valid_edges]  # [2, total_E]

        # Add per-graph node offsets
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(valid_edges.sum(1))
        powerline_edge_index_batch += offsets.unsqueeze(0)

        return self.gnn_feature_extractor(x=x, batch=batch, powerline_edge_index=powerline_edge_index_batch.to(dtype=torch.long))


class BaselineFeatureExtractorSB3(BaseFeaturesExtractor):
    """
    Wraps the BaselineGNN to be compatible with the sb3 API.
    """

    def __init__(
            self,
            observation_space: GraphObservationSpace,
            hidden_dim: int,
            out_dim: int,
            num_layers: int = 2,
            dropout_prob: float = 0.0,
    ):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim=out_dim)
        self.gnn: BaselineGNN = BaselineGNN(
            x_dim=observation_space.x_dim if hasattr(observation_space, 'x_dim') else 8, # TODO fix hack
            hidden_dim=hidden_dim,
            x_out_dim=out_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            residual=True,
        )

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        node_features_batch = observations[NODES]  # [B, N, node_in_dim]
        edge_index_batch = observations[EDGE_INDEX]  # [B, 2, E_max]
        edge_mask = observations[EDGE_MASK]  # [B, E_max]

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

        return self.gnn(x=x, batch=batch, edge_index=edge_index_batch.to(dtype=torch.long))

class RLlibGNNModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: GraphObservationSpace,
                 action_space: Discrete,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.gnn: BaselineGNN = BaselineGNN(
            x_dim=obs_space.x_dim if hasattr(obs_space, 'x_dim') else 8, # TODO fix hack
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

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
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

class RLlibRAGNNModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: GraphObservationSpace,
                 action_space: Discrete,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.ragnn: RAFeatureExtractor = RAFeatureExtractor(
            x_dim=obs_space.x_dim if hasattr(obs_space, 'x_dim') else 8, # TODO fix hack
            use_graphormer=kwargs['encoder'].get('use_graphormer', False),
            hidden_dim_enc=kwargs['encoder']['hidden_dim'],
            num_layers_enc=kwargs['encoder']['num_layers'],
            num_attention_heads_enc=kwargs['encoder'].get('num_attention_heads', kwargs['encoder'].get('num_edge_types', 1)),
            num_edge_types=kwargs['encoder'].get('num_edge_types', 2),
            max_degree=kwargs['encoder'].get('max_degree', None),
            max_path_distance=kwargs['encoder'].get('max_path_distance', None),
            hidden_dim_gnn=kwargs['gnn']['hidden_dim'],
            x_out_dim=kwargs['gnn']['out_dim'],
            num_layers_gnn=kwargs['gnn']['num_layers'],
            dropout_prob=kwargs['gnn'].get('dropout_prob', 0.0),
            residual=kwargs['gnn'].get('residual', True),
            tau=0.5#kwargs['sampling'].get('tau_end', kwargs['sampling'].get('tau', 1.0))
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
        self.batched_p_z_given_x: Optional[Tensor] = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
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
        gnn_out, self.batched_p_z_given_x = self.ragnn(x=x, batch=batch, powerline_edge_index=edge_index_batch.to(dtype=torch.long))

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


class RLlibNRIGNNModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: GraphObservationSpace,
                 action_space: Discrete,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.nri_gnn: NRIBasedGNN = NRIBasedGNN(
            x_dim=obs_space.x_dim if hasattr(obs_space, 'x_dim') else 8, # TODO fix hack
            hidden_dim_gnn=kwargs['gnn']['hidden_dim'],
            x_out_dim=kwargs['gnn']['out_dim'],
            num_layers_gnn=kwargs['gnn']['num_layers'],
            dropout_prob=kwargs['gnn'].get('dropout_prob', 0.0),
            residual=kwargs['gnn'].get('residual', True),
            edge_probs_path=kwargs['gnn']['edge_probs_path'],
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
        self.batched_p_z_given_x: Optional[Tensor] = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        node_features_batch = input_dict["obs"][NODES]  # [B, N, node_in_dim]

        B, N, _ = node_features_batch.shape
        device = node_features_batch.device

        # Flatten nodes
        x = node_features_batch.reshape(B * N, -1)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # NRIGNN to produce graph-level representation [B, gnn_out_dim]
        gnn_out = self.nri_gnn(x=x, batch=batch)

        # Pass GNN output through FCN (which expects input_dict format)
        mlp_input_dict = {"obs": gnn_out}
        logits, _ = self.mlp(mlp_input_dict, state, seq_lens)
        return logits, []

    def value_function(self) -> Tensor:
        # RLlib expects shape [B]
        return self.mlp.value_function()


ModelCatalog.register_custom_model("gnn_model", RLlibGNNModel)
ModelCatalog.register_custom_model("ragnn_model", RLlibRAGNNModel)
ModelCatalog.register_custom_model("nrignn_model", RLlibNRIGNNModel)