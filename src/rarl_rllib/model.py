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
import abc

from torch import Tensor, nn


class RARLModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def get_posterior(self) -> Tensor:
        """
        Returns the posterior distribution p(z|x) from the most recent forward pass. Shape: [B, E, K]
        Note that a forward call has to be performed first before this method can return anything and thus that calling
        this method does not cause an extra forward pass through the network.
        :return: Posterior distribution tensor of shape [BATCH, NUM_EDGES, NUM_EDGE_TYPES].
        """


    @abc.abstractmethod
    def set_tau(self, tau: float):
        """
        Sets the current sampling parameter tau
        :param tau: the new tau
        """
