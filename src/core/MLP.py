import numpy as np
from torch import nn, Tensor
import torch.nn.functional as f


class MLP(nn.Module):
    """
    Two-layer fully-connected ELU net with batch norm.
    This is a refactored version of the MLP used by Kipf et al. (https://github.com/ethanfetaya/NRI/blob/master/modules.py).
    """

    def __init__(self, input_features: int, hidden_dim: int, output_features: int, dropout_prob: float = 0., do_batch_norm: bool = True):
        """
        Creates a two layer fully-connected ELU net with batch norm.
        L1 -> Dropout -> L2 -> Batch norm

        :param input_features: The number of input features.
        :param hidden_dim: The number of features in the hidden layer.
        :param output_features: The number of output features.
        :param dropout_prob: The probability of dropout (neuron inactivity).
        :param do_batch_norm: Whether to apply batch normalization.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_features)
        self.bn = nn.BatchNorm1d(output_features)
        self.dropout_prob = dropout_prob
        self.do_batch_norm = do_batch_norm

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _batch_norm(self, inputs: Tensor) -> Tensor:
        """
        Applies batch norm to the input tensor. More specifically every dimension except the last dimension is considered
        to be part of the batching.
        :param inputs: the input tensor
        :return: the output tensor of same shape as inputs with normalized data
        """
        batch_dimensions = inputs.shape[:-1] # every dimension except the last feature dimension is part of the batch
        flattened_batch_size = int(np.prod(batch_dimensions)) # if we flatten these batch dimension they have this size
        x = inputs.view(flattened_batch_size, -1) # flatten the tensor
        x = self.bn(x) # apply batch norm
        return x.view(inputs.shape) # shape it back

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forwards the input through this module. The input can have multiple batch dimensions.
        :param inputs: the input tensor [..., input_features]
        :return: the output tensor [..., output_features]
        """
        x = f.elu(self.fc1(inputs))
        x = f.dropout(x, self.dropout_prob, training=self.training)
        x = f.elu(self.fc2(x))
        return self._batch_norm(x) if self.do_batch_norm else x
