"""
rarl – Relation-Aware Reinforcement Learning (core package).

Pure PyTorch / PyG implementation. No RL framework dependencies.

Sub-modules
-----------
rarl.nn              – Neural network modules (encoder, GNN, feature extractor)
rarl.nn.graphormer   – Optional Graphormer-based encoder
rarl.graph           – Graph utility functions
rarl.prior           – Prior distribution construction
rarl.annealing       – Annealing schedule (framework-agnostic)
rarl.loss            – KL-divergence regularisation loss
"""

from .graph import fully_connected_edge_index, fully_connected_edge_index_per_batch
from .prior import get_prior_tensor
from .annealing import cosine_decay_schedule, AnnealingState
from .loss import compute_ra_kl_loss
from .nn import *
