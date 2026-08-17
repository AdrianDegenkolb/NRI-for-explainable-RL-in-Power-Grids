"""
Module to pretrain an Encoder to predict the prior distribution. Use with caution to prevent collapse of mutual information.
"""
import dataclasses
import time
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from grid2op.gym_compat import GymEnv
from numpy._typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from core.constants import set_seed
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, NODES, make_observation_converter
from grid2op_env.utils import make_g2op_env
from rarl import GraphormerNRIEncoder, fully_connected_edge_index, get_prior_tensor, compute_ra_kl_loss
from rarl.prior import get_priors


@dataclasses.dataclass
class PretrainingResults:
    losses_per_epoch: List[float]
    time_ms_per_epoch: List[float]


class _PretrainingDataset(Dataset):
    """Dataset of (node_features, powerline_edges, prior, graph_mask) tuples.

    Each sample carries its own prior and graph mask because the grid topology
    (and therefore the prior) can change between observations due to line disconnections.

    :param node_features: Per-observation node feature tensors, each [N, x_dim].
    :param powerline_edges: Per-observation powerline edge indices, each [2, E_i].
    :param priors: Per-observation prior distributions over the FC edge set, each [E_fc, K].
    :param graph_masks: Per-observation boolean masks identifying powerline edges, each [E_fc].
    """

    def __init__(
        self,
        node_features: List[Tensor],
        powerline_edges: List[Tensor],
        priors: List[Tensor],
        graph_masks: List[Tensor],
    ):
        self.node_features = node_features
        self.powerline_edges = powerline_edges
        self.priors = priors
        self.graph_masks = graph_masks

    def __len__(self) -> int:
        return len(self.node_features)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.node_features[idx], self.powerline_edges[idx], self.priors[idx], self.graph_masks[idx]


class Pretrainer:
    """
    Pretrains a GraphormerNRIEncoder to reproduce the prior edge-type distribution.

    The encoder is initialized to output a posterior close to the prior p(z|A)
    before RL training begins. This gives the downstream GNN agent a sensible
    starting structure rather than a near-uniform one.

    Use caution: extended pretraining collapses mutual information between node
    features and edge predictions (encoder learns to ignore input).

    :param env_config: Config dict accepted by ``make_g2op_env`` and
        ``make_observation_converter`` (must include ``env_name``, ``grid2op_kwargs``,
        ``observation_space``, etc.).
    :param prior_prob_for_graph_edge: Prior probability that a powerline edge is active.
    :param temperature: Controls expected number of discovered latent edges (see ``get_priors``).
    :param num_edge_types: Number of discrete edge types K.
    :param beta: KL weight for powerline (graph) edges, passed to ``compute_ra_kl_loss``.
    :param beta_non_graph: KL weight for latent edges, passed to ``compute_ra_kl_loss``.
    :param lr: Adam learning rate for the encoder.
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(
        self,
        env_config: dict[str, Any],
        prior_prob_for_graph_edge: float,
        temperature: float,
        num_edge_types: int,
        beta: float = 1.0,
        beta_non_graph: float = 1.0,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = False,
        seed: int = 42
    ):
        self._env_config = env_config
        self._prior_prob_for_graph_edge = prior_prob_for_graph_edge
        self._temperature = temperature
        self._num_edge_types = num_edge_types
        self._beta = beta
        self._beta_non_graph = beta_non_graph
        self._lr = lr
        self._device = torch.device(device)
        self.verbose = verbose
        self.seed = seed
        set_seed(seed)

    def _sample_observations(self, num_observations: int) -> List[Tuple[NDArray, NDArray]]:
        """
        Creates an environment and samples several states from this environment. Returns a list of these states as
        [(node_features1, edge_index1), ... (node_features_n, edge_index_n)]

        Observations are normalized online (running mean/variance updated per call to ``to_gym``).

        :param num_observations: The number (n) of observed states
        :return: List of ``(node_features [N, x_dim], edge_index [2, E])`` pairs
        """
        env_g2op = make_g2op_env(self._env_config)
        env_gym = GymEnv(env_g2op, with_forecast=True)
        obs_converter = make_observation_converter(env_gym, self._env_config)

        observations = []
        g2op_obs = env_g2op.reset(seed=self.seed)
        loop_range = tqdm(range(num_observations), desc="Sample observations") if self.verbose else range(num_observations)
        for _ in loop_range:
            gym_obs = obs_converter.to_gym(g2op_obs)
            node_features = gym_obs[NODES]                      # [N, x_dim]
            edge_index_padded = gym_obs[EDGE_INDEX]             # [2, max_E]
            edge_mask = gym_obs[EDGE_MASK]                      # [max_E] bool
            edge_index = edge_index_padded[:, edge_mask]        # [2, E_actual]
            observations.append((node_features, edge_index))

            action = env_g2op.action_space({})
            g2op_obs, _, done, _ = env_g2op.step(action)
            if done:
                g2op_obs = env_g2op.reset()

        env_g2op.close()
        return observations

    def _build_dataset(self, observations: List[Tuple[NDArray, NDArray]]) -> _PretrainingDataset:
        """
        Iterates over the observations, considers the current edge_indices and transforms them into prior distributions
        with existing methods. Then stores the results as dataset in the format [((node_features1, edge_index1), prior_distribution_1), ...]

        The prior is recomputed per observation since line disconnections change the active
        topology and therefore the graph-edge count used to derive the prior.

        :param observations: A list of sampled observations ``[(node_features [N, x_dim], edge_index [2, E]), ...]``
        :return: A dataset of ``(node_features, powerline_edges, prior, graph_mask)`` tuples
        """
        first_x, _ = observations[0]
        N = first_x.shape[0]
        all_edges = fully_connected_edge_index(N)   # [2, N*(N-1)], same for all observations

        node_features_list, powerline_edges_list, priors_list, graph_masks_list = [], [], [], []

        loop_range = tqdm(observations, desc="Build dataset") if self.verbose else observations
        for x, edge_index in loop_range:
            graph_edges = torch.from_numpy(edge_index).long()  # [2, E_graph]
            num_graph_edges = graph_edges.shape[1]
            num_non_graph_edges = all_edges.shape[1] - num_graph_edges

            g_prior, ng_prior = get_priors(
                prob_graph_edges_exist=self._prior_prob_for_graph_edge,
                num_graph_edges=num_graph_edges,
                num_non_graph_edges=num_non_graph_edges,
                temperature=self._temperature,
            )
            prior, graph_mask = get_prior_tensor(
                graph_edges=graph_edges,
                all_edges=all_edges,
                prior_for_graph_edges=g_prior,
                prior_for_non_graph_edges=ng_prior,
                num_edge_types=self._num_edge_types,
                return_mask=True,
            )  # prior: [E_fc, K], graph_mask: [E_fc]

            node_features_list.append(torch.from_numpy(x).float())
            powerline_edges_list.append(graph_edges)
            priors_list.append(prior)
            graph_masks_list.append(graph_mask)

        return _PretrainingDataset(
            node_features=node_features_list,
            powerline_edges=powerline_edges_list,
            priors=priors_list,
            graph_masks=graph_masks_list,
        )

    def fit(
        self,
        encoder: GraphormerNRIEncoder,
        dataset: _PretrainingDataset,
        num_epochs: int,
    ) -> PretrainingResults:
        """
        Fits the encoder to reproduce the prior distributions for every sample in the dataset. Uses the KL divergence just
        as in the loss computation

        Trains with batch_size=1 to accommodate variable powerline edge counts across observations.

        :param encoder: The encoder of the RAPPO module
        :param dataset: The dataset to train on
        :param num_epochs: The epochs
        :return: PretrainingResults with per-epoch mean loss and elapsed time in ms
        """
        encoder = encoder.to(self._device)
        encoder.train()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self._lr)

        # batch_size=1: powerline edge counts may differ between observations
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

        losses_per_epoch: List[float] = []
        times_per_epoch: List[float] = []

        loop_range = tqdm(range(num_epochs), f"Train {num_epochs} epochs") if self.verbose else range(num_epochs)
        for _ in loop_range:
            t_start = time.perf_counter()
            epoch_loss = 0.0

            for x, powerline_edge_index, prior, graph_mask in loader:
                x = x.to(self._device)                                         # [N, x_dim]
                powerline_edge_index = powerline_edge_index.to(self._device)   # [2, E']
                prior = prior.to(self._device)                                 # [E_fc, K]
                graph_mask = graph_mask.to(self._device)                       # [E_fc]

                optimizer.zero_grad()
                logits = encoder(x=x, powerline_edge_index=powerline_edge_index)  # [E_fc, K]
                posterior = F.softmax(logits, dim=-1)                              # [E_fc, K]

                # Add batch dim: compute_ra_kl_loss expects [B, E, K] and [B, E]
                kl_loss, _ = compute_ra_kl_loss(
                    posteriors=posterior.unsqueeze(0),
                    prior_tensor=prior.unsqueeze(0),
                    graph_edge_masks=graph_mask.unsqueeze(0),
                    beta=self._beta,
                    beta_non_graph=self._beta_non_graph,
                )
                kl_loss.backward()
                optimizer.step()
                epoch_loss += kl_loss.item()

            losses_per_epoch.append(epoch_loss / len(loader))
            times_per_epoch.append((time.perf_counter() - t_start) * 1000)

        return PretrainingResults(
            losses_per_epoch=losses_per_epoch,
            time_ms_per_epoch=times_per_epoch,
        )

    def run(self, encoder: GraphormerNRIEncoder, num_observations: int, num_epochs: int) -> PretrainingResults:
        """
        Pretrains the encoder to reconstruct the prior
        :param encoder: the encoder
        :param num_observations: how many observations should be contained in the dataset
        :param num_epochs: how many epochs should the training go on for
        :return: PretrainingResults
        """
        observations = self._sample_observations(num_observations=num_observations)
        dataset = self._build_dataset(observations)
        return self.fit(encoder, dataset, num_epochs)
