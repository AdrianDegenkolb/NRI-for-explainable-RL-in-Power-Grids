"""Gumbel-Softmax: differentiable sampling from a categorical distribution."""

import time

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GumbelSoftmax(nn.Module):
    """
    Differentiable sampling from a categorical distribution using the
    Gumbel-Softmax / Concrete distribution trick.

    Temperature *tau* controls the sharpness of the samples:

    * High tau (e.g. 2.0) → soft, uniform-like samples — good for early training.
    * Low tau (e.g. 0.1) → near-discrete one-hot samples — good for evaluation.

    Set ``hard=True`` to obtain one-hot samples with straight-through gradients.

    :param tau: Initial temperature parameter (>0).
    :param eps: Clipping constant preventing log(0) in Gumbel noise sampling.
    """

    def __init__(self, tau: float = 1.0, eps: float = 1e-10):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self._timings: dict[str, float] = {}

    def _sample_gumbel(self, shape: torch.Size) -> Tensor:
        t0 = time.perf_counter()
        u = torch.rand(shape).float()
        result = -torch.log(-torch.log(u + self.eps) + self.eps)
        self._timings["gumbel_noise_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def forward(self, logits: Tensor, hard: bool = False) -> Tensor:
        """
        Draw a (optionally hard) Gumbel-Softmax sample.

        :param logits: Unnormalized log-probabilities [..., K].
        :param hard: Return one-hot samples with straight-through gradients.
        :return: Soft (or hard) categorical samples [..., K].
        """
        t0 = time.perf_counter()
        noise = self._sample_gumbel(logits.size()).to(device=logits.device)
        y = F.softmax((logits + noise) / self.tau, dim=-1)

        if hard:
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y

        self._timings["gumbel_softmax_ms"] = (time.perf_counter() - t0) * 1000
        return y
