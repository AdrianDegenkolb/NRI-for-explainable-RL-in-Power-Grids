"""
Framework-agnostic annealing schedule used to control:
  - beta  : weight of the KL regularization term
  - tau   : Gumbel-Softmax temperature

Both quantities are annealed with the same three-phase cosine schedule:

  Phase 1 (0 – 40 % of training):   hold at start value
  Phase 2 (40 – 80 % of training):  cosine decay from start to end value
  Phase 3 (80 – 100 % of training): hold at end value
"""

import math


def cosine_decay_schedule(
    current_step: int,
    total_steps: int,
    start_val: float,
    end_val: float,
) -> float:
    """
    Three-phase cosine annealing schedule.

    :param current_step: Current training step (e.g. total environment steps).
    :param total_steps: Total number of steps over which to anneal.
    :param start_val: Value at the beginning of training.
    :param end_val: Value at the end of training.
    :return: Annealed value at *current_step*.
    """
    if total_steps <= 0:
        return end_val

    progress = current_step / total_steps

    if progress < 0.4:
        return start_val

    if progress > 0.8:
        return end_val

    # Map [0.4, 0.8] → [0, 1]
    decay_progress = (progress - 0.4) / 0.4
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return end_val + (start_val - end_val) * cosine_factor


class AnnealingState:
    """
    Stateful wrapper around :func:`cosine_decay_schedule`.

    Tracks the current values of beta (graph), beta (non-graph), and tau
    so that framework-specific callbacks only need to call :meth:`step`.

    Example::

        state = AnnealingState(
            beta_start=0.0, beta_end=5.0,
            beta_non_graph_start=0.0, beta_non_graph_end=5.0,
            tau_start=2.0, tau_end=0.5,
            total_steps=1_000_000,
        )
        # Inside your training loop / callback:
        state.step(current_timesteps)
        model.set_tau(state.tau)
        loss = ppo_loss + state.beta * kl_graph + state.beta_non_graph * kl_latent
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_non_graph_start: float,
        beta_non_graph_end: float,
        tau_start: float,
        tau_end: float,
        total_steps: int,
        beta_anneal_steps: int | None = None,
        tau_anneal_steps: int | None = None,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_non_graph_start = beta_non_graph_start
        self.beta_non_graph_end = beta_non_graph_end
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.beta_anneal_steps = beta_anneal_steps if beta_anneal_steps is not None else total_steps
        self.tau_anneal_steps = tau_anneal_steps if tau_anneal_steps is not None else total_steps

        # Initialize at start values
        self.beta: float = beta_start
        self.beta_non_graph: float = beta_non_graph_start
        self.tau: float = tau_start

    def step(self, current_step: int) -> None:
        """Update all annealed values given the current training step."""
        self.beta = cosine_decay_schedule(
            current_step, self.beta_anneal_steps, self.beta_start, self.beta_end
        )
        self.beta_non_graph = cosine_decay_schedule(
            current_step, self.beta_anneal_steps,
            self.beta_non_graph_start, self.beta_non_graph_end,
        )
        self.tau = cosine_decay_schedule(
            current_step, self.tau_anneal_steps, self.tau_start, self.tau_end
        )
