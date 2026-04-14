"""
AnnealingCallback for RLlib: anneals beta (KL weight) and tau (Gumbel temperature).

The callback uses the framework-agnostic :class:`rarl.annealing.AnnealingState`
internally and propagates updated values to all RLlib workers after each
training iteration.

Config keys read from ``policy.config``::

    relation_awareness:
      beta_start: 0.0
      beta_end: 5.0
      beta_non_graph_edges_start: 0.0
      beta_non_graph_edges_end: 5.0
      beta_anneal_timesteps: null    # defaults to total_timesteps
      tau_anneal_timesteps: null     # defaults to total_timesteps

    model:
      custom_model_config:
        sampling:
          tau_start: 2.0
          tau_end: 0.5

Designed to be composed with other RLlib callbacks::

    from ray.rllib.algorithms.callbacks import make_multi_callbacks
    callbacks = make_multi_callbacks([AnnealingCallback, MyOtherCallback])
"""

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from src.rarl.annealing import AnnealingState

_POLICY_ID = "default_policy"


def _get_policy(algorithm: Algorithm, policy_id: str = _POLICY_ID):
    """Try multiple policy IDs to handle different algorithm setups."""
    for pid in (policy_id, "reinforcement_learning_policy", "default_policy"):
        p = algorithm.get_policy(pid)
        if p is not None:
            return p
    return None


class AnnealingCallback(DefaultCallbacks):
    """
    Anneals ``current_beta``, ``current_beta_non_graph``, and ``current_tau``
    on all workers using a three-phase cosine schedule.

    See :func:`rarl.annealing.cosine_decay_schedule` for the schedule details.
    """

    _state: AnnealingState | None = None

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        policy = _get_policy(algorithm)
        if policy is None or not hasattr(policy, "current_beta"):
            return

        ra_cfg = policy.config.get("relation_awareness", {})
        samp_cfg = policy.config["model"]["custom_model_config"].get("sampling", ra_cfg)
        total = algorithm.config.get("total_timesteps", 1_000_000)

        self._state = AnnealingState(
            beta_start=ra_cfg.get("beta_start", 0.0),
            beta_end=ra_cfg.get("beta_end", ra_cfg.get("beta", 1.0)),
            beta_non_graph_start=ra_cfg.get("beta_non_graph_edges_start", 0.0),
            beta_non_graph_end=ra_cfg.get("beta_non_graph_edges_end", ra_cfg.get("beta", 1.0)),
            tau_start=samp_cfg.get("tau_start", 2.0),
            tau_end=samp_cfg.get("tau_end", samp_cfg.get("temperature", 1.0)),
            total_steps=total,
            beta_anneal_steps=ra_cfg.get("beta_anneal_timesteps", total),
            tau_anneal_steps=ra_cfg.get("tau_anneal_timesteps", total),
        )

        # Set initial values on all workers
        self._sync(algorithm, step=0)

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        if self._state is None:
            return

        current_step = result.get("timesteps_total", 0)
        self._state.step(current_step)
        self._sync(algorithm, step=current_step)

    def _sync(self, algorithm: Algorithm, step: int) -> None:
        """Push current annealing values to local + remote workers."""
        if self._state is None:
            return

        beta = self._state.beta
        beta_ng = self._state.beta_non_graph
        tau = self._state.tau

        def _update(worker):
            p = worker.policy_map.get("default_policy") or \
                worker.policy_map.get("reinforcement_learning_policy")
            if p is None or not hasattr(p, "current_beta"):
                return
            p.current_beta_graph = beta
            p.current_beta_non_graph = beta_ng
            p.current_tau = tau
            if hasattr(p, "model") and hasattr(p.model, "set_tau"):
                p.model.set_tau(tau)

        algorithm.workers.local_worker().call(_update)
        algorithm.workers.foreach_worker(_update)
