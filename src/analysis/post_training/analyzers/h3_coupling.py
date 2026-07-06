from __future__ import annotations
from pathlib import Path

from analysis.analyze_latent_graphs.hypo3_action_effect_coupling import Hypothesis3verifier
from analysis.post_training.interfaces import EpisodeAnalyzer, StepContext


class H3ActionEffectAnalyzer(EpisodeAnalyzer):
    """
    Wraps Hypothesis3verifier as an EpisodeAnalyzer.
    The out_dir passed to save() is IGNORED; data is saved to self._verifier.outdir
    (which is set at construction time via the out_dir constructor parameter).
    """

    def __init__(self, out_dir: Path) -> None:
        self._verifier = Hypothesis3verifier(out_dir=out_dir)

    def on_episode_start(self, episode_id: str, max_steps: int) -> None:
        self._verifier.on_new_episode(episode_id)

    def on_step(self, ctx: StepContext) -> None:
        if ctx.posterior is None or not ctx.is_rl_step:
            return
        self._verifier.on_rl_step(
            posterior=ctx.posterior,
            prior=ctx.prior,
            powergrid_graph=None,
            observation=ctx.obs,
            environment=ctx.env,
            action=ctx.action,
        )

    def on_episode_end(self, steps_survived: int, max_steps: int) -> None:
        pass

    def save(self, out_dir: Path) -> None:
        self._verifier.on_evaluation_end()
