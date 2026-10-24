"""
Training curves of an RLlib/Tune run: selected TensorBoard scalars of a trial as one table with a row
per training iteration, e.g. for the Senior notebook (experiments/senior_results.ipynb).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

LEARNER = "ray/tune/info/learner/reinforcement_learning_policy/learner_stats/"
# Label -> tag of the scalars the Senior notebook plots.
SENIOR_TAGS = {
    "vf_loss": LEARNER + "vf_loss",
    "vf_explained_var": LEARNER + "vf_explained_var",
    "kl": LEARNER + "kl",
    "entropy": LEARNER + "entropy",
    "policy_loss": LEARNER + "policy_loss",
    # Mean last grid2op step of the episodes: training (sampled actions) and val env (greedy, every few iterations).
    "train_survival": "ray/tune/custom_metrics/grid2op_end_mean",
    "eval_survival": "ray/tune/evaluation/custom_metrics/grid2op_end_mean",
}


def load_scalars(trial_dir: Path, tags: dict[str, str]) -> pd.DataFrame:
    """Scalars of every event file in a Tune trial directory, one row per logged step.

    Args:
        trial_dir: Tune trial directory (contains `events.out.tfevents.*`).
        tags: Column label -> TensorBoard tag; tags the run did not log are left out.

    Returns:
        Columns `iteration` (1-based training iteration), `timesteps` (Tune's step: agent steps
        sampled so far) and one per found tag; NaN where a tag was not logged in an iteration
        (e.g. evaluation, which runs every few iterations). Later event files win on equal steps.
    """
    columns: dict[str, pd.Series] = {}
    for path in sorted(trial_dir.glob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={SCALARS: 0})
        events.Reload()
        found = set(events.Tags()[SCALARS])
        for label, tag in tags.items():
            if tag not in found:
                continue
            series = pd.Series({e.step: e.value for e in events.Scalars(tag)}, dtype=float)
            columns[label] = series if label not in columns else series.combine_first(columns[label])
    table = pd.DataFrame(columns).sort_index()
    table.index.name = "timesteps"
    table = table.reset_index()
    table.insert(0, "iteration", range(1, len(table) + 1))
    return table
