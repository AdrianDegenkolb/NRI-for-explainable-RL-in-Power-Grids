"""
Graph ablation experiment.

Evaluates trained agents with shuffled edge indices (same node features,
same degree, different connectivity) to measure how much the policy relies
on the graph structure vs. node features alone.

Results are saved to experiments/survival/obs_spaces/ablation/.
"""

import json
import sys
from pathlib import Path

import grid2op
import gymnasium as gym
import numpy as np
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend
from tqdm import tqdm

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.evaluate import evaluate_agent
from core.loading import AgentSpec, load_agent_from_spec
from grid2op_env.observation_converter import (
    ObservationConverter, T,
    EDGE_INDEX, EDGE_MASK, EDGES, EDGE_TYPE, EDGE_WEIGHTS,
)

RESULTS = ROOT / "experiments" / "survival" / "obs_spaces" / "ablation"
RESULTS.mkdir(exist_ok=True, parents=True)

METHODS = {
    "Default": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/default/CustomPPO*/",
    },
    "Elements": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/elements/CustomPPO*/",
    },
    "Elements + LODF": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/elements_lodf/CustomPPO*/",
    },
    "Heterogeneous": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/heterogenous/CustomPPO*/",
    },
    "LODF": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/lodf/CustomPPO*/",
    },
    "PTDF": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/ptdf/CustomPPO*/",
    },
    "Substation": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/substation/CustomPPO*/",
    },
    "Substation + PTDF": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/substation_ptdf/CustomPPO*/",
    },
    "Substation + Zbus": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/substation_zbus/CustomPPO*/",
    },
    "Zbus": {
        "glob": "results/2026_08_17_compare_graph_obs_spaces_IEEE14/zbus/CustomPPO*/",
    },
}


class GraphAblationWrapper(ObservationConverter):
    """Wraps a graph observation converter and shuffles edge connectivity.

    The node features and edge count are preserved; only the mapping from
    source to target nodes is shuffled. This isolates how much the policy
    depends on the graph structure vs. node features alone.

    Args:
        graph_observation_converter: the underlying converter to wrap
        seed: RNG seed for reproducible shuffling
    """

    def __init__(self, graph_observation_converter: ObservationConverter, seed: int = 42):
        self.converter = graph_observation_converter
        self.rng = np.random.default_rng(seed=seed)
        num_edges = graph_observation_converter.observation_space[EDGE_INDEX].shape[-1]
        self._perm = np.arange(num_edges)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.converter.observation_space

    def reset_obs(self) -> None:
        self.converter.reset_obs()
        num_edges = len(self._perm)
        self._perm = self.rng.permutation(num_edges)

    def to_gym(self, g2op_obs: BaseObservation) -> T:
        """Convert observation and apply a fixed per-episode edge permutation.

        The same permutation is held for the entire episode (drawn in reset_obs)
        so the rewired topology is consistent across timesteps.

        Args:
            g2op_obs: raw Grid2Op observation

        Returns:
            gym observation with permuted edge connectivity
        """
        obs = self.converter.to_gym(g2op_obs)
        obs[EDGE_INDEX] = obs[EDGE_INDEX][:, self._perm]
        obs[EDGE_MASK] = obs[EDGE_MASK][self._perm]
        for key in (EDGES, EDGE_TYPE, EDGE_WEIGHTS):
            if key in obs:
                obs[key] = obs[key][self._perm]
        return obs

    def normalize(self, gym_obs: T) -> T:
        return self.converter.normalize(gym_obs)

    def close(self) -> None:
        self.converter.close()


def best_checkpoint(path: Path, metric: str = "grid2op_end_mean") -> tuple[str, float]:
    """Return the checkpoint name with the highest value for `metric`.

    Args:
        path: path to checkpoint_results.json
        metric: metric key to maximise

    Returns:
        (checkpoint_name, metric_value) for the best checkpoint
    """
    data = json.loads(path.read_text())
    return max(
        ((name, stats[metric]) for name, stats in data.items()),
        key=lambda item: item[1],
    )


def evaluate_method(method: str, cfg: dict) -> None:
    """Evaluate the first seed of a method under graph ablation.

    Args:
        method: display name of the method
        cfg: config dict with 'glob' key
    """
    dirs = sorted(ROOT.glob(cfg["glob"]))
    if not dirs:
        print(f"No checkpoints found for {method}")
        return
    d = dirs[0]
    try:
        agent, env, gym_env = load_agent_from_spec(
            AgentSpec(
                name=method,
                load_path=d,
                checkpoint_name=best_checkpoint(d / "checkpoint_results.json")[0],
            )
        )
        if not env.name.endswith("_test"):
            env = grid2op.make(
                env.name.removesuffix("_val").removesuffix("_train") + "_test",
                backend=LightSimBackend()
            )

        gym_env.observation_converter = GraphAblationWrapper(
            gym_env.observation_converter, seed=42
        )
        evaluate_agent(
            agent=agent,
            env=env,
            path_results=RESULTS / f"{method}_0",
            num_episodes=50,
            max_episode_length=8064,
            verbose=False,
        )
    except Exception as e:
        print(f"Error while evaluating {d}:\n{e}")


def main() -> None:
    for method, cfg in tqdm(METHODS.items(), desc="Methods"):
        print(f"\n--- {method} ---")
        evaluate_method(method, cfg)


if __name__ == "__main__":
    main()
