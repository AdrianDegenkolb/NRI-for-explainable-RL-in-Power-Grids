import os
from pathlib import Path
from typing import List

from experiments.utils import AgentSpec
from src.analysis.cross_validate_models.cross_validate import CrossValidateResult, cross_validate, \
    save_cross_validate_results, compute_reconfiguration_frequency_data, compute_cross_validation_data, \
    compute_failing_edges_data, repaint_failing_edges, repaint_reconfiguration_frequency, \
    repaint_cross_validation_results


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from itertools import product

    cwd = os.getcwd()

    model1 = AgentSpec(name="RAPPO", load_path=Path(cwd, "results/agents/CustomPPO_0_426b7_2026-01-19_10-28-48"), checkpoint_name="checkpoint_000020")
    model2 = AgentSpec(name="MLP", load_path=Path(cwd, "results/agents/CustomPPO_0_48ac9_2026-01-19_14-39-31_MLP"), checkpoint_name="checkpoint_000020")
    model3 = AgentSpec(name="GNN", load_path=Path(cwd, "results/agents/CustomPPO_0_4cbd2_2026-01-19_14-39-38_GNN"), checkpoint_name="checkpoint_000023")

    num_episodes = 50

    results_dir = Path("results/cross_validation")
    save_results_to = results_dir / "cross_validate_models.json"
    save_heatmap_to = results_dir / "cross_validate_models.svg"
    save_connectivity_to = results_dir / "failing_edges_connectivity.svg"
    save_rho_to = results_dir / "failing_edges_rho.svg"

    compute_data = False

    models = [model1, model2, model3]
    pairs = [(m1, m2) for m1, m2 in product(models, models) if m1.name != m2.name]

    if compute_data:
        results: List[CrossValidateResult] = []
        with ProcessPoolExecutor(max_workers=1) as ex:
            futures = [ex.submit(cross_validate, m1, m2, num_episodes) for (m1, m2) in pairs]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                print(f"Cross-validation between {result.failing_agent.name} and {result.backup_agent.name}: {result.additional_timesteps}")

        save_cross_validate_results(results, save_path=save_results_to)
        compute_reconfiguration_frequency_data(results, save_dir=results_dir)
        compute_cross_validation_data(results, save_dir=results_dir)
        compute_failing_edges_data(results, save_dir=results_dir)

    repaint_failing_edges(results_dir, save_path_connectivity=save_connectivity_to, save_path_rho=save_rho_to, show=True)
    repaint_failing_edges(results_dir, save_path_connectivity=save_connectivity_to.with_suffix(".png"), save_path_rho=save_rho_to.with_suffix(".png"), show=False)
    repaint_reconfiguration_frequency(results_dir, save_dir=results_dir, show=True)
    repaint_reconfiguration_frequency(results_dir, save_dir=results_dir, show=False)
    repaint_cross_validation_results(results_dir, save_path=save_heatmap_to.with_suffix(".png"), show=False)
    repaint_cross_validation_results(results_dir, save_path=save_heatmap_to.with_suffix(".svg"), show=True)


if __name__ == "__main__":
    import logging
    logging.getLogger("src.ra_agents.RAFeatureExtractor").setLevel(logging.ERROR)
    logging.getLogger("pandapower.convert_format").setLevel(logging.WARNING)
    main()