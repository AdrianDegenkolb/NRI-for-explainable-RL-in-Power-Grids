from pathlib import Path

from core.evaluate import evaluate_rllib_checkpoint
from core.loading import AgentSpec
from core.constants import RL_POLICY


def main():
    model1 = AgentSpec(name="RAPPO",
                       load_path=Path("results/agents/CustomPPO_0_426b7_2026-01-19_10-28-48"),
                       checkpoint_name="checkpoint_000020")
    model2 = AgentSpec(name="MLP",
                       load_path=Path("results/agents/CustomPPO_0_48ac9_2026-01-19_14-39-31_MLP"),
                       checkpoint_name="checkpoint_000020")
    model3 = AgentSpec(name="GNN",
                       load_path=Path("results/agents/CustomPPO_0_4cbd2_2026-01-19_14-39-38_GNN"),
                       checkpoint_name="checkpoint_000023")
    model4 = AgentSpec(name="RAPPO_optuna",
                       load_path=Path("results/agents/CustomPPO_3087133_5b7242a3_2026-02-03_14-17-38/"),
                       checkpoint_name="checkpoint_000022")

    num_episodes = 50
    for dataset in ["l2rpn_case14_sandbox_val_local", "l2rpn_case14_sandbox_test_local"]:
        print(f"Evaluating on dataset: {dataset}")
        eval_env_name = dataset
        env_suffix = "_".join(eval_env_name.split('_')[-2:])

        for model in [model2]:
            print(f"\nEvaluating model: {model.name}")
            model_id = str(model.load_path).split('/')[-1]
            save_path = Path(f"results/evaluations/{model.name}/{model_id}/{env_suffix}")
            try:
                print(f"Result will we saved under: {save_path}")
                evaluate_rllib_checkpoint(
                    checkpoint_path=model.load_path,
                    policy_name=RL_POLICY,
                    checkpoint_name=model.checkpoint_name,
                    env_name_override=eval_env_name,
                    num_episodes=num_episodes,
                    save_to_path=save_path,
                    visualize=False
                )
            except Exception as e:
                print(f"Error evaluating model {model.name} on dataset {dataset}: {e}")

if __name__ == "__main__":
    main()