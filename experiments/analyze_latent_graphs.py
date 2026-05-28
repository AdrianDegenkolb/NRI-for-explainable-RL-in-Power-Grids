import logging
from pathlib import Path

from analysis.analyze_latent_graphs.agent_analysis_framework import LatentGraphAnalysisAgent
from analysis.analyze_latent_graphs.hypo1_electrical_coupling import Hypothesis1verifier
from analysis.analyze_latent_graphs.hypo2_risk_coupling import Hypothesis2verifier
from analysis.analyze_latent_graphs.hypo3_action_effect_coupling import Hypothesis3verifier
from experiments.utils import load_agent_from_spec, AgentSpec

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    analyzer_to_run = [
        Hypothesis1verifier(),
        Hypothesis2verifier(),
        Hypothesis3verifier()
    ]
    agent_spec = AgentSpec(
        name="RAPPO",
        load_path=Path("/home/adrian/Dev/NRI-for-explainable-RL-in-Power-Grids/results/2026_05_22_IEEE14/rappo/CustomPPO_RARL_4778599_edd16_2026-05-22_22-00-49"),
        checkpoint_name="checkpoint_000000",
    )
    env_name = "l2rpn_case14_sandbox_test"
    compute_data = True
    num_episodes = 50

    if compute_data:
        agent, env, gym_env = load_agent_from_spec(agent_spec=agent_spec, env_name=env_name)
        analysis_agent = LatentGraphAnalysisAgent(agent, gym_env, analyzer_to_run)
        logger.info("Agent loaded! Starting episodes...\n")
        analysis_agent.analyze(num_episodes=num_episodes)

    for analyzer in analyzer_to_run:
        analyzer.print_summary_from_saved()

    for analyzer in analyzer_to_run:
        analyzer.repaint()
