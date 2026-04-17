import logging

from experiments import LatentGraphAnalysisAgent
from experiments import Hypothesis2verifier
from experiments import Hypothesis3verifier
from experiments.utils import load_agent_from_spec, AgentSpec

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    analyzer_to_run = [
        #Hypothesis1verifier(),
        Hypothesis2verifier(),
        Hypothesis3verifier()
    ]
    agent_spec = AgentSpec(
        name="RAPPO",
        load_path="/home/adrian/Schreibtisch/1901/1901_rappo_with_anneal_different_betas/CustomPPO_0_426b7_2026-01-19_10-28-48",
        checkpoint_name="checkpoint_000020",
    )
    env_name = "l2rpn_case14_sandbox_test"
    compute_data = False
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
