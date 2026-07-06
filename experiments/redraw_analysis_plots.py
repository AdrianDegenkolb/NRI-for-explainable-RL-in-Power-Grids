"""
Redraw all post-training analysis figures from saved .npy data.

Usage:
    PYTHONPATH=$(pwd)/src python experiments/redraw_analysis_plots.py \
        --data-dir results/experiments/2026_07_06_IEEE36/rappo/trial_xxx/analysis \
        --env-name l2rpn_ieee_neurips_2020_track1_small_test
"""
import argparse
import logging
from pathlib import Path

from analysis.post_training.analyzers.congestion import redraw_plots as redraw_congestion
from analysis.post_training.analyzers.actions import redraw_plots as redraw_actions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _repaint_hypothesis(label: str, verifier_cls, out_dir: Path, subdir: str) -> None:
    """Instantiate verifier pointed at out_dir and call repaint() if data exists."""
    probe = out_dir / subdir
    if not probe.exists():
        logger.warning("No %s/ found in %s — skipping %s repaint.", subdir, out_dir, label)
        return
    logger.info("Repainting %s plots from %s", label, probe)
    verifier_cls(out_dir=out_dir).repaint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Top-level analysis output directory (contains congestion/, actions/, h1/, h2/, h3/).")
    parser.add_argument("--env-name", type=str, default=None,
                        help="Grid2Op env name for grid layout. Falls back to env_name.txt in each sub-dir.")
    args = parser.parse_args()

    data_dir = args.data_dir
    env_name = args.env_name

    congestion_dir = data_dir / "congestion"
    if congestion_dir.exists():
        logger.info("Redrawing congestion plots from %s", congestion_dir)
        redraw_congestion(congestion_dir, env_name)
    else:
        logger.warning("No congestion/ directory found, skipping.")

    actions_dir = data_dir / "actions"
    if actions_dir.exists():
        logger.info("Redrawing action plots from %s", actions_dir)
        redraw_actions(actions_dir, env_name)
    else:
        logger.warning("No actions/ directory found, skipping.")

    # H1 / H2 / H3 — lazy-import so the script works even without encoder results
    h1_dir = data_dir / "h1"
    if h1_dir.exists():
        from analysis.analyze_latent_graphs.hypo1_electrical_coupling import Hypothesis1verifier
        # H1 writes into out_dir/ptdf_coupling/ by default; repaint reads from there
        _repaint_hypothesis("H1", Hypothesis1verifier, h1_dir, "ptdf_coupling")
    else:
        logger.info("No h1/ directory — skipping H1 repaint.")

    h2_dir = data_dir / "h2"
    if h2_dir.exists():
        from analysis.analyze_latent_graphs.hypo2_risk_coupling import Hypothesis2verifier
        # H2 writes into out_dir/risk_coupling/
        _repaint_hypothesis("H2", Hypothesis2verifier, h2_dir, "risk_coupling")
    else:
        logger.info("No h2/ directory — skipping H2 repaint.")

    h3_dir = data_dir / "h3"
    if h3_dir.exists():
        from analysis.analyze_latent_graphs.hypo3_action_effect_coupling import Hypothesis3verifier
        # H3 writes into out_dir/action_effect/
        _repaint_hypothesis("H3", Hypothesis3verifier, h3_dir, "action_effect")
    else:
        logger.info("No h3/ directory — skipping H3 repaint.")

    logger.info("Done. Figures written back to their respective subdirectories.")


if __name__ == "__main__":
    main()
