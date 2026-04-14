import logging
import tempfile
from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch

LOGS_PATH = Path("results/logs")
MODELS_PATH = Path("results/models")
NRI_DATASETS_PATH = Path("results/nri_datasets")
EVAL_PATH = Path("results/evaluations")
EDGE_PROBS_PATH = Path("results/edge_probs")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

_testing = False

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_experiment_name(experiment_name: Optional[str]):
    global LOGS_PATH, MODELS_PATH, NRI_DATASETS_PATH, EVAL_PATH, EDGE_PROBS_PATH
    if experiment_name is not None and not _testing:
        LOGS_PATH = Path("results/experiments", experiment_name, "logs")
        MODELS_PATH = Path("results/experiments", experiment_name, "models")
        NRI_DATASETS_PATH = Path("results/experiments", experiment_name, "nri_datasets")
        EVAL_PATH = Path("results/experiments", experiment_name, "evaluations")
        EDGE_PROBS_PATH = Path("results/experiments", experiment_name, "edge_probs")


def enable_test_mode():
    global _testing
    _testing = True
    global LOGS_PATH, MODELS_PATH, NRI_DATASETS_PATH, EVAL_PATH, EDGE_PROBS_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        LOGS_PATH = Path(tmpdir, "logs")
        MODELS_PATH = Path(tmpdir, "models")
        NRI_DATASETS_PATH = Path(tmpdir, "nri_datasets")
        EVAL_PATH = Path(tmpdir, "evaluations")
        EDGE_PROBS_PATH = Path(tmpdir, "edge_probs")