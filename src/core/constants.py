import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

LOGS_PATH = Path("results/logs")
MODELS_PATH = Path("results/models")
NRI_DATASETS_PATH = Path("results/nri_datasets")
EVAL_PATH = Path("results/evaluations")
EDGE_PROBS_PATH = Path("results/edge_probs")

# agent and policy keys:
DO_NOTHING_AGENT = "do_nothing_agent"
RL_AGENT = "reinforcement_learning_agent"
HIGH_LEVEL_AGENT = "high_level_agent"
DO_NOTHING_POLICY = "do_nothing_policy"
RL_POLICY = "reinforcement_learning_policy"
HIGH_LEVEL_POLICY = "high_level_policy"
RAPPO_POLICY = "rappo_torch_policy"
RASAC_POLICY = "rasac_torch_policy"
RADQN_POLICY = "radqn_torch_policy"
DQN_GNN_POLICY = "dqn_gnn_torch_policy"
DQN_MLP_POLICY = "dqn_mlp_torch_policy"

class Style:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


SEED = 42


def set_seed(seed):
    global SEED
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


set_seed(42)

_testing = False

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
