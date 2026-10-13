"""
Imitation pretraining of the RL policy network: cross-entropy between the policy's logits and the
dataset's `policy_target` (one-hot for hard labels, a distribution for soft labels).

The network is built through RLlib's ModelCatalog exactly as PPO builds the RL_POLICY model, so
the trained `state_dict` loads into the policy unchanged (`rarl_rllib.callback.ImitationInitCallback`).
Only the policy logits are trained; the value branch is left to the critic warm-up in RL.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.utils.data import DataLoader, Dataset

from imitation.tutor import DO_NOTHING_INDEX

logger = logging.getLogger(__name__)

# An action counts as "good" if its simulated max rho is within this of the best valid action.
GOOD_ACTION_TOLERANCE = 0.01


@dataclass
class PretrainConfig:
    """Optimisation settings.

    Attributes:
        epochs: Maximum epochs.
        batch_size: Samples per gradient step.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        patience: Stop after this many epochs without a lower validation loss.
        device: Torch device.
        seed: Seed for initialisation and shuffling.
    """

    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    device: str = "cpu"
    seed: int = 42


def build_policy_model(obs_space: gym.spaces.Space, n_actions: int, model_config: dict) -> TorchModelV2:
    """Build the RL_POLICY network as RLlib's PPO policy does.

    Args:
        obs_space: The RL agent's observation space (the converter's).
        n_actions: Discrete policy actions incl. do-nothing.
        model_config: The `model` dict of the RLlib config, with the PPO custom model name.

    Returns:
        The torch model; its `state_dict` matches the PPO policy's `model`.
    """
    # PPO's model defaults, not RLlib's MODEL_DEFAULTS: PPO sets vf_share_layers=False, which adds a
    # separate value network (`_value_branch_separate`) that the state_dict must contain.
    config = copy.deepcopy(PPOConfig().model)
    config.update(copy.deepcopy(model_config))
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=gym.spaces.Discrete(n_actions),
        num_outputs=n_actions,
        model_config=config,
        framework="torch",
    )


def policy_logits(model: TorchModelV2, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Action logits [B, A] for a batch of converted observations."""
    logits, _ = model.forward({"obs": obs}, [], None)
    return logits


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy against target distributions (equals the usual CE for one-hot targets)."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (_to_device(v, device) if isinstance(v, dict) else v.to(device)) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: TorchModelV2, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    """Validation metrics.

    Always: loss, accuracy (argmax = argmax of target), top5 (target argmax among the 5 highest
    logits), do_nothing_rate (share of argmax = do-nothing). If the dataset has simulated max rho:
    good_action_rate (argmax within GOOD_ACTION_TOLERANCE of the best valid action), regret (mean
    rho excess of the argmax over the best valid action; invalid choices count as missing),
    invalid_rate (argmax not valid).
    """
    model.eval()
    sums: dict[str, float] = {}
    n_regret = 0
    n = 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = _to_device(batch, device)
        logits = policy_logits(model, batch["obs"])
        target = batch["target"]
        pred = logits.argmax(dim=-1)
        top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        best_target = target.argmax(dim=-1)
        b = len(pred)
        stats = {
            "loss": soft_cross_entropy(logits, target).item() * b,
            "accuracy": (pred == best_target).sum().item(),
            "top5": (top5 == best_target.unsqueeze(-1)).any(dim=-1).sum().item(),
            "do_nothing_rate": (pred == DO_NOTHING_INDEX).sum().item(),
        }
        if "max_rho" in batch:
            rho = torch.where(batch["valid"], batch["max_rho"], torch.full_like(batch["max_rho"], float("inf")))
            best = rho.min(dim=-1).values
            chosen = rho.gather(1, pred.unsqueeze(-1)).squeeze(-1)
            has_best = torch.isfinite(best)
            chosen_valid = torch.isfinite(chosen) & has_best
            stats["good_action_rate"] = (chosen_valid & (chosen - best <= GOOD_ACTION_TOLERANCE)).sum().item()
            stats["invalid_rate"] = (~torch.isfinite(chosen) & has_best).sum().item()
            stats["regret"] = (chosen - best)[chosen_valid].sum().item()
            n_regret += int(chosen_valid.sum().item())
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
        n += b
    metrics = {k: v / n for k, v in sums.items() if k != "regret"}
    if "regret" in sums:
        metrics["regret"] = sums["regret"] / max(n_regret, 1)
    return metrics


def train(model: TorchModelV2, train_set: Dataset, val_set: Dataset, config: PretrainConfig) -> tuple[dict, list[dict]]:
    """Train with early stopping on validation loss.

    Returns:
        `(best_state_dict, history)`; history has one dict per epoch with train_loss and `val_*` metrics.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)

    best_loss, best_state, epochs_without_gain = np.inf, copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = _to_device(batch, device)
            loss = soft_cross_entropy(policy_logits(model, batch["obs"]), batch["target"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch["target"])
            count += len(batch["target"])
        val = evaluate(model, val_set, config.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / count, **{f"val_{k}": v for k, v in val.items()}})
        logger.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in history[-1].items()))

        if val["loss"] < best_loss:
            best_loss, best_state, epochs_without_gain = val["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                logger.info(f"Early stop after epoch {epoch}: no lower validation loss for {config.patience} epochs")
                break
    model.load_state_dict(best_state)
    return best_state, history
