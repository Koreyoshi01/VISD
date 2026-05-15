from __future__ import annotations

from typing import Sequence

import torch


def _resolve_reward_weights(
    *,
    num_rewards: int,
    reward_weights: Sequence[float] | torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if reward_weights is None:
        return torch.ones(num_rewards, device=device, dtype=dtype)

    if isinstance(reward_weights, torch.Tensor):
        weights = reward_weights.to(device=device, dtype=dtype)
    else:
        if len(reward_weights) == 0:
            return torch.ones(num_rewards, device=device, dtype=dtype)
        weights = torch.tensor(list(reward_weights), device=device, dtype=dtype)

    if weights.numel() != num_rewards:
        raise ValueError(
            "Number of reward weights must match number of reward functions, "
            f"got {weights.numel()} vs {num_rewards}."
        )
    return weights


def compute_rollout_advantages(
    *,
    rewards_per_func: torch.Tensor,
    num_generations: int,
    reward_weights: Sequence[float] | torch.Tensor | None = None,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rewards_per_func.dim() != 2:
        raise ValueError(
            "rewards_per_func must be a 2D tensor of shape [batch_size * num_generations, num_rewards], "
            f"got shape {tuple(rewards_per_func.shape)}"
        )
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if rewards_per_func.size(0) % num_generations != 0:
        raise ValueError(
            "The first dimension of rewards_per_func must be divisible by num_generations, "
            f"got {rewards_per_func.size(0)} vs {num_generations}."
        )

    rewards_per_func = torch.nan_to_num(rewards_per_func)
    weights = _resolve_reward_weights(
        num_rewards=rewards_per_func.size(1),
        reward_weights=reward_weights,
        device=rewards_per_func.device,
        dtype=rewards_per_func.dtype,
    )

    weighted_rewards = (rewards_per_func * weights.unsqueeze(0)).sum(dim=1)
    mean_grouped_rewards = weighted_rewards.view(-1, num_generations).mean(dim=1)
    std_grouped_rewards = weighted_rewards.view(-1, num_generations).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)

    advantages = (weighted_rewards - mean_grouped_rewards) / (std_grouped_rewards + eps)

    return advantages, weighted_rewards, mean_grouped_rewards, std_grouped_rewards
