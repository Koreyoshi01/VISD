from __future__ import annotations

from typing import Any

import torch


def _safe_reward_value(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_phase2_group_state(
    reward_values: list[float | None],
    *,
    success_reward_threshold: float = 0.999,
) -> str:
    known_rewards = [reward for reward in reward_values if reward is not None]
    if len(known_rewards) != len(reward_values) or len(known_rewards) == 0:
        return "mixed"
    if all(reward >= success_reward_threshold for reward in known_rewards):
        return "all_success"
    if all(reward < success_reward_threshold for reward in known_rewards):
        return "all_failure"
    return "mixed"


def build_phase2_requested_masks(
    *,
    supervision_rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    success_reward_threshold: float = 0.999,
) -> dict[str, Any]:
    target_scores = [
        _safe_reward_value(
            (row.get("supervision") or {}).get("target_score")
        )
        if _safe_reward_value((row.get("supervision") or {}).get("target_score")) is not None
        else _safe_reward_value((row.get("reward_breakdown") or {}).get("reward"))
        for row in supervision_rows
    ]
    group_state = resolve_phase2_group_state(
        target_scores,
        success_reward_threshold=success_reward_threshold,
    )

    target_mask: list[float] = []
    reweight_mask: list[float] = []
    hindsight_available = 0
    feedback_available = 0

    for row in supervision_rows:
        supervision = row.get("supervision") or {}
        should_target = bool(supervision.get("should_target", False))
        has_hindsight_target = bool(supervision.get("has_hindsight_target", False))
        feedback_text = supervision.get("feedback_text")

        hindsight_available += 1 if has_hindsight_target else 0
        feedback_available += 1 if isinstance(feedback_text, str) and feedback_text.strip() else 0

        sample_target = should_target
        target_mask.append(1.0 if sample_target else 0.0)
        reweight_mask.append(1.0 if sample_target else 0.0)

    batch_size = max(1, len(supervision_rows))
    requested_target_mask = torch.tensor(target_mask, device=device, dtype=dtype)
    requested_reweight_mask = torch.tensor(reweight_mask, device=device, dtype=dtype)

    metrics = {
        "phase2/requested_sample_fraction": float(requested_target_mask.mean().item())
        if requested_target_mask.numel() > 0
        else 0.0,
        "phase2/target_sample_fraction": float(requested_target_mask.mean().item()) if requested_target_mask.numel() > 0 else 0.0,
        "phase2/reweight_sample_fraction": float(requested_reweight_mask.mean().item()) if requested_reweight_mask.numel() > 0 else 0.0,
        "phase2/group_all_success_fraction": 1.0 if group_state == "all_success" else 0.0,
        "phase2/group_all_failure_fraction": 1.0 if group_state == "all_failure" else 0.0,
        "phase2/group_mixed_fraction": 1.0 if group_state == "mixed" else 0.0,
        "phase2/group_avg_target_score": float(sum(score for score in target_scores if score is not None) / len(target_scores))
        if target_scores and all(score is not None for score in target_scores)
        else 0.0,
        "phase2/hindsight_available_fraction": float(hindsight_available / batch_size),
        "phase2/feedback_available_fraction": float(feedback_available / batch_size),
    }
    return {
        "group_state": group_state,
        "requested_target_mask": requested_target_mask,
        "requested_reweight_mask": requested_reweight_mask,
        "metrics": metrics,
    }


def build_effective_phase2_masks(
    *,
    requested_target_mask: torch.Tensor,
    requested_reweight_mask: torch.Tensor,
    valid_indices: list[int],
) -> dict[str, Any]:
    effective_target_mask = torch.zeros_like(requested_target_mask)
    effective_reweight_mask = torch.zeros_like(requested_reweight_mask)

    for index in valid_indices:
        effective_target_mask[index] = requested_target_mask[index]
        effective_reweight_mask[index] = requested_reweight_mask[index]

    metrics = {
        "phase2/requested_target_sample_fraction": float(requested_target_mask.mean().item())
        if requested_target_mask.numel() > 0
        else 0.0,
        "phase2/effective_sample_fraction": float(effective_target_mask.mean().item())
        if effective_target_mask.numel() > 0
        else 0.0,
        "phase2/effective_target_sample_fraction": float(effective_target_mask.mean().item())
        if effective_target_mask.numel() > 0
        else 0.0,
        "phase2/effective_reweight_sample_fraction": float(effective_reweight_mask.mean().item())
        if effective_reweight_mask.numel() > 0
        else 0.0,
    }
    return {
        "effective_target_mask": effective_target_mask,
        "effective_reweight_mask": effective_reweight_mask,
        "metrics": metrics,
    }


__all__ = [
    "build_effective_phase2_masks",
    "build_phase2_requested_masks",
    "resolve_phase2_group_state",
]
