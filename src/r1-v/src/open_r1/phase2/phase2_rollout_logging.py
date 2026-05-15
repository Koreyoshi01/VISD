from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel


def get_reward_func_name(reward_func) -> str:
    if isinstance(reward_func, PreTrainedModel):
        return reward_func.config._name_or_path.split("/")[-1]
    return getattr(reward_func, "__name__", reward_func.__class__.__name__)


def build_rollout_monitor_metrics(
    *,
    rewards_per_func: torch.Tensor,
    reward_names: list[str],
    num_generations: int,
    success_reward_threshold: float,
) -> dict[str, float]:
    if rewards_per_func.numel() == 0:
        return {}

    reward_matrix = rewards_per_func.detach().float()
    total_rewards = reward_matrix.sum(dim=1)
    grouped_total_rewards = total_rewards.view(-1, num_generations)
    metrics = {
        "rollout/group_all_zero_fraction": float((grouped_total_rewards <= 0).all(dim=1).float().mean().item()),
        "rollout/group_any_nonzero_fraction": float((grouped_total_rewards > 0).any(dim=1).float().mean().item()),
        "rollout/group_all_success_fraction": float(
            (grouped_total_rewards >= success_reward_threshold).all(dim=1).float().mean().item()
        ),
        "rollout/group_mean_total_reward": float(grouped_total_rewards.mean().item()),
    }
    for index, reward_name in enumerate(reward_names):
        reward_values = reward_matrix[:, index]
        metrics[f"rollout/nonzero_fraction/{reward_name}"] = float((reward_values > 0).float().mean().item())
        metrics[f"rollout/mean/{reward_name}"] = float(reward_values.mean().item())
    return metrics


def _truncate_text(value: Any, max_chars: int | None = 0) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def build_phase2_rollout_trace_row(
    *,
    global_step: int,
    sample_id: str | None,
    rollout_index: int,
    source: str | None,
    task: str | None,
    student_output: str | None,
    reward_breakdown: dict[str, Any] | None,
    reward_metadata: dict[str, Any] | None,
    supervision: dict[str, Any] | None,
    group_state: str | None,
    process_feedback_status: str,
    process_feedback_error_type: str | None = None,
    process_feedback_error_message: str | None = None,
    process_feedback_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reward_breakdown = dict(reward_breakdown or {})
    reward_metadata = dict(reward_metadata or {})
    supervision = dict(supervision or {})
    process_feedback_result = dict(process_feedback_result or {})
    feedback_text = process_feedback_result.get("feedback") or supervision.get("feedback_text")

    return {
        "global_step": int(global_step),
        "sample_id": sample_id,
        "rollout_index": int(rollout_index),
        "source": source,
        "task": task,
        "routing_group_state": group_state,
        "should_target": bool(supervision.get("should_target", False)),
        "has_hindsight_target": bool(supervision.get("has_hindsight_target", False)),
        "target_score": supervision.get("target_score"),
        "native_reward": reward_breakdown.get("reward"),
        "answer_semantic_score": reward_breakdown.get("answer_semantic_score"),
        "answer_window_score": reward_breakdown.get("answer_window_score"),
        "answer_box_score": reward_breakdown.get("answer_box_score"),
        "temporal_grounding_score": reward_breakdown.get("temporal_grounding_score"),
        "spatial_grounding_score": reward_breakdown.get("spatial_grounding_score"),
        "format_score": reward_breakdown.get("format_score"),
        "reward_feedback": reward_metadata.get("feedback"),
        "student_output": _truncate_text(student_output),
        "teacher_feedback_text": _truncate_text(feedback_text),
        "judge_available": bool(feedback_text),
        "judge_feedback": _truncate_text(process_feedback_result.get("feedback")),
        "process_feedback_status": process_feedback_status,
        "process_feedback_error_type": process_feedback_error_type,
        "process_feedback_error_message": _truncate_text(process_feedback_error_message, 240),
    }
