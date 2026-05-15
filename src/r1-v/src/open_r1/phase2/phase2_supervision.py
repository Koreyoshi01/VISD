from __future__ import annotations

from typing import Any

def _config_get(script_args, key: str, default):
    return getattr(script_args, key, default) if script_args is not None else default


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_feedback_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def has_reusable_feedback_signal(
    feedback_text: str | None,
    *,
    judge_metadata: dict[str, Any] | None = None,
    answer_semantic_score: Any = None,
) -> bool:
    return _normalize_feedback_text(feedback_text) is not None


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _build_reward_breakdown_feedback(
    *,
    reward_breakdown: dict[str, float | None],
    include_reward_breakdown: bool,
) -> str | None:
    if not include_reward_breakdown:
        return None

    parts = []
    reward = reward_breakdown.get("reward")
    answer_score = reward_breakdown.get("answer_semantic_score")
    temporal_score = reward_breakdown.get("temporal_grounding_score")
    spatial_score = reward_breakdown.get("spatial_grounding_score")

    if reward is not None:
        parts.append(f"Reward={reward:.4f}")
    if answer_score is not None:
        parts.append(f"Answer correctness score={answer_score:.4f}")
    if temporal_score is not None:
        parts.append(f"Temporal grounding score={temporal_score:.4f}")
    if spatial_score is not None:
        parts.append(f"Spatial grounding score={spatial_score:.4f}")
    if not parts:
        return None
    return " ".join(parts)


def _merge_feedback_texts(*parts: str | None) -> str | None:
    merged = []
    seen = set()
    for part in parts:
        if not isinstance(part, str):
            continue
        text = part.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    if not merged:
        return None
    return " ".join(merged)


def _has_hindsight_target(privileged_info: dict[str, Any]) -> bool:
    return bool(
        privileged_info.get("ground_truth_window") is not None
        or privileged_info.get("answer")
    )


def resolve_phase2_target_score(
    *,
    task: Any,
    reward_breakdown: dict[str, float | None],
    answer_semantic_score: float | None,
    success_metric: str = "task_aware",
) -> float | None:
    metric_mode = str(success_metric or "task_aware").lower()
    total_reward = _safe_float(reward_breakdown.get("reward"))
    format_score = _safe_float(reward_breakdown.get("format_score")) or 0.0
    semantic = _safe_float(answer_semantic_score)
    temporal_answer = _safe_float(reward_breakdown.get("answer_window_score"))
    spatial_answer = _safe_float(reward_breakdown.get("answer_box_score"))
    temporal_reasoning = _safe_float(reward_breakdown.get("temporal_grounding_score"))
    spatial_reasoning = _safe_float(reward_breakdown.get("spatial_grounding_score"))

    if metric_mode in {"native_total", "reward"}:
        return total_reward

    task_name = str(task or "")
    reward_minus_format = None if total_reward is None else max(0.0, total_reward - format_score)

    if metric_mode == "answer_only":
        if semantic is not None:
            return semantic
        if task_name == "temporal QA":
            candidates = [value for value in (temporal_answer, temporal_reasoning) if value is not None]
            return max(candidates) if candidates else reward_minus_format
        if task_name == "visual QA":
            candidates = [value for value in (spatial_answer, spatial_reasoning) if value is not None]
            return max(candidates) if candidates else reward_minus_format
        return reward_minus_format

    if task_name == "temporal QA":
        candidates = [value for value in (temporal_answer, temporal_reasoning) if value is not None]
        return max(candidates) if candidates else reward_minus_format

    if task_name == "temporal QA (MCQ)":
        temporal_signal_candidates = [value for value in (temporal_answer, temporal_reasoning) if value is not None]
        temporal_signal = max(temporal_signal_candidates) if temporal_signal_candidates else None
        if semantic is not None and temporal_signal is not None:
            return min(semantic, temporal_signal)
        return semantic if semantic is not None else (temporal_signal if temporal_signal is not None else reward_minus_format)

    if task_name == "visual QA":
        candidates = [value for value in (spatial_answer, spatial_reasoning) if value is not None]
        return max(candidates) if candidates else reward_minus_format

    if task_name == "temporal-spatial free-form QA":
        grounding_candidates = [value for value in (temporal_reasoning, spatial_reasoning) if value is not None]
        grounding_signal = max(grounding_candidates) if grounding_candidates else None
        if semantic is not None and grounding_signal is not None:
            return min(semantic, grounding_signal)
        return grounding_signal if grounding_signal is not None else (semantic if semantic is not None else reward_minus_format)

    if "General video QA" in task_name:
        return semantic if semantic is not None else reward_minus_format

    return semantic if semantic is not None else reward_minus_format


def build_phase2_supervision(
    *,
    example: dict[str, Any],
    privileged_info: dict[str, Any],
    reward_breakdown: dict[str, float | None],
    reward_metadata: dict[str, Any] | None = None,
    script_args=None,
    extra_feedback_text: str | None = None,
    generated_process_feedback_text: str | None = None,
    generated_process_feedback_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    reward_metadata = reward_metadata or {}
    include_text_feedback = bool(
        getattr(script_args, "phase2_teacher_include_text_feedback", True)
        if script_args is not None
        else True
    )
    include_reward_breakdown = bool(
        getattr(script_args, "phase2_teacher_include_reward_breakdown_in_feedback", False)
        if script_args is not None
        else False
    )
    success_metric = (
        getattr(script_args, "phase2_teacher_success_metric", "task_aware")
        if script_args is not None
        else "task_aware"
    )

    feedback_seed = _normalize_feedback_text(example.get("feedback_seed"))
    reward_feedback = _normalize_feedback_text(example.get("feedback")) or _normalize_feedback_text(
        reward_metadata.get("feedback")
    )
    generated_feedback = None
    if isinstance(generated_process_feedback_result, dict):
        generated_feedback = _normalize_feedback_text(generated_process_feedback_result.get("feedback"))
    process_feedback = (
        generated_feedback
        or _normalize_feedback_text(generated_process_feedback_text)
        or _normalize_feedback_text(example.get("process_feedback"))
    )

    base_feedback = None
    if include_text_feedback:
        base_feedback = _merge_feedback_texts(process_feedback, reward_feedback, feedback_seed)
    reward_breakdown_feedback = _build_reward_breakdown_feedback(
        reward_breakdown=reward_breakdown,
        include_reward_breakdown=include_reward_breakdown,
    )
    merged_feedback = _merge_feedback_texts(base_feedback, reward_breakdown_feedback, extra_feedback_text)

    answer_semantic_score = _first_non_none(
        _safe_float(example.get("answer_semantic_score")),
        _safe_float(reward_metadata.get("acc_score")),
        _safe_float(reward_breakdown.get("answer_semantic_score")),
    )
    judge_metadata = {}

    target_score = resolve_phase2_target_score(
        task=example.get("task") or privileged_info.get("task"),
        reward_breakdown=reward_breakdown,
        answer_semantic_score=answer_semantic_score,
        success_metric=success_metric,
    )

    has_hindsight_target = _has_hindsight_target(privileged_info)
    should_target = True

    supervision = {
        "feedback_text": merged_feedback,
        "judge_metadata": judge_metadata,
        "answer_semantic_score": answer_semantic_score,
        "target_score": target_score,
        "success_metric": success_metric,
        "has_hindsight_target": has_hindsight_target,
        "should_target": should_target,
        "reward_breakdown": reward_breakdown,
    }
    metrics = {
        "phase2/target_sample_fraction": 1.0 if should_target else 0.0,
        "phase2/effective_sample_fraction": 1.0 if should_target else 0.0,
        "phase2/hindsight_available_fraction": 1.0 if has_hindsight_target else 0.0,
        "phase2/feedback_available_fraction": 1.0 if merged_feedback else 0.0,
        "phase2/target_score_mean": float(target_score) if target_score is not None else 0.0,
        "phase2/semantic_score_available_fraction": 1.0 if answer_semantic_score is not None else 0.0,
    }
    return supervision, metrics


__all__ = [
    "build_phase2_supervision",
    "has_reusable_feedback_signal",
    "resolve_phase2_target_score",
]
