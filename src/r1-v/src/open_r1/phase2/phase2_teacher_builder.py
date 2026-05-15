from __future__ import annotations

from typing import Any


ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"


def has_hindsight_supervision(privileged_info: dict[str, Any] | None) -> bool:
    privileged_info = privileged_info or {}
    return bool(
        privileged_info.get("answer")
        or privileged_info.get("ground_truth_window")
        or privileged_info.get("gold_keyframes")
    )


def build_hindsight_hint(
    *,
    privileged_info: dict[str, Any] | None,
    feedback_text: str | None = None,
    answer_semantic_score: Any = None,
    judge_metadata: dict[str, Any] | None = None,
) -> str:
    privileged_info = privileged_info or {}
    judge_metadata = judge_metadata or {}

    parts: list[str] = [
        "[Hidden hindsight evidence for phase-2 teacher view]",
        "Use the following verified evidence only as hidden supervision.",
    ]

    answer = privileged_info.get("answer")
    ground_truth_window = privileged_info.get("ground_truth_window")
    keyframes = privileged_info.get("gold_keyframes") or []

    if answer:
        parts.append(f"Verified answer: {answer}")
    if isinstance(ground_truth_window, (list, tuple)) and len(ground_truth_window) == 2:
        parts.append(f"Verified temporal window: [{ground_truth_window[0]}, {ground_truth_window[1]}]")
    if keyframes:
        timestamps = []
        for frame in keyframes:
            timestamp = frame.get("timestamp", frame.get("time"))
            if timestamp is not None:
                timestamps.append(str(timestamp))
        if timestamps:
            parts.append(f"Verified keyframe timestamps: {', '.join(timestamps)}")
    if feedback_text:
        parts.append(f"Corrective feedback: {feedback_text}")

    return "\n" + "\n".join(parts) + "\n"


def inject_hindsight_hint_into_prompt(
    prompt_text: str,
    hint: str,
    *,
    assistant_marker: str = ASSISTANT_MARKER,
) -> str:
    if not hint.strip():
        return prompt_text

    if assistant_marker in prompt_text:
        marker_pos = prompt_text.rfind(assistant_marker)
        return prompt_text[:marker_pos] + hint + prompt_text[marker_pos:]
    return prompt_text + hint


__all__ = [
    "ASSISTANT_MARKER",
    "build_hindsight_hint",
    "has_hindsight_supervision",
    "inject_hindsight_hint_into_prompt",
]
