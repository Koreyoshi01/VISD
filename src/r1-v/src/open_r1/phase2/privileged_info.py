from __future__ import annotations

import ast
import re
from copy import deepcopy
from typing import Any


def _normalize_time_window(value: Any) -> list[float] | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            value = parsed
        else:
            match = re.search(r"<t>(\d+\.?\d*)</t>s to <t>(\d+\.?\d*)</t>s", text)
            if match:
                return [float(match.group(1)), float(match.group(2))]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _normalize_answer_window_for_task(task: Any, answer: Any) -> list[float] | None:
    if task not in {"temporal QA", "temporal QA (MCQ)"}:
        return None
    if isinstance(answer, str):
        text = answer.strip()
        if "\n[" in text:
            text = text.split("\n", 1)[1].strip()
        return _normalize_time_window(text)
    return _normalize_time_window(answer)


def _normalize_key_frame(frame: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(frame)
    if "path" in normalized and "image" not in normalized:
        normalized["image"] = normalized["path"]
    if "time" in normalized and "timestamp" not in normalized:
        normalized["timestamp"] = normalized["time"]
    return normalized


def _build_keyframe_object_evidence(
    key_frames: list[dict[str, Any]],
    key_items: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if not key_frames or not isinstance(key_items, dict):
        return evidence

    for frame in key_frames:
        frame_idx = frame.get("idx")
        frame_time = frame.get("timestamp", frame.get("time"))
        if frame_idx is None or frame_time is None:
            continue
        objects = key_items.get(str(frame_idx))
        if not isinstance(objects, dict) or not objects:
            continue
        evidence.append(
            {
                "frame_idx": int(frame_idx),
                "time": float(frame_time),
                "objects": deepcopy(objects),
            }
        )
    return evidence


def build_privileged_info(example: dict[str, Any] | None) -> dict[str, Any]:
    """Build a VISD-style privileged-info view from an STGR row."""
    example = dict(example or {})
    key_frames = example.get("key_frames") or []
    normalized_key_frames = [
        _normalize_key_frame(frame)
        for frame in key_frames
        if isinstance(frame, dict)
    ]
    key_items = deepcopy(example.get("key_items") or {})

    answer = example.get("answer")
    answer_window = (
        _normalize_answer_window_for_task(example.get("task"), example.get("answer"))
    )
    return {
        "question": example.get("question"),
        "task": example.get("task"),
        "type": example.get("type"),
        "sample_id": example.get("id"),
        "source": example.get("source"),
        "answer": answer,
        "ground_truth_window": answer_window,
        "gold_keyframes": normalized_key_frames,
        "has_gold_keyframes": bool(normalized_key_frames),
        "key_items": key_items,
        "keyframe_object_evidence": _build_keyframe_object_evidence(normalized_key_frames, key_items),
    }
