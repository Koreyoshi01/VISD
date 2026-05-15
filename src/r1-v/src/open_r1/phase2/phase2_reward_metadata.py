from __future__ import annotations

from typing import Any


def merge_reward_metadata_rows(
    reward_metadata_rows: list[dict[str, list[Any]]] | None,
    reward_metadata_part: dict[str, list[Any]],
    *,
    expected_length: int,
) -> list[dict[str, list[Any]]]:
    if reward_metadata_rows is None:
        reward_metadata_rows = [{} for _ in range(expected_length)]
    if len(reward_metadata_rows) != expected_length:
        raise ValueError(
            "reward_metadata_rows length does not match expected_length: "
            f"{len(reward_metadata_rows)} != {expected_length}"
        )

    for key, values in reward_metadata_part.items():
        padded_values = list(values[:expected_length])
        if len(padded_values) < expected_length:
            padded_values.extend([None] * (expected_length - len(padded_values)))
        for row_index, value in enumerate(padded_values):
            reward_metadata_rows[row_index].setdefault(key, []).append(value)

    return reward_metadata_rows


def normalize_reward_func_output(output_reward_func) -> tuple[list[float], dict[str, list[Any]]]:
    scores: list[float] = []
    metadata: dict[str, list[Any]] = {}

    for item in output_reward_func:
        row_index = len(scores)
        if isinstance(item, dict):
            score = item.get("score", item.get("reward", 0.0))
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                scores.append(0.0)

            row_keys = {key for key in item.keys() if key not in {"score", "reward"}}
            for key in list(metadata.keys()):
                if key not in row_keys:
                    metadata[key].append(None)
            for key, value in item.items():
                if key in {"score", "reward"}:
                    continue
                if key not in metadata:
                    metadata[key] = [None] * row_index
                metadata.setdefault(key, []).append(value)
        else:
            try:
                scores.append(float(item))
            except (TypeError, ValueError):
                scores.append(0.0)
            for key in metadata:
                metadata[key].append(None)

    return scores, metadata


def summarize_reward_metadata(reward_metadata: dict[str, list[Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in reward_metadata.items():
        cleaned = [value for value in values if value is not None]
        if not cleaned:
            continue
        first = cleaned[0]
        if isinstance(first, str):
            summary[key] = first.strip() if first.strip() else None
            continue
        try:
            numeric = [float(value) for value in cleaned]
        except (TypeError, ValueError):
            summary[key] = first
            continue
        summary[key] = sum(numeric) / len(numeric)
    return {key: value for key, value in summary.items() if value is not None}
