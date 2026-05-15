from __future__ import annotations

import torch


def normalize_importance_sampling_level(
    importance_sampling_level: str | None,
    *,
    gspo_default: bool = True,
) -> str:
    if importance_sampling_level is None or str(importance_sampling_level).strip() == "":
        return "sequence" if gspo_default else "token"

    normalized = str(importance_sampling_level).strip().lower()
    alias_map = {
        "grpo": "token",
        "token": "token",
        "gspo": "sequence",
        "sequence": "sequence",
        "gspo_token": "sequence_token",
        "gspo-token": "sequence_token",
        "sequence_token": "sequence_token",
        "sequence-token": "sequence_token",
    }
    if normalized not in alias_map:
        raise ValueError(
            "Unsupported importance_sampling_level: "
            f"{importance_sampling_level}. Expected one of "
            "'token'/'grpo', 'sequence'/'gspo', or 'sequence_token'/'gspo_token'."
        )
    return alias_map[normalized]


def build_log_importance_weights(
    *,
    per_token_logps: torch.Tensor,
    response_mask: torch.Tensor,
    importance_sampling_level: str,
    old_per_token_logps: torch.Tensor | None = None,
) -> torch.Tensor:
    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()
    if tuple(old_per_token_logps.shape) != tuple(per_token_logps.shape):
        raise ValueError(
            "old_per_token_logps must match per_token_logps shape, "
            f"got {tuple(old_per_token_logps.shape)} vs {tuple(per_token_logps.shape)}"
        )
    if tuple(response_mask.shape) != tuple(per_token_logps.shape):
        raise ValueError(
            "response_mask must match per_token_logps shape, "
            f"got {tuple(response_mask.shape)} vs {tuple(per_token_logps.shape)}"
        )

    normalized_level = normalize_importance_sampling_level(importance_sampling_level)
    mask = response_mask.to(device=per_token_logps.device, dtype=per_token_logps.dtype)
    log_ratio = per_token_logps - old_per_token_logps

    if normalized_level == "token":
        return log_ratio
    if normalized_level == "sequence":
        seq_level = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return seq_level.unsqueeze(-1)
    if normalized_level == "sequence_token":
        seq_level = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        seq_level = seq_level.detach().unsqueeze(-1)
        return (per_token_logps - per_token_logps.detach()) + seq_level

    raise ValueError(
        "Unsupported importance_sampling_level: "
        f"{importance_sampling_level}. Expected one of "
        "'token'/'grpo', 'sequence'/'gspo', or 'sequence_token'/'gspo_token'."
    )
