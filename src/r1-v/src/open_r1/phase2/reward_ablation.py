from __future__ import annotations

from typing import Any

import torch


def _reward_func_name(reward_func: Any) -> str:
    config = getattr(reward_func, "config", None)
    name_or_path = getattr(config, "_name_or_path", None)
    if isinstance(name_or_path, str) and name_or_path:
        return name_or_path.split("/")[-1]
    return getattr(reward_func, "__name__", reward_func.__class__.__name__)


def apply_reward_ablation_mask(
    *,
    rewards_per_func: torch.Tensor,
    reward_funcs: list[Any],
    script_args: Any | None,
) -> torch.Tensor:
    if script_args is None:
        return rewards_per_func

    masked = rewards_per_func.clone()
    wo_spatial = bool(getattr(script_args, "wo_spatial", False))
    wo_tempspatial = bool(getattr(script_args, "wo_tempspatial", False))
    wo_acc = bool(getattr(script_args, "wo_acc", False))

    if not any([wo_spatial, wo_tempspatial, wo_acc]):
        return masked

    for index, reward_func in enumerate(reward_funcs):
        reward_name = _reward_func_name(reward_func)
        disable = False
        if wo_acc and "ans_acc" in reward_name:
            disable = True
        if wo_tempspatial and (
            "thk_temporal_point" in reward_name
            or "thk_temporal_segment" in reward_name
            or "thk_spatial" in reward_name
        ):
            disable = True
        elif wo_spatial and "thk_spatial" in reward_name:
            disable = True
        if disable:
            masked[:, index] = torch.zeros_like(masked[:, index])

    return masked
