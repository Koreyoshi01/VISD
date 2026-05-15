from __future__ import annotations

from typing import Literal


Phase2TrainerMode = Literal["base", "phase2_teacher"]


def resolve_phase2_trainer_mode(
    *,
    enable_phase2_teacher: bool | None = None,
) -> Phase2TrainerMode:
    if bool(enable_phase2_teacher):
        return "phase2_teacher"
    return "base"


def normalize_phase2_advantage_clip(raw_value: float | int | str | None) -> float | None:
    if raw_value is None:
        return None
    value = float(raw_value)
    return None if value < 0 else value

