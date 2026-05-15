from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback


@dataclass
class Phase2RuntimeScheduleConfig:
    reweighting_initial_lambda: float = 1.0
    reweighting_anneal_steps: int = 0
    teacher_disable_after_step: int = 0
    process_feedback_initial_enabled: bool = True
    process_feedback_disable_after_step: int = 0
    runtime_step_offset: int = 0


@dataclass
class Phase2CheckpointScheduleConfig:
    dense_until_step: int = 0
    dense_interval: int = 0
    sparse_interval: int = 0


def _compute_reweighting_lambda(*, initial_lambda: float, anneal_steps: int, step: int) -> float:
    if anneal_steps <= 0:
        return float(initial_lambda)
    if step <= 0:
        return float(initial_lambda)
    if step > anneal_steps:
        return 0.0
    remaining_ratio = max(float(anneal_steps - (step - 1)), 0.0) / float(anneal_steps)
    return float(initial_lambda) * remaining_ratio


def _resolve_checkpoint_save(step: int, *, dense_until_step: int, dense_interval: int, sparse_interval: int) -> bool:
    if step <= 0:
        return False
    if dense_until_step > 0 and step <= dense_until_step:
        return dense_interval > 0 and step % dense_interval == 0
    return sparse_interval > 0 and step % sparse_interval == 0


class Phase2RuntimeScheduleCallback(TrainerCallback):
    def __init__(
        self,
        *,
        trainer: Any,
        runtime_config: Phase2RuntimeScheduleConfig,
        checkpoint_config: Phase2CheckpointScheduleConfig,
    ) -> None:
        self.trainer = trainer
        self.runtime_config = runtime_config
        self.checkpoint_config = checkpoint_config
        self._last_runtime_state: dict[str, float] = {}

    def _record_runtime_metrics(self) -> None:
        metrics = getattr(self.trainer, "_metrics", None)
        if not isinstance(metrics, dict):
            return
        for key, value in self._last_runtime_state.items():
            metrics.setdefault(key, []).append(float(value))

    def on_step_begin(self, args, state, control, **kwargs):
        current_step = int(state.global_step) + 1 + int(self.runtime_config.runtime_step_offset)
        runtime = self.runtime_config

        current_lambda = _compute_reweighting_lambda(
            initial_lambda=runtime.reweighting_initial_lambda,
            anneal_steps=runtime.reweighting_anneal_steps,
            step=current_step,
        )
        teacher_enabled = not (
            runtime.teacher_disable_after_step > 0 and current_step > runtime.teacher_disable_after_step
        )
        process_feedback_enabled = bool(runtime.process_feedback_initial_enabled) and not (
            runtime.process_feedback_disable_after_step > 0
            and current_step > runtime.process_feedback_disable_after_step
        )

        self.trainer.phase2_reweighting_mixing_lambda = float(current_lambda)
        self.trainer.phase2_inject_teacher_signal = bool(teacher_enabled)
        self.trainer.phase2_teacher_model_update_enabled = bool(teacher_enabled)
        self.trainer.phase2_process_feedback_enable = bool(process_feedback_enabled)

        self._last_runtime_state = {
            "phase2/runtime_effective_step": float(current_step),
            "phase2/runtime_reweighting_lambda": float(current_lambda),
            "phase2/runtime_teacher_enabled": 1.0 if teacher_enabled else 0.0,
            "phase2/runtime_teacher_model_update_enabled": 1.0 if teacher_enabled else 0.0,
            "phase2/runtime_process_feedback_enabled": 1.0 if process_feedback_enabled else 0.0,
            "phase2/runtime_pure_grpo_mode": 1.0 if (not teacher_enabled and current_lambda == 0.0) else 0.0,
        }
        return control

    def on_step_end(self, args, state, control, **kwargs):
        completed_step = int(state.global_step) + int(self.runtime_config.runtime_step_offset)
        teacher_model_updated = False
        maybe_update_teacher = getattr(self.trainer, "_maybe_update_teacher_model", None)
        if callable(maybe_update_teacher):
            teacher_model_updated = bool(maybe_update_teacher(completed_step=completed_step))
        self._last_runtime_state["phase2/runtime_teacher_model_updated"] = 1.0 if teacher_model_updated else 0.0
        self._record_runtime_metrics()

        if _resolve_checkpoint_save(
            completed_step,
            dense_until_step=int(self.checkpoint_config.dense_until_step),
            dense_interval=int(self.checkpoint_config.dense_interval),
            sparse_interval=int(self.checkpoint_config.sparse_interval),
        ):
            control.should_save = True

        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps > 0 and completed_step >= max_steps:
            control.should_save = True
        return control


__all__ = [
    "Phase2CheckpointScheduleConfig",
    "Phase2RuntimeScheduleCallback",
    "Phase2RuntimeScheduleConfig",
]
