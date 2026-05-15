"""Standalone GRPO trainer for VISD feedback-conditioned teacher replay."""

import copy
import json
import os
import re
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import PreTrainedModel, is_wandb_available

from configs.data_root import DATA_ROOT
from src.open_r1.phase2.grpo_trainer import Qwen2VLGRPOTrainer
from src.open_r1.phase2.grpo_trainer import is_conversational, maybe_apply_chat_template
from src.open_r1.phase2.grpo_trainer import temporarily_set_model_eval
from src.open_r1.phase2.phase2_teacher_builder import (
    build_hindsight_hint,
    has_hindsight_supervision,
    inject_hindsight_hint_into_prompt,
)
from src.open_r1.phase2.phase2_batch_routing import (
    build_effective_phase2_masks,
    build_phase2_requested_masks,
)
from src.open_r1.phase2.phase2_host_batching import validate_phase2_host_batch
from src.open_r1.phase2.phase2_supervision import build_phase2_supervision
from src.open_r1.phase2.phase2_process_feedback import build_process_feedback_result
from src.open_r1.phase2.phase2_process_feedback_config import (
    resolve_phase2_process_feedback_request_kwargs,
)
from src.open_r1.phase2.phase2_policy_objective import (
    build_log_importance_weights,
    normalize_importance_sampling_level,
)
from src.open_r1.phase2.phase2_reward_metadata import (
    merge_reward_metadata_rows,
    normalize_reward_func_output,
    summarize_reward_metadata,
)
from src.open_r1.phase2.phase2_rollout_logging import (
    build_rollout_monitor_metrics,
    get_reward_func_name,
)
from src.open_r1.phase2.phase2_reweighting import (
    apply_phase2_reweighting,
    build_phase2_reweighting_teacher_reward_raw,
)
from src.open_r1.phase2.privileged_info import build_privileged_info
from src.open_r1.phase2.reward_ablation import apply_reward_ablation_mask
from src.open_r1.vision_process import (
    build_dummy_image_input,
    build_dummy_video_input,
    MIN_DUMMY_VIDEO_HEIGHT,
    MIN_DUMMY_VIDEO_WIDTH,
    process_vision_info,
)
from trl.models import unwrap_model_for_generation

ROOT = os.path.join(DATA_ROOT, "videos")
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")

if is_wandb_available():
    import wandb


_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL)


def _normalize_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum()
    if denom.item() <= 0:
        return 0.0
    return float(((values * mask).sum() / denom).item())


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum()
    if denom.item() <= 1:
        return 0.0
    mean = (values * mask).sum() / denom
    var = (((values - mean) ** 2) * mask).sum() / denom
    return float(torch.sqrt(var).item())


def _build_response_thirds_mask(response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    response_mask_bool = response_mask > 0
    token_rank = response_mask_bool.long().cumsum(dim=-1) - 1
    lengths = response_mask_bool.long().sum(dim=-1, keepdim=True).clamp(min=1)
    position_ratio = token_rank.to(dtype=torch.float32) / lengths.to(dtype=torch.float32)
    front = response_mask_bool & (position_ratio < (1.0 / 3.0))
    mid = response_mask_bool & (position_ratio >= (1.0 / 3.0)) & (position_ratio < (2.0 / 3.0))
    tail = response_mask_bool & (position_ratio >= (2.0 / 3.0))
    return front, mid, tail


def build_response_entropy_metrics(
    *,
    response_entropy: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    entropy = response_entropy.detach().float()
    mask = response_mask.detach().float()
    front_mask, mid_mask, tail_mask = _build_response_thirds_mask(mask)
    front_mask = front_mask.to(dtype=mask.dtype)
    mid_mask = mid_mask.to(dtype=mask.dtype)
    tail_mask = tail_mask.to(dtype=mask.dtype)
    metrics = {
        "phase2/response_entropy_mean": _masked_mean(entropy, mask),
        "phase2/response_entropy_std": _masked_std(entropy, mask),
        "phase2/response_entropy_front_mean": _masked_mean(entropy, front_mask),
        "phase2/response_entropy_mid_mean": _masked_mean(entropy, mid_mask),
        "phase2/response_entropy_tail_mean": _masked_mean(entropy, tail_mask),
    }
    metrics["phase2/response_entropy_tail_minus_front"] = (
        metrics["phase2/response_entropy_tail_mean"] - metrics["phase2/response_entropy_front_mean"]
    )
    return metrics


def build_teacher_signal_metrics(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    target_mask: torch.Tensor,
    student_entropy: torch.Tensor,
    teacher_entropy: torch.Tensor | None = None,
) -> dict[str, float]:
    response_mask = response_mask.detach().float()
    target_mask = target_mask.detach().float()
    token_mask = response_mask * target_mask.unsqueeze(-1)
    gap = (teacher_log_probs.detach() - student_log_probs.detach()).float()
    abs_gap = gap.abs()

    front_mask, mid_mask, tail_mask = _build_response_thirds_mask(response_mask)
    front_mask = token_mask * front_mask.to(dtype=token_mask.dtype)
    mid_mask = token_mask * mid_mask.to(dtype=token_mask.dtype)
    tail_mask = token_mask * tail_mask.to(dtype=token_mask.dtype)

    metrics = {
        "phase2/teacher_gap_mean": _masked_mean(gap, token_mask),
        "phase2/teacher_gap_std": _masked_std(gap, token_mask),
        "phase2/teacher_gap_abs_mean": _masked_mean(abs_gap, token_mask),
        "phase2/teacher_gap_positive_fraction": _masked_mean((gap > 0).float(), token_mask),
        "phase2/teacher_gap_front_mean": _masked_mean(gap, front_mask),
        "phase2/teacher_gap_mid_mean": _masked_mean(gap, mid_mask),
        "phase2/teacher_gap_tail_mean": _masked_mean(gap, tail_mask),
    }
    metrics["phase2/teacher_gap_tail_minus_front"] = (
        metrics["phase2/teacher_gap_tail_mean"] - metrics["phase2/teacher_gap_front_mean"]
    )

    if teacher_entropy is not None:
        teacher_entropy = teacher_entropy.detach().float()
        student_entropy = student_entropy.detach().float()
        entropy_gap = teacher_entropy - student_entropy
        metrics["phase2/teacher_entropy_mean"] = _masked_mean(teacher_entropy, token_mask)
        metrics["phase2/teacher_entropy_gap_mean"] = _masked_mean(entropy_gap, token_mask)
        metrics["phase2/teacher_entropy_gap_abs_mean"] = _masked_mean(entropy_gap.abs(), token_mask)

    return metrics


def build_output_health_metrics(
    *,
    student_outputs: list[str],
    reward_breakdowns: list[dict[str, Any]],
    completion_lengths: list[int] | None = None,
    max_completion_length: int | None = None,
) -> dict[str, float]:
    total = max(1, len(student_outputs))
    empty_outputs = 0
    answer_tag_present = 0
    incomplete_answer_tag = 0
    truncated_outputs = 0
    format_success = 0

    for index, output in enumerate(student_outputs):
        text = (output or "").strip()
        if not text:
            empty_outputs += 1
        if _ANSWER_TAG_PATTERN.search(text):
            answer_tag_present += 1
        else:
            has_answer_open = "<answer>" in text
            has_answer_close = "</answer>" in text
            if has_answer_open or has_answer_close:
                incomplete_answer_tag += 1

        has_think_open = "<think>" in text
        has_think_close = "</think>" in text
        has_answer_open = "<answer>" in text
        has_answer_close = "</answer>" in text
        completion_length = None
        if completion_lengths is not None and index < len(completion_lengths):
            completion_length = completion_lengths[index]
        likely_hit_length_cap = (
            max_completion_length is not None
            and completion_length is not None
            and int(completion_length) >= int(max_completion_length)
        )
        if likely_hit_length_cap and ((has_think_open and not has_think_close) or (has_answer_open and not has_answer_close)):
            truncated_outputs += 1

        format_score = None
        if index < len(reward_breakdowns):
            format_score = (reward_breakdowns[index] or {}).get("format_score")
        try:
            if format_score is not None and float(format_score) > 0:
                format_success += 1
        except (TypeError, ValueError):
            pass

    answer_tag_present_fraction = float(answer_tag_present) / float(total)
    format_success_fraction = float(format_success) / float(total)
    return {
        "phase2/answer_tag_present_fraction": answer_tag_present_fraction,
        "phase2/answer_tag_missing_fraction": 1.0 - answer_tag_present_fraction,
        "phase2/incomplete_answer_tag_fraction": float(incomplete_answer_tag) / float(total),
        "phase2/truncated_fraction": float(truncated_outputs) / float(total),
        "phase2/empty_output_fraction": float(empty_outputs) / float(total),
        "phase2/format_success_fraction": format_success_fraction,
        "phase2/format_failure_fraction": 1.0 - format_success_fraction,
    }


def should_reuse_process_feedback_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return _normalize_optional_text(payload.get("feedback")) is not None


def resolve_teacher_forward_indices(
    target_indices: list[int],
    *,
    num_sequences: int,
    force_forward: bool,
) -> list[int]:
    if target_indices:
        return list(target_indices)
    if force_forward and num_sequences > 0:
        return [0]
    return []


def build_phase2_rollout_trace_row(
    *,
    global_step: int,
    sample_id: str | None,
    original_sample_id: str | None,
    rollout_index: int,
    source: str | None,
    task: str | None,
    student_output: str | None,
    reward_breakdown: dict[str, Any] | None,
    reward_metadata: dict[str, Any] | None,
    supervision: dict[str, Any] | None,
    group_state: str | None,
    process_feedback_status: str | None,
    process_feedback_error_type: str | None,
    process_feedback_error_message: str | None,
    process_feedback_result: dict[str, Any] | None,
    vision_error_media_path: str | None = None,
) -> dict[str, Any]:
    reward_breakdown = reward_breakdown or {}
    reward_metadata = reward_metadata or {}
    supervision = supervision or {}
    reward_metadata_feedback = _normalize_optional_text(reward_metadata.get("feedback"))
    supervision_feedback = _normalize_optional_text(supervision.get("feedback_text"))
    reused_reward_payload = should_reuse_process_feedback_payload(reward_metadata)
    judge_feedback = _normalize_optional_text((process_feedback_result or {}).get("feedback")) or (
        reward_metadata_feedback if reused_reward_payload else None
    )
    judge_available = bool(
        judge_feedback
    )

    return {
        "global_step": int(global_step),
        "sample_id": sample_id,
        "original_sample_id": original_sample_id,
        "rollout_index": int(rollout_index),
        "source": source,
        "task": task,
        "student_output": (student_output or "").strip(),
        "reward_total": reward_breakdown.get("reward"),
        "target_score": supervision.get("target_score"),
        "success_metric": supervision.get("success_metric"),
        "answer_semantic_score": reward_breakdown.get("answer_semantic_score"),
        "answer_window_score": reward_breakdown.get("answer_window_score"),
        "answer_box_score": reward_breakdown.get("answer_box_score"),
        "temporal_grounding_score": reward_breakdown.get("temporal_grounding_score"),
        "spatial_grounding_score": reward_breakdown.get("spatial_grounding_score"),
        "format_score": reward_breakdown.get("format_score"),
        "reward_metadata_feedback": reward_metadata_feedback,
        "routing_group_state": group_state,
        "should_target": bool(supervision.get("should_target", False)),
        "has_hindsight_target": bool(supervision.get("has_hindsight_target", False)),
        "teacher_feedback_text": supervision_feedback,
        "judge_available": judge_available,
        "judge_feedback": judge_feedback,
        "process_feedback_status": process_feedback_status,
        "process_feedback_error_type": process_feedback_error_type,
        "process_feedback_error_message": (
            (process_feedback_error_message or "")[:240] if process_feedback_error_message else None
        ),
        "vision_error_media_path": vision_error_media_path,
    }


def build_local_sample_skip_loss(model, device: torch.device) -> torch.Tensor:
    """Return a zero loss that still touches local params so DDP/ZeRO can sync zero grads."""
    zero = None
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        term = parameter.reshape(-1)[:1].sum() * 0.0
        zero = term if zero is None else zero + term
    if zero is None:
        return torch.zeros((), device=device, requires_grad=True)
    return zero


def build_local_sample_skip_token_tensor(
    model,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a zero token tensor that stays attached to local params for aligned backward."""
    zero = build_local_sample_skip_loss(model, device=device).to(dtype=dtype)
    return zero.reshape(1, 1).expand(batch_size, max(int(seq_len), 1))


def build_local_sample_skip_generation_outputs(
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    num_generations: int,
    eos_token_id,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a minimal completion tensor so sample-local generate failures can stay on the zero-loss path."""
    repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    resolved_eos_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
    eos_column = torch.full(
        (repeated_prompt_ids.size(0), 1),
        int(resolved_eos_token_id),
        dtype=repeated_prompt_ids.dtype,
        device=repeated_prompt_ids.device,
    )
    prompt_completion_ids = torch.cat([repeated_prompt_ids, eos_column], dim=1)
    return prompt_completion_ids, repeated_prompt_mask


def extract_phase2_runtime_media_path(input_copy: list[dict[str, Any]] | None) -> str | None:
    if not input_copy:
        return None
    if not isinstance(input_copy[0], dict):
        return None
    content = input_copy[0].get("content") or []
    if not content or not isinstance(content[0], dict):
        return None
    media_path = content[0].get("video") or content[0].get("image")
    if media_path is None:
        return None
    return str(media_path)


def ensure_phase2_runtime_media_exists(input_copy: list[dict[str, Any]] | None) -> None:
    media_path = extract_phase2_runtime_media_path(input_copy)
    if media_path is None:
        return
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"missing media: {media_path}")


def classify_phase2_vision_read_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "vision_media_missing"

    message = str(exc).lower()
    missing_markers = (
        "no such file or directory",
        "does not exist",
        "not found",
        "missing media",
    )
    if any(marker in message for marker in missing_markers):
        return "vision_media_missing"

    decode_markers = (
        "threaded_decoder",
        "resource temporarily unavailable",
        "vision decode failed",
        "error sending packet",
        "failed initializing scaling graph",
        "cannot decode",
        "decode failed",
        "error while decoding",
    )
    if any(marker in message for marker in decode_markers):
        return "vision_decode_failed"
    return "vision_unknown_failed"


def prune_none_key_items(example: dict[str, Any]) -> None:
    if "key_items" not in example:
        return
    keys_to_remove = []
    for key, item in example["key_items"].items():
        if item is None:
            keys_to_remove.append(key)
        elif isinstance(item, dict):
            sub_keys_to_remove = [k for k, v in item.items() if v is None]
            for k in sub_keys_to_remove:
                del item[k]
    for key in keys_to_remove:
        del example["key_items"][key]


def build_phase2_runtime_prompt_state(processing_class, example: dict[str, Any]) -> tuple[list[Any], list[str], list[dict[str, Any]]]:
    prompts = [example["prompt"]]
    prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"]]
    input_copy = [copy.deepcopy(example["prompt"][1])]

    if example["source"] == "videoespresso_train_video":
        input_copy[0]["content"][0]["video"] = os.path.join(VIDEO_ESPRESSO_ROOT, example["video_path"])
    elif example["source"] == "timerft":
        input_copy[0]["content"][0]["video"] = os.path.join(TIMERFT_ROOT, example["video_path"])
    elif example["source"] == "gqa":
        input_copy[0]["content"][0]["image"] = os.path.join(GQA_ROOT, example["image_path"])
    elif "STR" in example["source"]:
        video_root = STR_PLM_DATA if "STR_plm" in example["source"] else STR_DATA
        input_copy[0]["content"][0]["video"] = os.path.join(video_root, example["video_path"])
    elif "TVG" in example["source"]:
        input_copy[0]["content"][0]["video"] = os.path.join(TVG_ROOT, example["video_path"])
    elif "videor1" in example["source"]:
        input_copy[0]["content"][0]["video"] = os.path.join(GENERAL_VIDEO_ROOT, example["video_path"])
    else:
        raise ValueError(f"Invalid source: {example['source']}")

    return prompts, prompts_text, input_copy


def build_phase2_text_only_skip_prompt_state(
    example: dict[str, Any],
    *,
    vision_error_type: str | None,
    vision_error_message: str | None,
) -> tuple[list[dict[str, Any]], str]:
    question = _normalize_optional_text(example.get("question")) or "No question was provided."
    sample_id = _normalize_optional_text(example.get("phase2_runtime_original_id")) or _normalize_optional_text(
        example.get("id")
    ) or "unknown_sample"
    source = _normalize_optional_text(example.get("phase2_runtime_original_source")) or _normalize_optional_text(
        example.get("source")
    ) or "unknown_source"
    task = _normalize_optional_text(example.get("phase2_runtime_original_task")) or _normalize_optional_text(
        example.get("task")
    ) or "unknown_task"
    error_type = _normalize_optional_text(vision_error_type) or "vision_read_failed"
    error_message = _normalize_optional_text(vision_error_message) or "unknown vision read failure"
    prompt_text = (
        "This training sample hit a local vision read failure and must be skipped for optimization.\n"
        f"sample_id: {sample_id}\n"
        f"source: {source}\n"
        f"task: {task}\n"
        f"vision_error_type: {error_type}\n"
        f"vision_error_message: {error_message[:240]}\n"
        f"question: {question}\n"
        "Reply exactly with <answer>skip</answer>."
    )
    prompt = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    return prompt, prompt_text


def build_dummy_vision_inputs_for_failure(
    example: dict[str, Any],
) -> tuple[list[Image.Image] | None, list[torch.Tensor] | None, dict[str, Any]]:
    """Keep failed samples on a vision-enabled branch so ZeRO-3 sees aligned module usage."""
    source = str(example.get("source") or "").lower()
    if source == "gqa":
        return [build_dummy_image_input({})], None, {}
    dummy_video, sample_fps = build_dummy_video_input({}, return_video_sample_fps=True)
    return None, [dummy_video], {"fps": [sample_fps]}


def resolve_keyframe_fallback_image_size(image_size: tuple[int, int] | None) -> tuple[int, int]:
    if image_size is None:
        return int(MIN_DUMMY_VIDEO_WIDTH), int(MIN_DUMMY_VIDEO_HEIGHT)
    width, height = image_size
    return int(width), int(height)


def build_dummy_keyframe_tensor(image_size: tuple[int, int] | None) -> torch.Tensor:
    width, height = resolve_keyframe_fallback_image_size(image_size)
    return torch.zeros((3, int(height), int(width)), dtype=torch.uint8)


def load_keyframe_tensors_with_fallback(
    *,
    example: dict[str, Any],
    image_size: tuple[int, int] | None,
    key_frame_root: str,
) -> tuple[list[tuple[int, torch.Tensor]], list[dict[str, Any]]]:
    resolved_image_size = resolve_keyframe_fallback_image_size(image_size)
    key_frames = []
    failed_keyframes: list[dict[str, Any]] = []
    for key_frame_index, key_frame in enumerate(example.get("key_frames", [])):
        kf_path = os.path.join(key_frame_root, key_frame["path"])
        try:
            kf = Image.open(kf_path).convert("RGB")
            resized_kf = kf.resize(resolved_image_size)
            resized_kf = np.array(resized_kf)
            resized_kf = np.transpose(resized_kf, (2, 0, 1))
            kf_tensor = torch.from_numpy(resized_kf)
        except Exception as exc:
            kf_tensor = build_dummy_keyframe_tensor(resolved_image_size)
            failed_keyframes.append(
                {
                    "key_frame_index": key_frame_index,
                    "kind": "image",
                    "path": key_frame.get("path"),
                    "vision_error_type": classify_phase2_vision_read_error(exc),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        key_frames.append((round(key_frame["time"]), kf_tensor))
    return key_frames, failed_keyframes


def build_vision_read_failed_supervision_rows(
    *,
    num_sequences: int,
    vision_error_message: str | None,
    vision_error_type: str,
    success_reward_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    rows = []
    for _ in range(num_sequences):
        rows.append(
            {
                "reward_breakdown": {
                    "reward": 0.0,
                    "answer_semantic_score": 0.0,
                    "answer_window_score": 0.0,
                    "answer_box_score": 0.0,
                    "temporal_grounding_score": 0.0,
                    "spatial_grounding_score": 0.0,
                    "format_score": 0.0,
                },
                "reward_metadata": {},
                "supervision": {
                    "should_target": False,
                    "has_hindsight_target": False,
                    "target_score": 0.0,
                    "success_metric": "vision_read_failed",
                    "feedback_text": None,
                    "judge_metadata": {},
                },
                "process_feedback_status": f"skipped_{vision_error_type}",
                "process_feedback_error_type": vision_error_type,
                "process_feedback_error_message": vision_error_message,
                "process_feedback_result": None,
            }
        )
    requested_routing = build_phase2_requested_masks(
        supervision_rows=rows,
        device=torch.device("cpu"),
        dtype=torch.float32,
        success_reward_threshold=success_reward_threshold,
    )
    metrics = {
        "phase2/vision_read_failed_local_fraction": 1.0,
        "phase2/vision_read_failed_loss_skipped_fraction": 1.0,
        "phase2/vision_media_missing_local_fraction": 1.0 if vision_error_type == "vision_media_missing" else 0.0,
        "phase2/vision_decode_failed_local_fraction": 1.0 if vision_error_type == "vision_decode_failed" else 0.0,
        "phase2/vision_unknown_failed_local_fraction": 1.0 if vision_error_type == "vision_unknown_failed" else 0.0,
        "phase2/process_feedback_enabled": 0.0,
        "phase2/process_feedback_requested_fraction": 0.0,
        "phase2/process_feedback_available_fraction": 0.0,
        "phase2/process_feedback_error_fraction": 0.0,
        "phase2/process_feedback_reused_from_reward_fraction": 0.0,
        "phase2/judge_feedback_available_fraction": 0.0,
        "phase2/group_avg_reward": 0.0,
    }
    metrics.update(requested_routing["metrics"])
    return rows, requested_routing, metrics


def should_force_vision_read_fail(
    *,
    local_rank: int,
    current_step: int,
    consumed: bool,
) -> tuple[bool, bool]:
    """Debug-only hook for deterministic vision-read failure injection."""
    target_rank_raw = os.getenv("PHASE2_DEBUG_FORCE_VISION_FAIL_LOCAL_RANK")
    if target_rank_raw is None or str(target_rank_raw).strip() == "":
        return False, consumed

    try:
        target_rank = int(str(target_rank_raw).strip())
    except (TypeError, ValueError):
        return False, consumed
    if local_rank != target_rank:
        return False, consumed

    target_step_raw = os.getenv("PHASE2_DEBUG_FORCE_VISION_FAIL_STEP")
    if target_step_raw is not None and str(target_step_raw).strip() != "":
        try:
            target_step = int(str(target_step_raw).strip())
        except (TypeError, ValueError):
            return False, consumed
        if current_step != target_step:
            return False, consumed

    once_raw = str(os.getenv("PHASE2_DEBUG_FORCE_VISION_FAIL_ONCE", "true")).strip().lower()
    fail_once = once_raw in {"1", "true", "yes", "y", "on"}
    if fail_once and consumed:
        return False, consumed
    return True, (consumed or fail_once)


def repeat_prompt_vision_inputs(
    prompt_inputs: dict[str, Any],
    *,
    num_repeats: int,
) -> dict[str, Any]:
    """Repeat whichever vision tensors are actually present in prompt_inputs."""
    if "pixel_values" in prompt_inputs and "image_grid_thw" in prompt_inputs:
        prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(num_repeats, 1)
        prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(num_repeats, 1)
    elif "pixel_values_videos" in prompt_inputs and "video_grid_thw" in prompt_inputs:
        prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(num_repeats, 1)
        prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(num_repeats, 1)
    return prompt_inputs


def phase2_debug_progress_enabled() -> bool:
    value = str(os.getenv("PHASE2_DEBUG_PROGRESS", "")).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def phase2_debug_progress(message: str, *, local_rank: int | None = None) -> None:
    if not phase2_debug_progress_enabled():
        return
    rank = local_rank if local_rank is not None else -1
    print(f"[Phase2Debug][local_rank={rank}] {message}", flush=True)


class Qwen2VLGRPOTrainerVISD(Qwen2VLGRPOTrainer):
    """GSPO backbone plus a separate same-response phase2 hindsight teacher branch."""

    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        script_args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        max_pixels=12845056,
        min_pixels=3136,
        attn_implementation="flash_attention_2",
        gspo=True,
        phase2_enable_ref_model: bool = True,
        phase2_enable_teacher_model: bool = False,
        phase2_teacher_update_mode: str = "off",
        phase2_teacher_update_rate: float = 0.05,
        phase2_teacher_update_interval: int = 10,
        phase2_importance_sampling_level: Optional[str] = None,
        phase2_teacher_feedback_text: Optional[str] = None,
        phase2_base_reward_mode: str = "native",
        phase2_teacher_success_metric: str = "task_aware",
        phase2_inject_teacher_signal: bool = True,
        phase2_reweighting_mixing_lambda: float = 1.0,
        phase2_reweighting_weight_mode: str = "sampled",
        phase2_reweighting_topk: int = 8,
        phase2_reweighting_topk_gamma: float = 1.0,
        phase2_reweighting_weight_clip: Optional[float] = 0.2,
        phase2_teacher_max_prompt_length: int = 18432,
        phase2_process_feedback_enable: bool = False,
        phase2_process_feedback_timeout: float = 30.0,
        phase2_process_feedback_scope: str = "target_only",
        phase2_process_feedback_model: Optional[str] = None,
        phase2_process_feedback_base_url: Optional[str] = None,
        phase2_process_feedback_api_key: Optional[str] = None,
        phase2_process_feedback_max_feedback_chars: int = 1000,
        phase2_teacher_gamma: float = 1.0,
        phase2_reward_metadata_enable: bool = True,
        phase2_rollout_trace_enable: bool = False,
        phase2_rollout_trace_wandb_steps: int = 10,
        phase2_rollout_trace_max_groups: int = 1,
        **legacy_kwargs,
    ):
        phase2_importance_sampling_level = normalize_importance_sampling_level(
            phase2_importance_sampling_level,
            gspo_default=bool(gspo),
        )
        if legacy_kwargs:
            unknown_keys = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected phase2 trainer kwargs: {unknown_keys}")

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            script_args=script_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            attn_implementation=attn_implementation,
            gspo=phase2_importance_sampling_level != "token",
            phase2_enable_ref_model=phase2_enable_ref_model,
            phase2_enable_teacher_model=phase2_enable_teacher_model,
            phase2_teacher_update_mode=phase2_teacher_update_mode,
            phase2_teacher_update_rate=phase2_teacher_update_rate,
            phase2_teacher_update_interval=phase2_teacher_update_interval,
        )
        self.phase2_importance_sampling_level = phase2_importance_sampling_level
        self.phase2_enable_ref_model = bool(phase2_enable_ref_model)
        self.phase2_enable_teacher_model = bool(phase2_enable_teacher_model)
        self.phase2_teacher_update_mode = str(phase2_teacher_update_mode or "off").lower()
        self.phase2_teacher_update_rate = float(phase2_teacher_update_rate)
        self.phase2_teacher_update_interval = max(1, int(phase2_teacher_update_interval))
        self.phase2_teacher_feedback_text = phase2_teacher_feedback_text
        self.phase2_base_reward_mode = str(phase2_base_reward_mode or "native").lower()
        self.phase2_teacher_success_metric = str(phase2_teacher_success_metric or "task_aware").lower()
        self.phase2_inject_teacher_signal = bool(phase2_inject_teacher_signal)
        self.phase2_reweighting_mixing_lambda = float(phase2_reweighting_mixing_lambda)
        self.phase2_reweighting_weight_mode = str(phase2_reweighting_weight_mode or "sampled").lower()
        self.phase2_reweighting_topk = max(1, int(phase2_reweighting_topk))
        self.phase2_reweighting_topk_gamma = float(phase2_reweighting_topk_gamma)
        if self.phase2_reweighting_weight_mode not in {"sampled", "topk_interpolate"}:
            raise ValueError(
                "phase2_reweighting_weight_mode must be 'sampled' or 'topk_interpolate', "
                f"got {phase2_reweighting_weight_mode!r}"
            )
        if not 0.0 <= self.phase2_reweighting_topk_gamma <= 1.0:
            raise ValueError(
                "phase2_reweighting_topk_gamma must be within [0, 1], "
                f"got {self.phase2_reweighting_topk_gamma}"
            )
        self.phase2_reweighting_weight_clip = phase2_reweighting_weight_clip
        self.phase2_teacher_max_prompt_length = int(phase2_teacher_max_prompt_length)
        self.phase2_process_feedback_enable = bool(phase2_process_feedback_enable)
        self.phase2_process_feedback_timeout = float(phase2_process_feedback_timeout)
        self.phase2_process_feedback_scope = str(phase2_process_feedback_scope or "target_only").lower()
        self.phase2_process_feedback_model = (
            str(phase2_process_feedback_model).strip() if phase2_process_feedback_model else None
        )
        self.phase2_process_feedback_base_url = (
            str(phase2_process_feedback_base_url).strip() if phase2_process_feedback_base_url else None
        )
        self.phase2_process_feedback_api_key = (
            str(phase2_process_feedback_api_key).strip() if phase2_process_feedback_api_key else None
        )
        self.phase2_process_feedback_max_feedback_chars = int(phase2_process_feedback_max_feedback_chars)
        self.phase2_teacher_gamma = float(phase2_teacher_gamma)
        self.phase2_reward_metadata_enable = bool(phase2_reward_metadata_enable)
        self.phase2_rollout_trace_enable = bool(phase2_rollout_trace_enable)
        self.phase2_rollout_trace_wandb_steps = max(1, int(phase2_rollout_trace_wandb_steps))
        self.phase2_rollout_trace_max_groups = max(1, int(phase2_rollout_trace_max_groups))

        print(
            "[Phase2Teacher] Enabled VISD feedback-conditioned teacher replay: "
            f"importance_sampling_level={self.phase2_importance_sampling_level}, "
            "teacher_signal_mode=reweighting, "
            f"reweighting_weight_mode={self.phase2_reweighting_weight_mode}, "
            f"base_reward_mode={self.phase2_base_reward_mode}"
        )
        self._phase2_rollout_trace_path = None

    def _get_phase2_rollout_trace_rank(self) -> int:
        accelerator = getattr(self, "accelerator", None)
        rank = getattr(accelerator, "process_index", None)
        if rank is None:
            rank = getattr(getattr(self, "args", None), "process_index", None)
        if rank is None:
            rank = os.environ.get("RANK")
        try:
            return int(rank)
        except (TypeError, ValueError):
            return 0

    def _resolve_phase2_rollout_trace_path(self) -> str:
        if self._phase2_rollout_trace_path:
            return self._phase2_rollout_trace_path
        rank = self._get_phase2_rollout_trace_rank()
        output_dir = getattr(getattr(self, "args", None), "output_dir", None)
        if output_dir:
            trace_path = os.path.join(output_dir, f"phase2_rollout_trace.rank{rank}.jsonl")
        else:
            trace_path = os.path.join(os.getcwd(), "tmp_test", f"phase2_rollout_trace.rank{rank}.jsonl")
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        self._phase2_rollout_trace_path = trace_path
        return trace_path

    def _append_phase2_rollout_trace_rows(self, trace_rows: list[dict[str, Any]]) -> None:
        if not getattr(self, "phase2_rollout_trace_enable", False) or not trace_rows:
            return
        trace_path = self._resolve_phase2_rollout_trace_path()
        with open(trace_path, "a", encoding="utf-8") as handle:
            for row in trace_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _log_phase2_rollout_trace_to_wandb(self, trace_rows: list[dict[str, Any]], *, step: int) -> None:
        if (
            not getattr(self, "phase2_rollout_trace_enable", False)
            or not trace_rows
            or not self.is_world_process_zero()
            or not is_wandb_available()
        ):
            return
        if wandb.run is None:
            return
        trace_wandb_steps = max(1, int(getattr(self, "phase2_rollout_trace_wandb_steps", 10)))
        if step % trace_wandb_steps != 0:
            return
        max_trace_groups = max(1, int(getattr(self, "phase2_rollout_trace_max_groups", 1)))
        max_rows = max(1, max_trace_groups * int(self.num_generations))
        trace_rows = trace_rows[:max_rows]
        columns = [
            "sample_id",
            "rollout_index",
            "source",
            "task",
            "reward_total",
            "target_score",
            "answer_semantic_score",
            "format_score",
            "routing_group_state",
            "should_target",
            "judge_available",
            "process_feedback_status",
            "process_feedback_error_type",
            "student_output",
            "judge_feedback",
        ]
        data = [
            [
                row.get("sample_id"),
                row.get("rollout_index"),
                row.get("source"),
                row.get("task"),
                row.get("reward_total"),
                row.get("target_score"),
                row.get("answer_semantic_score"),
                row.get("format_score"),
                row.get("routing_group_state"),
                row.get("should_target"),
                row.get("judge_available"),
                row.get("process_feedback_status"),
                row.get("process_feedback_error_type"),
                row.get("student_output"),
                row.get("judge_feedback"),
            ]
            for row in trace_rows
        ]
        wandb.log({"rollout/latest_samples": wandb.Table(columns=columns, data=data)}, step=step)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        validate_phase2_host_batch(inputs)
        inputs = [copy.deepcopy(example) for example in inputs]
        local_rank = int(getattr(self.accelerator, "local_process_index", -1))
        prune_none_key_items(inputs[0])
        prompts, prompts_text, input_copy = build_phase2_runtime_prompt_state(self.processing_class, inputs[0])
        input_copy = self.remove_none_from_data(input_copy)
        original_sample_id = inputs[0].get("id")
        original_source = inputs[0].get("source")
        original_task = inputs[0].get("task")
        original_media_path = extract_phase2_runtime_media_path(input_copy)

        current_step = self.state.global_step + 1
        total_steps = self.state.max_steps
        vision_read_failed = False
        vision_error_type = None
        vision_error_message = None
        try:
            ensure_phase2_runtime_media_exists(input_copy)
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            force_vision_fail, consumed_flag = should_force_vision_read_fail(
                local_rank=int(getattr(self.accelerator, "local_process_index", 0)),
                current_step=int(current_step),
                consumed=bool(getattr(self, "_phase2_debug_force_vision_fail_consumed", False)),
            )
            if force_vision_fail:
                self._phase2_debug_force_vision_fail_consumed = consumed_flag
                raise RuntimeError(
                    f"debug forced vision read failure at local_rank="
                    f"{int(getattr(self.accelerator, 'local_process_index', 0))}, step={int(current_step)}"
                )
            if image_inputs is not None:
                inputs[0]["image_size_refine"] = (image_inputs[0].size[0], image_inputs[0].size[1])
                inputs[0]["prompt_text_final"] = prompts_text[0]
            if video_inputs is not None:
                inputs[0]["video_sample_fps"] = video_kwargs["fps"][0]
                inputs[0]["video_duration"] = video_inputs[0].size(0) / video_kwargs["fps"][0]
                inputs[0]["image_size"] = (video_inputs[0].size(3), video_inputs[0].size(2))
                inputs[0]["prompt_text_final"] = prompts_text[0]
        except Exception as exc:
            vision_read_failed = True
            vision_error_type = classify_phase2_vision_read_error(exc)
            vision_error_message = str(exc)
            print(
                "process_vision_info error, using text-only skip sample, "
                f"vision_error_type={vision_error_type}, original_sample_id={original_sample_id}, "
                f"media_path={original_media_path}, error={exc}"
            )
            image_inputs, video_inputs, video_kwargs = build_dummy_vision_inputs_for_failure(inputs[0])
            inputs[0]["prompt_text_final"] = prompts_text[0]
            if image_inputs is not None:
                inputs[0]["image_size_refine"] = (image_inputs[0].size[0], image_inputs[0].size[1])
            if video_inputs is not None:
                inputs[0]["video_sample_fps"] = video_kwargs["fps"][0]
                inputs[0]["video_duration"] = video_inputs[0].size(0) / video_kwargs["fps"][0]
                inputs[0]["image_size"] = (video_inputs[0].size(3), video_inputs[0].size(2))
        phase2_debug_progress(
            f"after process_vision_info vision_read_failed={vision_read_failed} "
            f"has_image={image_inputs is not None} has_video={video_inputs is not None}",
            local_rank=local_rank,
        )
        if vision_read_failed:
            text_only_prompt, text_only_prompt_text = build_phase2_text_only_skip_prompt_state(
                inputs[0],
                vision_error_type=vision_error_type,
                vision_error_message=vision_error_message,
            )
            prompts = [text_only_prompt]
            prompts_text = [text_only_prompt_text]
            inputs[0]["prompt"] = text_only_prompt
            inputs[0]["prompt_text_final"] = text_only_prompt_text
            inputs[0]["phase2_runtime_text_only_skip"] = True
            image_inputs = None
            video_inputs = None
            video_kwargs = {}
        inputs[0]["step_percent"] = current_step / total_steps

        multi_image = video_inputs is not None
        if multi_image:
            if vision_read_failed or inputs[0]["task"] != "temporal-spatial free-form QA":
                frame_prompt = ""
                ori_idx = 0
                while ori_idx < len(video_inputs[0]):
                    time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                    frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                    ori_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                prompts_text[0] = prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )
                inputs[0]["prompt_text_final"] = prompts_text[0]
                image_inputs = [video_inputs[0]]
            else:
                width, height = video_inputs[0].size(3), video_inputs[0].size(2)
                image_size = (width, height)

                if inputs[0]["source"] == "videoespresso_train_video":
                    key_frame_root = VIDEO_ESPRESSO_KF_ROOT
                elif "STR_plm" in inputs[0]["source"]:
                    key_frame_root = STR_PLM_KF_ROOT
                else:
                    key_frame_root = STR_KF_ROOT

                key_frames, failed_keyframes = load_keyframe_tensors_with_fallback(
                    example=inputs[0],
                    image_size=image_size,
                    key_frame_root=key_frame_root,
                )
                if failed_keyframes:
                    first_failed_keyframe = failed_keyframes[0]
                    vision_read_failed = True
                    vision_error_type = str(first_failed_keyframe.get("vision_error_type") or "vision_decode_failed")
                    vision_error_message = (
                        f"{len(failed_keyframes)} keyframe image(s) replaced with dummy tensors; "
                        f"first_failed_path={first_failed_keyframe.get('path')}, "
                        f"first_error={first_failed_keyframe.get('error_message')}"
                    )
                    print(
                        "process_vision_info keyframe fallback replaced failed keyframes, "
                        f"original_sample_id={original_sample_id}, failed_keyframes={failed_keyframes}"
                    )

                frame_prompt = ""
                refined_image_inputs = []
                kf_idx = 0
                ori_idx = 0
                frame_idx = 1
                while ori_idx < len(video_inputs[0]):
                    time_now = int(ori_idx / video_kwargs["fps"][0])
                    if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                        refined_image_inputs.append(key_frames[kf_idx][1])
                        time_now = round(key_frames[kf_idx][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        kf_idx += 1
                    else:
                        refined_image_inputs.append(video_inputs[0][ori_idx])
                        time_now = round(ori_idx / video_kwargs["fps"][0], 1)
                        frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        ori_idx += 1
                    frame_idx += 1
                frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                image_inputs = [torch.stack(refined_image_inputs)]
                prompts_text[0] = prompts_text[0].replace(
                    "<|vision_start|><|video_pad|><|vision_end|>", frame_prompt
                )
                inputs[0]["prompt_text_final"] = prompts_text[0]

        if multi_image:
            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs,
            )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_inputs = self._move_prompt_inputs_to_device(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        local_sample_skip_error_type = "vision_read_failed" if vision_read_failed else None
        local_sample_skip_error_message = vision_error_message if vision_read_failed else None

        with temporarily_set_model_eval(model):
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                phase2_debug_progress("before generate", local_rank=local_rank)
                local_generate_failed = False
                local_generate_failed_message = None
                try:
                    prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
                    phase2_debug_progress("after generate", local_rank=local_rank)
                except Exception as exc:
                    print(f"Error during student generate: {exc}. Zeroing only this local sample.")
                    local_generate_failed = True
                    local_generate_failed_message = str(exc)

                generation_failed_globally = local_generate_failed
                if dist.is_available() and dist.is_initialized():
                    generation_failed_tensor = torch.tensor(
                        [1 if local_generate_failed else 0],
                        device=prompt_ids.device,
                        dtype=torch.int64,
                    )
                    dist.all_reduce(generation_failed_tensor, op=dist.ReduceOp.MAX)
                    generation_failed_globally = bool(generation_failed_tensor.item())

                if generation_failed_globally:
                    if not local_generate_failed:
                        print("Peer rank hit student generate error. Zeroing this local sample to keep ranks aligned.")
                    local_sample_skip_error_type = "generation_failed"
                    local_sample_skip_error_message = local_generate_failed_message or "peer rank hit student generate failure"
                    prompt_completion_ids, prompt_mask = build_local_sample_skip_generation_outputs(
                        prompt_ids,
                        prompt_mask,
                        num_generations=self.num_generations,
                        eos_token_id=self.processing_class.eos_token_id,
                    )
                    phase2_debug_progress("after generate global fallback", local_rank=local_rank)
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if local_sample_skip_error_type == "generation_failed":
            vision_read_failed = True
            vision_error_type = vision_error_type or "vision_unknown_failed"
            vision_error_message = vision_error_message or local_sample_skip_error_message
            supervision_rows, requested_routing, supervision_metrics = build_vision_read_failed_supervision_rows(
                num_sequences=completion_ids.size(0),
                vision_error_message=local_sample_skip_error_message,
                vision_error_type=vision_error_type,
                success_reward_threshold=self._get_phase2_success_reward_threshold(),
            )
            final_loss = build_local_sample_skip_loss(model, device=device)
            self._metrics["completion_length"].append(
                float(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
            )
            vision_failed_tensor = torch.tensor([1.0], device=device, dtype=torch.float32)
            vision_missing_tensor = torch.tensor(
                [1.0 if vision_error_type == "vision_media_missing" else 0.0],
                device=device,
                dtype=torch.float32,
            )
            vision_decode_tensor = torch.tensor(
                [1.0 if vision_error_type == "vision_decode_failed" else 0.0],
                device=device,
                dtype=torch.float32,
            )
            vision_unknown_tensor = torch.tensor(
                [1.0 if vision_error_type == "vision_unknown_failed" else 0.0],
                device=device,
                dtype=torch.float32,
            )
            self._metrics["phase2/vision_read_failed_count"].append(
                float(self.accelerator.gather_for_metrics(vision_failed_tensor).sum().item())
            )
            self._metrics["phase2/vision_media_missing_count"].append(
                float(self.accelerator.gather_for_metrics(vision_missing_tensor).sum().item())
            )
            self._metrics["phase2/vision_decode_failed_count"].append(
                float(self.accelerator.gather_for_metrics(vision_decode_tensor).sum().item())
            )
            self._metrics["phase2/vision_unknown_failed_count"].append(
                float(self.accelerator.gather_for_metrics(vision_unknown_tensor).sum().item())
            )
            for key, value in supervision_metrics.items():
                self._metrics[key].append(value)
            phase2_debug_progress(
                f"returning early generation_failed loss={float(final_loss.detach().item())}",
                local_rank=local_rank,
            )
            return final_loss

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        prompt_inputs = repeat_prompt_vision_inputs(
            prompt_inputs,
            num_repeats=len(prompt_completion_ids),
        )
        prompt_inputs.pop("second_per_grid_ts", None)

        local_sample_skip_error_type = "vision_read_failed" if vision_read_failed else None
        local_sample_skip_error_message = vision_error_message if vision_read_failed else None

        try:
            phase2_debug_progress("before student logps", local_rank=local_rank)
            per_token_logps, per_token_entropy = self._get_per_token_logps_and_entropy(
                model,
                prompt_completion_ids,
                logits_to_keep=completion_ids.size(1),
                **prompt_inputs,
            )
            phase2_debug_progress("after student logps", local_rank=local_rank)
        except Exception as exc:
            print(f"Error computing per_token_logps: {exc}. Zeroing only this local sample.")
            local_sample_skip_error_type = "student_logps_failed"
            local_sample_skip_error_message = str(exc)
            full_seq_len = completion_ids.size(1)
            per_token_logps = build_local_sample_skip_token_tensor(
                model,
                batch_size=prompt_completion_ids.size(0),
                seq_len=full_seq_len,
                device=device,
                dtype=torch.float32,
            )[:, prompt_length - 1 :]
            per_token_entropy = torch.zeros_like(per_token_logps)

        with torch.inference_mode():
            if self.phase2_enable_ref_model:
                try:
                    phase2_debug_progress("before ref logps", local_rank=local_rank)
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(
                            self.ref_model,
                            prompt_completion_ids,
                            logits_to_keep=completion_ids.size(1),
                            **prompt_inputs,
                        )
                    else:
                        with self.accelerator.unwrap_model(model).disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                model,
                                prompt_completion_ids,
                                logits_to_keep=completion_ids.size(1),
                                **prompt_inputs,
                            )
                    phase2_debug_progress("after ref logps", local_rank=local_rank)
                    x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
                    per_token_kl = torch.exp(x_clamped) - x_clamped - 1
                except Exception as exc:
                    print(f"Error computing ref_per_token_logps: {exc}. Zeroing only this local sample.")
                    local_sample_skip_error_type = "ref_logps_failed"
                    local_sample_skip_error_message = str(exc)
                    per_token_kl = torch.zeros_like(per_token_logps)
            else:
                per_token_kl = torch.zeros_like(per_token_logps)

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        prompts_repeated = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts_repeated), len(self.reward_funcs), device=device)
        reward_metadata_rows = [{} for _ in range(len(prompts_repeated))]
        local_sample_skip_required = local_sample_skip_error_type is not None
        if not local_sample_skip_required:
            for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                if self.script_args is not None:
                    reward_kwargs["spatial_iou_mode"] = getattr(self.script_args, "spatial_iou_mode", "max")
                    reward_kwargs["identity_match_mode"] = getattr(self.script_args, "identity_match_mode", "none")
                    reward_kwargs["spatial_norm_mode"] = getattr(self.script_args, "spatial_norm_mode", "all")
                    reward_kwargs["correct_tempgate"] = getattr(self.script_args, "correct_tempgate", True)
                reward_kwargs["return_metadata"] = self.phase2_reward_metadata_enable
                output_reward_func = reward_func(prompts=prompts_repeated, completions=completions, **reward_kwargs)
                reward_scores, reward_metadata_part = normalize_reward_func_output(output_reward_func)
                rewards_per_func[:, i] = torch.tensor(reward_scores, dtype=torch.float32, device=device)
                reward_metadata_rows = merge_reward_metadata_rows(
                    reward_metadata_rows,
                    reward_metadata_part,
                    expected_length=len(prompts_repeated),
                )

        rewards_per_func = self._apply_reward_ablation_mask(rewards_per_func)
        if self.phase2_base_reward_mode in {"none", "teacher_only", "disabled"}:
            rewards_per_func = torch.zeros_like(rewards_per_func)
            self._metrics["phase2/base_reward_disabled"].append(1.0)
        else:
            self._metrics["phase2/base_reward_disabled"].append(0.0)
        advantages, rewards, mean_grouped_rewards, std_grouped_rewards = self._compute_rollout_advantages(
            rewards_per_func,
        )
        response_mask = completion_mask[:, : per_token_logps.size(1)].to(dtype=per_token_logps.dtype)
        base_token_advantages = advantages.unsqueeze(1).expand(-1, response_mask.size(1)).to(dtype=per_token_logps.dtype)
        base_token_advantages = base_token_advantages * response_mask

        log_importance_weights = build_log_importance_weights(
            per_token_logps=per_token_logps,
            response_mask=response_mask,
            importance_sampling_level=self.phase2_importance_sampling_level,
        )

        response_entropy_metrics = build_response_entropy_metrics(
            response_entropy=per_token_entropy[:, : response_mask.size(1)],
            response_mask=response_mask,
        )
        for key, value in response_entropy_metrics.items():
            self._metrics[key].append(value)

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        reward_names = [get_reward_func_name(reward_func) for reward_func in self.reward_funcs]
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_names[i]
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        rollout_metrics = build_rollout_monitor_metrics(
            rewards_per_func=rewards_per_func,
            reward_names=reward_names,
            num_generations=self.num_generations,
            success_reward_threshold=self._get_phase2_success_reward_threshold(),
        )
        for key, value in rollout_metrics.items():
            self._metrics[key].append(value)

        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        self._metrics["all_wrong"].append(wrong_devices.sum().item() / num_devices)
        self._metrics["all_correct"].append(correct_devices.sum().item() / num_devices)

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        student_outputs_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        privileged_info = build_privileged_info(inputs[0])
        if local_sample_skip_required:
            supervision_rows, requested_routing, supervision_metrics = build_vision_read_failed_supervision_rows(
                num_sequences=completion_ids.size(0),
                vision_error_message=local_sample_skip_error_message,
                vision_error_type=vision_error_type or "vision_read_failed",
                success_reward_threshold=self._get_phase2_success_reward_threshold(),
            )
        else:
            supervision_rows, requested_routing, supervision_metrics = self._build_phase2_supervision_rows(
                example=inputs[0],
                privileged_info=privileged_info,
                rewards_per_func=rewards_per_func,
                reward_metadata=reward_metadata_rows,
                student_outputs=student_outputs_text,
            )
        teacher_branch = self._compute_phase2_teacher_branch(
            model=model,
            final_prompt_text=inputs[0].get("prompt_text_final") or prompts_text[0],
            completion_ids=completion_ids,
            completion_mask=response_mask,
            student_per_token_logps=per_token_logps[:, : response_mask.size(1)],
            student_per_token_entropy=per_token_entropy[:, : response_mask.size(1)],
            base_token_advantages=base_token_advantages,
            rewards=rewards,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            video_kwargs=video_kwargs,
            multi_image=multi_image,
            privileged_info=privileged_info,
            supervision_rows=supervision_rows,
            requested_routing=requested_routing,
        )
        phase2_debug_progress("after teacher branch", local_rank=local_rank)
        trace_rows = [
            build_phase2_rollout_trace_row(
                global_step=current_step,
                sample_id=inputs[0].get("phase2_runtime_original_id") or inputs[0].get("id"),
                original_sample_id=inputs[0].get("phase2_runtime_original_id"),
                rollout_index=index,
                source=inputs[0].get("phase2_runtime_original_source") or inputs[0].get("source"),
                task=inputs[0].get("phase2_runtime_original_task") or inputs[0].get("task"),
                student_output=student_outputs_text[index] if index < len(student_outputs_text) else None,
                reward_breakdown=row.get("reward_breakdown"),
                reward_metadata=row.get("reward_metadata"),
                supervision=row.get("supervision"),
                group_state=requested_routing.get("group_state"),
                process_feedback_status=row.get("process_feedback_status"),
                process_feedback_error_type=row.get("process_feedback_error_type"),
                process_feedback_error_message=row.get("process_feedback_error_message"),
                process_feedback_result=row.get("process_feedback_result"),
                vision_error_media_path=inputs[0].get("phase2_runtime_original_media_path"),
            )
            for index, row in enumerate(supervision_rows)
        ]
        output_health_metrics = build_output_health_metrics(
            student_outputs=student_outputs_text,
            reward_breakdowns=[row.get("reward_breakdown") or {} for row in supervision_rows],
            completion_lengths=completion_mask.sum(dim=1).detach().cpu().tolist(),
            max_completion_length=self.max_completion_length,
        )
        for key, value in output_health_metrics.items():
            self._metrics[key].append(value)
        self._append_phase2_rollout_trace_rows(trace_rows)
        self._log_phase2_rollout_trace_to_wandb(trace_rows, step=current_step)

        effective_advantages = teacher_branch["effective_advantages"]
        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * effective_advantages
        per_token_loss2 = coef_2 * effective_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss + self.beta * per_token_kl[:, : effective_advantages.size(1)]
        policy_loss = ((per_token_loss * response_mask).sum(-1) / response_mask.sum(-1).clamp(min=1.0)).mean()
        final_loss = policy_loss + teacher_branch["extra_loss"]
        if local_sample_skip_required:
            # Keep the same forward/backward graph shape across ranks, but zero out
            # the failed sample's contribution so it does not update the model.
            final_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        phase2_debug_progress(
            f"returning loss vision_read_failed={vision_read_failed} "
            f"local_sample_skip_error_type={local_sample_skip_error_type} "
            f"loss={float(final_loss.detach().item()) if torch.is_tensor(final_loss) else final_loss}",
            local_rank=local_rank,
        )

        vision_failed_tensor = torch.tensor([1.0 if vision_read_failed else 0.0], device=device, dtype=torch.float32)
        vision_missing_tensor = torch.tensor(
            [1.0 if vision_error_type == "vision_media_missing" else 0.0],
            device=device,
            dtype=torch.float32,
        )
        vision_decode_tensor = torch.tensor(
            [1.0 if vision_error_type == "vision_decode_failed" else 0.0],
            device=device,
            dtype=torch.float32,
        )
        vision_unknown_tensor = torch.tensor(
            [1.0 if vision_error_type == "vision_unknown_failed" else 0.0],
            device=device,
            dtype=torch.float32,
        )
        self._metrics["phase2/vision_read_failed_count"].append(
            float(self.accelerator.gather_for_metrics(vision_failed_tensor).sum().item())
        )
        self._metrics["phase2/vision_media_missing_count"].append(
            float(self.accelerator.gather_for_metrics(vision_missing_tensor).sum().item())
        )
        self._metrics["phase2/vision_decode_failed_count"].append(
            float(self.accelerator.gather_for_metrics(vision_decode_tensor).sum().item())
        )
        self._metrics["phase2/vision_unknown_failed_count"].append(
            float(self.accelerator.gather_for_metrics(vision_unknown_tensor).sum().item())
        )

        for key, value in supervision_metrics.items():
            self._metrics[key].append(value)
        for key, value in teacher_branch["metrics"].items():
            self._metrics[key].append(value)
        return final_loss

    def _build_teacher_prompt_inputs(self, teacher_prompts_text, *, image_inputs, video_inputs, video_kwargs, multi_image):
        num_prompts = len(teacher_prompts_text)
        local_images = image_inputs
        local_videos = video_inputs
        if num_prompts > 1 and image_inputs is not None and len(image_inputs) == 1:
            local_images = [copy.deepcopy(image_inputs[0]) for _ in range(num_prompts)]
        if num_prompts > 1 and video_inputs is not None and len(video_inputs) == 1:
            local_videos = [copy.deepcopy(video_inputs[0]) for _ in range(num_prompts)]

        if multi_image:
            teacher_prompt_inputs = self.processing_class(
                text=copy.deepcopy(teacher_prompts_text),
                images=local_images,
                videos=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            teacher_prompt_inputs = self.processing_class(
                text=copy.deepcopy(teacher_prompts_text),
                images=local_images,
                videos=local_videos,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                **video_kwargs,
            )
        teacher_prompt_inputs = super()._prepare_inputs(teacher_prompt_inputs)
        teacher_prompt_inputs = self._move_prompt_inputs_to_device(teacher_prompt_inputs)
        prompt_limit = self.phase2_teacher_max_prompt_length or self.max_prompt_length
        if prompt_limit is not None:
            teacher_prompt_inputs["input_ids"] = teacher_prompt_inputs["input_ids"][:, -prompt_limit:]
            teacher_prompt_inputs["attention_mask"] = teacher_prompt_inputs["attention_mask"][:, -prompt_limit:]
        return teacher_prompt_inputs

    def _apply_reward_ablation_mask(self, rewards_per_func):
        return apply_reward_ablation_mask(
            rewards_per_func=rewards_per_func,
            reward_funcs=self.reward_funcs,
            script_args=self.script_args,
        )

    def _build_reward_breakdown(self, rewards_per_func):
        if rewards_per_func.dim() == 1:
            reward_values = rewards_per_func
        else:
            reward_values = rewards_per_func.mean(dim=0)
        values = {
            "reward": float(rewards_per_func.sum().item()) if rewards_per_func.dim() == 1 else float(rewards_per_func.sum(dim=1).mean().item()),
            "answer_semantic_score": None,
            "answer_window_score": None,
            "answer_box_score": None,
            "temporal_grounding_score": None,
            "spatial_grounding_score": None,
            "format_score": None,
        }
        for index, reward_func in enumerate(self.reward_funcs):
            reward_func_name = get_reward_func_name(reward_func)
            reward_value = float(reward_values[index].item())
            if "ans_acc" in reward_func_name:
                values["answer_semantic_score"] = reward_value
            elif "ans_tiou" in reward_func_name:
                values["answer_window_score"] = reward_value
            elif "ans_viou" in reward_func_name:
                values["answer_box_score"] = reward_value
            elif "thk_temporal_point" in reward_func_name or "thk_temporal_segment" in reward_func_name:
                existing = values["temporal_grounding_score"] or 0.0
                values["temporal_grounding_score"] = existing + reward_value
            elif "thk_spatial" in reward_func_name:
                values["spatial_grounding_score"] = reward_value
            elif "format" in reward_func_name:
                values["format_score"] = reward_value
        return values

    def _extract_reward_metadata_for_index(
        self,
        reward_metadata_rows: list[dict[str, list]],
        index: int,
    ) -> dict[str, object]:
        if index >= len(reward_metadata_rows):
            return {}
        return summarize_reward_metadata(reward_metadata_rows[index])

    def _build_teacher_forward_inputs(self, teacher_prompt_inputs, completion_mask):
        prompt_ids = teacher_prompt_inputs["input_ids"]
        prompt_attention_mask = teacher_prompt_inputs["attention_mask"]
        num_sequences = completion_mask.size(0)
        prompt_batch_matches = prompt_ids.size(0) == num_sequences
        if prompt_batch_matches:
            batched_prompt_ids = prompt_ids
            batched_prompt_attention_mask = prompt_attention_mask
        else:
            batched_prompt_ids = prompt_ids.repeat(num_sequences, 1)
            batched_prompt_attention_mask = prompt_attention_mask.repeat(num_sequences, 1)

        teacher_attention_mask = torch.cat(
            [
                batched_prompt_attention_mask,
                completion_mask.to(device=batched_prompt_attention_mask.device, dtype=batched_prompt_attention_mask.dtype),
            ],
            dim=1,
        )
        teacher_fwd_inputs = {"attention_mask": teacher_attention_mask}
        for key, value in teacher_prompt_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                if prompt_batch_matches:
                    teacher_fwd_inputs[key] = value
                else:
                    teacher_fwd_inputs[key] = value.repeat(num_sequences, *([1] * (value.dim() - 1)))
            else:
                teacher_fwd_inputs[key] = copy.deepcopy(value)
        teacher_fwd_inputs.pop("second_per_grid_ts", None)
        return batched_prompt_ids, teacher_fwd_inputs

    def _get_phase2_teacher_replay_model(self, fallback_model):
        if self.phase2_enable_teacher_model and getattr(self, "teacher_model", None) is not None:
            return self.teacher_model
        return fallback_model

    def _build_phase2_reweighting_topk_reward_raw(
        self,
        *,
        model,
        final_prompt_text: str,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        target_indices: list[int],
        min_len: int,
        student_trimmed: torch.Tensor,
        teacher_log_probs_target: torch.Tensor,
        teacher_token_logits_target: torch.Tensor,
        image_inputs,
        video_inputs,
        video_kwargs,
        multi_image: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = {
            "phase2/reweighting_weight_mode_sampled": 0.0,
            "phase2/reweighting_weight_mode_topk_interpolate": 1.0,
            "phase2/reweighting_topk_sampled_in_teacher_topk_fraction": 0.0,
            "phase2/reweighting_topk_gap_abs_mean": 0.0,
            "phase2/reweighting_topk_gamma": float(self.phase2_reweighting_topk_gamma),
        }
        if not target_indices:
            empty = torch.zeros((0, min_len), device=student_trimmed.device, dtype=student_trimmed.dtype)
            return empty, metrics

        prompt_texts = [final_prompt_text for _ in target_indices]
        student_prompt_inputs = self._build_teacher_prompt_inputs(
            prompt_texts,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            video_kwargs=video_kwargs,
            multi_image=multi_image,
        )
        student_prompt_length = student_prompt_inputs["input_ids"].size(1)
        target_completion_mask = completion_mask[target_indices]
        student_prompt_ids, student_fwd_inputs = self._build_teacher_forward_inputs(
            student_prompt_inputs,
            target_completion_mask,
        )
        student_prompt_completion_ids = torch.cat([student_prompt_ids, completion_ids[target_indices]], dim=1)

        with torch.no_grad():
            _, _, student_target_logits = self._get_per_token_logps_and_entropy(
                model,
                student_prompt_completion_ids,
                logits_to_keep=completion_ids[target_indices].size(1),
                compute_entropy=False,
                return_token_logits=True,
                **student_fwd_inputs,
            )
        student_target_logits = student_target_logits[:, :min_len, :]
        sampled_token_ids = completion_ids[target_indices, :min_len]

        teacher_reward_raw_target, aux = build_phase2_reweighting_teacher_reward_raw(
            student_log_probs=student_trimmed[target_indices],
            teacher_log_probs=teacher_log_probs_target,
            weight_mode="topk_interpolate",
            student_logits=student_target_logits,
            teacher_logits=teacher_token_logits_target,
            sampled_token_ids=sampled_token_ids,
            top_k=self.phase2_reweighting_topk,
            topk_gamma=self.phase2_reweighting_topk_gamma,
            include_sampled_token_in_topk=True,
        )

        target_response_mask = target_completion_mask[:, :min_len].to(dtype=student_trimmed.dtype)
        denom = target_response_mask.sum()
        if denom.item() > 0:
            sampled_in_teacher_topk = aux["sampled_in_teacher_topk"].to(
                device=student_trimmed.device,
                dtype=student_trimmed.dtype,
            )
            gap_abs = (aux["topk_gap"] - aux["sampled_gap"]).abs().to(
                device=student_trimmed.device,
                dtype=student_trimmed.dtype,
            )
            metrics["phase2/reweighting_topk_sampled_in_teacher_topk_fraction"] = float(
                ((sampled_in_teacher_topk * target_response_mask).sum() / denom).item()
            )
            metrics["phase2/reweighting_topk_gap_abs_mean"] = float(((gap_abs * target_response_mask).sum() / denom).item())
        return teacher_reward_raw_target.to(device=student_trimmed.device, dtype=student_trimmed.dtype), metrics

    def _get_phase2_success_reward_threshold(self) -> float:
        script_args = getattr(self, "script_args", None)
        if script_args is None:
            return 0.999
        neutral_value = getattr(script_args, "phase2_teacher_success_reward_threshold", None)
        if neutral_value is not None:
            return float(neutral_value)
        return 0.999

    def _should_request_process_feedback(
        self,
        *,
        probe_supervision: dict[str, Any],
    ) -> tuple[bool, str]:
        has_hindsight_target = bool(probe_supervision.get("has_hindsight_target"))
        should_target = bool(probe_supervision.get("should_target"))
        scope = str(getattr(self, "phase2_process_feedback_scope", "target_only") or "target_only").lower()

        if not has_hindsight_target:
            return False, "no_hindsight_target"

        if scope == "all_with_hindsight":
            request_allowed = True
        else:
            request_allowed = should_target

        if not request_allowed:
            return False, "scope_skipped"
        return True, "requested"

    def _build_phase2_supervision_rows(
        self,
        *,
        example,
        privileged_info,
        rewards_per_func,
        reward_metadata,
        student_outputs,
    ) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
        rows = []
        metric_totals: dict[str, float] = {}
        process_requested = 0
        process_available = 0
        process_errors = 0
        process_reused = 0
        judge_feedback_count = 0
        reward_totals = rewards_per_func.sum(dim=1)
        finite_reward_totals = reward_totals[torch.isfinite(reward_totals)]
        group_avg_reward = float(finite_reward_totals.mean().item()) if finite_reward_totals.numel() > 0 else None

        for index in range(rewards_per_func.size(0)):
            reward_breakdown = self._build_reward_breakdown(rewards_per_func[index])
            reward_metadata_row = self._extract_reward_metadata_for_index(reward_metadata, index)
            generated_process_feedback_result = None
            process_feedback_status = "disabled"
            process_feedback_error_type = None
            process_feedback_error_message = None

            if should_reuse_process_feedback_payload(reward_metadata_row):
                process_reused += 1
                process_feedback_status = "reused"
                judge_feedback_count += 1
            elif self.phase2_process_feedback_enable and index < len(student_outputs):
                probe_supervision, _ = build_phase2_supervision(
                    example=example,
                    privileged_info=privileged_info,
                    reward_breakdown=reward_breakdown,
                    reward_metadata=reward_metadata_row,
                    script_args=self.script_args,
                    extra_feedback_text=self.phase2_teacher_feedback_text,
                )
                request_process_feedback, process_feedback_status = self._should_request_process_feedback(
                    probe_supervision=probe_supervision,
                )
                if request_process_feedback:
                    process_requested += 1
                    try:
                        process_feedback_request_kwargs = resolve_phase2_process_feedback_request_kwargs(self)
                        generated_process_feedback_result = build_process_feedback_result(
                            question=privileged_info.get("question"),
                            answer=privileged_info.get("answer"),
                            ground_truth_window=privileged_info.get("ground_truth_window"),
                            student_output=student_outputs[index],
                            keyframe_object_evidence=privileged_info.get("keyframe_object_evidence"),
                            task=privileged_info.get("task"),
                            timeout=self.phase2_process_feedback_timeout,
                            **process_feedback_request_kwargs,
                            max_feedback_chars=self.phase2_process_feedback_max_feedback_chars,
                        )
                    except Exception as exc:
                        process_errors += 1
                        generated_process_feedback_result = None
                        process_feedback_status = "error"
                        process_feedback_error_type = type(exc).__name__
                        process_feedback_error_message = str(exc).strip()[:500] or process_feedback_error_type
                    if generated_process_feedback_result and generated_process_feedback_result.get("feedback"):
                        judge_feedback_count += 1
                        process_available += 1
                        process_feedback_status = "generated"
                    elif process_feedback_status != "error":
                        process_feedback_status = "empty"
            elif not self.phase2_process_feedback_enable:
                process_feedback_status = "disabled"
            else:
                process_feedback_status = "missing_student_output"

            supervision, row_metrics = build_phase2_supervision(
                example=example,
                privileged_info=privileged_info,
                reward_breakdown=reward_breakdown,
                reward_metadata=reward_metadata_row,
                script_args=self.script_args,
                extra_feedback_text=self.phase2_teacher_feedback_text,
                generated_process_feedback_text=(
                    generated_process_feedback_result.get("feedback")
                    if isinstance(generated_process_feedback_result, dict)
                    else None
                ),
                generated_process_feedback_result=generated_process_feedback_result,
            )
            rows.append(
                {
                    "supervision": supervision,
                    "reward_breakdown": reward_breakdown,
                    "reward_metadata": reward_metadata_row,
                    "process_feedback_status": process_feedback_status,
                    "process_feedback_error_type": process_feedback_error_type,
                    "process_feedback_error_message": process_feedback_error_message,
                    "process_feedback_result": generated_process_feedback_result,
                }
            )
            for key, value in row_metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + float(value)

        row_count = max(1, len(rows))
        metrics = {key: value / row_count for key, value in metric_totals.items()}
        requested_routing = build_phase2_requested_masks(
            supervision_rows=rows,
            device=rewards_per_func.device,
            success_reward_threshold=self._get_phase2_success_reward_threshold(),
        )
        for row in rows:
            row["phase2_group_state"] = requested_routing["group_state"]
        metrics["phase2/process_feedback_enabled"] = 1.0 if self.phase2_process_feedback_enable else 0.0
        metrics["phase2/process_feedback_requested_fraction"] = float(process_requested) / row_count
        metrics["phase2/process_feedback_available_fraction"] = float(process_available) / row_count
        metrics["phase2/process_feedback_error_fraction"] = float(process_errors) / row_count
        metrics["phase2/process_feedback_reused_from_reward_fraction"] = float(process_reused) / row_count
        metrics["phase2/judge_feedback_available_fraction"] = float(judge_feedback_count) / row_count
        metrics["phase2/group_avg_reward"] = group_avg_reward if group_avg_reward is not None else 0.0
        metrics.update(requested_routing["metrics"])
        return rows, requested_routing, metrics

    def _compute_phase2_teacher_branch(
        self,
        *,
        model,
        final_prompt_text,
        completion_ids,
        completion_mask,
        student_per_token_logps,
        student_per_token_entropy,
        base_token_advantages,
        rewards,
        image_inputs,
        video_inputs,
        video_kwargs,
        multi_image,
        privileged_info,
        supervision_rows,
        requested_routing=None,
    ):
        zero = student_per_token_logps.sum() * 0.0
        working_advantages = base_token_advantages.clone()
        if requested_routing is None:
            requested_routing = build_phase2_requested_masks(
                supervision_rows=supervision_rows,
                device=student_per_token_logps.device,
                dtype=student_per_token_logps.dtype,
                success_reward_threshold=self._get_phase2_success_reward_threshold(),
            )
        requested_target_mask = requested_routing["requested_target_mask"].to(
            device=student_per_token_logps.device,
            dtype=student_per_token_logps.dtype,
        )
        requested_reweight_mask = requested_routing["requested_reweight_mask"].to(
            device=student_per_token_logps.device,
            dtype=student_per_token_logps.dtype,
        )
        metrics = {
            "phase2/teacher_enabled": 1.0,
            "phase2/teacher_hindsight_available": 0.0,
            "phase2/teacher_targeted": 0.0,
            "phase2/teacher_triggered": 0.0,
            "phase2/teacher_forward_sample_fraction": 0.0,
            "phase2/teacher_replay_dropped_count": 0.0,
        }

        has_hindsight = 1.0 if has_hindsight_supervision(privileged_info) else 0.0
        metrics["phase2/teacher_hindsight_available"] = has_hindsight
        if not self.phase2_inject_teacher_signal:
            metrics["phase2/teacher_signal_disabled"] = 1.0
            metrics["phase2/effective_sample_fraction"] = 0.0
            return {"extra_loss": zero, "effective_advantages": working_advantages, "metrics": metrics}

        finite_rewards = rewards[torch.isfinite(rewards)]
        group_avg_reward = float(finite_rewards.mean().item()) if finite_rewards.numel() > 0 else None
        metrics["phase2/group_avg_reward"] = group_avg_reward if group_avg_reward is not None else 0.0

        teacher_prompts = []
        target_indices = []
        for index, row in enumerate(supervision_rows):
            supervision = row["supervision"]
            should_target = bool(supervision.get("should_target", False))
            should_apply = should_target and supervision.get("has_hindsight_target", False)
            if not should_apply:
                continue

            hint = build_hindsight_hint(
                privileged_info=privileged_info,
                feedback_text=supervision.get("feedback_text"),
                answer_semantic_score=supervision.get("answer_semantic_score"),
                judge_metadata=supervision.get("judge_metadata"),
            )
            if not hint.strip():
                continue
            teacher_prompts.append(inject_hindsight_hint_into_prompt(final_prompt_text, hint))
            target_indices.append(index)

        metrics["phase2/teacher_targeted"] = (
            float(requested_target_mask.mean().item()) if requested_target_mask.numel() > 0 else 0.0
        )

        local_has_teacher_targets = bool(target_indices)
        teacher_forward_required = local_has_teacher_targets
        if dist.is_available() and dist.is_initialized():
            forward_flag = torch.tensor(
                [1 if local_has_teacher_targets else 0],
                device=student_per_token_logps.device,
                dtype=torch.int64,
            )
            dist.all_reduce(forward_flag, op=dist.ReduceOp.MAX)
            teacher_forward_required = bool(forward_flag.item())

        forward_indices = resolve_teacher_forward_indices(
            target_indices,
            num_sequences=completion_ids.size(0),
            force_forward=teacher_forward_required,
        )
        if not forward_indices:
            effective_masks = build_effective_phase2_masks(
                requested_target_mask=requested_target_mask,
                requested_reweight_mask=requested_reweight_mask,
                valid_indices=[],
            )
            metrics.update(effective_masks["metrics"])
            metrics["phase2/teacher_replay_dropped_count"] = float(requested_target_mask.sum().item())
            return {"extra_loss": zero, "effective_advantages": working_advantages, "metrics": metrics}

        teacher_prompt_texts = teacher_prompts if local_has_teacher_targets else [final_prompt_text]
        target_completion_ids = completion_ids[forward_indices]
        target_completion_mask = completion_mask[forward_indices]
        teacher_prompt_inputs = self._build_teacher_prompt_inputs(
            teacher_prompt_texts,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            video_kwargs=video_kwargs,
            multi_image=multi_image,
        )
        teacher_prompt_length = teacher_prompt_inputs["input_ids"].size(1)
        prompt_ids, teacher_fwd_inputs = self._build_teacher_forward_inputs(
            teacher_prompt_inputs,
            target_completion_mask,
        )
        teacher_prompt_completion_ids = torch.cat([prompt_ids, target_completion_ids], dim=1)

        use_topk_reweight = self.phase2_reweighting_weight_mode == "topk_interpolate"
        try:
            # Keep teacher replay forward out of autograd while avoiding
            # inference-mode tensors that can break ZeRO-3/checkpointed backward.
            with torch.no_grad():
                if use_topk_reweight:
                    teacher_per_token_logps, teacher_per_token_entropy, teacher_token_logits = (
                        self._get_per_token_logps_and_entropy(
                            self._get_phase2_teacher_replay_model(model),
                            teacher_prompt_completion_ids,
                            logits_to_keep=target_completion_ids.size(1),
                            return_token_logits=True,
                            **teacher_fwd_inputs,
                        )
                    )
                else:
                    teacher_per_token_logps, teacher_per_token_entropy = self._get_per_token_logps_and_entropy(
                        self._get_phase2_teacher_replay_model(model),
                        teacher_prompt_completion_ids,
                        logits_to_keep=target_completion_ids.size(1),
                        **teacher_fwd_inputs,
                    )
                    teacher_token_logits = None
            if teacher_token_logits is not None:
                teacher_token_logits = teacher_token_logits[:, :, :]
        except Exception as exc:
            print(f"[Phase2Teacher] Error computing hindsight teacher logps: {exc}")
            metrics["phase2/teacher_error"] = 1.0
            metrics["phase2/teacher_replay_dropped_count"] = float(requested_target_mask.sum().item())
            return {"extra_loss": zero, "effective_advantages": working_advantages, "metrics": metrics}

        min_len = min(
            teacher_per_token_logps.size(1),
            student_per_token_logps.size(1),
            completion_mask.size(1),
            working_advantages.size(1),
        )
        if min_len <= 0:
            metrics["phase2/teacher_replay_dropped_count"] = float(requested_target_mask.sum().item())
            return {"extra_loss": zero, "effective_advantages": working_advantages, "metrics": metrics}

        student_trimmed = student_per_token_logps[:, :min_len]
        entropy_trimmed = student_per_token_entropy[:, :min_len]
        response_mask = completion_mask[:, :min_len].to(dtype=student_trimmed.dtype)
        working_advantages = working_advantages[:, :min_len]
        teacher_log_probs_full = torch.zeros_like(student_trimmed)
        teacher_entropy_full = torch.zeros_like(entropy_trimmed)
        effective_masks = build_effective_phase2_masks(
            requested_target_mask=requested_target_mask,
            requested_reweight_mask=requested_reweight_mask,
            valid_indices=target_indices,
        )
        effective_target_mask = effective_masks["effective_target_mask"]
        for local_idx, batch_idx in enumerate(target_indices):
            teacher_log_probs_full[batch_idx] = teacher_per_token_logps[local_idx, :min_len]
            teacher_entropy_full[batch_idx] = teacher_per_token_entropy[local_idx, :min_len]
        teacher_reward_raw_full = (teacher_log_probs_full.detach() - student_trimmed.detach()).to(dtype=student_trimmed.dtype)
        topk_reweight_metrics = {
            "phase2/reweighting_weight_mode_sampled": 1.0,
            "phase2/reweighting_weight_mode_topk_interpolate": 0.0,
            "phase2/reweighting_topk_sampled_in_teacher_topk_fraction": 0.0,
            "phase2/reweighting_topk_gap_abs_mean": 0.0,
            "phase2/reweighting_topk_gamma": float(self.phase2_reweighting_topk_gamma),
        }
        if use_topk_reweight and target_indices and teacher_token_logits is not None:
            try:
                teacher_reward_raw_target, topk_reweight_metrics = self._build_phase2_reweighting_topk_reward_raw(
                    model=model,
                    final_prompt_text=final_prompt_text,
                    completion_ids=completion_ids,
                    completion_mask=completion_mask,
                    target_indices=target_indices,
                    min_len=min_len,
                    student_trimmed=student_trimmed,
                    teacher_log_probs_target=teacher_per_token_logps[:, :min_len],
                    teacher_token_logits_target=teacher_token_logits[:, :min_len, :],
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    video_kwargs=video_kwargs,
                    multi_image=multi_image,
                )
                teacher_reward_raw_full[target_indices] = teacher_reward_raw_target
            except Exception as exc:
                print(f"[Phase2Teacher] Falling back to sampled reweighting weight after top-k replay error: {exc}")
                topk_reweight_metrics["phase2/reweighting_weight_mode_sampled"] = 1.0
                topk_reweight_metrics["phase2/reweighting_weight_mode_topk_interpolate"] = 0.0
                topk_reweight_metrics["phase2/reweighting_topk_fallback_sampled"] = 1.0
        metrics.update(topk_reweight_metrics)

        metrics["phase2/teacher_triggered"] = (
            float(effective_target_mask.mean().item()) if effective_target_mask.numel() > 0 else 0.0
        )
        metrics["phase2/teacher_forward_sample_fraction"] = metrics["phase2/teacher_triggered"]
        metrics["phase2/teacher_replay_dropped_count"] = float(
            requested_target_mask.sum().item() - effective_target_mask.sum().item()
        )
        metrics["phase2/teacher_prompt_length"] = float(teacher_prompt_length)
        metrics.update(effective_masks["metrics"])
        metrics.update(
            build_teacher_signal_metrics(
                student_log_probs=student_trimmed,
                teacher_log_probs=teacher_log_probs_full,
                response_mask=response_mask,
                target_mask=effective_target_mask,
                student_entropy=entropy_trimmed,
                teacher_entropy=teacher_entropy_full,
            )
        )

        extra_loss = zero
        working_advantages, _, teacher_metrics = apply_phase2_reweighting(
            base_token_advantages=working_advantages,
            student_log_probs=student_trimmed,
            teacher_log_probs=teacher_log_probs_full,
            teacher_reward_raw=teacher_reward_raw_full,
            response_mask=response_mask,
            target_mask=effective_target_mask,
            weight_clip=self.phase2_reweighting_weight_clip,
            mixing_lambda=self.phase2_reweighting_mixing_lambda,
        )
        metrics.update(teacher_metrics)

        return {"extra_loss": extra_loss, "effective_advantages": working_advantages, "metrics": metrics}
