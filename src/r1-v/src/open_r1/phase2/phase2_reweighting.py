from __future__ import annotations

import torch


def _compute_topk_support_log_ratio(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    top_k: int,
    include_sampled_token: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must be rank-3 tensors [batch, seq, vocab].")
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise ValueError(
            "student_logits and teacher_logits must have identical shape, "
            f"got {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if sampled_token_ids.ndim != 2:
        raise ValueError("sampled_token_ids must be rank-2 [batch, seq].")
    if tuple(sampled_token_ids.shape) != tuple(student_logits.shape[:2]):
        raise ValueError(
            "sampled_token_ids must match the [batch, seq] prefix of logits, "
            f"got {tuple(sampled_token_ids.shape)} vs {tuple(student_logits.shape[:2])}"
        )

    vocab_size = int(student_logits.size(-1))
    k = max(1, min(int(top_k), vocab_size))
    sampled_index = sampled_token_ids.to(device=student_logits.device, dtype=torch.long).unsqueeze(-1)

    teacher_topk_logits, teacher_topk_ids = torch.topk(teacher_logits, k=k, dim=-1)
    student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids)

    sampled_teacher_logits = torch.gather(teacher_logits, dim=-1, index=sampled_index).squeeze(-1)
    sampled_student_logits = torch.gather(student_logits, dim=-1, index=sampled_index).squeeze(-1)
    sampled_in_teacher_topk = (teacher_topk_ids == sampled_index).any(dim=-1)

    if not include_sampled_token and not bool(sampled_in_teacher_topk.all().item()):
        raise ValueError("sampled token is not contained in teacher top-k while include_sampled_token=False.")

    teacher_support_log_denom = torch.logsumexp(teacher_topk_logits, dim=-1)
    student_support_log_denom = torch.logsumexp(student_topk_logits, dim=-1)
    if include_sampled_token:
        teacher_support_log_denom = torch.where(
            sampled_in_teacher_topk,
            teacher_support_log_denom,
            torch.logaddexp(teacher_support_log_denom, sampled_teacher_logits),
        )
        student_support_log_denom = torch.where(
            sampled_in_teacher_topk,
            student_support_log_denom,
            torch.logaddexp(student_support_log_denom, sampled_student_logits),
        )

    teacher_support_log_prob = sampled_teacher_logits - teacher_support_log_denom
    student_support_log_prob = sampled_student_logits - student_support_log_denom
    log_ratio = teacher_support_log_prob - student_support_log_prob
    return log_ratio, {"sampled_in_teacher_topk": sampled_in_teacher_topk}


def build_phase2_reweighting_teacher_reward_raw(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weight_mode: str = "sampled",
    student_logits: torch.Tensor | None = None,
    teacher_logits: torch.Tensor | None = None,
    sampled_token_ids: torch.Tensor | None = None,
    top_k: int = 8,
    topk_gamma: float = 1.0,
    include_sampled_token_in_topk: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    if tuple(student_log_probs.shape) != tuple(teacher_log_probs.shape):
        raise ValueError(
            "student_log_probs and teacher_log_probs must match, "
            f"got {tuple(student_log_probs.shape)} vs {tuple(teacher_log_probs.shape)}"
        )

    weight_mode_normalized = str(weight_mode or "sampled").lower()
    sampled_gap = (teacher_log_probs.detach() - student_log_probs.detach()).to(
        device=student_log_probs.device,
        dtype=student_log_probs.dtype,
    )
    if weight_mode_normalized == "sampled":
        return sampled_gap, {"sampled_gap": sampled_gap, "weight_mode": 0.0}
    if weight_mode_normalized != "topk_interpolate":
        raise ValueError(f"Unsupported reweighting weight_mode={weight_mode!r}")

    if student_logits is None or teacher_logits is None or sampled_token_ids is None:
        raise ValueError("topk_interpolate mode requires student_logits, teacher_logits, and sampled_token_ids.")

    topk_gap, aux = _compute_topk_support_log_ratio(
        student_logits=student_logits.detach(),
        teacher_logits=teacher_logits.detach(),
        sampled_token_ids=sampled_token_ids.detach(),
        top_k=top_k,
        include_sampled_token=include_sampled_token_in_topk,
    )
    gamma = float(topk_gamma)
    teacher_reward_raw = ((1.0 - gamma) * topk_gap) + (gamma * sampled_gap)
    return teacher_reward_raw.to(device=student_log_probs.device, dtype=student_log_probs.dtype), {
        "sampled_gap": sampled_gap,
        "topk_gap": topk_gap.to(device=student_log_probs.device, dtype=student_log_probs.dtype),
        "sampled_in_teacher_topk": aux["sampled_in_teacher_topk"].to(device=student_log_probs.device),
        "weight_mode": 1.0,
        "topk_gamma": gamma,
    }


def _safe_fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = float(denominator.item()) if isinstance(denominator, torch.Tensor) else float(denominator)
    if denom <= 0:
        return 0.0
    num = float(numerator.item()) if isinstance(numerator, torch.Tensor) else float(numerator)
    return num / denom


def apply_phase2_reweighting(
    *,
    base_token_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_reward_raw: torch.Tensor | None = None,
    response_mask: torch.Tensor,
    target_mask: torch.Tensor,
    weight_clip: float | None = 0.2,
    mixing_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    expected_shape = tuple(base_token_advantages.shape)
    for name, value in {
        "student_log_probs": student_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "response_mask": response_mask,
    }.items():
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} must match base_token_advantages shape, got {tuple(value.shape)} vs {expected_shape}")

    dtype = base_token_advantages.dtype
    device = base_token_advantages.device
    response_mask = response_mask.to(device=device, dtype=dtype)
    target_mask = target_mask.to(device=device, dtype=dtype)
    if teacher_reward_raw is None:
        teacher_reward_raw = (teacher_log_probs.detach() - student_log_probs.detach()).to(device=device, dtype=dtype)
    elif tuple(teacher_reward_raw.shape) != expected_shape:
        raise ValueError(
            "teacher_reward_raw must match base_token_advantages shape, "
            f"got {tuple(teacher_reward_raw.shape)} vs {expected_shape}"
        )
    else:
        teacher_reward_raw = teacher_reward_raw.to(device=device, dtype=dtype)
    base_token_mask = response_mask * target_mask.unsqueeze(-1)
    token_mask = base_token_mask

    sign_advantages = torch.sign(base_token_advantages).to(dtype)
    raw_weights = torch.exp(sign_advantages * teacher_reward_raw)
    if weight_clip is not None:
        clipped_weights = raw_weights.clamp(min=1.0 - float(weight_clip), max=1.0 + float(weight_clip))
    else:
        clipped_weights = raw_weights

    mixing = max(0.0, min(1.0, float(mixing_lambda)))
    selected_weights = (1.0 - mixing) + mixing * clipped_weights
    effective_weights = torch.ones_like(base_token_advantages)
    effective_weights = effective_weights + token_mask * (selected_weights - 1.0)
    reweighted_advantages = base_token_advantages * effective_weights

    token_denom = token_mask.sum()
    base_token_denom = base_token_mask.sum()
    response_token_denom = response_mask.sum()
    if token_denom.item() > 0:
        mean_weight = float((effective_weights * token_mask).sum().item() / token_denom.item())
        nontrivial_fraction = float(
            (((effective_weights - 1.0).abs() > 1e-6) * token_mask.bool()).float().sum().item() / token_denom.item()
        )
        clipped_fraction = float((((raw_weights != clipped_weights) * token_mask.bool()).float().sum().item()) / token_denom.item())
    else:
        mean_weight = 1.0
        nontrivial_fraction = 0.0
        clipped_fraction = 0.0

    return reweighted_advantages, effective_weights, {
        "phase2/teacher_signal_mode_reweighting": 1.0,
        "phase2/reweighting_target_fraction": float(target_mask.mean().item()) if target_mask.numel() > 0 else 0.0,
        "phase2/reweighting_mean": mean_weight,
        "phase2/reweighting_nontrivial_fraction": nontrivial_fraction,
        "phase2/reweighting_clipped_fraction": clipped_fraction,
        "phase2/reweighting_effective_token_fraction": _safe_fraction(token_mask.sum(), response_token_denom),
        "phase2/reweighting_effective_token_fraction_within_target": _safe_fraction(token_mask.sum(), base_token_denom),
    }
