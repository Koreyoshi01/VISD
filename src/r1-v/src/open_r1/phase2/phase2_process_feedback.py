from __future__ import annotations

import re

import torch

from custom_rewards.unified_judge import judge_answer_and_feedback

MISSING_ANSWER_TAG_FEEDBACK = (
    "The student output does not contain an <answer> tag. "
    "The thinking section may be too long, and the answer may have been truncated before it was produced."
)


def _extract_model_answer(student_output: str) -> str | None:
    if not student_output:
        return None
    match = re.search(r"<answer>(.*?)</answer>", student_output, flags=re.DOTALL)
    if match:
        answer = re.sub(r"\s+", " ", match.group(1)).strip()
        if answer:
            return answer
    return None


def decode_student_output_texts(tokenizer, responses: torch.Tensor, response_mask: torch.Tensor) -> list[str]:
    texts: list[str] = []
    for token_ids, mask in zip(responses, response_mask, strict=True):
        valid_ids = token_ids[mask.to(dtype=torch.bool)].detach().cpu().tolist()
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True) if valid_ids else "")
    return texts


def normalize_verified_answer_for_judge(
    verified_answer: str | None,
    ground_truth_window: list[float] | None,
) -> str:
    text = (verified_answer or "").strip()
    return text if text else ""


def build_process_feedback_result(
    *,
    question: str | None,
    answer: str | None,
    ground_truth_window: list[float] | None,
    student_output: str | None,
    keyframe_object_evidence: list[dict[str, object]] | None = None,
    task: str | None = None,
    timeout: float = 30.0,
    model_override: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_feedback_chars: int = 1000,
) -> dict[str, object]:
    verified_answer = normalize_verified_answer_for_judge(answer, ground_truth_window)
    student_output = (student_output or "").strip()
    empty_result = {
        "feedback": None,
        "raw_response": "",
        "model_answer": None,
    }
    if not question or not verified_answer or not student_output:
        return empty_result

    model_answer = _extract_model_answer(student_output)

    result = judge_answer_and_feedback(
        question=question,
        standard_answer=verified_answer,
        model_answer=model_answer,
        student_output=student_output,
        keyframe_object_evidence=keyframe_object_evidence,
        task=task,
        timeout=timeout,
        model_override=model_override,
        api_base=api_base,
        api_key=api_key,
        max_feedback_chars=max_feedback_chars,
    )
    feedback_parts: list[str] = []
    if model_answer is None:
        feedback_parts.append(MISSING_ANSWER_TAG_FEEDBACK)
    feedback = result.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        feedback_parts.append(feedback.strip())

    merged_feedback = None
    if feedback_parts:
        seen = set()
        ordered_parts = []
        for part in feedback_parts:
            if part not in seen:
                seen.add(part)
                ordered_parts.append(part)
        merged_feedback = " ".join(ordered_parts).strip() or None
    return {
        "feedback": merged_feedback,
        "raw_response": result.get("raw_response", "") if isinstance(result, dict) else "",
        "model_answer": model_answer,
    }


def build_process_feedback_text(
    *,
    question: str | None,
    answer: str | None,
    ground_truth_window: list[float] | None,
    student_output: str | None,
    keyframe_object_evidence: list[dict[str, object]] | None = None,
    task: str | None = None,
    timeout: float = 30.0,
    model_override: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_feedback_chars: int = 1000,
) -> str | None:
    result = build_process_feedback_result(
        question=question,
        answer=answer,
        ground_truth_window=ground_truth_window,
        student_output=student_output,
        keyframe_object_evidence=keyframe_object_evidence,
        task=task,
        timeout=timeout,
        model_override=model_override,
        api_base=api_base,
        api_key=api_key,
        max_feedback_chars=max_feedback_chars,
    )
    feedback = result.get("feedback")
    return feedback.strip() if isinstance(feedback, str) and feedback.strip() else None
