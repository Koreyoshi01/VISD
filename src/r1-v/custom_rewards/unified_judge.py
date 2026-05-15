from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import requests
from openai import OpenAI


DEFAULT_TIMEOUT = 30.0

DEFAULT_POSITIVE_FEEDBACK = "The final answer is correct, and the reasoning broadly supports it."


def normalize_api_base(api_base: str | None) -> str:
    api_base = (api_base or "").strip().rstrip("/")
    if not api_base:
        raise ValueError("LLM_AS_A_JUDGE_BASE is empty")
    if api_base.endswith("/v1"):
        return api_base
    return f"{api_base}/v1"


def get_request_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@lru_cache(maxsize=8)
def resolve_model_name(api_base: str, api_key: str, model_override: str | None, timeout: float = DEFAULT_TIMEOUT) -> str:
    if model_override:
        return model_override
    response = requests.get(
        f"{api_base}/models",
        headers=get_request_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload["data"][0]["id"]


@lru_cache(maxsize=4)
def get_client(api_base: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=api_base)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = _strip_code_fence(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _compress_feedback(text: str | None, max_chars: int) -> str | None:
    if not isinstance(text, str):
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if text.lower() in {"none", "n/a", "na", "no issue", "correct", "aligned", "answer consistent"}:
        return None
    return text[:max_chars].strip() or None


def normalize_judge_result_payload(payload: dict[str, Any] | None, *, max_feedback_chars: int) -> dict[str, Any]:
    payload = payload or {}
    feedback = _compress_feedback(payload.get("feedback"), max_feedback_chars)
    if feedback is None:
        feedback = _compress_feedback(payload.get("raw_response"), max_feedback_chars)
    if feedback is None and payload:
        feedback = DEFAULT_POSITIVE_FEEDBACK[:max_feedback_chars].strip() or None
    return {
        "feedback": feedback,
        "raw_response": payload.get("raw_response", ""),
    }


def _build_task_aware_user_guidance(task: str | None) -> str:
    task = (task or "").strip()
    if task == "temporal QA":
        return (
            "Task-aware instructions for temporal QA:\n"
            "- This task asks for the precise time period that answers the question.\n"
            "- The standard answer is one raw text block representing the verified temporal window.\n"
            "- The student's final answer should be interpreted as the student's proposed temporal window.\n"
            "- The expected student answer format is a temporal span such as `From <t>start_time</t>s to <t>end_time</t>s`.\n"
            "- Evaluate this task in two steps.\n"
            "- Step 1: judge the final temporal window itself. Say whether it matches the verified interval or whether it is too broad, too narrow, too early, too late, or focused on the wrong event.\n"
            "- Step 2: judge why the student arrived at that window. Use the reasoning text to explain whether the student tracked the wrong event, tracked the right event but set the boundaries badly, or used reasoning that was too broad or incomplete to support a precise interval.\n"
            "- Internally assess the temporal overlap between the student's window and the verified window, similar to a rough IoU-style overlap judgment.\n"
            "- If the windows overlap strongly and refer to the same event, you may describe the timing as roughly aligned or partly correct, while still stating whether it is broader, narrower, earlier, or later than the verified interval.\n"
            "- If the overlap is weak, or if the student's window covers mostly the wrong part of the video, say the temporal answer is basically wrong.\n"
            "- If there is essentially no meaningful overlap, say the temporal answer is completely wrong.\n"
            "- Judge the temporal window strictly.\n"
            "- Do not call the final answer fully correct unless the student's proposed window matches the verified interval closely enough for this task.\n"
            "- Do not let broad event-level semantic relevance compensate for a wrong, overly broad, overly narrow, too early, or too late temporal window.\n"
        )
    if task == "temporal QA (MCQ)":
        return (
            "Task-aware instructions for temporal QA (MCQ):\n"
            "- This task requires a final answer with two required parts: the correct option letter and the correct temporal window.\n"
            "- The standard answer is one raw text block. In that raw block, the first line is the verified option letter and the second line is the verified temporal window.\n"
            "- The student's final answer is also one raw text block and may express the option and time window in a different textual format.\n"
            "- The expected student answer format is a temporal span followed by the option, such as `From <t>start_time</t>s to <t>end_time</t>s.` and `Correct Option: X`.\n"
            "- Evaluate this task in three steps.\n"
            "- Step 1: extract and judge the option letter by itself.\n"
            "- Step 2: extract and judge the temporal window by itself.\n"
            "- Step 3: combine the two results into one overall judgment, then explain why the reasoning led to that option/window combination.\n"
            "- Before judging, extract the option and the temporal window from both raw answer blocks, then compare both parts jointly.\n"
            "- The option and the time window are both required parts of the answer. Even if the option is correct, a wrong or missing time window means the final answer is not fully correct.\n"
            "- Judge the option and the temporal window independently and strictly.\n"
            "- Do not let a correct option compensate for a wrong time window, and do not let a plausible time window compensate for a wrong option.\n"
            "- If one part is correct and the other part is wrong, say the final answer is only partly correct and explicitly state which part is wrong.\n"
            "- For the temporal part, internally assess the overlap between the student's window and the verified window, similar to a rough IoU-style overlap judgment.\n"
            "- If the option is correct and the temporal overlap is strong for the same event, you may describe the temporal part as roughly aligned or partly correct, but still mention boundary drift if it exists.\n"
            "- If the option is correct but the temporal overlap is weak, or the student focused on the wrong part of the video, say the temporal part is wrong and the overall answer is only partly correct.\n"
            "- If there is essentially no meaningful overlap between the student's window and the verified window, say the temporal part is completely wrong.\n"
            "- Do not treat the time window as optional just because the natural-language question focuses on purpose or content.\n"
            "- Use the reasoning text to explain why the student got the option right or wrong and why the student got the temporal window right or wrong.\n"
        )
    if task == "visual QA":
        return (
            "Task-aware instructions for visual QA:\n"
            "- The standard answer uses object-plus-box format.\n"
            "- The expected student answer format is raw object-plus-box text such as `<obj>object_name</obj><box>[x1, y1, x2, y2]</box>`.\n"
            "- Judge whether the student's final answer identifies the correct object and a plausible bounding box.\n"
            "- If the reasoning mentions boxes or objects that do not align with the answer, mention that briefly.\n"
        )
    if task == "temporal-spatial free-form QA":
        return (
            "Task-aware instructions for temporal-spatial free-form QA:\n"
            "- The standard answer is free-form text and should be judged semantically rather than by exact wording.\n"
            "- The expected student final answer is free-form natural language in the answer block.\n"
            "- The reasoning is expected to be grounded with object, box, and time references when relevant.\n"
            "- If keyframe object evidence is provided, use it to check whether the reasoning points to the right keyframe time, object, and spatial region.\n"
            "- Keyframe object evidence is a hidden reference list of entries like `{frame_idx, time, objects}`, where `objects` maps object names to one or more boxes.\n"
        )
    if task == "General video QA MCQ":
        return (
            "Task-aware instructions for General video QA MCQ:\n"
            "- The standard answer is the correct option letter only.\n"
            "- The expected student final answer is the option letter only.\n"
            "- Judge whether the student's final option is correct and whether the reasoning supports that choice.\n"
            "- Do not require time windows or boxes unless the student voluntarily uses them.\n"
        )
    if task == "General video QA Free-form":
        return (
            "Task-aware instructions for General video QA Free-form:\n"
            "- The standard answer is free-form text and should be judged semantically rather than by exact wording.\n"
            "- The expected student final answer is free-form natural language.\n"
            "- Judge whether the student's final answer actually answers the question and whether the reasoning supports it.\n"
        )
    return ""


def _render_reference_block(
    *,
    task: str | None,
    question: str,
    standard_answer: str,
    keyframe_object_evidence: list[dict[str, Any]] | None,
) -> str:
    lines = ["## Task And Reference Context"]
    task_guidance = _build_task_aware_user_guidance(task)
    if task_guidance:
        lines.append(task_guidance.rstrip())
    lines.extend([
        f"- Task type: {task or 'unknown'}",
        f"- Question: {question}",
        f"- Standard answer: {standard_answer}",
    ])
    lines.append(
        "- Keyframe object evidence: "
        + (json.dumps(keyframe_object_evidence, ensure_ascii=False) if keyframe_object_evidence else "null")
    )
    if keyframe_object_evidence:
        lines.append(
            "- Keyframe object evidence is hidden grounding reference. Use it only to judge whether the student's reasoning points to the right time, object, and spatial region."
        )
    return "\n".join(lines)


def _render_student_block(
    *,
    model_answer: str | None,
    student_output: str,
) -> str:
    return "\n".join(
        [
            "## Student Submission",
            f"- Student final answer: {model_answer if model_answer else 'null (no <answer> tag found)'}",
            f"- Student full output: {student_output}",
        ]
    )


def _build_judge_prompts(
    *,
    question: str,
    standard_answer: str,
    model_answer: str | None,
    student_output: str,
    keyframe_object_evidence: list[dict[str, Any]] | None,
    task: str | None,
) -> tuple[str, str]:
    has_answer_tag = bool((model_answer or "").strip())
    system_prompt = (
        "## Role\n"
        "You are a strict video and image QA judge.\n\n"
        "## Mission\n"
        "You are judging a student model's response. "
        "Your feedback will be given to a teacher model, which will use it to better guide the student model. "
        "Judge whether the student's final answer is correct, whether the student's reasoning broadly supports that final answer, "
        "and what the single most likely high-level source of error is when the response is not fully correct.\n\n"
        "## Input Boundaries\n"
        "- Use only the information provided in the prompt.\n"
        "- You do not see the original video frames directly.\n"
        "- Do not pretend to verify frame-by-frame visual alignment that is not supported by the provided text or hidden grounding evidence.\n"
        "- For structured answers, follow the task instructions and field explanations given in the user prompt.\n"
        "- Treat semantically equivalent natural-language answers as correct even if the wording differs.\n\n"
        "## Output Format\n"
        "- Return strict JSON only with key: feedback.\n"
        "- Do not return markdown, prose outside JSON, or extra keys.\n\n"
        "## Feedback Requirements\n"
        "- The feedback should be concise, factual, correction-oriented, and usually two to four sentences.\n"
        "- Mention only the issues that actually appear; do not give an exhaustive recap.\n"
        "- If the response is fully correct and no clear issue is observed, provide a brief positive feedback sentence stating that the final answer is correct and the reasoning broadly supports it.\n\n"
        "## Evaluation Focus\n"
        "1. Answer diagnosis.\n"
        "Say whether the student final answer is fully correct, partly correct, or wrong. Briefly name the problematic part. "
        "For natural-language answers, focus on semantic meaning rather than exact wording. For structured answers, say which part is wrong or missing.\n"
        "2. Reasoning-versus-answer consistency.\n"
        "Judge whether the student's reasoning broadly supports the final answer. If the reasoning points to one event, object, text clue, time range, or spatial reference but the final answer states another, say so. If the reasoning is too weak, too broad, or too incomplete to justify the final answer, say that clearly.\n"
        "3. High-level error cause.\n"
        "When the response is not fully correct, pick the single most likely high-level cause and mention it briefly. Prefer one of these categories: the reasoning focused on the wrong event, time span, or object; the reasoning was too broad or lacked enough evidence to support such a specific answer; the reasoning was mostly on the right track but the final answer overreached, drifted, or stated the wrong thing; or the output was incomplete and never produced a real final answer. Do not invent fine-grained step-by-step visual mistakes when they are not well supported by the text.\n\n"
        "## Task\n"
        "Compare the verified answer against the student's final answer and reasoning text, then return the feedback JSON.\n"
    )
    if keyframe_object_evidence:
        system_prompt += "Hidden grounding evidence may be provided in the user prompt.\n"
    else:
        system_prompt += "No hidden grounding evidence is required for this sample.\n"

    if not has_answer_tag:
        system_prompt += (
            "The student output does not contain a final <answer> or </answer> tag. "
            "You must still analyze the student's reasoning text and provide corrective feedback. "
            "In the feedback, explicitly point out that the final answer was not produced and that the response "
            "may have been truncated because the thinking section was too long. "
            "Treat the missing final answer as an incomplete response.\n"
        )

    user_prompt = (
        "Please judge the following sample.\n\n"
        + _render_reference_block(
            task=task,
            question=question,
            standard_answer=standard_answer,
            keyframe_object_evidence=keyframe_object_evidence,
        )
        + "\n\n"
        + _render_student_block(
            model_answer=model_answer if has_answer_tag else None,
            student_output=student_output,
        )
        + "\n"
    )
    return system_prompt, user_prompt


def judge_answer_and_feedback(
    *,
    question: str,
    standard_answer: str,
    model_answer: str | None,
    student_output: str | None = None,
    keyframe_object_evidence: list[dict[str, Any]] | None = None,
    task: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    model_override: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_feedback_chars: int = 1000,
) -> dict[str, Any]:
    question = (question or "").strip()
    standard_answer = (standard_answer or "").strip()
    model_answer = (model_answer or "").strip() or None
    student_output = (student_output or "").strip()

    if not question or not standard_answer or not student_output:
        return {
            "feedback": None,
            "raw_response": "",
        }

    api_base = normalize_api_base(api_base or os.environ.get("LLM_AS_A_JUDGE_BASE", ""))
    api_key = api_key or os.environ.get("LLM_AS_A_JUDGE_API_KEY", "")
    model_name = resolve_model_name(
        api_base,
        api_key,
        model_override or os.environ.get("LLM_AS_A_JUDGE_MODEL"),
        float(timeout),
    )
    client = get_client(api_base, api_key)

    system_prompt, user_prompt = _build_judge_prompts(
        question=question,
        standard_answer=standard_answer,
        model_answer=model_answer,
        student_output=student_output,
        keyframe_object_evidence=keyframe_object_evidence,
        task=task,
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=float(timeout),
    )
    raw_response = response.output_text if hasattr(response, "output_text") else str(response)
    parsed = extract_json_object(raw_response) or {}
    parsed["raw_response"] = raw_response
    return normalize_judge_result_payload(parsed, max_feedback_chars=max_feedback_chars)
