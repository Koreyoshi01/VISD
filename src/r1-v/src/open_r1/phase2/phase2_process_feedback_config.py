from __future__ import annotations


def _resolve_config_value(config_like, key: str):
    if config_like is None:
        return None
    return getattr(config_like, key, None)


def resolve_phase2_process_feedback_request_kwargs(config_like) -> dict[str, str | None]:
    return {
        "model_override": _resolve_config_value(
            config_like,
            "phase2_process_feedback_model",
        ),
        "api_base": _resolve_config_value(
            config_like,
            "phase2_process_feedback_base_url",
        ),
        "api_key": _resolve_config_value(
            config_like,
            "phase2_process_feedback_api_key",
        ),
    }
