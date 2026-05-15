from __future__ import annotations

import os
from typing import Any


def ensure_pyav_compat() -> bool:
    """Bridge small API differences across PyAV versions."""
    try:
        import av  # type: ignore
    except Exception:
        return False

    if hasattr(av, "AVError"):
        return True

    fallback_error: Any = getattr(av, "FFmpegError", None)
    if fallback_error is None and hasattr(av, "error"):
        fallback_error = getattr(av.error, "FFmpegError", None) or getattr(av.error, "OSError", None)
    if fallback_error is None:
        return False

    av.AVError = fallback_error
    return True


def enable_torch_checkpoint_resume_compat() -> None:
    """
    Keep DeepSpeed checkpoint resume working under newer torch defaults.

    PyTorch 2.6 changed torch.load(..., weights_only=True) to be the default.
    DeepSpeed ZeRO checkpoints still rely on pickled config objects / enums.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    try:
        import torch.serialization as torch_serialization
    except Exception:
        return

    add_safe_globals = getattr(torch_serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    safe_globals: list[Any] = []
    try:
        from deepspeed.runtime.zero.config import (
            DeepSpeedZeroConfig,
            DeepSpeedZeroOffloadOptimizerConfig,
            DeepSpeedZeroOffloadParamConfig,
            OffloadDeviceEnum,
            ZeroStageEnum,
        )

        safe_globals.extend(
            [
                ZeroStageEnum,
                OffloadDeviceEnum,
                DeepSpeedZeroConfig,
                DeepSpeedZeroOffloadOptimizerConfig,
                DeepSpeedZeroOffloadParamConfig,
            ]
        )
    except Exception:
        pass

    try:
        from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

        safe_globals.append(OffloadStateTypeEnum)
    except Exception:
        pass

    if safe_globals:
        add_safe_globals(safe_globals)
