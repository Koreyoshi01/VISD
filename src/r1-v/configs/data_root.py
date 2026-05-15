from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]

# Prefer explicit overrides, then fall back to the repository root.
DATA_ROOT = os.environ.get("VISIONCOACH_DATA_ROOT") or os.environ.get("DATA_ROOT") or str(_REPO_ROOT)
