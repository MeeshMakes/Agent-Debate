"""Persist per-agent Ollama model selections across sessions."""
from __future__ import annotations

import json
from pathlib import Path

_PREFS_FILE = Path(__file__).parent / "model_prefs.json"

DEFAULT_LEFT_MODEL = "gemma3:27b"
DEFAULT_RIGHT_MODEL = "qwen3:30b"


def load_model_prefs() -> dict[str, str]:
    """Return saved model prefs or defaults."""
    if _PREFS_FILE.exists():
        try:
            data = json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
            return {
                "left_model": str(data.get("left_model", DEFAULT_LEFT_MODEL)),
                "right_model": str(data.get("right_model", DEFAULT_RIGHT_MODEL)),
            }
        except Exception:
            pass
    return {"left_model": DEFAULT_LEFT_MODEL, "right_model": DEFAULT_RIGHT_MODEL}


def save_model_prefs(left_model: str, right_model: str) -> None:
    """Persist model selections to disk."""
    _PREFS_FILE.write_text(
        json.dumps(
            {"left_model": left_model, "right_model": right_model},
            indent=2,
        ),
        encoding="utf-8",
    )
