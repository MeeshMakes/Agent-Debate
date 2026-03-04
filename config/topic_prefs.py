"""Persist selected debate topic across app restarts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PREFS_FILE = Path(__file__).parent / "topic_prefs.json"


def load_topic_prefs() -> dict[str, Any]:
    """Return saved topic selection data or empty defaults."""
    defaults: dict[str, Any] = {
        "title": "",
        "context": "",
        "tp_key": "",
        "repo_watchdog_meta": None,
    }
    if _PREFS_FILE.exists():
        try:
            data = json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
            repo_meta = data.get("repo_watchdog_meta")
            return {
                "title": str(data.get("title", "")),
                "context": str(data.get("context", "")),
                "tp_key": str(data.get("tp_key", "")),
                "repo_watchdog_meta": repo_meta if isinstance(repo_meta, dict) else None,
            }
        except Exception:
            pass
    return defaults


def save_topic_prefs(
    *,
    title: str,
    context: str,
    tp_key: str,
    repo_watchdog_meta: dict[str, Any] | None,
) -> None:
    """Persist selected debate configuration to disk."""
    payload = {
        "title": title,
        "context": context,
        "tp_key": tp_key,
        "repo_watchdog_meta": repo_watchdog_meta if isinstance(repo_watchdog_meta, dict) else None,
    }
    _PREFS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
