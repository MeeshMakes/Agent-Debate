"""TopicRefineWorker — QThread that rewrites a debate topic's description and talking
points after a session ends, using the session's resolution data.

The worker calls Ollama (same endpoint as the debate agents) and asks it to:
  - Incorporate what was learned / resolved
  - Flag what remains genuinely contested
  - Keep language concise and debate-ready

Output JSON: {"description": "...", "talking_points": ["...", ...], "session_brief": "..."}
This is saved via SessionManager.save_tp_refinement(tp_key, result).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal


class TopicRefineWorker(QThread):
    """Asks the LLM to refine a talking point after a debate session.

    Signals
    -------
    finished(dict)   — emit refined data dict on success
    failed(str)      — emit error message on failure
    """

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    _SYSTEM = (
        "You are a debate curriculum designer. "
        "Given a debate topic, its original talking points, and a summary of what was "
        "resolved in the most recent session, rewrite the description and talking points "
        "for the NEXT session. Rules:\n"
        "  - Preserve the conceptual core of the original topic.\n"
        "  - Integrate the key findings/truths from this session.\n"
        "  - Focus the new talking points on what remains unresolved or newly opened.\n"
        "  - Keep each talking point under 120 characters.\n"
        "  - Return ONLY valid JSON with no markdown fences, no prose outside the JSON.\n"
        "  - JSON shape: {\"description\": \"...\", "
        "\"talking_points\": [\"...\", ...], \"session_brief\": \"...\"}\n"
        "  - session_brief: one-sentence summary of this session's outcome (≤ 160 chars)."
    )

    def __init__(
        self,
        *,
        title: str,
        description: str,
        talking_points: list[str],
        resolution_data: dict,
        model: str,
        tp_key: str,
        session_manager,  # SessionManager — avoid circular import with type hint
        ollama_host: str = "http://localhost:11434",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._description = description
        self._talking_points = talking_points
        self._resolution_data = resolution_data
        self._model = model
        self._tp_key = tp_key
        self._sm = session_manager
        self._host = ollama_host.rstrip("/")

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            prompt = self._build_prompt()
            result = self._call_ollama(prompt)
            self._sm.save_tp_refinement(self._tp_key, result)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
        rd = self._resolution_data
        truths = rd.get("truths", [])
        problems = rd.get("problems", [])
        sub_topics = rd.get("sub_topics", [])

        tp_list = "\n".join(f"  - {tp}" for tp in self._talking_points)
        truth_list = "\n".join(f"  ✓ {t}" for t in truths) if truths else "  (none recorded)"
        problem_list = "\n".join(f"  ✗ {p}" for p in problems) if problems else "  (none recorded)"
        sub_list = "\n".join(f"  • {s}" for s in sub_topics) if sub_topics else "  (none)"

        return (
            f"TOPIC TITLE: {self._title}\n\n"
            f"CURRENT DESCRIPTION:\n{self._description}\n\n"
            f"CURRENT TALKING POINTS:\n{tp_list}\n\n"
            f"SESSION OUTCOME:\n"
            f"  Truths / agreements reached:\n{truth_list}\n"
            f"  Problems / tensions surfaced:\n{problem_list}\n"
            f"  Sub-topics explored:\n{sub_list}\n\n"
            "Now rewrite the description and talking points for the NEXT session."
        )

    def _call_ollama(self, user_msg: str) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 1024},
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())

        raw = body["message"]["content"].strip()

        # Strip optional markdown fence
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]

        result: dict[str, Any] = json.loads(raw)

        # Ensure required keys exist (with safe fallbacks)
        result.setdefault("description", self._description)
        if not isinstance(result.get("talking_points"), list):
            result["talking_points"] = self._talking_points
        result.setdefault("session_brief", "Session completed.")

        return result
