"""Topic Assistant Worker — LLM-backed agent that creates talking points.

Used by both:
  A) User-driven creation — user types a request, assistant generates a full talking point
  B) Autonomous creation — system promotes a segue concept, assistant builds the talking point

The worker calls Ollama to generate structured talking point data and saves it
to the autonomous talking point store.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


_SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"


class TopicAssistantWorker(QThread):
    """Background LLM call that generates a full talking point from a request.

    Signals
    -------
    finished(dict)  — {"title", "description", "talking_points": list[str], "origin": "user"|"autonomous"}
    failed(str)     — error message
    chat_reply(str) — conversational response to display in the assistant panel
    """

    finished   = pyqtSignal(dict)
    failed     = pyqtSignal(str)
    chat_reply = pyqtSignal(str)

    def __init__(
        self,
        user_message: str,
        *,
        model: str = "qwen3:30b",
        existing_titles: list[str] | None = None,
        analytics_summary: str = "",
        segue_context: str = "",
        origin: str = "user",           # "user" or "autonomous"
        ollama_host: str = "http://localhost:11434",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._user_message = user_message
        self._model = model
        self._existing_titles = existing_titles or []
        self._analytics_summary = analytics_summary
        self._segue_context = segue_context
        self._origin = origin
        self._ollama_host = ollama_host.rstrip("/")

    def run(self) -> None:
        try:
            result = self._run_inner()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_inner(self) -> dict:
        import urllib.request

        existing_block = ""
        if self._existing_titles:
            titles_text = "\n".join(f"  - {t}" for t in self._existing_titles[:30])
            existing_block = f"\n\nEXISTING TALKING POINTS (do NOT duplicate these):\n{titles_text}"

        analytics_block = ""
        if self._analytics_summary:
            analytics_block = f"\n\nANALYTICS CONTEXT:\n{self._analytics_summary[:2000]}"

        segue_block = ""
        if self._segue_context:
            segue_block = f"\n\nSEGUE BUFFER CONTEXT (concepts agents referenced laterally):\n{self._segue_context[:1500]}"

        system_prompt = (
            "You are the Topic Assistant — an expert debate architect embedded in a dual-agent "
            "debate system. Your job is to create new talking points for debates.\n\n"
            "When the user asks you to create a talking point, respond in TWO parts:\n\n"
            "PART 1: A brief conversational reply (1-3 sentences) confirming what you're creating.\n\n"
            "PART 2: A JSON block wrapped in ```json ... ``` containing:\n"
            '{\n'
            '  "title": "Short debate topic title (max 80 chars)",\n'
            '  "description": "2-4 paragraph rich description providing full context, background, '
            'the core disagreement, constraints, and desired depth for the agents",\n'
            '  "talking_points": [\n'
            '    "Sharp contestable claim #1 that agents can argue for/against",\n'
            '    "Sharp contestable claim #2...",\n'
            '    "... (5-8 talking points)"\n'
            '  ]\n'
            '}\n\n'
            "Each talking point must be a specific, contestable position — not a question, "
            "not a vague theme. Something debaters can fight over with evidence.\n"
            "The description should be detailed enough to ground a multi-turn debate.\n"
            "Do NOT duplicate any existing talking point listed below.\n"
            f"{existing_block}{analytics_block}{segue_block}"
        )

        user_content = self._user_message

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }

        url = f"{self._ollama_host}/api/chat"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        text = (
            resp_data.get("message", {}).get("content", "")
            or resp_data.get("response", "")
        ).strip()

        if not text:
            raise RuntimeError("LLM returned an empty response")

        # Parse out the JSON block
        result = self._parse_response(text)
        result["origin"] = self._origin

        # Emit the conversational part
        # Everything before the ```json block is the chat reply
        json_start = text.find("```json")
        if json_start > 0:
            chat_text = text[:json_start].strip()
        else:
            chat_text = f"Created talking point: **{result.get('title', 'New Topic')}**"
        self.chat_reply.emit(chat_text)

        return result

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Extract the JSON block from the LLM response."""
        # Try to find ```json ... ``` block
        import re
        json_match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        else:
            # Try to find raw JSON object
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                raw = text[brace_start:brace_end + 1]
            else:
                raise RuntimeError("Could not find JSON block in LLM response")

        data = json.loads(raw)
        title = data.get("title", "").strip()
        description = data.get("description", "").strip()
        talking_points = data.get("talking_points", [])

        if not title:
            raise RuntimeError("Generated talking point has no title")
        if not talking_points:
            raise RuntimeError("Generated talking point has no talking points")

        return {
            "title": title,
            "description": description,
            "talking_points": [tp.strip() for tp in talking_points if tp.strip()],
        }


class AutonomousPromotionWorker(QThread):
    """Background worker that reviews the segue buffer and promotes candidates.

    Fetches promotion candidates from the segue buffer, calls the LLM
    to generate full talking points for each, and saves them.

    If the segue buffer is empty / has no candidates, the worker finishes
    immediately with all_done(0).
    """

    promoted   = pyqtSignal(dict)     # emitted per promoted talking point
    progress   = pyqtSignal(str)      # status message
    all_done   = pyqtSignal(int)      # total count promoted
    failed     = pyqtSignal(str)

    def __init__(
        self,
        *,
        model: str = "qwen3:30b",
        existing_titles: list[str] | None = None,
        analytics_summary: str = "",
        threshold: int = 3,
        ollama_host: str = "http://localhost:11434",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._existing_titles = existing_titles or []
        self._analytics_summary = analytics_summary
        self._threshold = threshold
        self._ollama_host = ollama_host.rstrip("/")
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            from core.segue_buffer import get_segue_buffer
            buf = get_segue_buffer()
            candidates = buf.get_promotion_candidates(
                self._existing_titles, min_mentions=self._threshold
            )
        except Exception as exc:
            self.failed.emit(f"Could not load segue buffer: {exc}")
            return

        if not candidates:
            self.progress.emit("No promotion candidates in segue buffer yet.")
            self.all_done.emit(0)
            return

        self.progress.emit(f"Found {len(candidates)} candidate(s) to evaluate…")
        promoted_count = 0

        for entry in candidates:
            if self._stop:
                break

            self.progress.emit(f'Promoting "{entry.concept}"…')

            try:
                sessions_str = (
                    ", ".join(list(entry.source_sessions)[:5])
                    if entry.source_sessions else "unknown"
                )
                agents_str = (
                    ", ".join(set(entry.source_agents))
                    if entry.source_agents else "agents"
                )

                user_msg = (
                    f"Create a new debate talking point about: {entry.concept}\n\n"
                    f"Context from prior debates where agents referenced this:\n"
                    f"{(entry.surrounding_context or '')[:500]}\n\n"
                    f"This concept came up {entry.mention_count} times "
                    f"across sessions {sessions_str}. "
                    f"Agents {agents_str} referenced it "
                    f"while debating: {entry.related_tp or 'general discussion'}"
                )

                worker = TopicAssistantWorker(
                    user_msg,
                    model=self._model,
                    existing_titles=self._existing_titles,
                    analytics_summary=self._analytics_summary,
                    origin="autonomous",
                    ollama_host=self._ollama_host,
                )

                # Run synchronously within this thread
                result = worker._run_inner()
                result["origin"] = "autonomous"
                result["segue_concept"] = entry.concept
                result["segue_mentions"] = entry.mention_count
                result["segue_sessions"] = (
                    list(entry.source_sessions) if entry.source_sessions else []
                )

                self.promoted.emit(result)
                promoted_count += 1

                # Mark as promoted in the segue buffer
                buf.mark_promoted(entry.concept)

                # Add to existing titles to avoid duplicates in sequence
                self._existing_titles.append(result.get("title", ""))

            except Exception as exc:
                self.progress.emit(f'Skipped "{entry.concept}": {exc}')
                continue

        self.all_done.emit(promoted_count)
