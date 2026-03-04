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
        workspace_context: str = "",
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
        self._workspace_context = workspace_context
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

        workspace_block = ""
        if self._workspace_context:
            workspace_block = (
                "\n\nCURRENT WORKSPACE CONTEXT (active topic/repo/editor state):\n"
                f"{self._workspace_context[:3000]}"
            )

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
            "CRITICAL CONTEXT-PRESERVATION RULES:\n"
            "  - Preserve ALL major concerns, constraints, and goals from the user's request.\n"
            "  - Reframe intelligently, but do not drop the user's core intent.\n"
            "  - Include a subsection in description named: 'Preserved User Intent'.\n"
            "  - In that subsection, list the user's highest-priority concerns as explicit bullets.\n"
            "  - Ensure talking_points include these preserved concerns as contestable claims.\n"
            "CODE-LEVEL REFERENCE RULES (CRITICAL when the user mentions specific code):\n"
            "  - If the user mentions a specific function or file using backticks (e.g. `stable_bucket` in\n"
            "    `build_finetune_artifacts.py`), that EXACT identifier and file MUST appear verbatim,\n"
            "    with backticks preserved, in the relevant talking point.\n"
            "  - Do NOT abstract away code-level specifics. Do NOT replace `stable_bucket in\n"
            "    build_finetune_artifacts.py` with vague phrases like 'hash-based splitting logic'.\n"
            "  - Each talking point about a code concern MUST name: the specific function, the file it lives\n"
            "    in, and the exact technical problem. Format: `function()` in `file.py` — [issue] — [why it\n"
            "    matters for the debate].\n"
            "  - If the request mentions a repository path, begin the description with:\n"
            "    'Repository: [repo_path]' as the very first line.\n"
            "  - The FIRST element of the talking_points array MUST always be exactly:\n"
            "    \"\u2014 anchor every argument to these:\"\n"
            "    (This is the display header for the talking points list in the UI.)\n"
            "If the request involves software architecture, system design, UI, or UX, the description must include:\n"
            "  - a 'Plain-Text Diagram' subsection using ASCII arrows (e.g. [Panel] -> [Orchestrator] -> [Store])\n"
            "  - an 'Implementation Backlog' subsection with at least 5 concrete change items\n"
            "Each backlog item must name: target surface/component, user pain, exact interaction change, and acceptance check.\n"
            "Do NOT duplicate any existing talking point listed below.\n"
            f"{existing_block}{analytics_block}{segue_block}{workspace_block}"
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
        result = self._enforce_context_preservation(result, self._user_message)
        result = self._ensure_anchor_header(result)
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

    _ANCHOR_HDR = "\u2014 anchor every argument to these:"

    @staticmethod
    def _ensure_anchor_header(result: dict) -> dict:
        """Guarantee the first talking_points item is the anchor display header."""
        tps = list(result.get("talking_points") or [])
        hdr = TopicAssistantWorker._ANCHOR_HDR
        # Strip any existing anchor-like header to avoid duplication
        while tps and "anchor every argument" in str(tps[0]).lower():
            tps.pop(0)
        tps = [hdr] + tps
        return {**result, "talking_points": tps}

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

    @staticmethod
    def _enforce_context_preservation(result: dict, user_message: str) -> dict:
        description = str(result.get("description", "") or "").strip()
        talking_points = [str(tp).strip() for tp in result.get("talking_points", []) if str(tp).strip()]
        anchors = TopicAssistantWorker._extract_intent_anchors(user_message, max_items=8)

        if anchors:
            lowered_desc = description.lower()
            missing_anchors = [a for a in anchors if a.lower() not in lowered_desc]

            if missing_anchors:
                preserve_lines = "\n".join(f"- {a}" for a in missing_anchors[:6])
                if "preserved user intent" not in lowered_desc:
                    description = (
                        f"{description}\n\n"
                        "Preserved User Intent:\n"
                        f"{preserve_lines}"
                    ).strip()
                else:
                    description = (
                        f"{description}\n"
                        f"{preserve_lines}"
                    ).strip()

            # Ensure talking points retain user intent anchors as contestable claims.
            tp_blob = "\n".join(talking_points).lower()
            added = 0
            for anchor in anchors:
                if added >= 3:
                    break
                if anchor.lower() in tp_blob:
                    continue
                # Use code-specific template if anchor looks like a code reference
                import re as _re
                if _re.search(r'`[^`]+`', anchor):
                    talking_points.append(
                        f"{anchor} — this implementation detail must be addressed as a first-class concern rather than deferred."
                    )
                else:
                    talking_points.append(
                        f"The debate must directly address: {anchor}"
                    )
                added += 1

        # Keep list focused and non-empty.
        if len(talking_points) > 10:
            talking_points = talking_points[:10]

        return {
            "title": str(result.get("title", "") or "").strip(),
            "description": description,
            "talking_points": talking_points,
        }

    @staticmethod
    def _extract_intent_anchors(text: str, max_items: int = 8) -> list[str]:
        import re

        src = (text or "").strip()
        if not src:
            return []

        anchors: list[str] = []

        # Priority 0: sentences that contain backtick code references (function/file names).
        # Capture the FULL sentence so the specific identifier is preserved verbatim.
        code_ref_sentences = re.split(r"\n+", src)
        for raw in code_ref_sentences:
            s = " ".join(raw.split()).strip("-• ")
            if len(s) < 10 or len(s) > 400:
                continue
            if re.search(r"`[^`]+`", s):
                anchors.append(s)

        # Priority 1: markdown emphasis **...** as explicit user priorities.
        for item in re.findall(r"\*\*([^*]{3,120})\*\*", src):
            cleaned = " ".join(item.split()).strip(" .,:;-")
            if len(cleaned) >= 3:
                anchors.append(cleaned)

        # Priority 2: high-signal sentences that express goals/constraints.
        sentence_parts = re.split(r"(?<=[.!?])\s+|\n+", src)
        signal_terms = (
            "must", "need", "goal", "objective", "constraint", "cannot", "can't", "should",
            "preserve", "blockchain", "moral", "dataset", "agent", "governance", "api", "cli",
        )
        for raw in sentence_parts:
            s = " ".join(raw.split()).strip("-• ")
            if len(s) < 24 or len(s) > 220:
                continue
            low = s.lower()
            if any(term in low for term in signal_terms):
                anchors.append(s)

        # Deduplicate while preserving order.
        unique: list[str] = []
        seen: set[str] = set()
        for a in anchors:
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(a)
            if len(unique) >= max_items:
                break
        return unique

