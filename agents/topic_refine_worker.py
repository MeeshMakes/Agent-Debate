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
import re
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
        "You are a debate curriculum designer and evolution planner. "
        "Given a debate topic, current talking points, and full outcome telemetry from "
        "the most recent session, rewrite the description and talking points for the NEXT "
        "session. Rules:\n"
        "  - Preserve the conceptual core while strengthening specificity and depth.\n"
        "  - Integrate session truths as established context, not repeated debate questions.\n"
        "  - Use unresolved tensions, low-quality dimensions, verdict rationale, and transcript signals to shape new focus.\n"
        "  - Do NOT rehash solved/covered ground as primary new talking points.\n"
        "  - Keep each talking point under 170 characters and contestable.\n"
        "  - Keep 8-14 talking points total.\n"
        "  - Description must be dense, long-form continuation (roughly 1200+ chars), not a short summary.\n"
        "  - Description MUST include these exact section headers:\n"
        "      CONTINUATION BASELINE\n"
        "      ESTABLISHED AGREEMENTS\n"
        "      ACTIVE DISAGREEMENTS\n"
        "      OPEN PROBLEMS / UNCONCLUDED THREADS\n"
        "      NEXT DISCOVERY FRONTIER\n"
        "  - Return ONLY valid JSON with no markdown fences, no prose outside the JSON.\n"
        "  - JSON shape: {\"description\": \"...\", "
        "\"talking_points\": [\"...\", ...], \"session_brief\": \"...\"}\n"
        "  - session_brief: one-sentence summary of this session's evolution effect (≤ 160 chars)."
    )

    def __init__(
        self,
        *,
        title: str,
        description: str,
        talking_points: list[str],
        resolution_data: dict,
        scoring_data: dict | None = None,
        diagnostics_data: dict | None = None,
        graph_rows: list | None = None,
        session_brief_data: dict | None = None,
        continuation_context: dict | None = None,
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
        self._scoring_data = scoring_data or {}
        self._diagnostics_data = diagnostics_data or {}
        self._graph_rows = graph_rows or []
        self._session_brief_data = session_brief_data or {}
        self._continuation_context = continuation_context or {}
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
        total_memory_facts = rd.get("total_memory_facts", 0)

        sd = self._scoring_data if isinstance(self._scoring_data, dict) else {}
        winner = sd.get("winner", "")
        margin = sd.get("margin", "")
        reason = sd.get("reason", "")
        summary = sd.get("summary", "")
        astra_avg = sd.get("astra_avg", "")
        nova_avg = sd.get("nova_avg", "")
        turns = sd.get("turns", "")

        dd = self._diagnostics_data if isinstance(self._diagnostics_data, dict) else {}
        issues = dd.get("issues", []) if isinstance(dd.get("issues", []), list) else []
        event_counts = dd.get("event_counts", {}) if isinstance(dd.get("event_counts", {}), dict) else {}
        graph_nodes = dd.get("graph_nodes", 0)

        graph_counts: dict[str, int] = {}
        for row in self._graph_rows if isinstance(self._graph_rows, list) else []:
            node_type = ""
            if isinstance(row, (list, tuple)) and row:
                node_type = str(row[0]).strip().lower()
            elif isinstance(row, dict):
                node_type = str(row.get("type", "")).strip().lower()
            if not node_type:
                continue
            graph_counts[node_type] = graph_counts.get(node_type, 0) + 1

        prior_mode = str(self._session_brief_data.get("mode", "")).strip()
        prior_repo = str(self._session_brief_data.get("repo_path", "")).strip()
        prior_source = str(self._session_brief_data.get("source_session_id", "")).strip()

        tp_list = "\n".join(f"  - {tp}" for tp in self._talking_points)
        truth_list = "\n".join(f"  ✓ {t}" for t in truths) if truths else "  (none recorded)"
        problem_list = "\n".join(f"  ✗ {p}" for p in problems) if problems else "  (none recorded)"
        sub_list = "\n".join(f"  • {s}" for s in sub_topics) if sub_topics else "  (none)"
        issues_list = "\n".join(f"  ! {i}" for i in issues[:10]) if issues else "  (none)"
        graph_count_list = "\n".join(f"  • {k}: {v}" for k, v in sorted(graph_counts.items())) or "  (none)"

        cx = self._continuation_context if isinstance(self._continuation_context, dict) else {}
        original = cx.get("original", {}) if isinstance(cx.get("original", {}), dict) else {}
        current = cx.get("current", {}) if isinstance(cx.get("current", {}), dict) else {}
        covered_ground = cx.get("covered_ground", []) if isinstance(cx.get("covered_ground", []), list) else []
        cx_agreements = cx.get("agreements", []) if isinstance(cx.get("agreements", []), list) else []
        cx_disagreements = cx.get("disagreements", []) if isinstance(cx.get("disagreements", []), list) else []
        cx_open = cx.get("open_problems", []) if isinstance(cx.get("open_problems", []), list) else []
        cx_unresolved = cx.get("unresolved", []) if isinstance(cx.get("unresolved", []), list) else []
        cx_signals = cx.get("transcript_signals", []) if isinstance(cx.get("transcript_signals", []), list) else []
        cx_trend = cx.get("scoring_trend", []) if isinstance(cx.get("scoring_trend", []), list) else []

        def _fmt(items: list, max_items: int = 12) -> str:
            out: list[str] = []
            for raw in items[:max_items]:
                s = " ".join(str(raw).split()).strip()
                if s:
                    out.append(f"  - {s[:260]}")
            return "\n".join(out) if out else "  (none)"

        return (
            f"TOPIC TITLE: {self._title}\n\n"
            f"CURRENT DESCRIPTION:\n{self._description}\n\n"
            f"CURRENT TALKING POINTS:\n{tp_list}\n\n"
            "THREAD CONTINUATION INTELLIGENCE:\n"
            f"  thread_session_count={cx.get('thread_session_count', 0)}\n"
            f"  session_ids={cx.get('thread_session_ids', [])}\n\n"
            "ORIGINAL BASELINE (first session in thread):\n"
            f"  title={str(original.get('title', '')).strip()}\n"
            f"  description:\n{str(original.get('description', '')).strip()[:2600]}\n"
            "  talking_points:\n"
            f"{_fmt(original.get('talking_points', []) if isinstance(original.get('talking_points', []), list) else [], max_items=14)}\n\n"
            "CURRENT EVOLVED STATE (latest pre-refine):\n"
            f"  title={str(current.get('title', '')).strip()}\n"
            f"  description:\n{str(current.get('description', '')).strip()[:2200]}\n"
            "  talking_points:\n"
            f"{_fmt(current.get('talking_points', []) if isinstance(current.get('talking_points', []), list) else [], max_items=14)}\n\n"
            "COVERED GROUND (avoid repeating as primary points):\n"
            f"{_fmt(covered_ground, max_items=20)}\n\n"
            "THREAD AGREEMENTS:\n"
            f"{_fmt(cx_agreements, max_items=20)}\n\n"
            "THREAD DISAGREEMENTS:\n"
            f"{_fmt(cx_disagreements, max_items=20)}\n\n"
            "THREAD OPEN PROBLEMS:\n"
            f"{_fmt(cx_open, max_items=24)}\n\n"
            "THREAD UNRESOLVED QUESTIONS:\n"
            f"{_fmt(cx_unresolved, max_items=24)}\n\n"
            "TRANSCRIPT SIGNALS (recent key lines):\n"
            f"{_fmt(cx_signals, max_items=24)}\n\n"
            "SCORING TREND ACROSS THREAD:\n"
            f"{_fmt(cx_trend, max_items=12)}\n\n"
            "CURRENT SESSION BRIEF METADATA:\n"
            f"  mode={prior_mode or 'unknown'}\n"
            f"  repo_path={prior_repo or '(none)'}\n"
            f"  source_session_id={prior_source or '(none)'}\n\n"
            f"SESSION OUTCOME:\n"
            f"  Truths / agreements reached:\n{truth_list}\n"
            f"  Problems / tensions surfaced:\n{problem_list}\n"
            f"  Sub-topics explored:\n{sub_list}\n"
            f"  Total memory facts: {total_memory_facts}\n\n"
            "VERDICT + SCORING:\n"
            f"  winner={winner or '(unknown)'} margin={margin or '(unknown)'} turns={turns or '(unknown)'}\n"
            f"  astra_avg={astra_avg} nova_avg={nova_avg}\n"
            f"  reason: {reason or '(none)'}\n"
            f"  summary: {summary or '(none)'}\n\n"
            "SESSION DIAGNOSTICS:\n"
            f"  graph_nodes={graph_nodes}\n"
            f"  event_counts={json.dumps(event_counts, ensure_ascii=False)[:1200]}\n"
            f"  issues:\n{issues_list}\n"
            "GRAPH NODE DISTRIBUTION:\n"
            f"{graph_count_list}\n\n"
            "Rewrite the description and talking points for the NEXT session so the debate evolves organically "
            "instead of repeating prior framing. Preserve continuity but raise depth and contestability."
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

        raw = str(body.get("message", {}).get("content", "") or body.get("response", "")).strip()
        if not raw:
            raise RuntimeError("Topic refinement model returned empty content.")

        # Strip optional markdown fence
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]

        result: dict[str, Any] = self._parse_result_payload(raw)

        # Ensure required keys exist (with safe fallbacks)
        result.setdefault("description", self._description)
        if not isinstance(result.get("talking_points"), list):
            result["talking_points"] = self._talking_points
        result.setdefault("session_brief", "Session completed.")
        return self._postprocess_result(result)

    def _parse_result_payload(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        candidates: list[str] = [text]

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj_slice = text[start : end + 1].strip()
            if obj_slice and obj_slice not in candidates:
                candidates.insert(0, obj_slice)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return self._coerce_result_from_text(text)

    def _coerce_result_from_text(self, text: str) -> dict[str, Any]:
        lines = [" ".join(str(line).split()).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        talking_points: list[str] = []
        description_lines: list[str] = []

        for line in lines:
            tp = self._extract_talking_point_line(line)
            if tp:
                talking_points.append(tp)
            else:
                lower = line.lower()
                if lower in {"description", "talking points", "talking_points", "session_brief"}:
                    continue
                description_lines.append(line)

        description_seed = "\n".join(description_lines).strip()
        if not description_seed:
            description_seed = self._description

        session_brief = ""
        brief_match = re.search(r"session[_\s-]*brief\s*[:=]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if brief_match:
            session_brief = " ".join(brief_match.group(1).split()).strip()

        return {
            "description": description_seed,
            "talking_points": talking_points,
            "session_brief": session_brief,
        }

    @staticmethod
    def _extract_talking_point_line(line: str) -> str:
        bullet_patterns = (
            r"^[-*•]\s+(.*)$",
            r"^\d+[\.)]\s+(.*)$",
        )
        for pattern in bullet_patterns:
            match = re.match(pattern, line)
            if match:
                candidate = " ".join(match.group(1).split()).strip()
                if candidate:
                    return candidate
        return ""

    def _postprocess_result(self, result: dict[str, Any]) -> dict[str, Any]:
        desc = str(result.get("description", "") or "").strip() or self._description
        tps_raw = result.get("talking_points", [])
        tps = [" ".join(str(tp).split()).strip() for tp in tps_raw if " ".join(str(tp).split()).strip()] if isinstance(tps_raw, list) else []
        session_brief = str(result.get("session_brief", "") or "").strip()

        required_headers = (
            "CONTINUATION BASELINE",
            "ESTABLISHED AGREEMENTS",
            "ACTIVE DISAGREEMENTS",
            "OPEN PROBLEMS / UNCONCLUDED THREADS",
            "NEXT DISCOVERY FRONTIER",
        )
        if len(desc) < 900 or any(h not in desc for h in required_headers):
            desc = self._build_fallback_description(seed=desc)

        if len(tps) < 8:
            tps.extend(self._build_fallback_talking_points(existing=tps, desired=10))

        deduped: list[str] = []
        seen: set[str] = set()
        for tp in tps:
            p = tp[:170].strip()
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
            if len(deduped) >= 14:
                break

        if not session_brief:
            session_brief = "Debate evolved: established points retained; unresolved tensions promoted into next-step discovery prompts."
        if len(session_brief) > 160:
            session_brief = session_brief[:157].rstrip() + "..."

        return {
            "description": desc,
            "talking_points": deduped,
            "session_brief": session_brief,
        }

    def _build_fallback_description(self, seed: str = "") -> str:
        cx = self._continuation_context if isinstance(self._continuation_context, dict) else {}
        original = cx.get("original", {}) if isinstance(cx.get("original", {}), dict) else {}

        def _items(name: str, max_items: int) -> list[str]:
            raw = cx.get(name, [])
            if not isinstance(raw, list):
                return []
            out: list[str] = []
            seen: set[str] = set()
            for item in raw:
                s = " ".join(str(item).split()).strip()
                if not s:
                    continue
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
                if len(out) >= max_items:
                    break
            return out

        agreements = _items("agreements", 10)
        disagreements = _items("disagreements", 10)
        problems = _items("open_problems", 14)
        unresolved = _items("unresolved", 14)
        covered = _items("covered_ground", 12)

        def _block(rows: list[str]) -> str:
            if not rows:
                return "- (none)"
            return "\n".join(f"- {r}" for r in rows)

        baseline_seed = seed.strip()
        if len(baseline_seed) < 220:
            baseline_seed = (
                f"Original baseline: {str(original.get('description', '')).strip()[:500]}\n\n"
                f"Current continuation focus: {self._description[:700]}"
            ).strip()

        return (
            "CONTINUATION BASELINE\n"
            f"{baseline_seed}\n\n"
            "ESTABLISHED AGREEMENTS\n"
            f"{_block(agreements)}\n\n"
            "ACTIVE DISAGREEMENTS\n"
            f"{_block(disagreements)}\n\n"
            "OPEN PROBLEMS / UNCONCLUDED THREADS\n"
            f"{_block((problems + unresolved)[:16])}\n\n"
            "NEXT DISCOVERY FRONTIER\n"
            "The next session must preserve the baseline architecture while explicitly avoiding repeated coverage of already-settled points. "
            "Use covered-ground items only as brief context anchors, then push into deeper validation, failure-mode stress tests, and unresolved chains from this thread.\n"
            f"Covered-ground references to avoid rehashing:\n{_block(covered[:10])}"
        ).strip()

    def _build_fallback_talking_points(self, existing: list[str], desired: int) -> list[str]:
        cx = self._continuation_context if isinstance(self._continuation_context, dict) else {}
        pool: list[str] = []
        for key in ("open_problems", "unresolved", "disagreements", "transcript_signals"):
            raw = cx.get(key, [])
            if isinstance(raw, list):
                pool.extend(str(x).strip() for x in raw if str(x).strip())

        out: list[str] = []
        seen = {" ".join(s.split()).strip().lower() for s in existing if " ".join(s.split()).strip()}

        for item in pool:
            s = " ".join(item.split()).strip()
            if not s:
                continue
            point = f"Advance unresolved thread: {s[:138]}"
            key = point.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(point)
            if len(existing) + len(out) >= max(8, desired):
                break

        return out
