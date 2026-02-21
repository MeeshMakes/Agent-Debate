"""DebateSummaryWorker — QThread that generates the final debate verdict.

After every debate ends the orchestrator has:
    - scored every turn (quality composite per agent)
    - logged all arbiter interventions
    - accumulated all public messages

This worker sends that data to the LLM and asks it to:
    1. Declare a winner (or draw) with a clear reason
    2. Write a brief overall summary of the debate arc

Results are:
    - Emitted back as a dict for the ScoringPanel
    - Written to {session_path}/scoring_report.json (machine-readable)
    - Written to {session_path}/scoring_report.md  (human-readable)
    - Appended to the global analytics store at sessions/_analytics/scoring_log.jsonl
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

_OLLAMA_HOST = "http://localhost:11434"


class DebateSummaryWorker(QThread):
    """Single LLM call → winner + summary → saved to disk.

    Signals
    -------
    finished(dict)   success  → {"winner","margin","reason","summary",
                                  "astra_avg","nova_avg","turns"}
    failed(str)      error message
    """

    finished = pyqtSignal(dict)
    failed   = pyqtSignal(str)

    _SYSTEM = (
        "You are the Arbiter — the impartial judge of a structured AI debate.\n"
        "You have access to every agent's per-turn quality scores and every intervention "
        "you made during the debate. Your job: deliver one precise, honest verdict.\n\n"
        "Return ONLY valid JSON, no markdown fences, no prose outside:\n"
        "{\n"
        '  "winner": "Astra" | "Nova" | "Draw",\n'
        '  "margin": "clear" | "close" | "draw",\n'
        '  "reason": "<2-4 sentences explaining WHY this agent won>",\n'
        '  "summary": "<4-8 sentences covering the whole debate arc, key turning points, '
        'strongest arguments from each side, and what remains unresolved>"\n'
        "}"
    )

    def __init__(
        self,
        *,
        topic: str,
        left_agent: str,
        right_agent: str,
        turn_scores: list[dict],      # [{"turn":int,"agent":str,"composite":float,...}]
        arbiter_events: list[dict],   # [{"turn":int,"message":str,"echo":bool}]
        model: str = "qwen3:30b",
        session_path: str = "",
        ollama_host: str = _OLLAMA_HOST,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._topic        = topic
        self._left         = left_agent
        self._right        = right_agent
        self._turn_scores  = turn_scores
        self._arb_events   = arbiter_events
        self._model        = model
        self._session_path = session_path
        self._host         = ollama_host.rstrip("/")

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            result = self._run_inner()
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _run_inner(self) -> dict:
        import urllib.request

        # ── aggregate scores ──────────────────────────────────────────────
        left_scores  = [r["composite"] for r in self._turn_scores if r["agent"].lower() == self._left.lower()]
        right_scores = [r["composite"] for r in self._turn_scores if r["agent"].lower() == self._right.lower()]
        left_avg  = round(statistics.mean(left_scores),  3) if left_scores  else 0.0
        right_avg = round(statistics.mean(right_scores), 3) if right_scores else 0.0
        total_turns = len(self._turn_scores)

        # ── build prompt ──────────────────────────────────────────────────
        score_lines = "\n".join(
            f"  T{r['turn']} {r['agent']}: {r['composite']:.3f} "
            f"(rel={r.get('relevance',0.0):.2f} nov={r.get('novelty',0.0):.2f} "
            f"evi={r.get('evidence',0.0):.2f})"
            for r in self._turn_scores
        )
        arb_lines = "\n".join(
            f"  T{e['turn']} [{'ECHO' if e.get('echo') else 'ARBITER'}]: {e['message'][:200]}"
            for e in self._arb_events
        ) or "  No interventions."

        prompt = (
            f"DEBATE TOPIC: {self._topic}\n\n"
            f"AGENTS: {self._left} (left) vs {self._right} (right)\n\n"
            f"PER-TURN SCORES ({total_turns} turns):\n{score_lines}\n\n"
            f"AVERAGES: {self._left}={left_avg:.3f}  {self._right}={right_avg:.3f}\n\n"
            f"ARBITER INTERVENTIONS:\n{arb_lines}\n\n"
            "Based on the scores, the trend over turns, and the intervention record, "
            "deliver your verdict. Remember: score averages matter but so does momentum — "
            "an agent who starts weak but finishes strong may deserve the win."
        )

        # ── call Ollama ───────────────────────────────────────────────────
        body = json.dumps({
            "model":    self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = json.loads(resp.read().decode())
        content = raw["message"]["content"].strip()

        # ── parse JSON response ───────────────────────────────────────────
        try:
            verdict = json.loads(content)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            verdict = json.loads(m.group()) if m else {}

        result = {
            "winner":     verdict.get("winner", "Draw"),
            "margin":     verdict.get("margin", "close"),
            "reason":     verdict.get("reason", ""),
            "summary":    verdict.get("summary", ""),
            "astra_avg":  left_avg  if self._left.lower()  == "astra" else right_avg,
            "nova_avg":   right_avg if self._right.lower() == "nova"  else left_avg,
            "turns":      total_turns,
            "topic":      self._topic,
            "ts":         time.time(),
        }

        # ── save to disk ──────────────────────────────────────────────────
        self._save(result)

        return result

    def _save(self, result: dict) -> None:
        """Write scoring_report.json + scoring_report.md + append to analytics log."""
        # 1. Per-session files
        if self._session_path:
            sp = Path(self._session_path)
            sp.mkdir(parents=True, exist_ok=True)

            json_path = sp / "scoring_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            md_lines = [
                f"# Debate Verdict\n",
                f"**Topic:** {result['topic']}\n",
                f"**Winner:** {result['winner']} ({result['margin']} margin)\n\n",
                f"## Reason\n{result['reason']}\n\n",
                f"## Summary\n{result['summary']}\n\n",
                f"## Scores\n",
                f"- Astra avg: {result['astra_avg']:.3f}\n",
                f"- Nova avg:  {result['nova_avg']:.3f}\n",
                f"- Total turns: {result['turns']}\n",
            ]
            with open(sp / "scoring_report.md", "w", encoding="utf-8") as f:
                f.writelines(md_lines)

        # 2. Global analytics log
        analytics_dir = Path(__file__).parent.parent / "sessions" / "_analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)
        log_path = analytics_dir / "scoring_log.jsonl"
        record = dict(result)
        record["session_path"] = self._session_path
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
