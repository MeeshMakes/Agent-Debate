"""AnalyticsStore — reads all session data and presents a unified analytics view.

Data sources
------------
  sessions/*/session_meta.json      — topic, agents, models, timestamps, fact counts
  sessions/*/scoring_report.json    — winner, scores, summary
  sessions/_analytics/scoring_log.jsonl — lightweight global log (written by worker)

All session data is preserved even if the session folder is renamed or deleted —
the scoring_log.jsonl is the source of truth for the analytics panel.

Usage
-----
    from analytics.analytics_store import AnalyticsStore, get_analytics_store
    store = get_analytics_store()
    rows = store.all_sessions()          # list[AnalyticsRow]
    progression = store.score_series()  # {"astra": [floats], "nova": [floats]}
    totals = store.knowledge_totals()   # {"truths":n, "problems":n, ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_SESSIONS_ROOT    = Path(__file__).parent.parent / "sessions"
_ANALYTICS_DIR    = _SESSIONS_ROOT / "_analytics"
_SCORING_LOG_PATH = _ANALYTICS_DIR / "scoring_log.jsonl"


@dataclass
class AnalyticsRow:
    session_id:   str
    topic:        str
    winner:       str
    margin:       str
    astra_avg:    float
    nova_avg:     float
    turns:        int
    reason:       str
    summary:      str
    ts:           float
    status:       str    # complete | stopped | running | unknown
    left_facts:   int
    right_facts:  int
    sub_topics:   list[str] = field(default_factory=list)
    truths:       int = 0
    problems:     int = 0
    session_path: str = ""


class AnalyticsStore:
    """Reads all available session + scoring data and exposes analytics queries."""

    def all_sessions(self) -> list[AnalyticsRow]:
        """Return all debate sessions with scoring, newest first."""
        rows: dict[str, AnalyticsRow] = {}

        # ── 1. Read scoring_log.jsonl (global, even for deleted sessions) ──
        rows.update(self._read_scoring_log())

        # ── 2. Overlay with per-session data from sessions/ folders ──
        if _SESSIONS_ROOT.is_dir():
            for session_dir in sorted(
                _SESSIONS_ROOT.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                if not session_dir.is_dir() or session_dir.name.startswith("_"):
                    continue
                session_id = session_dir.name
                meta = self._read_session_meta(session_dir)
                score = self._read_scoring_report(session_dir)

                if session_id in rows:
                    # Enrich the scoring_log entry with meta details
                    r = rows[session_id]
                    if meta:
                        r.status      = meta.get("status", r.status)
                        r.left_facts  = meta.get("left_facts", r.left_facts)
                        r.right_facts = meta.get("right_facts", r.right_facts)
                        r.sub_topics  = meta.get("sub_topics", r.sub_topics)
                        r.truths      = meta.get("truths_count", r.truths)
                        r.problems    = meta.get("problems_count", r.problems)
                        r.ts          = meta.get("end_ts") or meta.get("start_ts", r.ts)
                    r.session_path = str(session_dir)
                else:
                    # Session exists but no scoring yet (e.g. stopped before end)
                    topic   = (meta or {}).get("topic", session_id)
                    winner  = (score or {}).get("winner", "—")
                    ts_val  = (meta or {}).get("end_ts") or (meta or {}).get("start_ts", 0.0)
                    rows[session_id] = AnalyticsRow(
                        session_id   = session_id,
                        topic        = topic,
                        winner       = winner,
                        margin       = (score or {}).get("margin", "—"),
                        astra_avg    = (score or {}).get("astra_avg", 0.0),
                        nova_avg     = (score or {}).get("nova_avg", 0.0),
                        turns        = (score or {}).get("turns", (meta or {}).get("turn_count", 0)),
                        reason       = (score or {}).get("reason", ""),
                        summary      = (score or {}).get("summary", ""),
                        ts           = ts_val or 0.0,
                        status       = (meta or {}).get("status", "unknown"),
                        left_facts   = (meta or {}).get("left_facts", 0),
                        right_facts  = (meta or {}).get("right_facts", 0),
                        sub_topics   = (meta or {}).get("sub_topics", []),
                        truths       = (meta or {}).get("truths_count", 0),
                        problems     = (meta or {}).get("problems_count", 0),
                        session_path = str(session_dir),
                    )

        result = sorted(rows.values(), key=lambda r: r.ts, reverse=True)
        return result

    def score_series(self) -> dict[str, list[float]]:
        """Return ordered Astra and Nova average score timelines (oldest → newest)."""
        sessions = sorted(self.all_sessions(), key=lambda r: r.ts)
        return {
            "astra": [r.astra_avg for r in sessions],
            "nova":  [r.nova_avg  for r in sessions],
        }

    def knowledge_totals(self) -> dict[str, int]:
        sessions = self.all_sessions()
        return {
            "debates":    len(sessions),
            "turns":      sum(r.turns      for r in sessions),
            "left_facts": sum(r.left_facts for r in sessions),
            "right_facts":sum(r.right_facts for r in sessions),
            "truths":     sum(r.truths     for r in sessions),
            "problems":   sum(r.problems   for r in sessions),
        }

    def win_counts(self) -> dict[str, int]:
        sessions = self.all_sessions()
        counts: dict[str, int] = {}
        for r in sessions:
            counts[r.winner] = counts.get(r.winner, 0) + 1
        return counts

    # ── private ──

    def _read_scoring_log(self) -> dict[str, AnalyticsRow]:
        rows: dict[str, AnalyticsRow] = {}
        if not _SCORING_LOG_PATH.exists():
            return rows
        with open(_SCORING_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    session_path = d.get("session_path", "")
                    session_id   = Path(session_path).name if session_path else str(d.get("ts", ""))
                    rows[session_id] = AnalyticsRow(
                        session_id   = session_id,
                        topic        = d.get("topic", "Unknown"),
                        winner       = d.get("winner", "—"),
                        margin       = d.get("margin", "—"),
                        astra_avg    = d.get("astra_avg", 0.0),
                        nova_avg     = d.get("nova_avg", 0.0),
                        turns        = d.get("turns", 0),
                        reason       = d.get("reason", ""),
                        summary      = d.get("summary", ""),
                        ts           = d.get("ts", 0.0),
                        status       = "complete",
                        left_facts   = 0,
                        right_facts  = 0,
                        session_path = session_path,
                    )
                except Exception:
                    continue
        return rows

    @staticmethod
    def _read_session_meta(session_dir: Path) -> dict | None:
        p = session_dir / "session_meta.json"
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _read_scoring_report(session_dir: Path) -> dict | None:
        p = session_dir / "scoring_report.json"
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


_store: AnalyticsStore | None = None


def get_analytics_store() -> AnalyticsStore:
    global _store
    if _store is None:
        _store = AnalyticsStore()
    return _store
