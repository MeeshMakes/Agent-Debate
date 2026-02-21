"""Session Manager — auto-names, creates, lists, and persists debate sessions.

Each debate run creates a unique session folder under sessions/.
Session folders are named: {topic_slug}_v{n}_{YYYYMMDD_HHMM}/

Inside each session:
    session_meta.json     — topic, agents, models, start/end, turn count, fact counts
    transcript.jsonl      — every DebateEvent emitted during the debate
    astra_memory.json     — Astra's SemanticMemory facts at session end
    nova_memory.json      — Nova's SemanticMemory facts at session end
    adaptive_prompts.json — AdaptivePromptStore snapshot for this topic
    graph.json            — DebateGraphManager rows
    uploads/              — copied files for ingestion
    ingested_dataset.json — tagged facts extracted from uploaded files

Topic-level (cross-session) data lives in sessions/_topic_threads/:
    {tp_key}.json         — refined topic description + talking points after each session
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


_SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"


@dataclass
class SessionMeta:
    session_id: str          # folder name
    topic: str
    left_agent: str
    right_agent: str
    left_model: str
    right_model: str
    start_ts: float
    end_ts: float = 0.0
    turn_count: int = 0
    left_facts: int = 0
    right_facts: int = 0
    sub_topics: list[str] = field(default_factory=list)
    truths_count: int = 0
    problems_count: int = 0
    status: str = "running"   # running | complete | stopped
    talking_point_key: str = ""  # sha1[:12] of topic_title+talking_point_text


class SessionManager:
    """Creates and manages debate sessions on disk."""

    def __init__(self, sessions_root: Path | None = None) -> None:
        self.root = sessions_root or _SESSIONS_ROOT
        self.root.mkdir(exist_ok=True)
        self.current_session: SessionMeta | None = None
        self.current_path: Path | None = None
        self._transcript_handle = None
        # Set before new_session() to tag this session with a talking_point_key
        self.pending_tp_key: str = ""
        # Topic threads folder for per-TP refinement storage
        self._tp_threads_dir = self.root / "_topic_threads"
        self._tp_threads_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle

    def new_session(
        self,
        topic: str,
        left_agent: str,
        right_agent: str,
        left_model: str,
        right_model: str,
    ) -> Path:
        """Create a new session folder and open transcript writer."""
        # Close any existing
        self.close_session()

        slug = _slugify(topic, max_len=42)
        version = self._next_version(slug)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        session_id = f"{slug}_v{version}_{ts_str}"

        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "uploads").mkdir(exist_ok=True)

        meta = SessionMeta(
            session_id=session_id,
            topic=topic,
            left_agent=left_agent,
            right_agent=right_agent,
            left_model=left_model,
            right_model=right_model,
            start_ts=time.time(),
            talking_point_key=self.pending_tp_key,
        )
        self.pending_tp_key = ""  # clear after use
        self.current_session = meta
        self.current_path = path
        self._save_meta()

        # Open transcript JSONL stream
        self._transcript_handle = open(path / "transcript.jsonl", "w", encoding="utf-8")

        return path

    def record_event(self, event_type: str, payload: dict) -> None:
        """Stream a debate event to transcript.jsonl."""
        if self._transcript_handle is None:
            return
        record = {
            "ts": time.time(),
            "event_type": event_type,
            "payload": {k: v for k, v in payload.items() if _is_serialisable(v)},
        }
        self._transcript_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._transcript_handle.flush()

    def finalise_session(
        self,
        turn_count: int,
        left_facts: int,
        right_facts: int,
        sub_topics: list[str],
        truths_count: int,
        problems_count: int,
        left_memory_data: list[dict] | None = None,
        right_memory_data: list[dict] | None = None,
        adaptive_data: dict | None = None,
        graph_rows: list | None = None,
    ) -> None:
        """Write final snapshots and close files."""
        if self.current_session is None:
            return
        meta = self.current_session
        meta.end_ts = time.time()
        meta.turn_count = turn_count
        meta.left_facts = left_facts
        meta.right_facts = right_facts
        meta.sub_topics = sub_topics
        meta.truths_count = truths_count
        meta.problems_count = problems_count
        meta.status = "complete"
        self._save_meta()

        path = self.current_path
        assert path is not None

        if left_memory_data:
            _write_json(path / "astra_memory.json", left_memory_data)
        if right_memory_data:
            _write_json(path / "nova_memory.json", right_memory_data)
        if adaptive_data:
            _write_json(path / "adaptive_prompts.json", adaptive_data)
        if graph_rows:
            _write_json(path / "graph.json", graph_rows)

        self.close_session()

    def mark_stopped(self) -> None:
        if self.current_session:
            self.current_session.status = "stopped"
            self._save_meta()
        self.close_session()

    def close_session(self) -> None:
        if self._transcript_handle:
            try:
                self._transcript_handle.close()
            except Exception:
                pass
            self._transcript_handle = None

    # ------------------------------------------------------------------
    # Session listing / management

    def list_sessions(self) -> list[SessionMeta]:
        """Return all sessions sorted newest-first."""
        sessions: list[SessionMeta] = []
        for p in sorted(self.root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and not p.name.startswith("_"):
                meta = self._load_meta(p)
                if meta:
                    sessions.append(meta)
        return sessions

    def list_sessions_for_tp(self, tp_key: str) -> list[SessionMeta]:
        """Return sessions tied to a specific talking_point_key, newest-first."""
        return [
            m for m in self.list_sessions()
            if m.talking_point_key == tp_key
        ]

    # ------------------------------------------------------------------
    # Talking-point generation (reset / archive without data loss)

    def get_tp_generation(self, base_tp_key: str) -> int:
        """Return the current generation index for this talking point (0 = original)."""
        data = _read_json(self._tp_threads_dir / f"{base_tp_key}_gen.json") or {}
        return int(data.get("gen", 0))

    def effective_tp_key(self, base_tp_key: str) -> str:
        """Return the key that should be written to new sessions right now.

        Generation 0 → base_tp_key  (unchanged, backward-compat)
        Generation N → base_tp_key_gN
        """
        if not base_tp_key:
            return base_tp_key
        gen = self.get_tp_generation(base_tp_key)
        return base_tp_key if gen == 0 else f"{base_tp_key}_g{gen}"

    def reset_tp_thread(self, base_tp_key: str) -> None:
        """Start a fresh session thread for this talking point.

        Old sessions are preserved on disk (their tp_key still points to the
        previous effective key).  The generation counter is incremented, so the
        next new session gets a new effective key and the history panel shows
        Session 1 again.  The AI refinement for the old thread is cleared so the
        topic presents itself without the old refinement badge.
        """
        if not base_tp_key:
            return
        gen = self.get_tp_generation(base_tp_key)
        _write_json(self._tp_threads_dir / f"{base_tp_key}_gen.json", {"gen": gen + 1})
        # Remove the refinement for the *old* effective key so the UI is clean
        old_eff = base_tp_key if gen == 0 else f"{base_tp_key}_g{gen}"
        old_ref = self._tp_threads_dir / f"{old_eff}.json"
        if old_ref.exists():
            old_ref.unlink(missing_ok=True)

    def clear_tp_data(self, base_tp_key: str) -> int:
        """Delete ALL sessions and refinements for every generation of this talking point.

        Returns the number of session folders deleted.
        """
        if not base_tp_key:
            return 0
        gen = self.get_tp_generation(base_tp_key)
        # Every effective key ever used for this talking point
        all_eff_keys: set[str] = {base_tp_key}
        for g in range(1, gen + 1):
            all_eff_keys.add(f"{base_tp_key}_g{g}")

        deleted = 0
        for session in self.list_sessions():
            if session.talking_point_key in all_eff_keys:
                if self.delete_session(session.session_id):
                    deleted += 1

        # Remove all refinement and generation tracking files
        for fname in ([f"{base_tp_key}_gen.json"] +
                      [f"{k}.json" for k in all_eff_keys]):
            p = self._tp_threads_dir / fname
            if p.exists():
                p.unlink(missing_ok=True)

        return deleted

    # ------------------------------------------------------------------
    # Topic refinement storage

    def save_tp_refinement(self, tp_key: str, data: dict) -> None:
        """Persist refined topic description / talking_points for a talking point."""
        if not tp_key:
            return
        self._tp_threads_dir.mkdir(exist_ok=True)
        _write_json(self._tp_threads_dir / f"{tp_key}.json", data)

    def load_tp_refinement(self, tp_key: str) -> dict | None:
        """Load the latest refinement for a talking point, or None."""
        if not tp_key:
            return None
        return _read_json(self._tp_threads_dir / f"{tp_key}.json")

    def delete_session(self, session_id: str) -> bool:
        """Delete a session folder and all its contents."""
        target = self.root / session_id
        if not target.exists():
            return False
        _rmtree(target)
        return True

    def load_session_transcript(self, session_id: str) -> list[dict]:
        path = self.root / session_id / "transcript.jsonl"
        if not path.exists():
            return []
        events: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events

    def load_session_memory(self, session_id: str) -> tuple[list[dict], list[dict]]:
        """Return (left_facts, right_facts) as raw dicts."""
        base = self.root / session_id
        left = _read_json(base / "astra_memory.json") or []
        right = _read_json(base / "nova_memory.json") or []
        return left, right

    def load_ingested_dataset(self, session_id: str) -> list[dict]:
        path = self.root / session_id / "ingested_dataset.json"
        return _read_json(path) or []

    def get_uploads_dir(self, session_id: str) -> Path:
        d = self.root / session_id / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Private helpers

    def _save_meta(self) -> None:
        if self.current_session is None or self.current_path is None:
            return
        m = self.current_session
        data = {
            "session_id": m.session_id,
            "topic": m.topic,
            "left_agent": m.left_agent,
            "right_agent": m.right_agent,
            "left_model": m.left_model,
            "right_model": m.right_model,
            "start_ts": m.start_ts,
            "end_ts": m.end_ts,
            "turn_count": m.turn_count,
            "left_facts": m.left_facts,
            "right_facts": m.right_facts,
            "sub_topics": m.sub_topics,
            "truths_count": m.truths_count,
            "problems_count": m.problems_count,
            "status": m.status,
            "talking_point_key": m.talking_point_key,
        }
        _write_json(self.current_path / "session_meta.json", data)

    def _load_meta(self, path: Path) -> SessionMeta | None:
        data = _read_json(path / "session_meta.json")
        if not data:
            return None
        try:
            # backward-compat: old sessions lack talking_point_key
            data.setdefault("talking_point_key", "")
            return SessionMeta(**data)
        except Exception:
            return None

    def _next_version(self, slug: str) -> int:
        existing = [
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith(slug)
        ]
        return len(existing) + 1


# ------------------------------------------------------------------
# Module-level singleton

_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


# ------------------------------------------------------------------
# Helpers

def _slugify(text: str, max_len: int = 42) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len]


def _is_serialisable(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _write_json(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rmtree(path: Path) -> None:
    """Recursively delete a folder."""
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()
