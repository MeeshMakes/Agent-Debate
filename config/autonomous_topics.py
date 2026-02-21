"""Autonomous Talking Point Store — manages talking points created by the system.

Storage: ``sessions/_autonomous_topics/`` directory.
Each autonomous talking point is a JSON file:
  {
    "title": "...",
    "description": "...",
    "talking_points": ["...", "..."],
    "origin": "autonomous",
    "created_ts": 1234567890.0,
    "segue_concept": "...",          # if promoted from segue buffer
    "segue_mentions": 3,
    "segue_sessions": ["session_a", "session_b"],
    "promoted_to_custom": false,
    "deleted": false
  }

File naming: ``{slugified_title}_{timestamp}.json``
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

_SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"
_AUTO_TOPICS_DIR = _SESSIONS_ROOT / "_autonomous_topics"


@dataclass
class AutonomousTopic:
    title:              str
    description:        str
    talking_points:     list[str]
    origin:             str = "autonomous"    # "autonomous" or "user"
    created_ts:         float = 0.0
    segue_concept:      str = ""
    segue_mentions:     int = 0
    segue_sessions:     list[str] = field(default_factory=list)
    promoted_to_custom: bool = False
    deleted:            bool = False
    filename:           str = ""              # set after save


def _slugify(text: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower().strip())
    return slug[:60].strip('_')


class AutonomousTopicStore:
    """Reads and writes autonomous talking points from disk."""

    def __init__(self) -> None:
        _AUTO_TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, topic: dict) -> AutonomousTopic:
        """Save a new autonomous talking point to disk.

        ``topic`` should have at minimum: title, description, talking_points.
        Returns the AutonomousTopic with filename set.
        """
        ts = time.time()
        title = topic.get("title", "Untitled")
        slug = _slugify(title)
        filename = f"{slug}_{int(ts)}.json"

        at = AutonomousTopic(
            title=title,
            description=topic.get("description", ""),
            talking_points=topic.get("talking_points", []),
            origin=topic.get("origin", "autonomous"),
            created_ts=ts,
            segue_concept=topic.get("segue_concept", ""),
            segue_mentions=topic.get("segue_mentions", 0),
            segue_sessions=topic.get("segue_sessions", []),
            filename=filename,
        )

        path = _AUTO_TOPICS_DIR / filename
        path.write_text(
            json.dumps(asdict(at), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return at

    def load_all(self, include_deleted: bool = False) -> list[AutonomousTopic]:
        """Load all autonomous talking points, newest first."""
        topics: list[AutonomousTopic] = []
        if not _AUTO_TOPICS_DIR.is_dir():
            return topics

        for p in sorted(_AUTO_TOPICS_DIR.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
            if not p.name.endswith(".json"):
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                at = AutonomousTopic(
                    title=d.get("title", ""),
                    description=d.get("description", ""),
                    talking_points=d.get("talking_points", []),
                    origin=d.get("origin", "autonomous"),
                    created_ts=d.get("created_ts", 0.0),
                    segue_concept=d.get("segue_concept", ""),
                    segue_mentions=d.get("segue_mentions", 0),
                    segue_sessions=d.get("segue_sessions", []),
                    promoted_to_custom=d.get("promoted_to_custom", False),
                    deleted=d.get("deleted", False),
                    filename=p.name,
                )
                if not include_deleted and at.deleted:
                    continue
                topics.append(at)
            except Exception:
                continue

        return topics

    def titles(self, include_deleted: bool = False) -> list[str]:
        """Return all autonomous talking point titles."""
        return [t.title for t in self.load_all(include_deleted)]

    def delete(self, filename: str) -> None:
        """Soft-delete an autonomous talking point (mark deleted: true)."""
        path = _AUTO_TOPICS_DIR / filename
        if not path.exists():
            return
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            d["deleted"] = True
            path.write_text(
                json.dumps(d, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def hard_delete(self, filename: str) -> None:
        """Permanently remove the file."""
        path = _AUTO_TOPICS_DIR / filename
        if path.exists():
            path.unlink()

    def promote_to_custom(self, filename: str) -> dict | None:
        """Mark as promoted to custom and return the data for integration."""
        path = _AUTO_TOPICS_DIR / filename
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            d["promoted_to_custom"] = True
            path.write_text(
                json.dumps(d, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return d
        except Exception:
            return None

    def get_by_filename(self, filename: str) -> AutonomousTopic | None:
        path = _AUTO_TOPICS_DIR / filename
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            return AutonomousTopic(
                title=d.get("title", ""),
                description=d.get("description", ""),
                talking_points=d.get("talking_points", []),
                origin=d.get("origin", "autonomous"),
                created_ts=d.get("created_ts", 0.0),
                segue_concept=d.get("segue_concept", ""),
                segue_mentions=d.get("segue_mentions", 0),
                segue_sessions=d.get("segue_sessions", []),
                promoted_to_custom=d.get("promoted_to_custom", False),
                deleted=d.get("deleted", False),
                filename=filename,
            )
        except Exception:
            return None


# ── Singleton ────────────────────────────────────────────────────────────────

_store: AutonomousTopicStore | None = None


def get_autonomous_store() -> AutonomousTopicStore:
    global _store
    if _store is None:
        _store = AutonomousTopicStore()
    return _store
