"""Segue Buffer — tracks lateral semantic references during debate.

When agents reference concepts that are outside the current talking point's scope,
those references are logged as segue candidates.  If a concept surfaces enough
times across turns/sessions and analytics confirm low coverage, it is promoted
to an autonomous talking point.

Storage: ``sessions/_analytics/segue_buffer.jsonl`` — one JSON line per entry.
Each entry tracks mention_count, source turns/sessions, and promotion status.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

_SESSIONS_ROOT = Path(__file__).parent.parent / "sessions"
_ANALYTICS_DIR = _SESSIONS_ROOT / "_analytics"
_BUFFER_PATH   = _ANALYTICS_DIR / "segue_buffer.jsonl"

# ── Config ───────────────────────────────────────────────────────────────────

MIN_SURROUNDING_TOKENS = 15      # ignore shallow one-word comparisons
PROMOTE_MIN_MENTIONS   = 3       # need ≥3 separate appearances to promote
SIMILARITY_EXISTING_THRESHOLD = 0.75  # don't create TP if one exists at ≥ this sim
MAX_BUFFER_SIZE = 10_000         # trim on startup


@dataclass
class SegueEntry:
    concept:            str
    source_sessions:    list[str] = field(default_factory=list)
    source_turns:       list[int] = field(default_factory=list)
    source_agents:      list[str] = field(default_factory=list)
    mention_count:      int = 1
    related_tp:         str = ""
    surrounding_context: str = ""
    ts:                 float = 0.0
    promoted:           bool = False
    deleted:            bool = False
    dismissed:          bool = False


class SegueBuffer:
    """In-memory segue buffer backed by a JSONL file on disk."""

    def __init__(self) -> None:
        self._entries: list[SegueEntry] = []
        self._load()

    # ── Public API ───────────────────────────────────────────────────────

    def log_segue(
        self,
        concept: str,
        session_id: str,
        turn: int,
        agent: str,
        related_tp: str,
        surrounding_context: str,
    ) -> None:
        """Record a lateral reference.  Deduplicates by concept text."""
        concept_norm = concept.strip().lower()
        if not concept_norm:
            return

        # Filter shallow uses
        tokens = surrounding_context.split()
        if len(tokens) < MIN_SURROUNDING_TOKENS:
            return

        # Check if we already have this concept
        existing = self._find(concept_norm)
        if existing is not None:
            existing.mention_count += 1
            if session_id not in existing.source_sessions:
                existing.source_sessions.append(session_id)
            existing.source_turns.append(turn)
            if agent not in existing.source_agents:
                existing.source_agents.append(agent)
            existing.ts = time.time()
            # Keep best surrounding context (longest)
            if len(surrounding_context) > len(existing.surrounding_context):
                existing.surrounding_context = surrounding_context
        else:
            entry = SegueEntry(
                concept=concept_norm,
                source_sessions=[session_id],
                source_turns=[turn],
                source_agents=[agent],
                mention_count=1,
                related_tp=related_tp,
                surrounding_context=surrounding_context,
                ts=time.time(),
            )
            self._entries.append(entry)

        self._save()

    def get_promotion_candidates(
        self,
        existing_tp_titles: Sequence[str] = (),
        *,
        min_mentions: int = PROMOTE_MIN_MENTIONS,
        coverage_threshold: float = 0.6,
    ) -> list[SegueEntry]:
        """Return entries ready for promotion to autonomous talking points.

        A candidate must:
          1. mention_count >= min_mentions
          2. not already promoted, deleted, or dismissed
          3. not significantly similar to any existing talking point title
        """
        candidates: list[SegueEntry] = []
        existing_lower = {t.lower() for t in existing_tp_titles}

        for entry in self._entries:
            if entry.promoted or entry.deleted or entry.dismissed:
                continue
            if entry.mention_count < min_mentions:
                continue
            # Simple similarity check: substring or high word overlap
            if self._similar_to_existing(entry.concept, existing_lower):
                continue
            candidates.append(entry)

        return candidates

    def mark_promoted(self, concept: str) -> None:
        entry = self._find(concept.strip().lower())
        if entry:
            entry.promoted = True
            self._save()

    def mark_dismissed(self, concept: str) -> None:
        entry = self._find(concept.strip().lower())
        if entry:
            entry.dismissed = True
            self._save()

    def mark_deleted(self, concept: str) -> None:
        entry = self._find(concept.strip().lower())
        if entry:
            entry.deleted = True
            self._save()

    def all_entries(self) -> list[SegueEntry]:
        return list(self._entries)

    def unresolved_entries(self) -> list[SegueEntry]:
        """Entries not yet promoted, deleted, or dismissed."""
        return [
            e for e in self._entries
            if not e.promoted and not e.deleted and not e.dismissed
        ]

    def reload(self) -> None:
        self._entries.clear()
        self._load()

    # ── Private ──────────────────────────────────────────────────────────

    def _find(self, concept_norm: str) -> SegueEntry | None:
        for e in self._entries:
            if e.concept == concept_norm:
                return e
        return None

    @staticmethod
    def _similar_to_existing(concept: str, existing_lower: set[str]) -> bool:
        """Quick check: is the concept covered by an existing talking point?"""
        if not existing_lower:
            return False
        # Exact substring match in any direction
        for title in existing_lower:
            if concept in title or title in concept:
                return True
        # Word-level Jaccard
        concept_words = set(concept.split())
        for title in existing_lower:
            title_words = set(title.split())
            if not concept_words or not title_words:
                continue
            intersection = concept_words & title_words
            union = concept_words | title_words
            if len(intersection) / len(union) >= SIMILARITY_EXISTING_THRESHOLD:
                return True
        return False

    def _load(self) -> None:
        if not _BUFFER_PATH.exists():
            return
        try:
            lines = _BUFFER_PATH.read_text(encoding="utf-8").strip().splitlines()
            # Trim to MAX_BUFFER_SIZE, pruning promoted/deleted first
            if len(lines) > MAX_BUFFER_SIZE:
                # Keep un-promoted entries, trim oldest promoted/deleted
                keep: list[str] = []
                prunable: list[str] = []
                for ln in lines:
                    try:
                        d = json.loads(ln)
                        if d.get("promoted") or d.get("deleted"):
                            prunable.append(ln)
                        else:
                            keep.append(ln)
                    except Exception:
                        continue
                # Keep all unresolved + tail of prunable
                allowed_prunable = MAX_BUFFER_SIZE - len(keep)
                lines = keep + prunable[-max(0, allowed_prunable):]

            for line in lines:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    self._entries.append(SegueEntry(
                        concept=d.get("concept", ""),
                        source_sessions=d.get("source_sessions", []),
                        source_turns=d.get("source_turns", []),
                        source_agents=d.get("source_agents", []),
                        mention_count=d.get("mention_count", 1),
                        related_tp=d.get("related_tp", ""),
                        surrounding_context=d.get("surrounding_context", ""),
                        ts=d.get("ts", 0.0),
                        promoted=d.get("promoted", False),
                        deleted=d.get("deleted", False),
                        dismissed=d.get("dismissed", False),
                    ))
                except Exception:
                    continue
        except Exception:
            pass

    def _save(self) -> None:
        _ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_BUFFER_PATH, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except Exception:
            pass


# ── Singleton ────────────────────────────────────────────────────────────────

_buffer: SegueBuffer | None = None


def get_segue_buffer() -> SegueBuffer:
    global _buffer
    if _buffer is None:
        _buffer = SegueBuffer()
    return _buffer


# ── Concept Extraction ──────────────────────────────────────────────────────

# Simple noun-phrase extractor: finds capitalized phrases and
# multi-word terms that look like named concepts.

_CONCEPT_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"     # Capitalized multi-word phrases
    r"|"
    r"\b([A-Z][a-z]{3,}(?:\s+[a-z]+){1,3})\b"    # Capitalized word + lowercase followers
)

# Common debate-noise words to filter out
_NOISE = {
    "this is", "that is", "there are", "we should", "i think",
    "for example", "in this", "on the", "it is", "they are",
    "however", "therefore", "furthermore", "nevertheless",
    "astra", "nova", "the point",
}


def extract_concepts(text: str) -> list[tuple[str, str]]:
    """Extract potential lateral concepts from a debate message.

    Returns list of (concept_phrase, surrounding_sentence) tuples.
    """
    results: list[tuple[str, str]] = []

    # Split into sentences
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 5:
            continue

        for match in _CONCEPT_RE.finditer(sentence):
            phrase = (match.group(1) or match.group(2) or "").strip()
            if not phrase or len(phrase) < 4:
                continue
            if phrase.lower() in _NOISE:
                continue
            results.append((phrase, sentence))

    return results
