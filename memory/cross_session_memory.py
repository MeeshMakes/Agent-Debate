"""Cross-session semantic memory.

When Semantic Awareness is ON, agents can query facts from every past session.
Results are tagged with their source (session_id, topic, agent) so the prompt
can show inner-monologue breadcrumbs like:
  [from "bible_vs_quran_v1" / Astra] Both texts are theological operating systems...

Architecture:
  - Loads all sessions lazily (index cached in memory, auto-invalidated on new session)
  - Keyword-overlap scoring identical to per-session SemanticMemory
  - Also loads ingested_dataset.json from each session
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.session_manager import SessionManager, get_session_manager, _read_json, _SESSIONS_ROOT

_STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "its", "our", "your", "their",
    "they", "them", "you", "she", "his", "her", "him", "who", "which", "what",
    "when", "how", "can", "will", "would", "could", "should", "may", "might",
    "also", "about", "into", "than", "then", "just", "only", "some", "more",
    "most", "very", "much", "each", "every", "all", "any", "both", "few",
    "such", "other", "over", "like", "one", "two", "three", "does", "did",
    "use", "used", "using",
})


@dataclass
class CrossSessionFact:
    text: str
    keywords: frozenset[str]
    source_session: str    # session folder name (human-readable)
    source_topic: str
    source_agent: str
    fact_type: str         # truth | problem | claim | sub_topic | verify | ingested
    score: float = 0.0


class CrossSessionMemory:
    """Queries semantic facts across all past sessions."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self._sm = session_manager or get_session_manager()
        self._index: list[CrossSessionFact] = []
        self._indexed_sessions: set[str] = set()
        self.enabled: bool = True   # toggled by Semantic Awareness button

    # ------------------------------------------------------------------
    # Index management

    def refresh_index(self, exclude_session_id: str | None = None) -> None:
        """Rebuild the cross-session index (skipping current active session)."""
        self._index.clear()
        self._indexed_sessions.clear()

        sessions = self._sm.list_sessions()
        for meta in sessions:
            if meta.session_id == exclude_session_id:
                continue
            if meta.status == "running":
                continue   # don't read mid-session data
            self._index_session(meta.session_id, meta.topic)

    def _index_session(self, session_id: str, topic: str) -> None:
        """Load and index one session's memory + ingested dataset."""
        left_facts, right_facts = self._sm.load_session_memory(session_id)

        for agent_name, facts_data in (("Astra", left_facts), ("Nova", right_facts)):
            for item in facts_data:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                fact_type = str(item.get("fact_type", "claim"))
                kw = _keywords(text)
                self._index.append(CrossSessionFact(
                    text=text[:500],
                    keywords=frozenset(kw),
                    source_session=session_id,
                    source_topic=topic,
                    source_agent=agent_name,
                    fact_type=fact_type,
                ))

        # Also index ingested documents
        ingested = self._sm.load_ingested_dataset(session_id)
        for item in ingested:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            kw = _keywords(text)
            self._index.append(CrossSessionFact(
                text=text[:500],
                keywords=frozenset(kw),
                source_session=session_id,
                source_topic=topic,
                source_agent="[ingested]",
                fact_type="ingested",
            ))

        self._indexed_sessions.add(session_id)

    # ------------------------------------------------------------------
    # Query API

    def recall(self, query: str, top_k: int = 8, min_score: float = 0.07) -> list[CrossSessionFact]:
        """Return top-k cross-session facts matching query."""
        if not self.enabled or not self._index:
            return []

        qkw = _keywords(query)
        if not qkw:
            return []

        scored: list[tuple[float, CrossSessionFact]] = []
        for fact in self._index:
            overlap = qkw & fact.keywords
            if not overlap:
                continue
            score = len(overlap) / max(len(qkw), 1)
            if score >= min_score:
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, fact in scored[:top_k]:
            fact.score = score
            results.append(fact)
        return results

    def build_context_block(self, query: str, top_k: int = 5) -> str:
        """Build a prompt block showing cross-session references."""
        results = self.recall(query, top_k=top_k)
        if not results:
            return ""

        lines = ["=== CROSS-SESSION MEMORY (from past debates) ==="]
        for fact in results:
            session_label = _pretty_session_label(fact.source_session)
            type_icon = {
                "truth": "✓",
                "problem": "✗",
                "sub_topic": "→",
                "verify": "?",
                "ingested": "📄",
            }.get(fact.fact_type, "·")
            lines.append(
                f"  {type_icon} [{session_label} / {fact.source_agent}] {fact.text[:200]}"
            )
        lines.append("=================================================")
        return "\n".join(lines)

    def build_monologue_refs(self, query: str, top_k: int = 3) -> list[str]:
        """Return short reference strings for inner-monologue display."""
        results = self.recall(query, top_k=top_k)
        refs = []
        for fact in results:
            label = _pretty_session_label(fact.source_session)
            refs.append(
                f"[{label} / {fact.source_agent}] {fact.text[:120]}..."
            )
        return refs

    @property
    def total_facts(self) -> int:
        return len(self._index)


# ------------------------------------------------------------------
# Singleton

_cross_memory: CrossSessionMemory | None = None


def get_cross_session_memory() -> CrossSessionMemory:
    global _cross_memory
    if _cross_memory is None:
        _cross_memory = CrossSessionMemory()
    return _cross_memory


# ------------------------------------------------------------------
# Helpers

def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP}


def _pretty_session_label(session_id: str) -> str:
    """bible_vs_quran_v1_20260220_1430 → 'Bible Vs Quran v1'"""
    # Strip date suffix
    parts = session_id.rsplit("_", 2)
    if len(parts) >= 3:
        slug = parts[0]
    else:
        slug = session_id
    return slug.replace("_", " ").title()
