"""Living Topic — an evolving debate brief that agents co-author in real time.

Starts as a seed (the user's talking point).  As the debate progresses:
  - Agents can ADD to it via   EXPAND-TOPIC: <text>
  - Conclusions auto-appended via append_conclusion()
  - Contradictions auto-appended via append_contradiction()
  - The document is passed back to agents each turn so they always have
    the full, current state of the debate brief

The living topic is the gravitational document of the debate — it starts
the argument, it grows with the argument, and it ends as an artefact
of everything the agents discovered together.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TopicEntry:
    kind: str    # seed | expansion | conclusion | contradiction | falsehood
    author: str  # agent name or "system"
    text: str
    turn: int


class LivingTopic:
    """Mutable, ever-growing topic document shared between agents."""

    MAX_CONTEXT_TOKENS = 2400   # rough char limit for prompt injection

    def __init__(self, seed: str) -> None:
        self._entries: list[TopicEntry] = [
            TopicEntry(kind="seed", author="system", text=seed.strip(), turn=0)
        ]

    # ── Write API ────────────────────────────────────────────────────────────

    def add_expansion(self, text: str, agent: str, turn: int) -> None:
        """Agent explicitly expands the topic document."""
        t = text.strip()
        if not t or len(t) < 12:
            return
        self._entries.append(TopicEntry(kind="expansion", author=agent, text=t[:800], turn=turn))

    def append_conclusion(self, text: str, agent: str, turn: int) -> None:
        self._entries.append(TopicEntry(kind="conclusion", author=agent, text=text[:500], turn=turn))

    def append_contradiction(self, claim_a: str, claim_b: str, agent_a: str, agent_b: str, turn: int) -> None:
        combined = f"[{agent_a}]: {claim_a[:200]}  ←→  [{agent_b}]: {claim_b[:200]}"
        self._entries.append(TopicEntry(kind="contradiction", author="system", text=combined, turn=turn))

    def append_falsehood(self, text: str, refuted_by: str, turn: int) -> None:
        self._entries.append(TopicEntry(kind="falsehood", author=refuted_by, text=text[:400], turn=turn))

    # ── Read API ─────────────────────────────────────────────────────────────

    @property
    def seed(self) -> str:
        for e in self._entries:
            if e.kind == "seed":
                return e.text
        return ""

    def to_document(self, max_chars: int | None = None) -> str:
        """Return the full living document as a formatted string for agent prompts."""
        limit = max_chars or self.MAX_CONTEXT_TOKENS
        sections: dict[str, list[TopicEntry]] = {
            "seed": [], "expansion": [], "conclusion": [],
            "contradiction": [], "falsehood": [],
        }
        for e in self._entries:
            sections.setdefault(e.kind, []).append(e)

        parts: list[str] = []

        seed = sections["seed"]
        if seed:
            parts.append("── DEBATE SEED ──────────────────────────────────────")
            parts.append(seed[0].text)

        if sections["expansion"]:
            parts.append("\n── TOPIC EXPANSIONS (added by agents during debate) ──")
            for e in sections["expansion"]:
                parts.append(f"[{e.author} | turn {e.turn}] {e.text}")

        if sections["conclusion"]:
            parts.append("\n── CONCLUSIONS REACHED ───────────────────────────────")
            for e in sections["conclusion"]:
                parts.append(f"✓ [{e.author} | turn {e.turn}] {e.text}")

        if sections["contradiction"]:
            parts.append("\n── OPEN CONTRADICTIONS ───────────────────────────────")
            for e in sections["contradiction"]:
                parts.append(f"⚡ [turn {e.turn}] {e.text}")

        if sections["falsehood"]:
            parts.append("\n── ESTABLISHED FALSEHOODS ────────────────────────────")
            for e in sections["falsehood"]:
                parts.append(f"✗ [refuted by {e.author} | turn {e.turn}] {e.text}")

        doc = "\n".join(parts)
        if len(doc) > limit:
            # Keep seed intact, truncate the rest
            seed_text = "\n".join(parts[:2]) if len(parts) > 1 else parts[0]
            tail = doc[-int(limit * 0.6):]
            doc = seed_text + "\n[...earlier entries omitted...]\n" + tail
        return doc

    def summary_line(self) -> str:
        """Short status line for UI display."""
        counts = {}
        for e in self._entries:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        parts = []
        if counts.get("expansion"):   parts.append(f"{counts['expansion']} expansions")
        if counts.get("conclusion"):  parts.append(f"{counts['conclusion']} conclusions")
        if counts.get("contradiction"): parts.append(f"{counts['contradiction']} contradictions")
        if counts.get("falsehood"):   parts.append(f"{counts['falsehood']} falsehoods")
        return "Living topic: " + (", ".join(parts) if parts else "seed only")

    def to_dict(self) -> list[dict]:
        return [{"kind": e.kind, "author": e.author, "text": e.text, "turn": e.turn}
                for e in self._entries]

    @classmethod
    def from_dict(cls, data: list[dict], seed: str = "") -> "LivingTopic":
        lt = cls(seed or "")
        lt._entries = []
        for d in data:
            lt._entries.append(TopicEntry(**d))
        return lt
