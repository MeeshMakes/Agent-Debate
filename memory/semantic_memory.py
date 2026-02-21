"""Semantic memory – keyword-overlap recall with structured fact storage.

Each agent accumulates facts, claims, sub-topics, and verified truths.
Before every turn the orchestrator queries recall() and injects results
into the agent's prompt so it can *remember and grow its understanding*.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_STOP = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "are",
        "was", "were", "been", "have", "has", "had", "not", "but",
        "its", "our", "your", "their", "they", "them", "you", "she",
        "his", "her", "him", "who", "which", "what", "when", "how",
        "can", "will", "would", "could", "should", "may", "might",
        "also", "about", "into", "than", "then", "just", "only",
        "some", "more", "most", "very", "much", "each", "every",
        "all", "any", "both", "few", "such", "other", "over", "like",
        "one", "two", "three", "does", "did", "use", "used", "using",
    }
)


@dataclass
class MemoryFact:
    text: str
    keywords: set[str]
    agent: str
    turn: int
    topic: str
    fact_type: str = "claim"     # claim | truth | problem | sub-topic | verify
    score: float = 0.0


class SemanticMemory:
    """One per agent — stores recalled facts and grows understanding."""

    def __init__(self) -> None:
        self.facts: list[MemoryFact] = []
        self.truths: list[str] = []
        self.problems: list[str] = []
        self.sub_topics: list[str] = []

    # -- tokenizer ---------------------------------------------------------

    @staticmethod
    def _keywords(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
        return {t for t in tokens if t not in _STOP}

    # -- store -------------------------------------------------------------

    def store(
        self,
        text: str,
        agent: str,
        turn: int,
        topic: str,
        fact_type: str = "claim",
    ) -> None:
        kw = self._keywords(text)
        self.facts.append(
            MemoryFact(
                text=text[:600],
                keywords=kw,
                agent=agent,
                turn=turn,
                topic=topic,
                fact_type=fact_type,
            )
        )
        if fact_type == "truth":
            self.truths.append(text[:400])
        elif fact_type == "problem":
            self.problems.append(text[:400])
        elif fact_type == "sub-topic":
            if text[:200] not in self.sub_topics:
                self.sub_topics.append(text[:200])

    # -- recall ------------------------------------------------------------

    def recall(self, query: str, top_k: int = 6) -> list[MemoryFact]:
        query_kw = self._keywords(query)
        if not query_kw:
            return self.facts[-top_k:] if self.facts else []

        scored: list[tuple[float, MemoryFact]] = []
        for fact in self.facts:
            overlap = query_kw & fact.keywords
            if not overlap:
                continue
            score = len(overlap) / max(len(query_kw), 1)
            scored.append((score, fact))

        scored.sort(key=lambda p: p[0], reverse=True)
        results: list[MemoryFact] = []
        for score, fact in scored[:top_k]:
            fact.score = round(score, 3)
            results.append(fact)
        return results

    # -- context string for prompt injection --------------------------------

    def recall_context(self, query: str, top_k: int = 6) -> str:
        facts = self.recall(query, top_k)
        if not facts:
            return "(no prior memories)"

        lines: list[str] = []
        for f in facts:
            lines.append(f"[{f.fact_type.upper()}] (turn {f.turn}, {f.agent}): {f.text[:300]}")
        return "\n".join(lines)

    # -- dataset export ----------------------------------------------------

    def build_dataset(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        for f in self.facts:
            by_type[f.fact_type] = by_type.get(f.fact_type, 0) + 1
            by_agent[f.agent] = by_agent.get(f.agent, 0) + 1
        return {
            "total_facts": len(self.facts),
            "truths": list(self.truths),
            "problems": list(self.problems),
            "sub_topics": list(self.sub_topics),
            "by_type": by_type,
            "by_agent": by_agent,
        }

    def to_dict(self) -> list[dict]:
        """Serialise all facts to a plain list of dicts (for session archiving)."""
        return [
            {
                "text": f.text,
                "agent": f.agent,
                "turn": f.turn,
                "topic": f.topic,
                "fact_type": f.fact_type,
            }
            for f in self.facts
        ]

    def save(self, path: Path) -> None:
        data = []
        for f in self.facts:
            data.append(
                {
                    "text": f.text,
                    "agent": f.agent,
                    "turn": f.turn,
                    "topic": f.topic,
                    "fact_type": f.fact_type,
                }
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            self.store(
                text=item["text"],
                agent=item["agent"],
                turn=item["turn"],
                topic=item["topic"],
                fact_type=item.get("fact_type", "claim"),
            )
