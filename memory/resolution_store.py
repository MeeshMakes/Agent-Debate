"""Resolution Store — persists conclusions, contradictions, and falsehoods.

Every debate can produce three classes of epistemic output:

  CONCLUSION    — both agents have reached at least 80% agreement on a point.
  CONTRADICTION — the agents hold positions that cancel each other out and
                  cannot currently be resolved.  Saved for future sessions.
  FALSEHOOD     — a claim was explicitly refuted by evidence or logic.

All three are saved to disk so they accumulate across sessions and become
a growing knowledge base the agents can query at the start of each debate.

Signals agents should emit in their prose:
  CONCLUDE: <settled point>
  CONTRADICT: <claim A> / <claim B>
  FALSE: <refuted statement>
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


_STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "its", "our", "your", "their",
    "they", "you", "who", "can", "will", "would", "could", "should", "also",
    "about", "into", "than", "then", "just", "only", "some", "more", "every",
    "all", "any", "both", "each", "such", "over", "like", "one", "does", "did",
    "use", "used", "using", "these", "those",
})


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Conclusion:
    text: str
    agent: str
    topic: str
    session_id: str
    turn: int
    ts: float = field(default_factory=time.time)
    keywords: list[str] = field(default_factory=list)


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    agent_a: str
    agent_b: str
    topic: str
    session_id: str
    turn: int
    ts: float = field(default_factory=time.time)
    keywords: list[str] = field(default_factory=list)


@dataclass
class Falsehood:
    text: str
    refuted_by: str
    topic: str
    session_id: str
    turn: int
    ts: float = field(default_factory=time.time)
    keywords: list[str] = field(default_factory=list)


# ── Store ────────────────────────────────────────────────────────────────────

class ResolutionStore:
    """Accumulates and retrieves conclusions, contradictions, and falsehoods."""

    def __init__(self, store_root: Path | None = None) -> None:
        root = store_root or Path(__file__).parent.parent / "sessions" / "_resolutions"
        root.mkdir(parents=True, exist_ok=True)
        self._root = root
        self.conclusions:   list[Conclusion]   = []
        self.contradictions: list[Contradiction] = []
        self.falsehoods:    list[Falsehood]    = []
        self._session_id: str = ""
        self._load_all()

    # ── Session lifecycle ────────────────────────────────────────────────────

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    # ── Append ──────────────────────────────────────────────────────────────

    def add_conclusion(self, text: str, agent: str, topic: str, turn: int) -> None:
        c = Conclusion(
            text=text.strip()[:600],
            agent=agent,
            topic=topic,
            session_id=self._session_id,
            turn=turn,
            keywords=_kw(text),
        )
        self.conclusions.append(c)
        self._save()

    def add_contradiction(
        self,
        claim_a: str, claim_b: str,
        agent_a: str, agent_b: str,
        topic: str, turn: int,
    ) -> None:
        c = Contradiction(
            claim_a=claim_a.strip()[:400],
            claim_b=claim_b.strip()[:400],
            agent_a=agent_a,
            agent_b=agent_b,
            topic=topic,
            session_id=self._session_id,
            turn=turn,
            keywords=_kw(claim_a + " " + claim_b),
        )
        self.contradictions.append(c)
        self._save()

    def add_falsehood(self, text: str, refuted_by: str, topic: str, turn: int) -> None:
        f = Falsehood(
            text=text.strip()[:500],
            refuted_by=refuted_by,
            topic=topic,
            session_id=self._session_id,
            turn=turn,
            keywords=_kw(text),
        )
        self.falsehoods.append(f)
        self._save()

    # ── Query ────────────────────────────────────────────────────────────────

    def recall_conclusions(self, query: str, top_k: int = 4) -> list[Conclusion]:
        return _score_list(self.conclusions, query, top_k, lambda c: c.text)

    def recall_contradictions(self, query: str, top_k: int = 3) -> list[Contradiction]:
        return _score_list(
            self.contradictions, query, top_k,
            lambda c: c.claim_a + " " + c.claim_b,
        )

    def recall_falsehoods(self, query: str, top_k: int = 3) -> list[Falsehood]:
        return _score_list(self.falsehoods, query, top_k, lambda f: f.text)

    def build_context_block(self, query: str) -> str:
        """Return a formatted prompt block for agent context."""
        concs  = self.recall_conclusions(query, top_k=4)
        contrs = self.recall_contradictions(query, top_k=3)
        falses = self.recall_falsehoods(query, top_k=3)
        if not concs and not contrs and not falses:
            return ""
        lines = ["═══ EPISTEMIC RECORD (accumulated across all sessions) ═══"]
        if concs:
            lines.append("CONCLUSIONS REACHED:")
            for c in concs:
                lines.append(f"  ✓ [{c.agent} | turn {c.turn}] {c.text[:200]}")
        if contrs:
            lines.append("OPEN CONTRADICTIONS (unresolved):")
            for c in contrs:
                lines.append(
                    f"  ⚡ [{c.agent_a} vs {c.agent_b} | turn {c.turn}]\n"
                    f"     {c.claim_a[:160]}\n"
                    f"     ↔ {c.claim_b[:160]}"
                )
        if falses:
            lines.append("ESTABLISHED FALSEHOODS:")
            for f in falses:
                lines.append(f"  ✗ [refuted by {f.refuted_by} | turn {f.turn}] {f.text[:200]}")
        lines.append("══════════════════════════════════════════════════════")
        return "\n".join(lines)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "conclusions": len(self.conclusions),
            "contradictions": len(self.contradictions),
            "falsehoods": len(self.falsehoods),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = {
            "conclusions":   [asdict(c) for c in self.conclusions],
            "contradictions": [asdict(c) for c in self.contradictions],
            "falsehoods":    [asdict(f) for f in self.falsehoods],
        }
        _write_json(self._root / "all_resolutions.json", data)

    def _load_all(self) -> None:
        data = _read_json(self._root / "all_resolutions.json") or {}
        self.conclusions   = [Conclusion(**d)   for d in data.get("conclusions",   [])]
        self.contradictions = [Contradiction(**d) for d in data.get("contradictions", [])]
        self.falsehoods    = [Falsehood(**d)    for d in data.get("falsehoods",    [])]


# ── Module singleton ─────────────────────────────────────────────────────────

_store: ResolutionStore | None = None


def get_resolution_store() -> ResolutionStore:
    global _store
    if _store is None:
        _store = ResolutionStore()
    return _store


# ── Helpers ──────────────────────────────────────────────────────────────────

def _kw(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    return list({t for t in tokens if t not in _STOP})


def _score_list(items, query, top_k, text_fn):
    qkw = set(_kw(query))
    if not qkw:
        return items[:top_k]
    scored = []
    for item in items:
        txt_kw = set(item.keywords)
        overlap = qkw & txt_kw
        if overlap:
            scored.append((len(overlap) / max(len(qkw), 1), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def _write_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
