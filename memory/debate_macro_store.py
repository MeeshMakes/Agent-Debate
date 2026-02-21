"""Debate Macro Store — learned behavioral patterns for argument quality.

Macros are *persistent, confidence-weighted behavioral rules* that every
agent sees at the start of each turn.  They accumulate across debates.

Categories
----------
CONSTRAINT     — rules that always fire, every turn (e.g. "no compliment openers")
WORKFLOW       — argument strategies for specific situations
ANTI_PATTERN   — moves that reliably erode debate quality
DOMAIN         — topic-scoped knowledge anchors

The store is seeded with proven debate-quality rules on first run.
Every time an argument strategy is used and the arbiter rates it highly,
its confidence rises.  When it fails, confidence falls.
Macros below `_DROP_THRESHOLD` are automatically suppressed (not deleted).

Persistence: ``memory/debate_macros.json``
"""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

_STORE_FILE = Path(__file__).parent / "debate_macros.json"
_DROP_THRESHOLD = 0.15   # macros below this confidence are treated as inactive
_MIN_KEYWORD_SCORE = 0.05


# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

class MacroCategory:
    CONSTRAINT   = "constraint"    # Always on — injected every turn
    WORKFLOW     = "workflow"      # Situational strategy
    ANTI_PATTERN = "anti_pattern"  # Things to actively avoid
    DOMAIN       = "domain"        # Topic-scoped knowledge


# ---------------------------------------------------------------------------
# Macro dataclass
# ---------------------------------------------------------------------------

@dataclass
class DebateMacro:
    id: str
    name: str
    category: str               # MacroCategory constant
    content: str                # The actual instruction / rule
    trigger_keywords: list[str] = field(default_factory=list)
    # Triggers this macro when ANY of these words appear in topic+context.
    # Empty list → macro fires if category == CONSTRAINT (always).

    confidence: float = 0.75    # 0.0–1.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: str | None = None
    created_at: str = field(default_factory=lambda: _ts())

    # ---- scoring ----

    def relevance(self, context: str) -> float:
        """Keyword-overlap relevance, confidence-weighted. 0.0–1.0."""
        if self.category == MacroCategory.CONSTRAINT:
            return self.confidence   # constraints always relevant

        if not self.trigger_keywords:
            return 0.0

        ctx_tokens = _tokenize(context)
        if not ctx_tokens:
            return 0.0

        hits = sum(1 for kw in self.trigger_keywords if kw.lower() in ctx_tokens)
        raw = hits / len(self.trigger_keywords)
        return math.sqrt(raw) * self.confidence   # sqrt softens sparsity penalty

    def record_use(self, success: bool) -> None:
        self.usage_count += 1
        self.last_used = _ts()
        if success:
            self.success_count += 1
            self.confidence = min(0.99, self.confidence + 0.03)
        else:
            self.failure_count += 1
            self.confidence = max(0.05, self.confidence - 0.06)

    def is_active(self) -> bool:
        return self.confidence >= _DROP_THRESHOLD

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DebateMacro":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class DebateMacroStore:
    """Persistent store of debate behavioral macros."""

    def __init__(self) -> None:
        self._macros: dict[str, DebateMacro] = {}
        self._load()
        if not self._macros:
            self._seed()

    # ------------------------------------------------------------------
    # Query

    def get_relevant_macros(
        self,
        topic: str,
        context: str = "",
        top_n: int = 4,
    ) -> list[DebateMacro]:
        """Return constraints (always) + top-N scored non-constraint macros."""
        combined_ctx = f"{topic} {context}".lower()

        constraints = [
            m for m in self._macros.values()
            if m.category == MacroCategory.CONSTRAINT and m.is_active()
        ]

        scored: list[tuple[float, DebateMacro]] = []
        for m in self._macros.values():
            if m.category == MacroCategory.CONSTRAINT:
                continue
            if not m.is_active():
                continue
            score = m.relevance(combined_ctx)
            if score >= _MIN_KEYWORD_SCORE:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:top_n]]

        # Deduplicate — constraints never appear in the scored list anyway
        return constraints + top

    def format_for_prompt(self, topic: str, context: str = "") -> str:
        """Return a formatted block ready for prompt injection. Empty string if nothing."""
        macros = self.get_relevant_macros(topic, context)
        if not macros:
            return ""

        lines = ["\n=== LEARNED ARGUMENT STRATEGIES (apply these this turn) ==="]
        for m in macros:
            tag = {"constraint": "RULE", "workflow": "STRATEGY",
                   "anti_pattern": "AVOID", "domain": "KNOWLEDGE"}.get(m.category, m.category.upper())
            lines.append(f"  [{tag}] {m.content}")
        lines.append("==========================================================\n")
        return "\n".join(lines)

    def all_macros(self) -> list[DebateMacro]:
        return list(self._macros.values())

    def get(self, macro_id: str) -> DebateMacro | None:
        return self._macros.get(macro_id)

    # ------------------------------------------------------------------
    # Write

    def add_macro(
        self,
        name: str,
        category: str,
        content: str,
        trigger_keywords: list[str] | None = None,
        confidence: float = 0.6,
    ) -> DebateMacro:
        m = DebateMacro(
            id=str(uuid.uuid4())[:8],
            name=name,
            category=category,
            content=content,
            trigger_keywords=trigger_keywords or [],
            confidence=confidence,
        )
        self._macros[m.id] = m
        self._save()
        return m

    def record_usage(self, macro_id: str, success: bool) -> None:
        m = self._macros.get(macro_id)
        if m:
            m.record_use(success)
            self._save()

    def record_arbiter_scores(
        self,
        agent_name: str,
        arbiter_note: str,
        score: float,
        topic: str,
        context: str = "",
    ) -> None:
        """Push arbiter feedback into macro confidence.

        Macros that were relevant to this context get credit if the score
        is positive, penalty if negative.  This is the learning hook.
        """
        relevant = self.get_relevant_macros(topic, context)
        threshold = 0.65   # arbiter score above which we treat as success
        success = score >= threshold
        for m in relevant:
            m.record_use(success)
        if relevant:
            self._save()

    # ------------------------------------------------------------------
    # Persistence

    def _load(self) -> None:
        if not _STORE_FILE.exists():
            return
        try:
            raw = json.loads(_STORE_FILE.read_text(encoding="utf-8"))
            for d in raw:
                m = DebateMacro.from_dict(d)
                self._macros[m.id] = m
        except Exception:
            self._macros = {}

    def _save(self) -> None:
        try:
            data = [m.to_dict() for m in self._macros.values()]
            _STORE_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Seed data

    def _seed(self) -> None:
        """Populate initial high-confidence macros on first run."""
        seeds: list[dict] = [
            # ---- Constraints (fire every turn) ----
            dict(
                name="no_compliment_opener",
                category=MacroCategory.CONSTRAINT,
                content=(
                    "Never open your response with an acknowledgment, compliment, or "
                    "reflection of what the other agent just said.  Start with YOUR idea.  "
                    "If your first sentence is about them, delete it and start again."
                ),
                trigger_keywords=[],
                confidence=0.92,
            ),
            dict(
                name="specificity_over_adjectives",
                category=MacroCategory.CONSTRAINT,
                content=(
                    "Every sentence must earn its place with a specific claim — a named concept, "
                    "a number, a mechanism, a named study, or a precise logical gap.  "
                    "Sentences that only say something is 'important', 'complex', or 'fascinating' "
                    "add zero information and must be deleted or replaced."
                ),
                trigger_keywords=[],
                confidence=0.90,
            ),
            dict(
                name="vary_sentence_structure",
                category=MacroCategory.CONSTRAINT,
                content=(
                    "If your previous response started with [I / The / This / That / It / "
                    "Consider / While / Although], you must start this one differently.  "
                    "Vary your rhetorical entry angle every single turn."
                ),
                trigger_keywords=[],
                confidence=0.85,
            ),

            # ---- Workflows (situational) ----
            dict(
                name="shift_to_mechanism_after_stalemate",
                category=MacroCategory.WORKFLOW,
                content=(
                    "When the same disagreement has been asserted twice without resolution, "
                    "stop asserting and explain the MECHANISM: why is your claimed causal "
                    "relationship true at the level of physics, biology, economics, or logic?  "
                    "Assertion loops are broken only by mechanistic explanation."
                ),
                trigger_keywords=["disagree", "claim", "argue", "assert", "mechanism", "cause", "why"],
                confidence=0.80,
            ),
            dict(
                name="close_before_opening",
                category=MacroCategory.WORKFLOW,
                content=(
                    "Before introducing a new sub-topic or line of argument, explicitly close "
                    "any thread you have raised and not resolved.  Use CONCLUDE: to mark it. "
                    "Abandoned threads erode debate coherence and let weak arguments survive unchallenged."
                ),
                trigger_keywords=["sub-topic", "branch", "new", "also", "additionally", "furthermore"],
                confidence=0.75,
            ),
            dict(
                name="calibrate_epistemic_confidence",
                category=MacroCategory.WORKFLOW,
                content=(
                    "State your confidence level explicitly when making empirical claims.  "
                    "'This is near-certain given X' vs 'This is speculative — here is why I raise it anyway.'  "
                    "Undifferentiated certainty across all claims destroys your credibility when one is wrong."
                ),
                trigger_keywords=["evidence", "study", "data", "certain", "proof", "fact", "know"],
                confidence=0.72,
            ),

            # ---- Anti-patterns ----
            dict(
                name="no_mirror_restate",
                category=MacroCategory.ANTI_PATTERN,
                content=(
                    "AVOID: Restating the opponent's argument back at them with minor rewording "
                    "is not a counterargument.  It burns a turn and signals you have no response.  "
                    "If you need to reference their claim, do it in one clause then immediately advance."
                ),
                trigger_keywords=["you said", "you argued", "your point", "you mentioned", "you claim"],
                confidence=0.82,
            ),
            dict(
                name="no_hedge_every_sentence",
                category=MacroCategory.ANTI_PATTERN,
                content=(
                    "AVOID: Hedging every sentence with 'perhaps', 'might', 'could possibly', "
                    "'it seems', 'one might argue'.  Use epistemic hedges only when you are "
                    "genuinely uncertain about the mechanism.  Pervasive hedging reads as evasion."
                ),
                trigger_keywords=["perhaps", "might", "possibly", "seems", "could", "may suggest"],
                confidence=0.78,
            ),
        ]

        for s in seeds:
            mid = str(uuid.uuid4())[:8]
            m = DebateMacro(
                id=mid,
                name=s["name"],
                category=s["category"],
                content=s["content"],
                trigger_keywords=s.get("trigger_keywords", []),
                confidence=s.get("confidence", 0.75),
            )
            self._macros[m.id] = m

        self._save()

    # ------------------------------------------------------------------
    # Iteration
    def __iter__(self) -> Iterator[DebateMacro]:
        return iter(self._macros.values())

    def __len__(self) -> int:
        return len(self._macros)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _ts() -> str:
    return str(int(time.time()))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: DebateMacroStore | None = None


def get_debate_macro_store() -> DebateMacroStore:
    global _store
    if _store is None:
        _store = DebateMacroStore()
    return _store
