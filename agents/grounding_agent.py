"""Grounding Agent — Cross-Reference Pass

After an agent produces a response, this module:

  1. Extracts every speculative, assumptive, diagnostic, or suggestive
     claim from the text — each isolated as a "claim entity".
  2. Semantically queries the loaded repo dataset for each claim.
  3. Returns a GroundingReport that can be injected into the next turn's
     context so the agent knows what already exists in the codebase.

This prevents agents from confidently proposing things that already exist,
or asserting that something is missing when it is merely named differently.

The pass is synchronous and dataset-only (no additional LLM calls).
Progress is communicated via an optional callback(done: int, total: int).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ingestion.dataset_context import DatasetContextProvider


# ── Claim-entity extraction ───────────────────────────────────────────────────

# Sentence-boundary split (keep the sentence text, not empty tokens).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Markers that flag a sentence as a claim that should be cross-referenced.
_SPECULATIVE   = re.compile(
    r"\b(might|could|may|perhaps|possibly|probably|seemingly|would suggest|"
    r"it appears|it seems|appear to|seems like|likely|unlikely|presumably)\b",
    re.IGNORECASE,
)
_ASSUMPTIVE    = re.compile(
    r"\b(assume|assuming|assumption|absence of|lacks|lack of|missing|no evidence|"
    r"not present|doesn.t exist|does not exist|without|no such|no .{1,20} mechanism|"
    r"no .{1,20} check|no .{1,20} test|no .{1,20} validation)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC    = re.compile(
    r"\b(the problem is|this represents|vulnerability|critical flaw|risk|"
    r"fails to|failure|insufficient|inadequate|incorrect|bug|issue|flaw|"
    r"broken|regression|cascading|compounding|error|unstable)\b",
    re.IGNORECASE,
)
_SUGGESTIVE    = re.compile(
    r"\b(I propose|we should|must implement|requires|recommend|should add|"
    r"should include|introduce|incorporate|needs to|need to add|"
    r"implement a|build a|create a|a suite of|new mechanism|new module|"
    r"new component|additional agent|an automated)\b",
    re.IGNORECASE,
)

_ALL_PATTERNS = [_SPECULATIVE, _ASSUMPTIVE, _DIAGNOSTIC, _SUGGESTIVE]

# Minimum and maximum claim length (chars) to avoid noise.
_MIN_CLAIM_CHARS = 24
_MAX_CLAIM_CHARS = 320


def extract_claim_entities(text: str) -> list[str]:
    """Return sentences from *text* that contain a speculative/assumptive/
    diagnostic/suggestive marker.  Each returned string is a single claim.
    """
    # Split on double-newlines (paragraphs) then on sentence boundaries
    raw_sents: list[str] = []
    for para in re.split(r"\n{2,}", text):
        for sent in _SENT_SPLIT_RE.split(para):
            s = sent.strip()
            # Remove markdown noise
            s = re.sub(r"\*{1,3}|_{1,3}|`{1,3}", "", s).strip()
            if _MIN_CLAIM_CHARS <= len(s) <= _MAX_CLAIM_CHARS:
                raw_sents.append(s)

    claims: list[str] = []
    seen: set[str] = set()
    for sent in raw_sents:
        if any(p.search(sent) for p in _ALL_PATTERNS):
            key = sent[:80].lower()
            if key not in seen:
                seen.add(key)
                claims.append(sent)

    return claims


# ── Semantic search against dataset ──────────────────────────────────────────

def _extract_keywords(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "have", "has", "had", "not", "but", "you", "your", "their",
        "there", "into", "will", "would", "could", "should", "while", "where",
        "which", "what", "does", "also", "been", "even", "just", "than",
        "more", "such", "any", "all", "its", "can", "may",
    }
    tokens = re.findall(r"[A-Za-z_]{4,}", text.lower())
    return {t for t in tokens if t not in stop}


def semantic_search_dataset(
    query: str,
    facts: list[dict],
    top_k: int = 5,
    min_overlap: int = 1,
) -> list[dict]:
    """Return up to *top_k* fact chunks most relevant to *query*.

    Scoring: keyword overlap × TF-IDF weight of the chunk.
    Returns list of dicts: {source_path, source_file, text, score}.
    """
    if not facts:
        return []

    query_kw = _extract_keywords(query)
    if not query_kw:
        return []

    scored: list[tuple[float, dict]] = []
    for fact in facts:
        fact_kw = set(fact.get("keywords", []))
        overlap = len(query_kw & fact_kw)
        if overlap < min_overlap:
            continue
        weight = float(fact.get("tfidf_weight", 0.5))
        score = (overlap / max(len(query_kw), 1)) * weight
        scored.append((score, fact))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    seen_paths: dict[str, int] = {}
    for score, fact in scored:
        sp = str(fact.get("source_path", fact.get("source_file", "unknown")))
        if seen_paths.get(sp, 0) >= 2:
            continue
        seen_paths[sp] = seen_paths.get(sp, 0) + 1
        results.append({
            "source_path": sp,
            "source_file": fact.get("source_file", sp),
            "text": fact.get("text", "").strip()[:300],
            "score": round(score, 3),
        })
        if len(results) >= top_k:
            break

    return results


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ClaimHit:
    """One claim entity and what the dataset found for it."""
    claim: str
    matches: list[dict] = field(default_factory=list)   # [{source_path, text, score}]

    @property
    def has_match(self) -> bool:
        return bool(self.matches)

    @property
    def best_match(self) -> dict | None:
        return self.matches[0] if self.matches else None


@dataclass
class GroundingReport:
    """Full report for one agent turn."""
    agent: str
    turn: int
    claims_checked: int
    hits: list[ClaimHit] = field(default_factory=list)

    @property
    def matched_hits(self) -> list[ClaimHit]:
        return [h for h in self.hits if h.has_match]

    @property
    def unmatched_hits(self) -> list[ClaimHit]:
        return [h for h in self.hits if not h.has_match]

    def to_context_block(self) -> str:
        """Build a text block to inject into the agent's next THINK prompt."""
        if not self.matched_hits:
            return ""

        lines = [
            "⚠ GROUNDING CROSS-REFERENCE (auto-generated — do NOT cite as new evidence):",
            "The following claims from your last turn were matched against the actual repo dataset.",
            "Before repeating or building on these claims, verify them against the cited source files.",
            "",
        ]
        for hit in self.matched_hits[:8]:
            m = hit.best_match
            lines.append(f"CLAIM: {hit.claim}")
            lines.append(
                f"  ↳ FOUND IN CODEBASE: {m['source_path']}  (relevance {m['score']})"
            )
            preview = m["text"][:160].replace("\n", " ")
            lines.append(f"     excerpt: …{preview}…")
            lines.append("")

        lines.append(
            "If the dataset shows this already exists, do NOT re-propose it as missing or new. "
            "Either confirm it exists and assess whether it is sufficient, "
            "or explain why the existing implementation is inadequate."
        )
        return "\n".join(lines)

    def to_event_payload(self) -> dict:
        return {
            "agent": self.agent,
            "turn": self.turn,
            "claims_checked": self.claims_checked,
            "matched": len(self.matched_hits),
            "unmatched": len(self.unmatched_hits),
            "hits": [
                {
                    "claim": h.claim[:120],
                    "matched": h.has_match,
                    "source": h.best_match["source_path"] if h.has_match else None,
                    "score": h.best_match["score"] if h.has_match else None,
                }
                for h in self.hits
            ],
        }


# ── Main grounding pass ───────────────────────────────────────────────────────

class GroundingAgent:
    """Runs the post-turn claim cross-reference pass.

    Usage (inside the orchestrator, after SPEAK):
        grounding = GroundingAgent()
        report = grounding.run(
            response_text=message,
            dataset_provider=self._dataset_provider,
            agent_name=current_speaker.name,
            turn_index=turn_index,
            progress_cb=lambda done, total: self._emit("grounding_progress", {...}),
        )
        # Store report for next-turn context injection
        self._grounding_reports[current_speaker.name] = report
    """

    def run(
        self,
        response_text: str,
        dataset_provider: DatasetContextProvider,
        agent_name: str = "",
        turn_index: int = 0,
        progress_cb: Callable[[int, int], None] | None = None,
        top_k_per_claim: int = 3,
        max_claims: int = 12,
    ) -> GroundingReport:
        """Extract claims and cross-reference each against the repo dataset.

        Returns a GroundingReport.  If no dataset is loaded, returns an
        empty report (claims_checked=0).
        """
        if not dataset_provider.loaded:
            return GroundingReport(agent=agent_name, turn=turn_index, claims_checked=0)

        claims = extract_claim_entities(response_text)[:max_claims]
        total = len(claims)

        if total == 0:
            return GroundingReport(agent=agent_name, turn=turn_index, claims_checked=0)

        facts = dataset_provider._facts
        hits: list[ClaimHit] = []

        for i, claim in enumerate(claims):
            matches = semantic_search_dataset(claim, facts, top_k=top_k_per_claim)
            hits.append(ClaimHit(claim=claim, matches=matches))
            if progress_cb:
                progress_cb(i + 1, total)

        return GroundingReport(
            agent=agent_name,
            turn=turn_index,
            claims_checked=total,
            hits=hits,
        )
