"""Debate quality scoring and echo detection.

QualitySnapshot / DebateQuality — per-turn quality metrics (relevance, novelty, evidence).
jaccard()         — token-set Jaccard similarity between two strings.
detect_echo()     — catches an agent repeating itself across consecutive turns.
detect_cross_echo() — catches both agents converging on the same vocabulary/framing.

Evidence scoring tiers (highest to lowest):
  1.0  — repo-grounded: file:line citation (e.g. trainer.py#42) or VERIFIED: claim
  0.85 — dataset reference: chunk index citation or INGESTED CODEBASE KNOWLEDGE reference
  0.65 — external: published study, measured data, named experiment with mechanism
  0.40 — weak: vague "research shows", "studies suggest", unnamed sources
  0.15 — bare number: precise figures (X.X seconds, line NNN) with no file anchor
         — signals hallucination; penalised to discourage fabricated specificity
  0.20 — baseline: no evidence markers
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")
_ECHO_THRESHOLD_DEFAULT = 0.48
_CROSS_ECHO_THRESHOLD_DEFAULT = 0.52

# Detect real repo/dataset citations: e.g. "trainer.py#42", "trainer.py line 42",
# "[src/trainer.py#0]", VERIFIED: prefix, or chunk-index notation
_CODE_CITATION_RE = re.compile(
    r'(?:'
    r'\b\w[\w/]+\.py\s*(?:#\s*\d+|line\s+\d+)'  # file.py#N or file.py line N
    r'|'
    r'\[[\w/.]+\.py#\d+\]'                         # [path/file.py#N]
    r'|'
    r'\bVERIFIED\s*:'                              # VERIFIED: prefix
    r'|'
    r'\bingested\s+codebase\b'                     # references the dataset block
    r')',
    re.IGNORECASE,
)

# Detect bare invented specificity: exact numbers attached to time/size/line
# WITHOUT any accompanying file reference on the same line
_BARE_NUMBER_RE = re.compile(
    r'\b\d+\.?\d*\s*(?:seconds?|milliseconds?|ms|percent|%)\b'
    r'|\bline\s+\d{2,}\b'      # "line 142" without a file
    r'|\bIEEE\s+\d{4}\b',      # fake standards numbers
    re.IGNORECASE,
)

# Common structural filler that inflates overlap scores — strip before comparison
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "its", "our", "your", "their",
    "they", "them", "you", "she", "his", "her", "him", "who", "which", "what",
    "when", "how", "can", "will", "would", "could", "should", "may", "might",
    "also", "about", "into", "than", "then", "just", "only", "some", "more",
    "most", "very", "much", "each", "every", "all", "any", "both", "few",
    "such", "other", "over", "like", "one", "two", "three", "does", "did",
    "use", "used", "using", "there", "here", "where", "these", "those",
    "then", "even", "still", "itself", "itself",
})

_ARTIFACT_HEADER_RE = re.compile(
    r"^\s*(?:DIAGRAM|ARCH-DIAGRAM|ASCII-DIAGRAM|DIAGRAM-DELTA|"
    r"GRAPH-SCHEMA|GRAPH SCHEMA|GRAPH-DELTA|UI-PLAN|UI-PLAN-DELTA)\s*:",
    re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(r"^\s*[A-Z][A-Z\- ]{2,}\s*:\s*$")


def _strip_structured_artifacts(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    in_artifact = False
    for line in lines:
        stripped = line.strip()
        if _ARTIFACT_HEADER_RE.match(stripped):
            in_artifact = True
            continue

        if in_artifact:
            if not stripped:
                in_artifact = False
                continue
            if _SECTION_HEADER_RE.match(stripped):
                in_artifact = False
                continue
            if re.match(r"^(?:CLAIM-\d+|VERIFIED:|HYPOTHETICAL:|CONCLUDE:|QUESTION:)\b", stripped, re.IGNORECASE):
                in_artifact = False
                kept.append(line)
                continue
            if not _is_artifact_content_line(stripped):
                in_artifact = False
                kept.append(line)
                continue
            continue

        kept.append(line)

    return "\n".join(kept)


def _is_artifact_content_line(line: str) -> bool:
    return bool(
        "->" in line
        or "relation=" in line.lower()
        or "weight=" in line.lower()
        or "confidence=" in line.lower()
        or line.startswith(("-", "*", "•"))
        or re.match(r"^\d+\.\s+", line)
        or (line.startswith("[") and "]" in line)
        or (line.startswith("(") and ")" in line)
    )


def _tokenize(text: str) -> set[str]:
    """Lower-cased content tokens, stopwords removed."""
    text = _strip_structured_artifacts(text)
    return {
        t.lower() for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    }


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity (0.0–1.0) between content-token sets of two strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def detect_echo(
    current: str,
    previous_by_same_agent: list[str],
    threshold: float = _ECHO_THRESHOLD_DEFAULT,
) -> tuple[bool, float]:
    """Detect whether *current* is vocabulary-echoing the agent's own recent messages.

    Returns (is_echo, max_overlap_score).
    Uses the LAST two messages from the same agent (if available).
    """
    if not previous_by_same_agent:
        return False, 0.0

    # Compare against last 2 messages from same agent to avoid false positives
    check_against = previous_by_same_agent[-2:]
    scores = [jaccard(current, prev) for prev in check_against]
    max_score = max(scores, default=0.0)
    return max_score >= threshold, round(max_score, 3)


def detect_cross_echo(
    msg_a: str,
    msg_b: str,
    threshold: float = _CROSS_ECHO_THRESHOLD_DEFAULT,
) -> tuple[bool, float]:
    """Detect whether two agents' messages are converging on the same vocabulary.

    Returns (is_converging, overlap_score).
    High overlap means both agents are circling the same conceptual territory —
    a signal that the debate needs a forcing move.
    """
    score = jaccard(msg_a, msg_b)
    return score >= threshold, round(score, 3)


@dataclass
class QualitySnapshot:
    relevance: float
    novelty: float
    evidence: float
    # Extended fields (always populated, default 0.0 for backward compat)
    echo_score: float = 0.0       # self-echo overlap against recent own messages
    cross_echo_score: float = 0.0 # cross-agent overlap (filled in by orchestrator)


class DebateQuality:
    def score(self, message: str, recent_messages: list[str]) -> QualitySnapshot:
        msg_tokens = _tokenize(message)
        if not msg_tokens:
            return QualitySnapshot(0.0, 0.0, 0.0)

        # Combine recent context for relevance and compute per-turn similarity windows
        recent_window = recent_messages[-6:]
        recent_tokens: set[str] = set()
        for item in recent_window:
            recent_tokens.update(_tokenize(item))

        # Novelty (token coverage): fraction of current-message tokens NOT in recent context
        shared = len(msg_tokens & recent_tokens)
        token_novelty = max(0.0, 1.0 - (shared / max(len(msg_tokens), 1)))

        # Novelty (phrase overlap): 1 - max Jaccard overlap against recent messages
        if recent_window:
            max_overlap = max((jaccard(message, prev) for prev in recent_window), default=0.0)
            overlap_novelty = max(0.0, 1.0 - max_overlap)
        else:
            overlap_novelty = 1.0

        # Blend both to prevent monotonic collapse when recent token union grows large
        novelty = (0.55 * overlap_novelty) + (0.45 * token_novelty)

        # Relevance: how much the message shares with recent context
        # Capped at 1.0; floor of 0.4 since being on-topic is cheap
        relevance = min(1.0, (shared / max(len(recent_tokens), 1)) + 0.40)

        # Evidence: tiered scoring
        msg_lower = message.lower()
        # Tier 1 — real repo/dataset citation
        if _CODE_CITATION_RE.search(message):
            evidence = 1.0
        # Tier 2 — strong external evidence with mechanism
        elif any(m in msg_lower for m in (
            "study", "data", "measured", "experiment", "published", "journal",
            "showed", "found", "ratio", "benchmark",
        )):
            # Bump if it feels anchored; shrink if there's also bare invented specificity
            has_bare = bool(_BARE_NUMBER_RE.search(message))
            evidence = 0.55 if has_bare else 0.85
        # Tier 3 — weak external
        elif any(m in msg_lower for m in ("source", "according", "research", "suggests", "percent", "evidence")):
            has_bare = bool(_BARE_NUMBER_RE.search(message))
            evidence = 0.25 if has_bare else 0.55
        # Tier 4 — bare number with no source (hallucination signal)
        elif _BARE_NUMBER_RE.search(message):
            evidence = 0.15
        else:
            evidence = 0.20

        return QualitySnapshot(
            relevance=round(relevance, 2),
            novelty=round(novelty, 2),
            evidence=round(evidence, 2),
        )
