"""Arbiter engine — Truth-Grounding Orchestrator.

Replaces the original drift-detector with three distinct functions:

1. Claim Interceptor
   Scans each message for code-specific claims (line numbers, method names,
   precise timing figures, invented standards) and warns when claims appear
   unverifiable against the ingested dataset.

2. Module Speculation Tracker
   Tracks which files agents have *actually cited from the dataset* vs
   *speculated about*.  Generates exploration directives toward unread modules.

3. Session Synthesizer
   Compresses the raw growing conclusion list into structured knowledge objects
   so context windows don't overflow.

The arbiter never judges who "won".  Output: confirmed findings, open questions,
retracted fabrications, exploration directives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"\bline[s]?\s+\d{1,5}\b", re.IGNORECASE)
_TIMING_RE = re.compile(
    r"\b\d+\.\d+\s*(?:seconds?|ms|milliseconds?)\b"
    r"|\b\d+\s*(?:ms|milliseconds?)\b",
    re.IGNORECASE,
)
_STD_RE = re.compile(r"\b(?:IEEE|RFC|ISO|IEC)\s+\d{3,}\b", re.IGNORECASE)
_CODE_ANCHOR_RE = re.compile(
    r"\b\w[\w/]+\.py\s*(?:#\s*\d+|line\s+\d+)"
    r"|\[[\w/.]+\.py#\d+\]"
    r"|\bVERIFIED\s*:"
    r"|\bingested\s+codebase\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(r"\bHYPOTHETICAL\s*:", re.IGNORECASE)
_PY_FILE_RE = re.compile(r"\b([\w/\\]+\.py)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArbiterDecision:
    intervention: bool
    message: str
    force_synthesis: bool
    fabrication_warning: bool = False
    exploration_directive: str = ""


@dataclass
class StructuredFinding:
    claim: str
    status: str       # "confirmed" | "refuted" | "uncertain"
    evidence: str     # e.g. "src/agents/trainer.py" or "none"
    verified_by: str  # "repo_read" | "dataset_chunk" | "inference" | "none"
    session: str
    agent: str


# ---------------------------------------------------------------------------
# Main arbiter
# ---------------------------------------------------------------------------

class ArbiterEngine:
    def __init__(self, drift_threshold: int = 2, synthesis_interval: int = 6) -> None:
        self._drift_threshold = drift_threshold
        self._synthesis_interval = synthesis_interval
        self._drift_count = 0

        self._cited_files: dict[str, set[str]] = {}
        self._speculated_files: dict[str, set[str]] = {}
        self._all_mentioned_files: set[str] = set()
        self._findings: list[StructuredFinding] = []
        self._session_label: str = "v?"

    def set_session(self, label: str) -> None:
        self._session_label = label

    def register_dataset_citation(self, agent: str, source_path: str) -> None:
        """Mark a file as actually read (from a dataset chunk) for this agent."""
        fname = _basename(source_path)
        if fname:
            self._cited_files.setdefault(agent, set()).add(fname)
            self._all_mentioned_files.add(fname)

    def evaluate(
        self,
        topic: str,
        turn_index: int,
        current_message: str,
        agent_name: str = "",
        dataset_chunks_used: list[str] | None = None,
    ) -> ArbiterDecision:
        if dataset_chunks_used:
            for path in dataset_chunks_used:
                self.register_dataset_citation(agent_name, path)

        # Track files mentioned (speculation detection)
        for f in _extract_mentioned_files(current_message):
            self._all_mentioned_files.add(f)
            if f not in self._cited_files.get(agent_name, set()):
                self._speculated_files.setdefault(agent_name, set()).add(f)

        # 1. Topic drift
        topic_terms = {w.lower().strip(".,!?;:") for w in topic.split() if len(w) > 2}
        msg_terms = {w.lower().strip(".,!?;:") for w in current_message.split() if len(w) > 2}
        if topic_terms & msg_terms:
            self._drift_count = 0
        else:
            self._drift_count += 1

        # 2. Synthesis checkpoint
        force_synthesis = turn_index > 0 and turn_index % self._synthesis_interval == 0

        # 3. Fabrication detection
        fabrication_warning, fab_msg = self._check_fabrications(current_message)

        # 4. Exploration directive (every 4 turns to avoid spam)
        directive = self._build_exploration_directive(agent_name) if turn_index % 4 == 0 else ""

        # Priority: drift > fabrication > synthesis > directive
        if self._drift_count >= self._drift_threshold:
            self._drift_count = 0
            return ArbiterDecision(
                intervention=True,
                force_synthesis=False,
                fabrication_warning=fabrication_warning,
                exploration_directive=directive,
                message=(
                    "REFOCUS: Last two turns drifted from the active talking point. "
                    "Anchor your next claim to a specific file or function from the ingested dataset."
                ),
            )

        if fabrication_warning:
            return ArbiterDecision(
                intervention=True,
                force_synthesis=force_synthesis,
                fabrication_warning=True,
                exploration_directive=directive,
                message=fab_msg,
            )

        if force_synthesis:
            return ArbiterDecision(
                intervention=True,
                force_synthesis=True,
                fabrication_warning=False,
                exploration_directive=directive,
                message=(
                    f"SYNTHESIS CHECKPOINT (turn {turn_index}): State your single strongest "
                    "*verified* finding (file + function + mechanism), then name the most "
                    "important unresolved question that requires reading a specific file to answer.\n"
                    + self._synthesis_note()
                ),
            )

        if directive:
            return ArbiterDecision(
                intervention=True,
                force_synthesis=False,
                fabrication_warning=False,
                exploration_directive=directive,
                message=directive,
            )

        return ArbiterDecision(intervention=False, force_synthesis=False, message="")

    def add_finding(
        self,
        claim: str,
        status: str,
        evidence: str = "none",
        verified_by: str = "none",
        agent: str = "",
    ) -> None:
        self._findings.append(
            StructuredFinding(
                claim=claim, status=status, evidence=evidence,
                verified_by=verified_by, session=self._session_label, agent=agent,
            )
        )

    def synthesize_findings(self) -> list[dict]:
        """Return structured findings as JSON-serialisable dicts."""
        return [
            {"claim": f.claim, "status": f.status, "evidence": f.evidence,
             "verified_by": f.verified_by, "session": f.session, "agent": f.agent}
            for f in self._findings
        ]

    def get_unread_modules(self) -> list[str]:
        cited: set[str] = set()
        for s in self._cited_files.values():
            cited.update(s)
        return sorted(self._all_mentioned_files - cited)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fabrications(self, message: str) -> tuple[bool, str]:
        if _CODE_ANCHOR_RE.search(message):
            return False, ""
        if _HYPOTHETICAL_RE.search(message):
            return False, ""

        issues: list[str] = []
        if hits := _LINE_RE.findall(message):
            issues.append(f"line references without a file anchor ({', '.join(hits[:3])})")
        if hits := _TIMING_RE.findall(message):
            issues.append(f"precise timing figures without a source ({', '.join(hits[:3])})")
        if hits := _STD_RE.findall(message):
            issues.append(f"unverifiable standards citations ({', '.join(hits[:3])})")

        if not issues:
            return False, ""

        return True, (
            f"\u26a0 UNVERIFIED CLAIMS: {'; '.join(issues)}. "
            "Cite the specific file and dataset chunk that supports these figures. "
            "If no dataset evidence exists, prefix the claim with HYPOTHETICAL: instead."
        )

    def _build_exploration_directive(self, agent_name: str) -> str:
        speculated = self._speculated_files.get(agent_name, set())
        cited = self._cited_files.get(agent_name, set())
        unread = sorted(speculated - cited)
        if len(unread) >= 2:
            files = ", ".join(f"`{f}`" for f in unread[:3])
            return (
                f"EXPLORATION DIRECTIVE: You referenced {files} without citing actual dataset "
                "content from those files. Your next turn must anchor at least one claim to a "
                "retrieved chunk from these files before introducing new claims about them."
            )
        return ""

    def _synthesis_note(self) -> str:
        confirmed = [f for f in self._findings if f.status == "confirmed"]
        uncertain = [f for f in self._findings if f.status == "uncertain"]
        lines = []
        if confirmed:
            lines.append(f"  Confirmed: {len(confirmed)} — latest: {confirmed[-1].claim[:100]}")
        if uncertain:
            lines.append(f"  Open: {len(uncertain)} unresolved question(s)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    if not path:
        return ""
    return re.split(r"[/\\]", path)[-1]


def _extract_mentioned_files(text: str) -> list[str]:
    return [_basename(m) for m in _PY_FILE_RE.findall(text) if m]
