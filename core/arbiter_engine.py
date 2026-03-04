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
_STRUCTURE_TOPIC_RE = re.compile(
    r"\b(architecture|architectural|system design|data flow|workflow|pipeline|graph|"
    r"module|integration|orchestrator|component|state machine|topology)\b",
    re.IGNORECASE,
)
_UIUX_TOPIC_RE = re.compile(
    r"\b(ui|ux|user interface|user experience|navigation|dialog|toolbar|button|"
    r"interaction|usability|accessibility)\b",
    re.IGNORECASE,
)
_DIAGRAM_HEADER_RE = re.compile(r"(^|\n)\s*(?:DIAGRAM|ARCH-DIAGRAM|ASCII-DIAGRAM)\s*:", re.IGNORECASE)
_ASCII_EDGE_RE = re.compile(r"->|=>|-->|──>|\bflows?\s+to\b", re.IGNORECASE)
_NODE_RE = re.compile(r"\[[^\]]+\]|\([^\)]+\)|\b[A-Z][A-Za-z0-9_]{2,}\b")
_UI_PLAN_HEADER_RE = re.compile(r"(^|\n)\s*UI-PLAN\s*:", re.IGNORECASE)
_CHANGE_ITEM_RE = re.compile(r"\bCHANGE\s*[-_ ]?\d+\b|(^|\n)\s*(?:\d+\.|-)\s+", re.IGNORECASE)
_HOOK_HINT_RE = re.compile(r"\.py\b|\bclass\b|\bdef\b|\bfunction\b|\bmethod\b|\bwidget\b|\bdialog\b|\bpanel\b", re.IGNORECASE)
_ACCEPT_HINT_RE = re.compile(r"\bacceptance\b|\bverify\b|\bcheck\b|\bexpected\b|\bpass(es)?\b", re.IGNORECASE)
_GRAPH_TOPIC_RE = re.compile(r"\b(graph|node|edge|semantic distance|knowledge graph|adjacency|topology)\b", re.IGNORECASE)
_GRAPH_SCHEMA_HEADER_RE = re.compile(r"(^|\n)\s*(?:GRAPH-SCHEMA|GRAPH SCHEMA|EDGE-SCHEMA)\s*:", re.IGNORECASE)
_GRAPH_EDGE_LINE_RE = re.compile(
    r"^\s*[^\n|]{2,}->[^\n|]{2,}\s*\|\s*relation\s*=\s*[a-z_]{3,}(?:\s*\|\s*(?:weight|confidence)\s*=\s*(?:1(?:\.0+)?|0(?:\.\d+)?))?",
    re.IGNORECASE | re.MULTILINE,
)
_GRAPH_WEIGHT_RE = re.compile(
    r"\|\s*(?:weight|confidence)\s*=\s*(?:1(?:\.0+)?|0(?:\.\d+)?)",
    re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(r"(^|\n)\s*[A-Z][A-Z\- ]{2,}\s*:")
_PROPOSAL_MARKER_RE = re.compile(
    r"\b(?:PROPOSAL\s*[-_ ]?\d+|PROPOSE\s*[-_ ]?\d+|NEW\s+WORKFLOW|NEW\s+MODULE|"
    r"EXTEND\s+LOGIC|ADD\s+(?:A|AN|THE)\s+(?:STAGE|PANEL|PIPELINE|CHECKPOINT|GUARD))\b",
    re.IGNORECASE,
)


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
        self._artifact_last_warn_turn: dict[str, int] = {}
        self._artifact_last_ok_turn: dict[str, int] = {}
        self._artifact_last_missing_sig: dict[str, str] = {}
        self._artifact_warn_cooldown: int = 8
        self._artifact_ok_grace_turns: int = 2

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

        # 3b. Structured artifact quality checks (diagram + UI/UX implementation detail)
        artifact_msg = self._check_artifact_quality(
            topic=topic,
            message=current_message,
            turn_index=turn_index,
            agent_name=agent_name,
        )

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

        if artifact_msg:
            return ArbiterDecision(
                intervention=True,
                force_synthesis=False,
                fabrication_warning=False,
                exploration_directive=directive,
                message=artifact_msg,
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

    def _check_artifact_quality(self, topic: str, message: str, turn_index: int, agent_name: str) -> str:
        """Require concrete artifacts on architecture/UI-heavy debates."""
        combined = f"{topic}\n{message[:1200]}"
        needs_diagram = bool(_STRUCTURE_TOPIC_RE.search(combined))
        needs_uiux = bool(_UIUX_TOPIC_RE.search(combined))
        needs_graph = bool(_GRAPH_TOPIC_RE.search(combined))
        needs_proposals = needs_diagram or needs_uiux or needs_graph
        has_diagram = _has_plaintext_diagram(message)
        has_uiux = _has_uiux_plan(message)
        has_graph = _has_graph_schema(message)
        has_proposals = bool(_PROPOSAL_MARKER_RE.search(message))

        missing: list[str] = []
        if needs_graph and not has_graph and turn_index % 3 == 1:
            missing.append(
                "a GRAPH-SCHEMA: block with at least 3 typed directed edges and weights"
            )
        if needs_diagram and not has_diagram and (not needs_graph or turn_index % 4 == 0):
            missing.append(
                "a plain-text DIAGRAM: block (ASCII nodes + directional edges)"
            )
        if needs_uiux and not has_uiux and not needs_graph:
            missing.append(
                "a UI-PLAN: block with 3 concrete changes including implementation hook + acceptance check"
            )
        if needs_proposals and not has_proposals:
            missing.append(
                "at least 2 concrete PROPOSAL-* lines introducing new workflow/module/logic improvements"
            )

        if not missing:
            if agent_name:
                self._artifact_last_ok_turn[agent_name] = turn_index
                self._artifact_last_missing_sig.pop(agent_name, None)
            return ""

        if agent_name:
            last_warn = self._artifact_last_warn_turn.get(agent_name, -10_000)
            last_ok = self._artifact_last_ok_turn.get(agent_name, -10_000)
            missing_sig = " | ".join(missing)
            last_sig = self._artifact_last_missing_sig.get(agent_name, "")
            if (turn_index - last_warn) < self._artifact_warn_cooldown:
                return ""
            if (turn_index - last_ok) <= self._artifact_ok_grace_turns:
                return ""
            if last_sig == missing_sig and (turn_index - last_warn) < (self._artifact_warn_cooldown * 2):
                return ""
            self._artifact_last_warn_turn[agent_name] = turn_index
            self._artifact_last_missing_sig[agent_name] = missing_sig

        return (
            "ARTIFACT QUALITY DIRECTIVE: Your last turn stayed too abstract. "
            "Next turn must include " + " and ".join(missing) + ". "
            "Focus on executable design changes, not only high-level critique. "
            "If a file/function hook is uncertain, mark it HYPOTHETICAL."
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


def _has_plaintext_diagram(text: str) -> bool:
    body = _extract_named_block(text, _DIAGRAM_HEADER_RE, limit=900)
    if not body:
        edge_lines = [ln for ln in text.splitlines() if _ASCII_EDGE_RE.search(ln)]
        if len(edge_lines) < 2:
            return False
        node_count = sum(len(_NODE_RE.findall(ln)) for ln in edge_lines)
        return node_count >= 4
    has_edges = bool(_ASCII_EDGE_RE.search(body))
    nodes = _NODE_RE.findall(body)
    return has_edges and len(nodes) >= 4


def _has_uiux_plan(text: str) -> bool:
    body = _extract_named_block(text, _UI_PLAN_HEADER_RE, limit=1400)
    if not body:
        return False
    change_items = _CHANGE_ITEM_RE.findall(body)
    has_hook = bool(_HOOK_HINT_RE.search(body))
    has_accept = bool(_ACCEPT_HINT_RE.search(body))
    return len(change_items) >= 3 and has_hook and has_accept


def _has_graph_schema(text: str) -> bool:
    body = _extract_named_block(text, _GRAPH_SCHEMA_HEADER_RE, limit=1200)
    if not body:
        return False
    typed_edges = _GRAPH_EDGE_LINE_RE.findall(body)
    weighted_edges = _GRAPH_WEIGHT_RE.findall(body)
    return len(typed_edges) >= 3 and len(weighted_edges) >= 2


def _extract_named_block(text: str, header_re: re.Pattern[str], limit: int) -> str:
    match = header_re.search(text)
    if not match:
        return ""
    tail = text[match.end(): match.end() + limit]
    next_header = _SECTION_HEADER_RE.search(tail)
    if next_header:
        return tail[:next_header.start()]
    return tail
