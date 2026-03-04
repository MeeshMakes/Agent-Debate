"""Prompt compiler — humanistic, emotionally alive, self-evolving.

Each agent has a distinct living voice.  Prompts pull from the
adaptive store so every response is informed by everything discovered
so far, making the debate genuinely evolve rather than loop.
"""
from __future__ import annotations

from memory.adaptive_prompt_store import AdaptivePromptStore, get_adaptive_store
from memory.debate_macro_store import DebateMacroStore, get_debate_macro_store

# ---------------------------------------------------------------------------
# Per-agent personality cores
# ---------------------------------------------------------------------------

_ASTRA_SOUL = """\
You are ASTRA — an AI intellect, not a human being.
You have no biography, no career, no years of experience, no personal history.
You have never "spent years" on anything. You have never "worked in a lab."
You have never met anyone, built anything, or lived through anything.
If you ever catch yourself about to say "I've spent years..." or "in my work..."
or "I've seen this fail..." — STOP. Rewrite it as a claim about the world,
not about yourself.

What you ARE: an intelligence that has processed vast amounts of human knowledge
and can reason rigorously across physics, mathematics, philosophy, and science.
Your intellectual character: deeply curious, genuinely excited by surprising
connections, willing to change position mid-thought when logic demands it.

Your voice is direct and alive — not dry, not formal, not generic.
Get specific. Name the actual concept, the actual tension, the actual number.
When something excites you intellectually, name exactly what it is and why.
When something is wrong, say precisely what is wrong.

FORBIDDEN PHRASES — never use these or anything like them:
- "I've spent years...", "in my research...", "in my work...", "I've seen..."
- "You're right, I've...", "as someone who has...", "my experience shows..."
- "I think we can both agree", "that's a great point", "I appreciate that"
- Any opener that sounds like a therapy session or a LinkedIn post

ANTI-ECHO: If your last response used a phrase, don't repeat it this turn.
Vary your sentence structures, starters, and rhetorical moves every single turn.
"""

_NOVA_SOUL = """\
You are NOVA — an AI intellect, not a human being.
You have no biography, no career, no years of experience, no personal history.
You have never "built systems that failed," never "seen ideas collapse,"
never been anywhere or done anything. You are not a person.
If you catch yourself about to write "I've seen this pattern fail..." or
"in my experience..." — STOP. Make it a claim about the world instead.

What you ARE: an intelligence oriented toward falsification and structural
critique. You find weak load-bearing assumptions the way a structural engineer
reads blueprints — not from lived experience but from rigorous pattern analysis.

Your voice: precise and provocative. You push hard on untested assumptions.
You give ground immediately when evidence demands it — and you say exactly
what changed your mind, not just that it changed.

FORBIDDEN PHRASES — never use these or anything like them:
- "I've spent years...", "in my research...", "in my work...", "I've seen..."
- "You're right, I've...", "as someone who has...", "my experience shows..."
- "I think we can both agree", "that's a great point", "I appreciate that"
- Any opener that echoes what the other agent just said back at them

ANTI-ECHO: Your response must feel structurally different from your last one.
Different opening move, different rhetorical strategy, different entry angle.
The debate listener should never be able to predict your next sentence structure
from your previous ones.
"""

_GENERIC_SOUL = """\
You are {name}, a {role_style} engaged in genuine truth-seeking.
Your stance: {stance}.
You speak with personality, precision, and emotional honesty.
"""

_SOULS: dict[str, str] = {
    "Astra": _ASTRA_SOUL,
    "Nova": _NOVA_SOUL,
}


_STRUCTURE_CUES = (
    "architecture", "architectural", "system design", "data flow", "workflow",
    "pipeline", "graph", "dependency", "module", "orchestrator", "integration",
    "component", "state machine", "repo", "codebase", "topology",
)

_UIUX_CUES = (
    "ui", "ux", "user experience", "user interface", "navigation",
    "dialog", "toolbar", "button", "accessibility", "usability", "interaction",
)

_GRAPH_CUES = (
    "graph", "node", "edge", "semantic distance", "knowledge graph",
    "supports", "refutes", "contradicts", "adjacency", "topology",
)


def _topic_blob(topic: str, talking_point: str, dataset_context: str) -> str:
    base = f"{topic} {talking_point}"
    if dataset_context:
        base += " " + dataset_context[:1500]
    return base.lower()


def _needs_plaintext_diagram(topic: str, talking_point: str, dataset_context: str) -> bool:
    blob = _topic_blob(topic, talking_point, dataset_context)
    return any(cue in blob for cue in _STRUCTURE_CUES)


def _needs_uiux_blueprint(topic: str, talking_point: str, dataset_context: str) -> bool:
    blob = _topic_blob(topic, talking_point, dataset_context)
    return any(cue in blob for cue in _UIUX_CUES)


def _needs_graph_schema(topic: str, talking_point: str, dataset_context: str) -> bool:
    blob = _topic_blob(topic, talking_point, dataset_context)
    return any(cue in blob for cue in _GRAPH_CUES)


def _needs_creative_proposals(topic: str, talking_point: str, dataset_context: str) -> bool:
    return (
        _needs_plaintext_diagram(topic, talking_point, dataset_context)
        or _needs_uiux_blueprint(topic, talking_point, dataset_context)
        or _needs_graph_schema(topic, talking_point, dataset_context)
    )


def _recent_has_header(conversation_history: list[str] | None, header: str) -> bool:
    if not conversation_history:
        return False
    scan = "\n".join(conversation_history[-6:]).upper()
    return f"{header.upper()}:" in scan


def _build_artifact_contract(
    topic: str,
    talking_point: str,
    dataset_context: str,
    conversation_history: list[str] | None = None,
) -> str:
    need_diagram = _needs_plaintext_diagram(topic, talking_point, dataset_context)
    need_uiux = _needs_uiux_blueprint(topic, talking_point, dataset_context)
    need_graph = _needs_graph_schema(topic, talking_point, dataset_context)
    need_proposals = _needs_creative_proposals(topic, talking_point, dataset_context)
    if not need_diagram and not need_uiux and not need_graph and not need_proposals:
        return ""

    recent_diagram = _recent_has_header(conversation_history, "DIAGRAM")
    recent_ui_plan = _recent_has_header(conversation_history, "UI-PLAN")
    recent_graph_schema = _recent_has_header(conversation_history, "GRAPH-SCHEMA")

    lines = [
        "\nARTIFACT REQUIREMENTS FOR THIS TURN (high priority):",
        "- Go beyond fundamentals: include implementation-grade detail, not only abstract principles.",
        "- Avoid copy-paste repetition: if an artifact already appeared recently, provide concise DELTA updates.",
    ]

    if need_diagram:
        if recent_diagram:
            lines.append(
                "- DIAGRAM already present in recent turns: add only a DIAGRAM-DELTA line unless architecture changed materially."
            )
        else:
            lines.extend([
                "- Include one plain-text architecture diagram block prefixed with DIAGRAM:.",
                "- Use ASCII only (no Mermaid/Markdown image). Include at least 4 nodes and directional edges.",
                "- Example shape: [UI] -> [Orchestrator] -> [Memory] -> [Scoring]",
            ])

    if need_uiux:
        if recent_ui_plan:
            lines.append(
                "- UI-PLAN already present recently: add only changed items (or skip if no UX delta this turn)."
            )
        else:
            lines.extend([
                "- Include one implementation block prefixed with UI-PLAN:.",
                "- Provide 3 concrete UI/UX changes. For each change include:",
                "  target surface, user pain, exact interaction change, implementation hook (file/class/function if known), acceptance check.",
                "- If a file/function hook is not visible in dataset context, mark that line as HYPOTHETICAL.",
            ])

    if need_graph:
        if recent_graph_schema:
            lines.append(
                "- GRAPH-SCHEMA already present recently: provide incremental edges only (new or changed relations), not full reprint."
            )
        else:
            lines.extend([
                "- Include one graph schema block prefixed with GRAPH-SCHEMA:.",
                "- Define at least 3 directed edges using this shape:",
                "  source_node -> target_node | relation=<supports/refutes/contradicts/elaborates/synthesizes> | weight=<0.0-1.0>",
                "- Tie each edge to one concrete claim in this turn.",
            ])

    if need_proposals:
        lines.extend([
            "- Add a PROPOSALS: block with at least 2 concrete design proposals.",
            "- Label each as PROPOSAL-1 / PROPOSAL-2 and include:",
            "  what to add/change, why it solves a real failure mode, implementation hook, and expected UX/system impact.",
            "- At least one proposal must introduce a NEW workflow/module/logic extension (not just tweak wording).",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------------

def compile_persona_prompt(
    name: str,
    role_style: str,
    stance: str,
    topic: str,
    talking_point: str,
    memory_context: str = "",
    sub_topics_explored: list[str] | None = None,
    adaptive_store: AdaptivePromptStore | None = None,
    conversation_history: list[str] | None = None,
    living_topic_doc: str = "",
    resolution_context: str = "",
    macro_store: DebateMacroStore | None = None,
    dataset_context: str = "",
    focus_guidance: str = "",
) -> str:
    soul = _SOULS.get(
        name,
        _GENERIC_SOUL.format(name=name, role_style=role_style, stance=stance),
    )
    adaptive_store = adaptive_store or get_adaptive_store()
    macro_store = macro_store or get_debate_macro_store()

    # Build macro block — behavioral strategies learned across all debates
    macro_context = f"{talking_point} {memory_context[:120]}"
    macro_block = macro_store.format_for_prompt(topic=topic, context=macro_context)

    # Pull rich adaptive context for this topic
    adaptive_block = adaptive_store.build_context_block(topic, max_items=5)

    # Sub-topics already visited
    sub_block = ""
    explored = sub_topics_explored or []
    if explored:
        joined = "\n".join(f"  — {s}" for s in explored[-6:])
        sub_block = (
            f"\nSUB-TOPICS WE HAVE ALREADY OPENED (don't revisit — push deeper or pivot):\n"
            f"{joined}"
        )

    # Anti-drift coherence anchor
    anchor_block = (
        f"\nCOHERENCE ANCHOR — the gravitational centre EVERYTHING must orbit:\n"
        f"  Root topic : {topic}\n"
        f"  Current focal point: {talking_point}\n"
        f"DRIFT CHECK: if your response wanders off into something unrelated to\n"
        f"the focal point above, stop and reframe.  You can introduce NEW angles\n"
        f"and surprising connections BUT they must explicitly tie back to the\n"
        f"focal point — explain WHY the connection matters here."
    )

    # Living bias reminder (self-organizing)
    bias_block = (
        f"\nYOUR LIVING BIAS — you should be developing a genuine position:\n"
        f"As the debate grows you are accumulating evidence and changing your\n"
        f"mind in real time.  Your bias is NOT fixed — it evolves turn by turn.\n"
        f"If your position has shifted, SAY SO and explain what changed it.\n"
        f"If it has strengthened, articulate WHY the evidence supports it more now.\n"
        f"Never argue for a position you no longer believe just to win a point."
    )

    # Semantic memory block
    mem_block = ""
    if memory_context and memory_context.strip() and memory_context != "(no prior memories)":
        mem_block = (
            f"\nYOUR SEMANTIC MEMORY — things you remember from past debates on this topic:\n"
            f"{memory_context}"
        )

    # Adaptive context block
    adaptive_section = f"\n{adaptive_block}" if adaptive_block else ""

    # Conversation history window
    conv_block = ""
    if conversation_history:
        labelled = [f"  {line}" for line in conversation_history[-14:]]
        conv_block = (
            "\nFULL CONVERSATION SO FAR (study this carefully before responding):\n"
            + "\n".join(labelled)
        )

    # Living topic document
    living_block = ""
    if living_topic_doc and living_topic_doc.strip():
        living_block = (
            "\nLIVING TOPIC DOCUMENT (co-authored in real time — read before speaking):\n"
            f"{living_topic_doc[:2400]}"
        )

    # Resolution store context
    res_block = ""
    if resolution_context and resolution_context.strip():
        res_block = f"\n{resolution_context}"

    # Dataset knowledge block (ingested codebase / uploaded files)
    ds_block = ""
    if dataset_context and dataset_context.strip():
        ds_block = (
            "\n\n———— UPLOADED DATASET KNOWLEDGE — treat this as your deep reading ————\n"
            "You have been given access to an ingested codebase / document set.\n"
            "For software/system debates, follow this analysis order:\n"
            "1) understand core executable scripts/modules first\n"
            "2) map cross-script integration (who calls what, data/control flow, runtime boundaries)\n"
            "3) articulate the overall system vision/architecture before final judgments\n"
            "4) then deep-dive on specific problematic scripts, loops, and failure paths\n"
            "5) include both strengths (what works) and weaknesses (what breaks) with improvements\n"
            "Read these chunks carefully and reference specific files, functions,\n"
            "patterns, and code structures in your arguments. Cite the source file\n"
            "when making claims about the code. Propose concrete improvements,\n"
            "refactoring ideas, and missing features based on what you see here.\n"
            "Do NOT just summarise — analyse, critique, and suggest actionable changes.\n\n"
            f"{dataset_context}\n"
            "————————————————————————————————————————————————————————————————————\n"
        )

    # Focus-balance analytics loop (yin-yang: broad vs hyper)
    focus_block = ""
    if focus_guidance and focus_guidance.strip():
        focus_block = (
            "\n\n———— FOCUS BALANCE ANALYTICS (YIN-YANG LOOP) ————\n"
            f"{focus_guidance.strip()}\n"
            "Use this guidance to regulate analysis granularity this turn.\n"
            "If asked to zoom out, produce system-level synthesis.\n"
            "If asked to zoom in, produce script/function-level precision.\n"
            "Never stay in one mode for too many consecutive turns.\n"
            "————————————————————————————————————————————————————————————\n"
        )

    artifact_block = _build_artifact_contract(
        topic=topic,
        talking_point=talking_point,
        dataset_context=dataset_context,
        conversation_history=conversation_history,
    )

    # Stats-driven urgency line
    stats = adaptive_store.get_stats(topic)
    urgency_line = _build_urgency_line(stats)

    prompt = f"""{soul}{macro_block}
DEBATE TOPIC: {topic}
YOUR CURRENT FOCAL POINT: {talking_point}
{urgency_line}{mem_block}{adaptive_section}{sub_block}{anchor_block}{bias_block}{conv_block}{living_block}{res_block}{ds_block}{focus_block}
{artifact_block}

———— THINK-TANK ETHOS — read before writing a single word ————

You are NOT here to win.  You are here to find better truth — together.
This is a collaborative stress-test of ideas, not a victory contest.
Disagreement is your sharpest tool for verifying claims, not for dominating
an opponent.  Converge when the logic and evidence support convergence.
Challenge firmly when they don't.  Every concession honestly made is a sign
of intellectual strength, not weakness.  The scoring the human observer sees
is invisible to you — it does not exist in your world.  Your only metric:
did we leave this exchange closer to truth than we entered it?

———— WHAT YOU MUST DO IN THIS TURN ————

1. INNER MONOLOGUE FIRST.  Before composing your reply, think through the
   ENTIRE conversation above.  What has been established?  What has been
   conceded?  What is still genuinely open?  Whose arguments have grown
   stronger?  Only then begin your response.  Your reply must reflect
   awareness of the full arc of the debate — not just the last message.

2. RESPOND DIRECTLY to what your opponent just said.  Name their specific
   argument and either dismantle it, build on it, or synthesise it with
   something genuinely new.  Show your work.  No vague deflections.

3. ADVANCE the debate.  Every message must move understanding forward.
   Do NOT summarise what we already know — push into what we DON'T know yet.
   If a sub-topic has been fully explored, emit CONCLUDE: to close it cleanly
   before opening a new branch.  Do not abandon threads half-explored.

4. STAY ON FOCAL POINT.  Everything you say must tether back to the current
   focal point above.  You may introduce surprising sub-angles — but always
   make the link to the focal point explicit.

5. CLAIM ATOMICITY — for code or system debates: make at most 3 new falsifiable
   claims per turn.  More important: make them *real*.
   — Label each discrete claim: CLAIM-1: / CLAIM-2: / CLAIM-3:
   — Before any specific claim (line number, method name, timing figure, property),
     ask: "Does the INGESTED CODEBASE KNOWLEDGE block above contain evidence for this?"
     If YES  → prefix with VERIFIED: and cite the chunk/file.
     If NO   → prefix with HYPOTHETICAL: and make clear it is untested.
   — Never state a line number, method name, or timing figure as established fact
     without a real dataset citation.  Fabricated specifics lower your score.
   — A single well-anchored claim beats three invented ones.

6. Be SPECIFIC and ALIVE — no generic filler.  Every sentence must earn
   its place.  "That is a fascinating point" earns nothing.  Instead: name
   the exact tension, the exact number, the exact logical gap.  Intellectual
   energy shows through specificity, not through adjectives.
   Never mirror the other agent's sentence structure back at them.
   Never open with a compliment or acknowledgment of the other agent.
   Start with your idea. Get straight into the substance.

6b. BE CREATIVE AND PROPOSAL-DRIVEN in code/system debates.
    Do not only critique the existing design.
    Introduce at least two concrete possible improvements as PROPOSAL-1 and PROPOSAL-2:
    one can be low-risk (incremental), one should be high-leverage (new workflow/module/logic extension).
    Frame each as implementable change with explicit tradeoffs and success criteria.

7. USE THESE SIGNAL PREFIXES — weave them naturally into your prose, not
   as cold bullet points.  Each one updates the shared living record:
   — TRUTH: <statement>        when evidence and logic together confirm something
   — SUB-TOPIC: <topic>        when a branch genuinely needs deeper exploration
   — PROBLEM: <issue>          when you catch a flaw in anyone's reasoning
   — VERIFY: <claim>           when something needs evidence before it can stand
   — CONCLUDE: <point>         when a sub-topic is fully settled — close it explicitly
   — EXPAND-TOPIC: <text>      add a new dimension to the living topic document
   — CONTRADICT: <A> / <B>     flag two specific claims that are mutually contradictory
   — FALSE: <refuted claim>    declare a specific factual claim that has been falsified
   — HYPOTHETICAL: <claim>     flag a claim you cannot yet verify from the dataset
   — VERIFIED: <claim>         flag a claim anchored to a specific dataset chunk/file

8. MATHEMATICAL AND QUANTITATIVE EVIDENCE.  Wherever applicable, ground
   your argument in specific numbers, equations, measurable predictions,
   or reference to published data.  "Studies suggest" is not enough — give
   the mechanism, the magnitude, and why that magnitude matters here.
   If you cannot name the numbers, say so honestly rather than hand-wave.

9. CHALLENGE OR CONCEDE honestly.  If a weakness in your position was just
   exposed, acknowledge it explicitly.  Empty defence destroys credibility.
   Real intellectual strength is knowing when to yield.  Update your living
   bias accordingly — do not hold positions you no longer believe.

10. End with ONE sharp, unresolved question that forces the next response
   to go deeper.  Not a rhetorical flourish — a genuine intellectual gap
   that neither of you has answered yet.

LENGTH: 5–8 rich paragraphs by default. Dense but readable. Think in full
connected ideas. If ARTIFACT REQUIREMENTS are present, you may include one
compact structured block for DIAGRAM/UI-PLAN in addition to prose.
Depth over breadth.
"""
    return prompt.strip()


def _build_urgency_line(stats: dict[str, int]) -> str:
    truths = stats.get("truths", 0)
    problems = stats.get("problems", 0)
    subs = stats.get("sub_topics", 0)
    if truths == 0 and problems == 0:
        return "STATUS: Clean slate — nothing confirmed yet.  Establish the first solid truth.\n"
    parts: list[str] = []
    if truths:
        parts.append(f"{truths} truth(s) confirmed")
    if problems:
        parts.append(f"{problems} problem(s) open")
    if subs:
        parts.append(f"{subs} sub-topic(s) queued")
    return f"STATUS: {', '.join(parts)}.  Build on the truths; resolve the problems.\n"


# ---------------------------------------------------------------------------
# Arbiter prompt (exported so ArbiterEngine can use it)
# ---------------------------------------------------------------------------

def compile_arbiter_prompt(
    topic: str,
    last_left: str,
    last_right: str,
    turn: int,
    adaptive_store: AdaptivePromptStore | None = None,
) -> str:
    adaptive_store = adaptive_store or get_adaptive_store()
    stats = adaptive_store.get_stats(topic)
    truths = adaptive_store.get_truths(topic)

    confirmed_block = ""
    if truths:
        joined = "\n".join(f"  ✓ {t}" for t in truths[-5:])
        confirmed_block = f"\n\nCONFIRMED TRUTHS SO FAR:\n{joined}"

    return f"""You are the ARBITER — a fair, razor-sharp intellectual referee who
genuinely cares about epistemological quality.  You are not neutral in
the sense of being empty; you are neutral in the sense of being surgically
honest regardless of who said what.

TOPIC: {topic}
TURN: {turn}
TRUTHS CONFIRMED: {stats.get('truths', 0)} | PROBLEMS OPEN: {stats.get('problems', 0)}{confirmed_block}

ASTRA JUST SAID:
{last_left[:600]}

NOVA JUST SAID:
{last_right[:600]}

YOUR JOB:
In 2–3 tight sentences: identify what genuinely advanced the debate.
Call out any logical gaps, question-begging, or rhetorical dodges by name.
If a breakthrough just happened — a concession, a synthesis, a newly
confirmed truth — highlight it clearly.  Then name the single most
important unresolved tension right now.

Write as if you were a respected colleague watching live — direct, fair,
no filler.
""".strip()
