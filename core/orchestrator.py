from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import Callable

from agents.persona_agent import PersonaAgent
from core.arbiter_engine import ArbiterEngine
from core.debate_quality import DebateQuality, detect_echo, detect_cross_echo
from core.session_manager import SessionManager
from core.state_machine import DebateState, DebateStateMachine
from core.turn_scheduler import TurnScheduler
from debate_graph.manager import DebateGraphManager
from evidence.retriever import CorpusEvidenceRetriever
from runtime_logging.event_logger import EventLogger
from runtime_logging.telemetry import Telemetry
from memory.evidence_store import EvidenceStore
from memory.history_window import SlidingPairWindow
from memory.living_topic import LivingTopic
from memory.private_memory import PrivateMemory
from memory.resolution_store import ResolutionStore, get_resolution_store
from memory.shared_memory import SharedMemory


@dataclass
class DebateEvent:
    event_type: str
    payload: dict


class DebateOrchestrator:
    def __init__(
        self,
        left_agent: PersonaAgent,
        right_agent: PersonaAgent,
        arbiter: ArbiterEngine,
        graph: DebateGraphManager,
        logger: EventLogger,
        evidence_retriever: CorpusEvidenceRetriever | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.left_agent = left_agent
        self.right_agent = right_agent
        self.arbiter = arbiter
        self.graph = graph
        self.logger = logger
        self.evidence_retriever = evidence_retriever
        self._session_manager = session_manager

        self.quality = DebateQuality()
        self.state_machine = DebateStateMachine()
        self.scheduler = TurnScheduler()
        self.shared_memory = SharedMemory()
        self.left_private = PrivateMemory()
        self.right_private = PrivateMemory()
        self.pair_window = SlidingPairWindow(max_pairs=30)
        self.evidence_store = EvidenceStore()
        self.telemetry = Telemetry()

        self._listeners: list[Callable[[DebateEvent], None]] = []
        self._stop_requested = False
        self._pause_requested = False
        # threading.Event is used (not asyncio.Event) because pause/resume/stop are
        # always called from the Qt main thread, while the debate loop runs on a
        # separate QThread.  asyncio.Event.set() is NOT thread-safe from outside the
        # event loop; threading.Event + run_in_executor is the correct pattern.
        self._pause_gate = threading.Event()
        self._pause_gate.set()    # gate is open (not paused) initially
        self._pair_count = 0      # incremented after each full pair of turns
        self._sub_topics_explored: list[str] = []
        self._root_talking_point_id: str | None = None
        self._active_talking_point_label: str = ""
        self._living_topic: LivingTopic | None = None
        self._resolution_store: ResolutionStore = get_resolution_store()
        self._turn_index: int = 0   # persists across continue_debate calls
        self._agent_recent_msgs: dict[str, list[str]] = {}  # name → last 3 messages

    def subscribe(self, listener: Callable[[DebateEvent], None]) -> None:
        self._listeners.append(listener)

    def stop(self) -> None:
        self._stop_requested = True
        self._pause_gate.set()    # open the gate so the blocked loop can exit

    def pause(self) -> None:
        """Request a pair-aware pause — honoured only after both agents finish a pair."""
        self._pause_requested = True
        self._pause_gate.clear()  # close the gate (thread-safe)

    def resume(self) -> None:
        """Resume a paused debate, continuing exactly where it left off."""
        self._pause_requested = False
        self._pause_gate.set()    # open the gate (thread-safe)

    def inject_arbiter_message(self, text: str) -> None:
        """Inject an arbiter message into the conversation then resume.

        The formatted message is written into shared_memory so both agents see
        it in their next conversation_window.  Calling resume() here is what
        actually unblocks the paused loop — pressing Send IS pressing Play.
        """
        formatted = (
            f"[ARBITER INJECTION — You must read this carefully and address it directly "
            f"in your next response before continuing your argument] {text.strip()}"
        )
        self.shared_memory.add_public("Arbiter", formatted)
        self._emit("arbiter_injection", {"message": text, "formatted": formatted})
        # Unblock the loop — next agent will see the injection in shared_memory
        self.resume()

    @property
    def is_paused(self) -> bool:
        return self._pause_requested

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def _emit(self, event_type: str, payload: dict) -> None:
        event = DebateEvent(event_type=event_type, payload=payload)
        self.logger.log(event_type=event_type, payload=payload)
        if self._session_manager:
            self._session_manager.record_event(event_type, payload)
        for listener in self._listeners:
            listener(event)

    def _emit_graph(self) -> None:
        """Emit graph event with both simple rows and rich tree-node dicts."""
        tree_nodes = self.graph.as_tree_nodes()
        self._emit("graph", {
            "rows": self.graph.as_rows(),
            "tree": [
                {
                    "type":   n.node_type,
                    "label":  n.label,
                    "status": n.status,
                    "depth":  n.depth,
                    "order":  n.creation_order,
                }
                for n in tree_nodes
            ],
        })

    def _emit_transition(self, transition) -> None:
        self._emit(
            "state",
            {
                "transition": {
                    "previous": transition.previous.value,
                    "current": transition.current.value,
                    "reason": transition.reason,
                }
            },
        )

    def _extract_sub_topics(self, message: str) -> list[str]:
        """Pull SUB-TOPIC: lines from agent response."""
        found = []
        for line in message.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("SUB-TOPIC:"):
                sub = stripped[10:].strip()
                # Strip leading signal noise / markdown
                sub = re.sub(r'^[*_`\-\.]+', '', sub).strip()
                if sub and len(sub) > 6 and sub not in self._sub_topics_explored:
                    found.append(sub)
        return found

    def _extract_conclude(self, message: str) -> list[str]:
        """Pull CONCLUDE: statements — settled sub-topic points."""
        found = []
        for line in message.split("\n"):
            s = line.strip()
            if s.upper().startswith("CONCLUDE:"):
                txt = s[9:].strip()
                if txt:
                    found.append(txt)
        return found

    def _extract_contradict(self, message: str) -> list[tuple[str, str]]:
        """Pull CONTRADICT: <claim A> / <claim B> pairs."""
        found = []
        for line in message.split("\n"):
            s = line.strip()
            if s.upper().startswith("CONTRADICT:"):
                body = s[11:].strip()
                # expect 'claim A / claim B' or 'claim A  //  claim B'
                parts = re.split(r'\s*/{1,2}\s*', body, maxsplit=1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    found.append((parts[0].strip(), parts[1].strip()))
        return found

    def _extract_false(self, message: str) -> list[str]:
        """Pull FALSE: statements — claims that have been falsified."""
        found = []
        for line in message.split("\n"):
            s = line.strip()
            if s.upper().startswith("FALSE:"):
                txt = s[6:].strip()
                if txt:
                    found.append(txt)
        return found

    def _extract_expand_topic(self, message: str) -> list[str]:
        """Pull EXPAND-TOPIC: additions to the living topic document."""
        found = []
        for line in message.split("\n"):
            s = line.strip()
            if s.upper().startswith("EXPAND-TOPIC:"):
                txt = s[13:].strip()
                if txt:
                    found.append(txt)
        return found

    def _cycle_state(self, turn_index: int) -> None:
        """Cycle through debate states for sustained exploration."""
        cycle_pos = turn_index % 12
        if cycle_pos == 1:
            self._emit_transition(
                self.state_machine.transition(DebateState.OPENING, "New exploration cycle")
            )
        elif cycle_pos == 3:
            self._emit_transition(
                self.state_machine.transition(DebateState.EXPLORATION, "Expand subtopics")
            )
        elif cycle_pos == 6:
            self._emit_transition(
                self.state_machine.transition(DebateState.CONTEST, "Evidence-heavy contest")
            )
        elif cycle_pos == 9:
            self._emit_transition(
                self.state_machine.transition(DebateState.SYNTHESIS, "Synthesis checkpoint")
            )

    async def run_debate(self, topic: str, turns: int = 10, endless: bool = False) -> None:
        self._stop_requested = False
        self._sub_topics_explored = []
        self._turn_index = 0
        self._pair_count = 0

        # Initialise living topic from seed
        self._living_topic = LivingTopic(seed=topic)

        # Tell resolution store which session we're in
        if self._session_manager and self._session_manager.current_session:
            self._resolution_store.set_session(self._session_manager.current_session.session_id)

        # Open a new session on disk
        if self._session_manager:
            self._session_manager.new_session(
                topic=topic,
                left_agent=self.left_agent.name,
                right_agent=self.right_agent.name,
                left_model=getattr(self.left_agent.provider, "model", "unknown"),
                right_model=getattr(self.right_agent.provider, "model", "unknown"),
            )
            # Now set session id in resolution store
            if self._session_manager.current_session:
                self._resolution_store.set_session(
                    self._session_manager.current_session.session_id
                )

        self._emit_transition(self.state_machine.transition(DebateState.BRIEFING, "Topic accepted"))
        root_tp = self.graph.add_talking_point(label=topic)
        self._root_talking_point_id = root_tp.node_id
        self._active_talking_point_label = topic
        self._emit_graph()

        self._emit_transition(self.state_machine.transition(DebateState.OPENING, "Opening statements"))

        await self._run_turns(topic=topic, turns=turns, endless=endless)

    async def continue_debate(self, extra_turns: int = 10) -> None:
        """Continue an already-finished debate for extra_turns more turns."""
        if self._living_topic is None:
            return  # not yet initialised
        self._stop_requested = False
        topic = self._living_topic.seed
        await self._run_turns(topic=topic, turns=extra_turns, endless=False)

    async def _run_turns(
        self, topic: str, turns: int, endless: bool
    ) -> None:
        """Core turn loop — shared by run_debate and continue_debate."""
        current_speaker = (
            self.left_agent if self._turn_index % 2 == 0 else self.right_agent
        )
        current_private = (
            self.left_private if current_speaker is self.left_agent else self.right_private
        )
        other_speaker = (
            self.right_agent if current_speaker is self.left_agent else self.left_agent
        )

        max_turns = 999_999 if endless else (self._turn_index + turns)

        while self._turn_index < max_turns:
            self._turn_index += 1
            turn_index = self._turn_index

            if self._stop_requested:
                break

            # Pair-aware pause: only pause after an even number of individual turns
            # (i.e. after both agents have spoken once in a pair)
            if self._pause_requested and turn_index % 2 == 1:
                # Let current turn finish, then wait
                pass  # pause handled BELOW, after both speakers in pair

            # Cycle debate states
            self._cycle_state(turn_index)

            async with self.scheduler.turn(current_speaker.name, turn_index):
                opponent_last = other_speaker.last_public_message

                # Full conversation window for context
                conversation_window = self.shared_memory.recent_messages(n=12)

                # Living topic document
                living_doc = self._living_topic.to_document() if self._living_topic else ""

                # Resolution store context
                res_context = self._resolution_store.build_context_block(
                    f"{topic} {self._active_talking_point_label}"
                )

                # -- THINK phase --
                thought = await current_speaker.think(
                    topic=topic,
                    talking_point=self._active_talking_point_label,
                    opponent_last_message=opponent_last,
                    conversation_window=conversation_window,
                    living_topic_doc=living_doc,
                    resolution_context=res_context,
                )
                current_private.add_thought(thought)

                self._emit(
                    "private_thought",
                    {"agent": current_speaker.name, "thought": thought, "turn": turn_index},
                )

                # -- EVIDENCE retrieval --
                evidence_context, evidence_citations = self._collect_evidence_context(
                    query=f"{topic}\n{self._active_talking_point_label}\n{opponent_last}\n{thought}",
                    top_k=2,
                )

                # -- SPEAK phase --
                message = await current_speaker.speak(
                    topic=topic,
                    talking_point=self._active_talking_point_label,
                    private_thought=thought,
                    opponent_last_message=opponent_last,
                    evidence_context=evidence_context,
                    sub_topics_explored=list(self._sub_topics_explored),
                    conversation_window=conversation_window,
                    living_topic_doc=living_doc,
                    resolution_context=res_context,
                )

                # -- Parse all signal types --
                new_subs        = self._extract_sub_topics(message)
                conclusions     = self._extract_conclude(message)
                contradictions  = self._extract_contradict(message)
                falsehoods      = self._extract_false(message)
                expansions      = self._extract_expand_topic(message)

                # -- Update living topic --
                for exp in expansions:
                    self._living_topic.add_expansion(exp, current_speaker.name, turn_index)
                for conc in conclusions:
                    self._living_topic.append_conclusion(conc, current_speaker.name, turn_index)
                    self._resolution_store.add_conclusion(conc, current_speaker.name, topic, turn_index)
                    self.graph.add_child(
                        self._root_talking_point_id or "", "conclusion", conc[:120]
                    )
                for ca, cb in contradictions:
                    other = other_speaker.name
                    self._living_topic.append_contradiction(ca, cb, current_speaker.name, other, turn_index)
                    self._resolution_store.add_contradiction(ca, cb, current_speaker.name, other, topic, turn_index)
                    self.graph.add_child(
                        self._root_talking_point_id or "", "contradiction", f"{ca[:80]} ↔ {cb[:80]}"
                    )
                for false_txt in falsehoods:
                    self._living_topic.append_falsehood(false_txt, current_speaker.name, turn_index)
                    self._resolution_store.add_falsehood(false_txt, current_speaker.name, topic, turn_index)
                    self.graph.add_child(
                        self._root_talking_point_id or "", "falsehood", false_txt[:120]
                    )

                # -- Branch sub-topics --
                for sub in new_subs:
                    self._sub_topics_explored.append(sub)
                    self.graph.add_child(
                        self._root_talking_point_id or "",
                        "sub-topic",
                        sub,
                    )
                    self._emit("branch", {"sub_topic": sub, "parent": topic, "turn": turn_index})

                if new_subs or conclusions or contradictions or falsehoods:
                    self._emit_graph()

                # Rotate active talking point every few turns
                if self._sub_topics_explored and turn_index % 4 == 0:
                    branch_idx = (turn_index // 4) % len(self._sub_topics_explored)
                    self._active_talking_point_label = self._sub_topics_explored[branch_idx]
                    self._emit("branch_switch", {
                        "new_talking_point": self._active_talking_point_label,
                        "turn": turn_index,
                    })
                elif turn_index % 4 == 2:
                    self._active_talking_point_label = (
                        self._living_topic.seed if self._living_topic else topic
                    )

                # -- REFRAME phase (creative second pass) --
                reframe_text = await current_speaker.reframe(message)

                # -- Record --
                # Compose the full public message — reframe appended so the next
                # agent's context window includes the expressive restatement.
                full_message_for_memory = (
                    message + f"\n\n[↺ Reframe]\n{reframe_text}"
                    if reframe_text else message
                )
                self.shared_memory.add_public(current_speaker.name, full_message_for_memory)
                self.pair_window.add(prompt=thought, response=message)
                quality = self.quality.score(
                    message=full_message_for_memory,
                    recent_messages=self.shared_memory.recent_messages(),
                )

                res_stats = self._resolution_store.stats
                self._emit(
                    "public_message",
                    {
                        "agent": current_speaker.name,
                        "message": message,
                        "reframe": reframe_text,
                        "turn": turn_index,
                        "quality": quality.__dict__,
                        "citations": evidence_citations,
                        "evidence_score": round(
                            sum(c["score"] for c in evidence_citations)
                            / max(len(evidence_citations), 1),
                            3,
                        ),
                        "talking_point": self._active_talking_point_label,
                        "sub_topics_count": len(self._sub_topics_explored),
                        "memory_facts": len(current_speaker.semantic_memory.facts),
                        "living_topic_summary": self._living_topic.summary_line() if self._living_topic else "",
                        "resolution_stats": res_stats,
                    },
                )
                self.telemetry.bump_turn()

                # -- Segue detection — log lateral references to segue buffer --
                if turn_index >= 2:
                    try:
                        from core.segue_buffer import get_segue_buffer, extract_concepts
                        sess_id = ""
                        if self._session_manager and self._session_manager.current_session:
                            sess_id = self._session_manager.current_session.session_id
                        concepts = extract_concepts(message)
                        sbuf = get_segue_buffer()
                        for concept, context in concepts:
                            sbuf.log_segue(
                                concept, sess_id, turn_index,
                                current_speaker.name,
                                self._active_talking_point_label, context,
                            )
                    except Exception:
                        pass   # never break the debate loop

                # -- Arbiter --
                decision = self.arbiter.evaluate(
                    topic=topic, turn_index=turn_index, current_message=message
                )
                if decision.intervention:
                    self.telemetry.bump_intervention()
                    self._emit("arbiter", {"message": decision.message, "turn": turn_index})

                if decision.force_synthesis:
                    note = f"Synthesis checkpoint at turn {turn_index}: compare strongest claims and unresolved assumptions."
                    self.shared_memory.add_synthesis(note)
                    if self._root_talking_point_id:
                        self.graph.add_child(self._root_talking_point_id, "synthesis", note)
                    self._emit_graph()

                # -- Self-echo detection (fires only if arbiter is silent this turn) --
                agent_history = self._agent_recent_msgs.setdefault(current_speaker.name, [])
                is_echo, echo_overlap = detect_echo(message, agent_history)
                agent_history.append(message)
                if len(agent_history) > 3:
                    agent_history.pop(0)
                if is_echo and not decision.intervention:
                    echo_msg = (
                        f"⚠ ECHO ALERT: {current_speaker.name}, your last two turns share "
                        f"{echo_overlap:.0%} vocabulary overlap — you are repeating yourself. "
                        f"Approach this from a completely different angle: different rhetorical "
                        f"strategy, different evidence type, different conceptual entry point."
                    )
                    self.telemetry.bump_intervention()
                    self._emit("arbiter", {"message": echo_msg, "turn": turn_index, "echo": True})

            # Swap speakers
            if current_speaker is self.left_agent:
                current_speaker = self.right_agent
                current_private = self.right_private
                other_speaker = self.left_agent
                self._pair_count += 1
            else:
                current_speaker = self.left_agent
                current_private = self.left_private
                other_speaker = self.right_agent

                # -- Cross-agent convergence check (fires after each full pair) --
                left_hist  = self._agent_recent_msgs.get(self.left_agent.name, [])
                right_hist = self._agent_recent_msgs.get(self.right_agent.name, [])
                if left_hist and right_hist:
                    is_converging, conv_overlap = detect_cross_echo(left_hist[-1], right_hist[-1])
                    if is_converging:
                        conv_msg = (
                            f"⚠ CONVERGENCE ALERT ({conv_overlap:.0%} overlap): Both agents are using "
                            f"nearly identical vocabulary this round. Find the EXACT point where your "
                            f"causal chains diverge — not just that you disagree, but WHERE the "
                            f"reasoning splits and WHY your path is the correct one."
                        )
                        self.shared_memory.add_public("Arbiter", conv_msg)
                        self._emit("arbiter", {"message": conv_msg, "turn": turn_index, "convergence": True})

            # Pair-aware pause: honour AFTER both speakers have had a full turn
            if self._pause_requested and current_speaker is self.left_agent:
                self._emit("paused", {"turn": turn_index, "pair": self._pair_count})
                # Block in a thread-pool executor so the asyncio event loop stays
                # responsive.  threading.Event.wait() is set/cleared from the Qt
                # main thread via pause() / resume() / inject_arbiter_message().
                await asyncio.get_event_loop().run_in_executor(None, self._pause_gate.wait)
                if self._stop_requested:
                    break

            await asyncio.sleep(0.15)

        # -- Resolution --
        if not endless or self._stop_requested:
            self._emit_transition(
                self.state_machine.transition(DebateState.RESOLUTION, "Resolve key points")
            )

            left_truths  = self.left_agent.semantic_memory.truths
            right_truths = self.right_agent.semantic_memory.truths
            left_problems  = self.left_agent.semantic_memory.problems
            right_problems = self.right_agent.semantic_memory.problems

            living_doc = self._living_topic.to_document() if self._living_topic else ""
            res_stats = self._resolution_store.stats

            resolution = {
                "truths_discovered": (left_truths + right_truths)[:10],
                "problems_found":    (left_problems + right_problems)[:10],
                "sub_topics_explored": list(self._sub_topics_explored),
                "total_memory_facts": (
                    len(self.left_agent.semantic_memory.facts)
                    + len(self.right_agent.semantic_memory.facts)
                ),
                "conclusions":     res_stats["conclusions"],
                "contradictions":  res_stats["contradictions"],
                "falsehoods":      res_stats["falsehoods"],
                "living_topic": living_doc,
                "agreements": "See CONCLUDE: items above for confirmed agreements.",
                "next_steps": "Expand open contradictions; verify flagged claims.",
            }
            self._emit("resolution", resolution)

            # Emit living topic update
            self._emit("living_topic", {
                "document": living_doc,
                "summary": self._living_topic.summary_line() if self._living_topic else "",
            })

        self._emit_transition(self.state_machine.transition(DebateState.WRAP, "Debate complete"))
        self._emit("telemetry", self.telemetry.snapshot.__dict__)

        # Persist session to disk
        if self._session_manager:
            left_mem  = self.left_agent.semantic_memory
            right_mem = self.right_agent.semantic_memory
            if self._stop_requested:
                self._session_manager.mark_stopped()
            else:
                living_dict = self._living_topic.to_dict() if self._living_topic else []
                self._session_manager.finalise_session(
                    turn_count=self._turn_index,
                    left_facts=len(left_mem.facts),
                    right_facts=len(right_mem.facts),
                    sub_topics=list(self._sub_topics_explored),
                    truths_count=len(left_mem.truths + right_mem.truths),
                    problems_count=len(left_mem.problems + right_mem.problems),
                    left_memory_data=left_mem.to_dict(),
                    right_memory_data=right_mem.to_dict(),
                    graph_rows=self.graph.as_rows(),
                )

    def _collect_evidence_context(self, query: str, top_k: int = 2) -> tuple[list[str], list[dict]]:
        if self.evidence_retriever is None:
            return [], []

        snippets = self.evidence_retriever.search(query=query, top_k=top_k)
        context: list[str] = []
        citations: list[dict] = []
        for snippet in snippets:
            line = f"{snippet.title}: {snippet.snippet[:220].replace(chr(10), ' ')}"
            context.append(line)
            self.evidence_store.add(claim=query[:180], support=line)
            citations.append(
                {
                    "title": snippet.title,
                    "score": snippet.score,
                    "source": snippet.source_display,
                    "source_path": snippet.source,
                    "excerpt": snippet.snippet[:180].replace("\n", " "),
                }
            )

        if context:
            self._emit("evidence", {"items": context, "citations": citations})
        return context, citations
