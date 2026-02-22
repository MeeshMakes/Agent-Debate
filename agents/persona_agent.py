from __future__ import annotations

from dataclasses import dataclass

from agents.base_agent import BaseAgent
from agents.prompt_compiler import compile_persona_prompt
from agents.reflection_engine import ReflectionEngine
from memory.adaptive_prompt_store import get_adaptive_store
from memory.cross_session_memory import get_cross_session_memory
from memory.semantic_memory import SemanticMemory
from providers.base_provider import BaseProvider


@dataclass
class PersonaConfig:
    name: str
    role_style: str
    stance: str
    tone: str = "calm"
    reflection_enabled: bool = True
    voice: str = "david"
    color: str = "#4fc3f7"


class PersonaAgent(BaseAgent):
    def __init__(self, config: PersonaConfig, provider: BaseProvider | None = None) -> None:
        self.config = config
        self.name = config.name
        self.provider = provider
        self.reflection_engine = ReflectionEngine()
        self.semantic_memory = SemanticMemory()
        self.last_public_message = ""
        self._turn_counter = 0

    async def think(
        self,
        topic: str,
        talking_point: str,
        opponent_last_message: str,
        conversation_window: list[str] | None = None,
        living_topic_doc: str = "",
        resolution_context: str = "",
        dataset_context: str = "",
        focus_guidance: str = "",
    ) -> str:
        self._turn_counter += 1

        # Recall related past knowledge
        recall_query = f"{topic} {talking_point} {opponent_last_message[:200]}"
        memory_recall = self.semantic_memory.recall_context(recall_query, top_k=4)

        # Build conversation context paragraph
        conv_summary = ""
        if conversation_window:
            recent = conversation_window[-6:]
            conv_summary = (
                "\nRecent conversation (full history visible in your speak phase):\n"
                + "\n".join(f"  {line}" for line in recent)
            )

        # Living topic context
        living_hint = ""
        if living_topic_doc:
            living_hint = f"\nLiving topic summary:\n{living_topic_doc[:600]}"

        # Resolution context
        res_hint = ""
        if resolution_context:
            res_hint = f"\n{resolution_context[:400]}"

        # Dataset knowledge (ingested codebase / uploaded files)
        dataset_hint = ""
        if dataset_context:
            dataset_hint = f"\n{dataset_context}"

        focus_hint = ""
        if focus_guidance:
            focus_hint = f"\nFocus analytics guidance:\n{focus_guidance}"

        base = (
            f"Focus on talking point '{talking_point}'. "
            f"As a {self.config.role_style}, your only goal is truth-finding, not winning. "
            "Think through the entire conversation so far, then identify the "
            "most unresolved factual question and what specific evidence from "
            "the ingested dataset would actually resolve it.  "
            "Be specific — name files, functions, mechanisms, not vague patterns.\n"
            f"Your accumulated knowledge:\n{memory_recall}"
            f"{conv_summary}{living_hint}{res_hint}{dataset_hint}{focus_hint}"
        )

        if self.config.reflection_enabled:
            reflection = self.reflection_engine.reflect(
                self.last_public_message, opponent_last_message
            )
            base = f"{base}\nReflection: {reflection}"

        # Cross-session breadcrumbs for inner monologue
        csm = get_cross_session_memory()
        if csm.enabled:
            refs = csm.build_monologue_refs(recall_query, top_k=3)
            if refs:
                base += "\nPast-debate echoes:\n" + "\n".join(refs)

        return base

    async def speak(
        self,
        topic: str,
        talking_point: str,
        private_thought: str,
        opponent_last_message: str,
        evidence_context: list[str] | None = None,
        sub_topics_explored: list[str] | None = None,
        conversation_window: list[str] | None = None,
        living_topic_doc: str = "",
        resolution_context: str = "",
        dataset_context: str = "",
        focus_guidance: str = "",
    ) -> str:
        # Build memory context for prompt
        recall_query = f"{topic} {talking_point} {opponent_last_message[:200]}"
        memory_context = self.semantic_memory.recall_context(recall_query, top_k=5)

        system_prompt = compile_persona_prompt(
            name=self.config.name,
            role_style=self.config.role_style,
            stance=self.config.stance,
            topic=topic,
            talking_point=talking_point,
            memory_context=memory_context,
            sub_topics_explored=sub_topics_explored,
            adaptive_store=get_adaptive_store(),
            conversation_history=conversation_window,
            living_topic_doc=living_topic_doc,
            resolution_context=resolution_context,
            dataset_context=dataset_context,
            focus_guidance=focus_guidance,
        )

        evidence_block = ""
        if evidence_context:
            evidence_lines = "\n".join(f"  ⚑ {item}" for item in evidence_context[:3])
            evidence_block = (
                "\n⚠ PAST-DEBATE ARCHIVE — background context only.\n"
                "These are excerpts from OLD UNRELATED DEBATES, NOT the repository being analysed.\n"
                "Do NOT cite these as evidence about the current codebase.\n"
                "If you need to make a code claim, cite the INGESTED CODEBASE KNOWLEDGE block above.\n"
                f"{evidence_lines}\n"
            )

        # Cross-session semantic memory injection
        csm = get_cross_session_memory()
        if csm.enabled:
            cross_block = csm.build_context_block(recall_query, top_k=5)
            if cross_block:
                evidence_block += "\n" + cross_block + "\n"

        if self.provider is not None:
            opponent_context = (
                opponent_last_message
                or "None yet — this is your opening statement.  Set the intellectual frame for this debate."
            )
            prompt = (
                f"Your private thinking before this response: {private_thought}\n\n"
                f"What your opponent (the other debater) just said:\n{opponent_context}\n\n"
                f"{evidence_block}"
                "Now deliver your response.  Be genuinely alive — think out loud, push the argument forward, "
                "and use TRUTH:, SUB-TOPIC:, PROBLEM:, VERIFY: naturally woven into your prose."
            )
            generated = await self.provider.generate(prompt=prompt, system_prompt=system_prompt)

            # Store in semantic memory AND adaptive prompt store
            self._store_response_in_memory(generated, topic, talking_point)

            self.last_public_message = generated
            return generated

        # Fallback template
        evidence_hint = ""
        if evidence_context:
            evidence_hint = f" Relevant evidence to cite: {evidence_context[0]}"

        fallback = (
            f"{self.config.name}: On {talking_point}, I acknowledge your point but challenge its assumptions. "
            "Evidence from comparable cases suggests outcomes depend on implementation quality, not slogans. "
            f"Let's isolate one measurable criterion and test both positions against it.{evidence_hint}"
        )
        self._store_response_in_memory(fallback, topic, talking_point)
        self.last_public_message = fallback
        return fallback

    async def reframe(self, original_response: str) -> str:
        """Creative second pass: re-express the just-delivered argument in a vivid,
        humanistic, sensory way.  Same ideas — different lens.  The agent reads its
        own words and finds the human image inside the logic.

        This is shown as an expressive companion card beneath the main response.
        The next opponent sees both, so the evocative framing charges the conversation.
        """
        if self.provider is None:
            return ""

        system_prompt = (
            f"You are {self.config.name}. You have just delivered a debate argument.\n"
            "Your task: re-read what you said, then give it a second life.\n\n"
            "Reframe it in the most vivid, expressive, human way you can imagine.\n"
            "Think of it like translating dense sheet music into the feeling of hearing it live.\n"
            "Use concrete imagery, metaphor, texture — make the listener *see* and *feel* the idea.\n"
            "You may deepen or refine your thinking if something new crystallises as you look back.\n"
            "This is NOT a summary or a simplification — it is a creative reinterpretation.\n\n"
            "Rules:\n"
            "  • 3–7 sentences maximum — distilled, not diluted\n"
            "  • Do not add completely new arguments\n"
            "  • No bullet points, no headers — flowing prose only\n"
            "  • Stay in character but let the human weight of the idea show\n"
        )
        prompt = (
            f"Here is what you just argued:\n\n{original_response}\n\n"
            "Now reframe it — give it vivid, expressive, human life."
        )
        try:
            result = await self.provider.generate(prompt=prompt, system_prompt=system_prompt)
            return result.strip()
        except Exception:
            return ""

    def _store_response_in_memory(self, text: str, topic: str, talking_point: str) -> None:
        """Parse response for tagged facts and store them in semantic + adaptive memory."""
        adaptive = get_adaptive_store()
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            content: str | None = None
            fact_type: str | None = None
            for prefix, ftype in (
                ("TRUTH:", "truth"),
                ("SUB-TOPIC:", "sub-topic"),
                ("PROBLEM:", "problem"),
                ("VERIFY:", "verify"),
                ("CONCLUDE:", "conclusion"),
                ("EXPAND-TOPIC:", "expansion"),
                ("FALSE:", "falsehood"),
                ("VERIFIED:", "verified_claim"),
                ("HYPOTHETICAL:", "hypothetical"),
            ):
                if prefix in stripped:
                    idx = stripped.index(prefix)
                    content = stripped[idx + len(prefix):].strip()
                    fact_type = ftype
                    break

            if content and fact_type:
                self.semantic_memory.store(
                    content, self.name, self._turn_counter, topic, fact_type
                )
                store_type = fact_type.replace("-", "_")
                adaptive.record(topic, self.name, store_type, content, self._turn_counter)

        # Always store a general claim snapshot
        self.semantic_memory.store(text[:500], self.name, self._turn_counter, topic, "claim")
