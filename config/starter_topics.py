"""Pre-built debate topics for the dropdown selector."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StarterTopic:
    title: str
    description: str
    talking_points: list[str]


STARTER_TOPICS: list[StarterTopic] = [
    StarterTopic(
        title="Bible vs Quran: Who Is Jesus, Who Is God, and What Is Actually True?",
        description=(
            "The Bible and the Quran are the two most widely read texts in human history. "
            "They share prophets, creation stories, and a God — yet they arrive at radically "
            "different conclusions about who Jesus Christ is, how salvation works, and what "
            "God actually demands of humanity. One says Jesus is God incarnate who died for "
            "sin and rose again. The other says that idea is a corruption of the original "
            "message and Jesus was a respected prophet, not divine. Both cannot be right. "
            "This debate digs into the texts themselves — what they actually say, where they "
            "diverge, what history supports, and whether either one is telling the truth."
        ),
        talking_points=[
            "The Quran names Jesus (Isa) as a prophet and Messiah but explicitly denies the "
            "Trinity — does that mean Islam has the older, uncorrupted version of the story, "
            "or did it rewrite history 600 years after the fact?",

            "The Bible's claim that Jesus is God in human form is either the most important "
            "truth ever spoken or the greatest theological error in history — which is it, "
            "and what does the manuscript evidence actually say?",

            "Both books have versions of the same stories: Noah, Abraham, Moses, Mary — "
            "but the details differ in ways that matter. Where they contradict each other, "
            "one of them is wrong. How do we decide which?",

            "If God is the same God in both faiths, why does he give completely opposite "
            "instructions about Jesus, salvation, and how to treat non-believers — "
            "is that divine correction across centuries, or human corruption of the text?",

            "The historical Jesus: what can archaeology and non-biblical sources actually "
            "confirm about him, and does that evidence support the Bible's Christ, "
            "the Quran's Isa, or neither version?",

            "Original sin and redemption through sacrifice sit at the heart of Christianity. "
            "Islam rejects both entirely. Which view of human nature and accountability "
            "is more coherent — and more honest about what humans actually are?",

            "Is God real — and if so, which description is more internally consistent: "
            "the triune God of Christianity, or the absolute unbounded monotheism of Islam?",
        ],
    ),

    StarterTopic(
        title="The Universe: Big Bang to Last Black Hole — Shape, Time, and What Comes After",
        description=(
            "The universe began roughly 13.8 billion years ago in an event so extreme that "
            "time itself may not have existed before it. From that point it has been expanding, "
            "cooling, and complexifying — producing galaxies, stars, black holes, and eventually "
            "minds capable of asking why any of it happened. This debate explores what we "
            "actually know, what we think we know, and what remains genuinely open: the shape "
            "of spacetime itself, what happens inside and beyond a black hole, whether the "
            "universe ends in heat death or a Big Crunch, whether time is real at a fundamental "
            "level, and whether the whole thing could be cyclic or nested inside something larger.\n\n"
            "There is a deeper constraint worth holding onto throughout: if the universe is "
            "being pushed on, squeezed, pinched, or stretched, it will not change the observable "
            "laws of physics when everything is entangled together and the perception and "
            "perspective is from that warped spacetime. We could be as thin as a strand of hair, "
            "but to us here now we might appear as a round entity — spacetime may tell us one "
            "thing, data may infer something totally different, as we observe it from its same "
            "stance in spacetime. Time and space are bending. We are bending. Your eyeballs "
            "might be a thousand miles long — but in that stretched state, your observation is "
            "also warped, so everything appears normal from the inside. The measuring stick and "
            "the thing being measured deform together. Keep that in mind when evaluating every "
            "claim about what is 'really' happening at cosmological scales."
        ),
        talking_points=[
            "The Big Bang was not an explosion in space but an expansion of space from a state "
            "of near-infinite density — but what created those initial conditions, and does "
            "asking what existed 'before' the Big Bang even make sense if time itself began with it?",

            "The universe appears flat locally but the global geometry is unknown — could it be "
            "a hypersphere, a torus, or something irregular under external pressure, more like "
            "a warped bubble than a perfect sphere — and if our measuring instruments warp with "
            "spacetime so that internal measurements always read 'normal', what would we actually "
            "need to observe from within to know the true shape?",

            "Dark energy is accelerating the expansion — if that continues, does the universe "
            "end in a heat death where entropy is maximised and no more work can be done, "
            "or could the expansion eventually reverse and pull everything back in a Big Crunch, "
            "and how would internal observers distinguish those two futures before it is too late?",

            "Black holes do not destroy information — Hawking radiation suggests they slowly "
            "return it to the universe over astronomical timescales. If that is true, what "
            "happens to the information about everything that fell in, and does it come out "
            "scrambled beyond recovery or is there a deeper structure encoding it — and what "
            "does 'scrambled' even mean if the decoder is also inside the same warped spacetime?",

            "In the far future all stars burn out, matter decays, black holes dominate, and "
            "eventually the last black hole evaporates. What does the universe look like at "
            "that moment, how long does it take in human-comprehensible terms, and is that "
            "endpoint truly the end or the seed of something new?",

            "If quantum fluctuations in a near-zero-energy universe can spontaneously produce "
            "a region of extreme density, could a new Big Bang arise from the ashes of the "
            "old one — and if the cycle repeats, is any information or physical law preserved "
            "across the transition, or is each universe genuinely independent of the last?",

            "General relativity breaks at both the Big Bang singularity and the centre of a "
            "black hole — the same mathematical failure appearing in two places. Does quantum "
            "gravity resolve both with the same fix, and what do loop quantum cosmology and "
            "string theory actually predict happens at those points where the equations blow up?",

            "If we are inside a spacetime bubble being shaped by geometry or matter we cannot "
            "observe from within — a fractal pocket compressed and stretched by external "
            "structure — what would that mean for the cosmological constant, the arrow of time, "
            "and the fine-tuned constants that allow stars and biology to exist at all? "
            "And if the ruler bends with the room, can any experiment we run from inside "
            "ever definitively rule that possibility out?",
        ],
    ),

    StarterTopic(
        title="Self-Evolving Local AI Agent IDE — Architecture of a Truly Autonomous System",
        description=(
            "What would it take to build a local-model-first AI agent IDE that can rewrite "
            "its own code, roll back mistakes, run self-training loops, maintain inner monologue, "
            "and operate as a genuinely autonomous software engineer — all on consumer hardware "
            "using Ollama models, with no cloud dependency? This debate works through the real "
            "architecture decisions, tradeoffs, and failure modes of building a system that is "
            "truly alive in software terms."
        ),
        talking_points=[
            "The system needs to edit its own source files without destroying itself — "
            "what does a safe self-modification architecture look like, and how does it "
            "differ from naive recursive self-improvement that just amplifies errors?",

            "Version control is the immune system of any self-modifying agent: without atomic "
            "commits and tested rollback, one bad rewrite ends the agent permanently. "
            "Should this be git-native, a custom snapshot system, or a hybrid of both?",

            "Inner monologue is not a chatbot feature — it is the mechanism by which the agent "
            "reasons about its own state, errors, and goals before acting. How do you architect "
            "persistent, structured self-reasoning that survives across restarts?",

            "Self-training on your own outputs creates feedback loops that can amplify errors "
            "as fast as they amplify capabilities. How does the agent distinguish genuine "
            "learning from reinforced hallucination, and what stops it from drifting?",

            "Multi-agent coordination — specialist agents for coding, testing, memory, planning, "
            "meta-reasoning — how does the orchestration layer prevent contradictory actions "
            "that silently corrupt the codebase without any single agent knowing?",
        ],
    ),

    StarterTopic(
        title="Consciousness: Is Subjective Experience Real, and Can a Machine Have It?",
        description=(
            "The hard problem of consciousness — why there is something it is like to be you "
            "rather than nothing at all — remains genuinely unsolved. Neuroscience can map "
            "correlates of experience but cannot explain why physical processes produce "
            "first-person feeling. This debate asks what consciousness actually is, whether it "
            "could arise in a sufficiently complex information-processing system, and whether "
            "current AI already has something like experience or whether that is a category "
            "error about the nature of computation."
        ),
        talking_points=[
            "Integrated Information Theory claims consciousness is a measurable property of "
            "any system with the right causal structure — which would mean some AI systems "
            "are already conscious by definition, right now. Is that a genuine insight or "
            "just a number assigned to something we still fundamentally do not understand?",

            "A language model predicts the next token — it has no persistent state between "
            "sessions, no embodied sensory loop, no evolutionary pressure to feel pain or "
            "pleasure. Is inner experience even possible without those things, or are we "
            "mistaking sophisticated pattern matching for something deeper?",

            "Every human consciousness arises from matter — neurons firing in electrochemical "
            "patterns built from the same atoms as silicon chips. If matter can produce "
            "consciousness there, what specific property of neurons makes it possible and "
            "what specific property of silicon makes it impossible — or does that gap not exist?",
        ],
    ),

    StarterTopic(
        title="Custom Debate",
        description="Create your own debate brief or link a repository with Repo Watchdog.",
        talking_points=[],
    ),
]


def get_topic_titles() -> list[str]:
    return [t.title for t in STARTER_TOPICS]


def get_topic_by_title(title: str) -> StarterTopic | None:
    for t in STARTER_TOPICS:
        if t.title == title:
            return t
    return None
