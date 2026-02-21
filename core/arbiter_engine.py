from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArbiterDecision:
    intervention: bool
    message: str
    force_synthesis: bool


class ArbiterEngine:
    def __init__(self, drift_threshold: int = 2, synthesis_interval: int = 6) -> None:
        self._drift_threshold = drift_threshold
        self._synthesis_interval = synthesis_interval
        self._drift_count = 0

    def evaluate(self, topic: str, turn_index: int, current_message: str) -> ArbiterDecision:
        topic_terms = {w.lower().strip(".,!?;:") for w in topic.split() if len(w) > 2}
        msg_terms = {w.lower().strip(".,!?;:") for w in current_message.split() if len(w) > 2}

        on_topic = bool(topic_terms & msg_terms)
        if on_topic:
            self._drift_count = 0
        else:
            self._drift_count += 1

        force_synthesis = turn_index > 0 and turn_index % self._synthesis_interval == 0

        if self._drift_count >= self._drift_threshold:
            self._drift_count = 0
            return ArbiterDecision(
                intervention=True,
                force_synthesis=False,
                message="Let’s refocus on the active talking point and anchor claims in evidence.",
            )

        if force_synthesis:
            return ArbiterDecision(
                intervention=True,
                force_synthesis=True,
                message="Checkpoint: both agents summarize strongest opposing point, then advance one refined argument.",
            )

        return ArbiterDecision(intervention=False, force_synthesis=False, message="")
