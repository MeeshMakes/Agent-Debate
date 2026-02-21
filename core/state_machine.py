from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DebateState(str, Enum):
    INIT = "init"
    BRIEFING = "briefing"
    OPENING = "opening"
    EXPLORATION = "exploration"
    CONTEST = "contest"
    SYNTHESIS = "synthesis"
    RESOLUTION = "resolution"
    WRAP = "wrap"


@dataclass
class StateTransition:
    previous: DebateState
    current: DebateState
    reason: str


class DebateStateMachine:
    def __init__(self) -> None:
        self._state = DebateState.INIT

    @property
    def state(self) -> DebateState:
        return self._state

    def transition(self, next_state: DebateState, reason: str) -> StateTransition:
        previous = self._state
        self._state = next_state
        return StateTransition(previous=previous, current=next_state, reason=reason)
