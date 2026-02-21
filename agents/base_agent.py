from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    name: str

    @abstractmethod
    async def think(self, topic: str, talking_point: str, opponent_last_message: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def speak(
        self,
        topic: str,
        talking_point: str,
        private_thought: str,
        opponent_last_message: str,
        evidence_context: list[str] | None = None,
        sub_topics_explored: list[str] | None = None,
    ) -> str:
        raise NotImplementedError
