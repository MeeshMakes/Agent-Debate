from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrivateMemory:
    thoughts: list[str] = field(default_factory=list)

    def add_thought(self, thought: str) -> None:
        self.thoughts.append(thought)

    def recent_thoughts(self, n: int = 10) -> list[str]:
        return self.thoughts[-n:]
