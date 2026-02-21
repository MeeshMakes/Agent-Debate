from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SharedMemory:
    transcript: list[str] = field(default_factory=list)
    synthesis_notes: list[str] = field(default_factory=list)

    def add_public(self, speaker: str, message: str) -> None:
        self.transcript.append(f"{speaker}: {message}")

    def add_synthesis(self, note: str) -> None:
        self.synthesis_notes.append(note)

    def recent_messages(self, n: int = 8) -> list[str]:
        return self.transcript[-n:]
