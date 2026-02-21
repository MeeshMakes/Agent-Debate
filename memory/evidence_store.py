from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    claim: str
    support: str


@dataclass
class EvidenceStore:
    items: list[EvidenceItem] = field(default_factory=list)

    def add(self, claim: str, support: str) -> None:
        self.items.append(EvidenceItem(claim=claim, support=support))

    def latest(self, n: int = 5) -> list[EvidenceItem]:
        return self.items[-n:]
