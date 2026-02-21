from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class PairRecord:
    prompt: str
    response: str


class SlidingPairWindow:
    def __init__(self, max_pairs: int = 20) -> None:
        self._pairs: deque[PairRecord] = deque(maxlen=max_pairs)

    def add(self, prompt: str, response: str) -> None:
        self._pairs.append(PairRecord(prompt=prompt, response=response))

    def to_context(self) -> list[str]:
        return [f"Prompt: {pair.prompt}\nResponse: {pair.response}" for pair in self._pairs]
