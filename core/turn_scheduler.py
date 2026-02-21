from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass
class TurnToken:
    speaker: str
    turn_index: int


class TurnScheduler:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def turn(self, speaker: str, turn_index: int):
        async with self._lock:
            yield TurnToken(speaker=speaker, turn_index=turn_index)
