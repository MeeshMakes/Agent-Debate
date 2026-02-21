from __future__ import annotations


class ReflectionEngine:
    def reflect(self, own_message: str, opponent_message: str) -> str:
        if not opponent_message:
            return "Open by framing one falsifiable claim and one practical consequence."
        return (
            "Strength in opponent view: "
            f"{opponent_message[:120]}... "
            "Next move: concede one valid point and challenge one unsupported assumption."
        )
