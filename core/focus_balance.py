from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import re


_ARCH_TERMS = (
    "architecture",
    "system",
    "workflow",
    "pipeline",
    "integration",
    "boundary",
    "data flow",
    "control flow",
    "end-to-end",
    "loop",
)


@dataclass
class FocusSnapshot:
    mode: str
    recommended_next_mode: str
    broad_turns: int
    hyper_turns: int
    balanced_turns: int
    window_size: int
    dominant_file: str


class FocusBalanceTracker:
    """Tracks broad-vs-hyper analysis balance over recent turns.

    Modes:
      - broad: system-level, architecture, cross-module reasoning
      - hyper: narrow file/function deep-dive
      - balanced: mixed mode
    """

    def __init__(self, window_size: int = 6) -> None:
        self._window_size = max(4, window_size)
        self._mode_window: deque[str] = deque(maxlen=self._window_size)
        self._file_window: deque[str] = deque(maxlen=self._window_size)

    def reset(self) -> None:
        self._mode_window.clear()
        self._file_window.clear()

    def observe_turn(
        self,
        *,
        message: str,
        talking_point: str,
        citations: list[dict] | None = None,
    ) -> FocusSnapshot:
        mode, dominant_file = self._classify_mode(
            message=message,
            talking_point=talking_point,
            citations=citations or [],
        )
        self._mode_window.append(mode)
        self._file_window.append(dominant_file)

        broad_turns = sum(1 for m in self._mode_window if m == "broad")
        hyper_turns = sum(1 for m in self._mode_window if m == "hyper")
        balanced_turns = sum(1 for m in self._mode_window if m == "balanced")

        recommended_next_mode = self._recommend_next_mode(
            broad_turns=broad_turns,
            hyper_turns=hyper_turns,
        )

        return FocusSnapshot(
            mode=mode,
            recommended_next_mode=recommended_next_mode,
            broad_turns=broad_turns,
            hyper_turns=hyper_turns,
            balanced_turns=balanced_turns,
            window_size=len(self._mode_window),
            dominant_file=dominant_file,
        )

    def build_guidance(self, talking_point: str) -> str:
        if not self._mode_window:
            return (
                "YIN-YANG FOCUS LOOP: start broad (system purpose, architecture, workflow loops), "
                "then ground claims with one specific script/function anchor."
            )

        broad_turns = sum(1 for m in self._mode_window if m == "broad")
        hyper_turns = sum(1 for m in self._mode_window if m == "hyper")
        balanced_turns = sum(1 for m in self._mode_window if m == "balanced")
        recommendation = self._recommend_next_mode(
            broad_turns=broad_turns,
            hyper_turns=hyper_turns,
        )

        if recommendation == "broad":
            return (
                "YIN-YANG FOCUS LOOP: you have been too hyper-focused recently. "
                "In this turn, zoom out and synthesize: system goal, architecture shape, "
                "workflow loops, and how your recent deep-dive changes the larger picture. "
                f"Current focal point remains: {talking_point}."
            )

        if recommendation == "hyper":
            return (
                "YIN-YANG FOCUS LOOP: you have been too broad recently. "
                "In this turn, zoom in on one concrete script/function-level mechanism, "
                "test one claim precisely, and explain exact failure/strength paths. "
                f"Current focal point remains: {talking_point}."
            )

        return (
            "YIN-YANG FOCUS LOOP: keep balance this turn — include one broad system-level "
            "insight and one precise code-level anchor (script/function/flow point)."
        )

    def _classify_mode(
        self,
        *,
        message: str,
        talking_point: str,
        citations: list[dict],
    ) -> tuple[str, str]:
        text = f"{talking_point}\n{message}".lower()

        source_paths = [str(c.get("source_path", "")).replace("\\", "/") for c in citations]
        source_paths = [p for p in source_paths if p]
        file_counts = Counter(source_paths)
        dominant_file = file_counts.most_common(1)[0][0] if file_counts else ""
        unique_files = len(file_counts)

        arch_hits = sum(1 for term in _ARCH_TERMS if term in text)
        function_hits = len(re.findall(r"\b[a-zA-Z_][\w]*\s*\(", text))
        function_hits += text.count("def ") + text.count("class ")

        if unique_files == 1 and (function_hits >= 2 or arch_hits == 0):
            return "hyper", dominant_file

        if unique_files >= 3 or arch_hits >= 2:
            return "broad", dominant_file

        if function_hits >= 2 and unique_files <= 2:
            return "hyper", dominant_file

        return "balanced", dominant_file

    def _recommend_next_mode(self, *, broad_turns: int, hyper_turns: int) -> str:
        total = len(self._mode_window)
        if total == 0:
            return "balanced"

        last_three = list(self._mode_window)[-3:]
        if len(last_three) == 3 and all(m == "hyper" for m in last_three):
            return "broad"
        if len(last_three) == 3 and all(m == "broad" for m in last_three):
            return "hyper"

        if hyper_turns / total >= 0.65:
            return "broad"
        if broad_turns / total >= 0.65:
            return "hyper"
        return "balanced"
