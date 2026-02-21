"""Live debate monitor — headless test runner.

Run from project root:
    python debug_debate_monitor.py

Prints every agent turn and arbiter commentary to the console.
Saves full transcript to logs/debug_transcript.txt
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app.bootstrap import build_orchestrator
from core.orchestrator import DebateEvent

TOPIC = "Bible vs Quran: Which text has more historical accuracy and moral clarity?"
LEFT_MODEL = "gemma3:27b"
RIGHT_MODEL = "qwen3:30b"
TURNS = 6

TRANSCRIPT_PATH = ROOT / "logs" / "debug_transcript.txt"
TRANSCRIPT_PATH.parent.mkdir(exist_ok=True)

_lines: list[str] = []


def _log(msg: str) -> None:
    print(msg, flush=True)
    _lines.append(msg)


def _save_transcript() -> None:
    TRANSCRIPT_PATH.write_text("\n".join(_lines), encoding="utf-8")


def on_event(event: DebateEvent) -> None:  # noqa: C901
    if event.event_type == "private_thought":
        agent = event.payload["agent"]
        thought = event.payload["thought"]
        turn = event.payload.get("turn", "?")
        _log(f"\n  💭 [{agent} / T{turn} private] {thought[:200]}")

    elif event.event_type == "public_message":
        agent = event.payload["agent"]
        message = event.payload["message"]
        turn = event.payload.get("turn", "?")
        talking_point = event.payload.get("talking_point", "")
        evidence_score = event.payload.get("evidence_score")
        bar = "=" * 70
        _log(f"\n{bar}")
        _log(f"  {agent.upper()} — Turn {turn}  |  {talking_point}")
        if evidence_score is not None:
            _log(f"  Evidence score: {evidence_score:.2f}")
        _log(bar)
        _log(message)
        _log("")

    elif event.event_type == "arbiter":
        msg = event.payload["message"]
        turn = event.payload.get("turn", "?")
        _log(f"\n  ⚖️  ARBITER [T{turn}]: {msg}")

    elif event.event_type == "branch":
        sub = event.payload.get("sub_topic", "")
        _log(f"\n  🌿 BRANCH: {sub}")

    elif event.event_type == "branch_switch":
        np = event.payload.get("new_talking_point", "")
        _log(f"\n  ↪ PIVOT to: {np}")

    elif event.event_type == "state":
        t = event.payload["transition"]
        _log(f"  [STATE] {t['previous']} → {t['current']}  ({t['reason']})")

    elif event.event_type == "resolution":
        truths = event.payload.get("truths_discovered", [])
        problems = event.payload.get("problems_found", [])
        subs = event.payload.get("sub_topics_explored", [])
        facts = event.payload.get("total_memory_facts", 0)
        _log("\n" + "=" * 70)
        _log("  RESOLUTION SUMMARY")
        _log("=" * 70)
        if truths:
            _log("TRUTHS DISCOVERED:")
            for t in truths:
                _log(f"  ✓ {t}")
        if problems:
            _log("OPEN PROBLEMS:")
            for p in problems:
                _log(f"  ✗ {p}")
        if subs:
            _log(f"SUB-TOPICS EXPLORED ({len(subs)}):")
            for s in subs:
                _log(f"  → {s}")
        _log(f"Total semantic memory facts: {facts}")


async def main() -> None:
    _log(f"=" * 70)
    _log(f"  AGENT DEBATE SYSTEM — LIVE MONITOR")
    _log(f"  Topic:  {TOPIC}")
    _log(f"  Astra:  {LEFT_MODEL}")
    _log(f"  Nova:   {RIGHT_MODEL}")
    _log(f"  Turns:  {TURNS}")
    _log(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"=" * 70)

    orch, _ = build_orchestrator(
        ROOT,
        left_model=LEFT_MODEL,
        right_model=RIGHT_MODEL,
    )
    orch.subscribe(on_event)

    try:
        await orch.run_debate(topic=TOPIC, turns=TURNS)
    except Exception as exc:
        _log(f"\n[ERROR during debate] {exc}")
        import traceback
        _log(traceback.format_exc())
    finally:
        _save_transcript()
        _log(f"\nTranscript saved → {TRANSCRIPT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
