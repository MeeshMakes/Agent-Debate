"""Adaptive, context-sensitive prompt store.

Agents are constantly discovering new sub-topics, truths, problems and
verification needs.  This store captures those learnings and weaves them
into future prompts so the debate *evolves* rather than repeating itself.

Persists to ``memory/adaptive_prompts.json`` on disk so knowledge
accumulates across sessions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_STORE_FILE = Path(__file__).parent / "adaptive_prompts.json"


class AdaptivePromptStore:
    """Topic-keyed store of prompt-enrichment context.

    For every topic seen we accumulate:
    - discovered truths
    - identified problems
    - sub-topics worth exploring
    - claims that need verification
    - evolved framing notes (auto-generated summaries)
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # public API

    def record(
        self,
        topic: str,
        agent: str,
        fact_type: str,          # truth | problem | sub_topic | verify | framing
        content: str,
        turn: int = 0,
    ) -> None:
        """Record a new insight discovered during debate."""
        tkey = self._topic_key(topic)
        bucket = self._data.setdefault(tkey, self._empty_bucket(topic))

        entry = {
            "agent": agent,
            "content": content,
            "turn": turn,
            "ts": int(time.time()),
        }

        if fact_type == "truth":
            _append_dedup(bucket["truths"], entry)
        elif fact_type == "problem":
            _append_dedup(bucket["problems"], entry)
        elif fact_type == "sub_topic":
            _append_dedup(bucket["sub_topics"], entry)
        elif fact_type == "verify":
            _append_dedup(bucket["verify_queue"], entry)
        elif fact_type == "framing":
            _append_dedup(bucket["framing_notes"], entry)

        bucket["last_updated"] = int(time.time())
        self._save()

    def build_context_block(self, topic: str, max_items: int = 6) -> str:
        """Return a rich prompt-injection block for the given topic."""
        tkey = self._topic_key(topic)
        bucket = self._data.get(tkey)
        if not bucket:
            return ""

        lines: list[str] = ["=== WHAT WE HAVE LEARNED SO FAR ==="]

        truths = [e["content"] for e in bucket["truths"][-max_items:]]
        if truths:
            lines.append("\nESTABLISHED TRUTHS (build on these — don't re-argue them):")
            lines.extend(f"  ✓ {t}" for t in truths)

        problems = [e["content"] for e in bucket["problems"][-max_items:]]
        if problems:
            lines.append("\nOPEN PROBLEMS (these still need resolution):")
            lines.extend(f"  ✗ {p}" for p in problems)

        subs = [e["content"] for e in bucket["sub_topics"][-max_items:]]
        if subs:
            lines.append("\nSUB-TOPICS RAISED (consider diving deeper into these):")
            lines.extend(f"  → {s}" for s in subs)

        verify = [e["content"] for e in bucket["verify_queue"][-4:]]
        if verify:
            lines.append("\nCLAIMS AWAITING VERIFICATION (challenge or confirm):")
            lines.extend(f"  ? {v}" for v in verify)

        framing = [e["content"] for e in bucket["framing_notes"][-3:]]
        if framing:
            lines.append("\nEVOLVED FRAMING NOTES:")
            lines.extend(f"  ⟳ {f}" for f in framing)

        lines.append("===========================================")
        return "\n".join(lines)

    def get_sub_topics(self, topic: str) -> list[str]:
        """All sub-topics discovered for this topic."""
        tkey = self._topic_key(topic)
        bucket = self._data.get(tkey, {})
        return [e["content"] for e in bucket.get("sub_topics", [])]

    def get_truths(self, topic: str) -> list[str]:
        tkey = self._topic_key(topic)
        bucket = self._data.get(tkey, {})
        return [e["content"] for e in bucket.get("truths", [])]

    def get_stats(self, topic: str) -> dict[str, int]:
        tkey = self._topic_key(topic)
        bucket = self._data.get(tkey, {})
        return {
            "truths": len(bucket.get("truths", [])),
            "problems": len(bucket.get("problems", [])),
            "sub_topics": len(bucket.get("sub_topics", [])),
            "verify": len(bucket.get("verify_queue", [])),
        }

    def all_topics(self) -> list[str]:
        return [v["topic"] for v in self._data.values()]

    # ------------------------------------------------------------------
    # persistence

    def _load(self) -> None:
        if _STORE_FILE.exists():
            try:
                self._data = json.loads(_STORE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self) -> None:
        try:
            _STORE_FILE.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # helpers

    @staticmethod
    def _topic_key(topic: str) -> str:
        return topic.strip().lower()[:80]

    @staticmethod
    def _empty_bucket(topic: str) -> dict[str, Any]:
        return {
            "topic": topic,
            "truths": [],
            "problems": [],
            "sub_topics": [],
            "verify_queue": [],
            "framing_notes": [],
            "last_updated": int(time.time()),
        }


def _append_dedup(lst: list[dict], entry: dict) -> None:
    """Append unless the same content already exists."""
    content = entry["content"].strip().lower()
    for existing in lst:
        if existing["content"].strip().lower() == content:
            return
    lst.append(entry)


# Module-level singleton so all agents share one store per process
_store: AdaptivePromptStore | None = None


def get_adaptive_store() -> AdaptivePromptStore:
    global _store
    if _store is None:
        _store = AdaptivePromptStore()
    return _store
