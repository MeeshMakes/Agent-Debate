from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvidenceSnippet:
    title: str
    snippet: str
    source: str
    source_display: str
    score: float = 0.0
    # Always "past-debate" for this retriever.  Agents must never treat these
    # as authoritative about the *current* repo being analysed.
    corpus_tag: str = "past-debate"


class CorpusEvidenceRetriever:
    def __init__(self, corpus_root: Path, max_items: int = 800) -> None:
        self.corpus_root = corpus_root
        self.max_items = max_items
        self._items: list[EvidenceSnippet] = []
        self._loaded = False

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z]{3,}", text.lower())}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.corpus_root.exists():
            return

        count = 0
        for container in sorted(self.corpus_root.glob("container_*")):
            dataset_file = container / "content" / "conversation.dataset.json"
            if not dataset_file.exists():
                continue
            try:
                data = json.loads(dataset_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            title = str(data.get("title", container.name))
            summary = str(data.get("summary", "")).strip()
            if not summary:
                continue

            self._items.append(
                EvidenceSnippet(
                    title=title,
                    snippet=summary[:1200],
                    source=str(dataset_file),
                    source_display=str(dataset_file.relative_to(self.corpus_root.parent)),
                )
            )
            count += 1
            if count >= self.max_items:
                break

    def search(self, query: str, top_k: int = 3) -> list[EvidenceSnippet]:
        self._load()
        if not self._items:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return self._items[:top_k]

        scored: list[tuple[float, EvidenceSnippet]] = []
        for item in self._items:
            item_terms = self._tokenize(item.title + " " + item.snippet)
            overlap = query_terms & item_terms
            if not overlap:
                continue
            score = (len(overlap) / max(len(query_terms), 1)) + (len(overlap) / max(len(item_terms), 1))
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[EvidenceSnippet] = []
        for score, item in scored[:top_k]:
            results.append(
                EvidenceSnippet(
                    title=item.title,
                    snippet=item.snippet,
                    source=item.source,
                    source_display=item.source_display,
                    score=round(score, 3),
                    corpus_tag="past-debate",
                )
            )
        return results
