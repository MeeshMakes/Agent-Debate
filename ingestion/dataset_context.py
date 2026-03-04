"""Dataset context provider — makes ingested datasets available to agents.

Loads a dataset of fact-chunks (with TF-IDF weights and keywords) and
provides methods to retrieve the most relevant chunks for a given query.
Agents treat this context like a LORA — they read the ingested codebase
data and reference it when thinking and formulating responses.

Usage:
    provider = DatasetContextProvider()
    provider.load_from_session(session_dir)      # loads ingested_dataset.json
    context = provider.get_context(query, top_k=12)   # returns formatted string
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class DatasetContextProvider:
    """Manages access to ingested dataset facts for agent context injection."""

    def __init__(self) -> None:
        self._facts: list[dict] = []
        self._dataset_name: str = ""
        self._loaded = False
        # Track which chunks each agent has already seen to rotate knowledge
        self._seen_indices: dict[str, set[int]] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded and bool(self._facts)

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def load_from_session(self, session_dir: Path) -> bool:
        """Load ingested_dataset.json from a session directory.
        Returns True if facts were loaded successfully.
        """
        ds_path = session_dir / "ingested_dataset.json"
        if not ds_path.exists():
            return False
        try:
            data = json.loads(ds_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._facts = data
            elif isinstance(data, dict):
                self._facts = data.get("facts", [])
                meta = data.get("_meta", {})
                self._dataset_name = meta.get("name", "")
            self._loaded = bool(self._facts)
            return self._loaded
        except Exception:
            return False

    def load_facts(self, facts: list[dict], name: str = "") -> None:
        """Load facts directly (e.g. from an IngestionWorker)."""
        self._facts = facts
        self._dataset_name = name
        self._loaded = bool(facts)

    def get_context(
        self,
        query: str,
        agent_name: str = "",
        top_k: int = 12,
        max_chars: int = 4000,
        prefer_unseen: bool = True,
        max_chunks_per_file: int = 2,
    ) -> str:
        """Return the most relevant dataset chunks as a formatted context block.

        Scoring: keyword overlap × TF-IDF weight, with a bonus for unseen chunks.
        If prefer_unseen is True, chunks the agent hasn't seen yet get 2× boost.
        """
        if not self._facts:
            return ""

        query_keywords = _extract_query_keywords(query)
        explicit_targets = _extract_query_file_targets(query)
        if not query_keywords and not explicit_targets:
            # Fallback: return highest-weighted chunks
            scored = [(i, f.get("tfidf_weight", 0.5)) for i, f in enumerate(self._facts)]
        else:
            scored = []
            for i, fact in enumerate(self._facts):
                fact_keywords = set(fact.get("keywords", []))
                overlap = len(query_keywords & fact_keywords)
                weight = fact.get("tfidf_weight", 0.5)
                source_path = str(fact.get("source_path", "")).strip()
                source_file = str(fact.get("source_file", "")).strip()
                score = (overlap + 0.1) * weight
                score += _explicit_target_score(explicit_targets, source_path, source_file)

                # Bonus for unseen chunks
                if prefer_unseen and agent_name:
                    seen = self._seen_indices.get(agent_name, set())
                    if i not in seen:
                        score *= 2.0

                scored.append((i, score))

        # Sort by score descending, then select with per-file diversity limits
        scored.sort(key=lambda x: x[1], reverse=True)
        selected: list[tuple[int, float]] = []
        selected_idx: set[int] = set()
        per_file_counts: dict[str, int] = {}

        # Explicit file references in the query should always be represented.
        if explicit_targets:
            forced = self._collect_forced_target_matches(
                explicit_targets=explicit_targets,
                prefer_unseen=prefer_unseen,
                agent_name=agent_name,
                max_chunks_per_file=max_chunks_per_file,
            )
            for idx, score in forced:
                fact = self._facts[idx]
                source_path = str(fact.get("source_path", "")).strip() or str(fact.get("source_file", "unknown")).strip()
                used = per_file_counts.get(source_path, 0)
                if used >= max_chunks_per_file:
                    continue
                selected.append((idx, score))
                selected_idx.add(idx)
                per_file_counts[source_path] = used + 1
                if len(selected) >= top_k:
                    break

        for idx, score in scored:
            if idx in selected_idx:
                continue
            fact = self._facts[idx]
            source_path = str(fact.get("source_path", "")).strip() or str(fact.get("source_file", "unknown")).strip()
            used = per_file_counts.get(source_path, 0)
            if used >= max_chunks_per_file:
                continue
            selected.append((idx, score))
            selected_idx.add(idx)
            per_file_counts[source_path] = used + 1
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            chosen = {idx for idx, _ in selected}
            for idx, score in scored:
                if idx in chosen:
                    continue
                selected.append((idx, score))
                if len(selected) >= top_k:
                    break

        # Build the formatted block
        lines: list[str] = []
        total_chars = 0
        for idx, _score in selected:
            fact = self._facts[idx]
            text = fact.get("text", "").strip()
            source = fact.get("source_file", "unknown")
            source_path = fact.get("source_path", source)
            chunk_i = fact.get("chunk_index", 0)

            line = f"[{source_path}#{chunk_i}] {text}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

            # Track as seen
            if agent_name:
                self._seen_indices.setdefault(agent_name, set()).add(idx)

        if not lines:
            return ""

        header = f"INGESTED CODEBASE KNOWLEDGE ({len(lines)} chunks from {self._dataset_name or 'uploaded files'})"
        return f"{header}:\n" + "\n".join(lines)

    def _collect_forced_target_matches(
        self,
        *,
        explicit_targets: list[str],
        prefer_unseen: bool,
        agent_name: str,
        max_chunks_per_file: int,
    ) -> list[tuple[int, float]]:
        forced: list[tuple[int, float]] = []
        seen_idx: set[int] = set()
        for target in explicit_targets:
            matches: list[tuple[int, float]] = []
            for i, fact in enumerate(self._facts):
                source_path = str(fact.get("source_path", "")).strip()
                source_file = str(fact.get("source_file", "")).strip()
                match_score = _single_target_match_score(target, source_path, source_file)
                if match_score <= 0:
                    continue
                score = fact.get("tfidf_weight", 0.5) + match_score
                if prefer_unseen and agent_name:
                    seen = self._seen_indices.get(agent_name, set())
                    if i not in seen:
                        score *= 2.0
                matches.append((i, score))

            matches.sort(key=lambda item: item[1], reverse=True)
            used_for_target = 0
            for idx, score in matches:
                if idx in seen_idx:
                    continue
                forced.append((idx, score + 100.0))
                seen_idx.add(idx)
                used_for_target += 1
                if used_for_target >= max_chunks_per_file:
                    break
        return forced

    def get_summary(self) -> str:
        """Return a brief summary of what's in the dataset (for think phase)."""
        if not self._facts:
            return ""
        # Collect unique source files
        sources = sorted(set(f.get("source_file", "") for f in self._facts))
        # Get top tags
        from collections import Counter
        tag_counter: Counter[str] = Counter()
        for f in self._facts:
            tag_counter.update(f.get("tags", []))
        top_tags = [t for t, _ in tag_counter.most_common(5)]

        summary = (
            f"Dataset: {self._dataset_name or 'uploaded'} "
            f"| {len(self._facts)} chunks from {len(sources)} files "
            f"| Tags: {', '.join(top_tags) if top_tags else 'none'}\n"
            f"Files: {', '.join(sources[:15])}"
        )
        if len(sources) > 15:
            summary += f" (+{len(sources) - 15} more)"
        return summary


def _extract_query_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a query string."""
    _STOP = frozenset({
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "been", "have", "has", "had", "not", "but", "its", "our", "your",
        "they", "them", "you", "one", "two", "can", "will", "would", "could",
        "also", "about", "into", "than", "then", "just", "only", "very", "each",
        "all", "any", "both", "few", "other", "over", "like", "does", "did",
        "should", "must", "what", "which", "where", "when", "how", "who",
        "being", "more", "some", "such", "these", "those", "their", "there",
    })
    tokens = re.findall(r"[A-Za-z_]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP}


def _extract_query_file_targets(text: str) -> list[str]:
    """Extract file-like path mentions from user/agent queries."""
    targets: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        norm = _normalize_path(raw)
        if not norm or norm in seen:
            return
        if norm.startswith("http://") or norm.startswith("https://"):
            return
        suffix = Path(norm).suffix.lower()
        if not suffix or len(suffix) > 11:
            return
        seen.add(norm)
        targets.append(norm)

    for raw in re.findall(r"(?:[A-Za-z]:)?[A-Za-z0-9_.\\/-]+[\\/][A-Za-z0-9_.\\/-]+", text):
        add(raw)
    for raw in re.findall(r"\b[A-Za-z0-9_.-]+\.[A-Za-z]{1,10}\b", text):
        add(raw)

    return targets


def _normalize_path(value: str) -> str:
    cleaned = value.strip().strip("`\"'[](){}<>")
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lower()
    return cleaned


def _single_target_match_score(target: str, source_path: str, source_file: str) -> float:
    target_norm = _normalize_path(target)
    source_norm = _normalize_path(source_path or source_file)
    source_file_norm = _normalize_path(source_file)
    target_name = Path(target_norm).name

    if not target_norm or not source_norm:
        return 0.0
    if source_norm == target_norm:
        return 10.0
    if source_norm.endswith(f"/{target_norm}"):
        return 8.0
    if source_file_norm and (source_file_norm == target_norm or source_file_norm == target_name):
        return 7.0
    if target_name and source_norm.endswith(f"/{target_name}"):
        return 6.5
    if target_name and Path(source_file_norm).stem == Path(target_name).stem:
        return 2.0
    return 0.0


def _explicit_target_score(explicit_targets: list[str], source_path: str, source_file: str) -> float:
    if not explicit_targets:
        return 0.0
    return max((_single_target_match_score(t, source_path, source_file) for t in explicit_targets), default=0.0)
