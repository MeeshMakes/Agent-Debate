"""Background ingestion worker (QThread).

Pipeline:
  1. Copies source files/folders to uploads/ inside the session.
  2. Runs file_ingester to chunk+tag everything into fact dicts.
  3. Semantically weights every fact chunk via TF-IDF (sklearn).
  4. Auto-generates a short dataset name from the top TF-IDF keywords.
  5. Saves dataset to TWO places:
       a. <session_dir>/ingested_dataset.json          (local, per-session)
       b. sessions/_datasets/<auto_name>.json          (global persistent store)
  6. Removes the copied uploads (originals on user disk are never touched).

The global _datasets/ store is what the rewrite tool reads when building
context for improving a talking point.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from ingestion.file_ingester import ingest_files
from core.session_manager import SessionManager, _write_json


# ---------------------------------------------------------------------------
# TF-IDF semantic weighting
# ---------------------------------------------------------------------------

def _apply_tfidf_weights(facts: list[dict]) -> tuple[list[dict], list[str]]:
    """Run TF-IDF across all fact texts; attach a tfidf_weight float to each.
    Returns (enriched_facts, top_keywords) where top_keywords are the 10
    highest-scoring terms globally (used for auto-naming).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        corpus = [f.get("text", "") for f in facts]
        if not corpus:
            return facts, []

        vec = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        X = vec.fit_transform(corpus)
        # Row-wise max TF-IDF score as the "weight" for that chunk
        row_maxes = np.asarray(X.max(axis=1).todense()).flatten()

        for i, fact in enumerate(facts):
            fact["tfidf_weight"] = float(round(row_maxes[i], 4))

        # Top global terms (column-wise mean, descending)
        feature_names = vec.get_feature_names_out()
        col_means = np.asarray(X.mean(axis=0)).flatten()
        top_indices = col_means.argsort()[::-1][:10]
        top_keywords = [str(feature_names[idx]) for idx in top_indices]

        return facts, top_keywords

    except Exception:
        # sklearn unavailable or other error — return facts unweighted
        top_kw = _fallback_keywords(facts)
        return facts, top_kw


def _fallback_keywords(facts: list[dict]) -> list[str]:
    """Simple word-frequency fallback if sklearn is absent."""
    from collections import Counter
    _STOP = {"the","a","an","is","are","was","were","in","of","to","and",
              "or","for","with","on","at","by","from","that","this","it",
              "be","as","have","has","had","not","but","so","if","its"}
    words: list[str] = []
    for f in facts:
        words.extend(
            w.lower() for w in re.findall(r"[a-z]{4,}", f.get("text","").lower())
            if w not in _STOP
        )
    counter = Counter(words)
    return [w for w, _ in counter.most_common(10)]


# ---------------------------------------------------------------------------
# Auto dataset name generation
# ---------------------------------------------------------------------------

def _make_dataset_name(top_keywords: list[str], source_names: list[str]) -> str:
    """Produce a short slug name like: ai_consciousness_ethics_2026"""
    slug_parts: list[str] = []
    for kw in top_keywords[:4]:
        part = re.sub(r"[^a-z0-9]+", "_", kw.lower()).strip("_")
        if part and len(part) > 2:
            slug_parts.append(part)
    if not slug_parts and source_names:
        # Fall back to source file names
        slug_parts = [re.sub(r"[^a-z0-9]+", "_", Path(n).stem.lower())[:20]
                      for n in source_names[:3]]
    stamp = time.strftime("%Y%m%d")
    name = ("_".join(slug_parts[:4]) or "dataset") + f"_{stamp}"
    return name[:80]


# ---------------------------------------------------------------------------
# Global dataset metadata helper
# ---------------------------------------------------------------------------

def get_datasets_dir(session_manager_root: Path) -> Path:
    """Return (creating if needed) the global datasets directory."""
    d = session_manager_root / "_datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_global_datasets(session_manager_root: Path) -> list[dict]:
    """Return metadata for every saved dataset (name, fact_count, ts, path)."""
    import json
    store = get_datasets_dir(session_manager_root)
    results = []
    for jf in sorted(store.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            meta = data.get("_meta", {})
            results.append({
                "name":       jf.stem,
                "path":       str(jf),
                "fact_count": meta.get("fact_count", len(data.get("facts", []))),
                "keywords":   meta.get("top_keywords", []),
                "created":    meta.get("created", ""),
            })
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# IngestionWorker
# ---------------------------------------------------------------------------

class IngestionWorker(QThread):
    progress    = pyqtSignal(str)    # status message
    finished    = pyqtSignal(int)    # number of facts ingested
    failed      = pyqtSignal(str)    # error message
    dataset_saved = pyqtSignal(str)  # emits the auto-generated dataset name

    def __init__(
        self,
        source_paths: list[Path],
        session_manager: SessionManager,
        session_id: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sources = source_paths
        self._sm = session_manager
        self._session_id = session_id

    def run(self) -> None:
        try:
            uploads_dir = self._sm.get_uploads_dir(self._session_id)

            # ── Step 1: copy source files to uploads/ ───────────────────────
            copied: list[Path] = []
            for src in self._sources:
                if not src.exists():
                    continue
                if src.is_file():
                    dest = uploads_dir / src.name
                    shutil.copy2(src, dest)
                    copied.append(dest)
                    self.progress.emit(f"Copied: {src.name}")
                elif src.is_dir():
                    for child in src.rglob("*"):
                        if child.is_file() and child.suffix.lower() in _SUPPORTED:
                            rel = child.relative_to(src.parent)
                            dest = (
                                uploads_dir
                                / str(rel).replace("\\", "_").replace("/", "_")
                            )
                            shutil.copy2(child, dest)
                            copied.append(dest)
                    self.progress.emit(f"Copied folder: {src.name}")

            if not copied:
                self.failed.emit("No supported files found to ingest.")
                return

            # ── Step 2: ingest (chunk + tag) ────────────────────────────────
            self.progress.emit(f"Ingesting {len(copied)} file(s)…")
            facts = ingest_files(copied)
            if not facts:
                self.failed.emit("Ingestion produced no facts from the provided files.")
                return

            # ── Step 3: TF-IDF semantic weighting ───────────────────────────
            self.progress.emit(f"Applying semantic weighting to {len(facts)} chunks…")
            facts, top_keywords = _apply_tfidf_weights(facts)

            # ── Step 4: auto-generate dataset name ──────────────────────────
            source_names = [p.name for p in self._sources]
            dataset_name = _make_dataset_name(top_keywords, source_names)

            # ── Step 5a: save per-session dataset ───────────────────────────
            session_out = self._sm.root / self._session_id / "ingested_dataset.json"
            _write_json(session_out, facts)
            self.progress.emit(f"Session dataset saved ({len(facts)} facts)")

            # ── Step 5b: save global persistent dataset ─────────────────────
            global_store = get_datasets_dir(self._sm.root)
            global_out = global_store / f"{dataset_name}.json"
            # If a file with that name already exists, append a counter
            counter = 1
            while global_out.exists():
                global_out = global_store / f"{dataset_name}_{counter}.json"
                counter += 1
            dataset_package = {
                "_meta": {
                    "name":         dataset_name,
                    "fact_count":   len(facts),
                    "top_keywords": top_keywords,
                    "source_files": source_names,
                    "created":      time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id":   self._session_id,
                },
                "facts": facts,
            }
            _write_json(global_out, dataset_package)
            self.progress.emit(
                f"Global dataset \"{dataset_name}\" saved to _datasets/"
            )
            self.dataset_saved.emit(dataset_name)

            # ── Step 6: clean up upload copies ──────────────────────────────
            for p in copied:
                p.unlink(missing_ok=True)
            self.progress.emit("Upload copies cleaned up")

            self.finished.emit(len(facts))

        except Exception as exc:
            self.failed.emit(f"Ingestion error: {exc}")


_SUPPORTED = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml",
    ".csv", ".html", ".htm", ".rst", ".log", ".pdf",
    ".js", ".ts", ".toml", ".ini", ".cfg", ".xml",
}
