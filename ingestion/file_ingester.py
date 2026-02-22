"""File ingestion engine.

Reads text files (txt, md, json, csv, py, html, pdf-text extraction),
chunks them, assigns semantic tags, and produces a list of tagged fact
dicts ready for use as a session dataset.

Supported file types (plain text):
    .txt  .md  .py  .json  .yaml  .yml  .csv  .html  .htm  .rst  .log

Supported rich types (content-aware):
    .pdf  — extracts text via pdfminer if installed, falls back to raw bytes
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 600       # characters per chunk
_CHUNK_OVERLAP = 80    # overlap between chunks

# Directories that should never be ingested (matched against any path component)
_EXCLUDED_DIRS = frozenset({
    "__pycache__", ".venv", "venv", ".git", "node_modules", ".pytest_cache",
    ".mypy_cache", ".tox", ".eggs", "dist", "build", ".idea", ".vscode",
    "env", ".env", ".story-reader", ".ai", ".morph", ".github",
    ".ruff_cache", "__pypackages__", "site-packages", ".cargo",
})

# Binary / non-text extensions to skip even if they appear in a valid folder
_EXCLUDED_EXTS = frozenset({
    ".exe", ".dll", ".so", ".dylib", ".pyd", ".pyc", ".pyo",
    ".whl", ".egg", ".tar", ".gz", ".bz2", ".zip", ".7z", ".rar",
    ".ico", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".mov", ".flac",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".class", ".o", ".obj", ".lib", ".a",
    ".lock", ".map",
})


def _is_excluded_path(p: Path) -> bool:
    """Return True if any path component is in the excluded dirs set."""
    for part in p.parts:
        if part.lower() in _EXCLUDED_DIRS:
            return True
    return False


def ingest_files(file_paths: list[Path]) -> list[dict]:
    """Read a list of files and return a flat list of tagged fact dicts."""
    all_facts: list[dict] = []
    for p in file_paths:
        if not p.exists() or not p.is_file():
            continue
        # Skip junk files
        if p.suffix.lower() in _EXCLUDED_EXTS:
            continue
        if _is_excluded_path(p):
            continue
        try:
            chunks = _read_and_chunk(p)
        except Exception as exc:
            chunks = [f"[Could not read {p.name}: {exc}]"]

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < 40:
                continue
            tags = _auto_tag(chunk)
            all_facts.append({
                "source_file": p.name,
                "chunk_index": i,
                "text": chunk[:600],
                "tags": tags,
                "fact_type": _infer_fact_type(chunk, tags),
                "keywords": list(_extract_keywords(chunk)),
                "ts": int(time.time()),
            })
    return all_facts


def _read_and_chunk(path: Path) -> list[str]:
    """Read a file and split into overlapping chunks."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".json":
        text = _flatten_json(path)
    elif ext in (".html", ".htm"):
        text = _strip_html(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    # Chunk by paragraphs first, then hard-cut
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) < _CHUNK_SIZE:
            buffer += " " + para
        else:
            if buffer:
                chunks.append(buffer.strip())
                # overlap: keep last _CHUNK_OVERLAP chars as start of next
                buffer = buffer[-_CHUNK_OVERLAP:] + " " + para
            else:
                # Para itself is huge — hard split
                for start in range(0, len(para), _CHUNK_SIZE - _CHUNK_OVERLAP):
                    chunks.append(para[start:start + _CHUNK_SIZE])
                buffer = ""
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks or [text[:_CHUNK_SIZE]]


def _read_pdf(path: Path) -> str:
    """Try pdfminer, fall back to raw read."""
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(str(path))
    except ImportError:
        pass
    try:
        raw = path.read_bytes()
        return raw.decode("latin-1", errors="replace")
    except Exception:
        return ""


def _flatten_json(path: Path) -> str:
    """Flatten a JSON file into human-readable text."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return _json_to_text(data)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def _json_to_text(obj: Any, depth: int = 0) -> str:
    if depth > 5:
        return str(obj)[:200]
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {_json_to_text(v, depth + 1)}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_json_to_text(item, depth + 1) for item in obj)
    return str(obj)


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_TAG_PATTERNS: list[tuple[str, list[str]]] = [
    ("religion", ["bible", "quran", "torah", "god", "allah", "jesus", "prophet", "faith",
                  "scripture", "theology", "prayer", "church", "mosque", "temple"]),
    ("science", ["evidence", "experiment", "hypothesis", "data", "research", "study",
                 "quantum", "physics", "biology", "chemistry", "evolution", "scientific"]),
    ("philosophy", ["truth", "ethics", "moral", "reason", "logic", "argument", "proof",
                    "knowledge", "consciousness", "existence", "reality", "epistemology"]),
    ("history", ["historical", "ancient", "century", "empire", "civilization", "war",
                 "period", "era", "dynasty", "archaeology"]),
    ("technology", ["software", "algorithm", "machine", "neural", "model", "training",
                    "code", "program", "system", "compute", "inference", "language"]),
    ("political", ["government", "policy", "democracy", "freedom", "rights", "law",
                   "constitution", "power", "society", "nation", "state"]),
]


def _auto_tag(text: str) -> list[str]:
    lower = text.lower()
    return [tag for tag, words in _TAG_PATTERNS if any(w in lower for w in words)]


def _infer_fact_type(text: str, tags: list[str]) -> str:
    lower = text.lower()
    if any(w in lower for w in ["truth", "proven", "verified", "confirmed", "established"]):
        return "truth"
    if any(w in lower for w in ["problem", "issue", "flaw", "fallacy", "contradiction"]):
        return "problem"
    if any(w in lower for w in ["should", "must", "question", "verify", "check"]):
        return "verify"
    return "claim"


_STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "its", "its", "our", "your",
    "they", "them", "you", "one", "two", "can", "will", "would", "could",
    "also", "about", "into", "than", "then", "just", "only", "very", "each",
    "all", "any", "both", "few", "other", "over", "like", "does", "did",
})


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    return {t for t in tokens if t not in _STOP}
