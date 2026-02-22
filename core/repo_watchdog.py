from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re

_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "out",
}

_SKIP_FILE_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
    ".7z",
    ".rar",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".mp4",
    ".mp3",
    ".wav",
    ".pdf",
    ".pyc",
}

_INCLUDE_FILE_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
}

_MAX_FILE_BYTES = 180_000
_MAX_SNIPPET_CHARS = 5_000


@dataclass
class RepoFileSummary:
    relative_path: str
    size: int
    line_count: int
    char_count: int
    mode: str
    semantic_hash: str
    tags: list[str]
    excerpt: str


@dataclass
class RepoSnapshot:
    repo_path: str
    generated_at: str
    included_files: int
    skipped_files: int
    skipped_reasons: dict[str, int]
    extension_counts: dict[str, int]
    top_large_files: list[dict]
    summaries: list[RepoFileSummary]


@dataclass
class RepoChangeSet:
    added: list[str]
    modified: list[str]
    deleted: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)


@dataclass
class RepoUpdateReport:
    updated_files: int
    deleted_files: int
    sampled_files: int
    full_files: int
    unreadable_files: int
    touched_paths: list[str]


class RepoWatchdog:
    def __init__(self, session_root: Path) -> None:
        self._state_root = session_root / "_repo_watchdog"
        self._state_root.mkdir(parents=True, exist_ok=True)

    def build_snapshot(self, repo_path: str) -> RepoSnapshot:
        repo = Path(repo_path).resolve()
        if not repo.exists() or not repo.is_dir():
            raise ValueError(f"Repository path does not exist: {repo}")

        summaries: list[RepoFileSummary] = []
        skipped_reasons: dict[str, int] = {}
        extension_counts: dict[str, int] = {}
        top_large_files: list[dict] = []
        included = 0
        skipped = 0

        for file_path in self._iter_repo_files(repo):
            rel = file_path.relative_to(repo).as_posix()
            reason = self._skip_reason(rel, file_path)
            if reason is not None:
                skipped += 1
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                skipped += 1
                skipped_reasons["stat_error"] = skipped_reasons.get("stat_error", 0) + 1
                continue

            ext = file_path.suffix.lower() or "<none>"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            if size > 80_000:
                top_large_files.append({"path": rel, "size": size})

            included += 1
            summaries.append(self._summarize_file(repo, file_path, rel, size))

        top_large_files.sort(key=lambda item: item["size"], reverse=True)
        snapshot = RepoSnapshot(
            repo_path=str(repo),
            generated_at=datetime.now().isoformat(timespec="seconds"),
            included_files=included,
            skipped_files=skipped,
            skipped_reasons=skipped_reasons,
            extension_counts=extension_counts,
            top_large_files=top_large_files[:20],
            summaries=summaries,
        )

        profile_dir = self._profile_dir(repo)
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "snapshot.json").write_text(
            json.dumps(
                {
                    **asdict(snapshot),
                    "summaries": [asdict(item) for item in snapshot.summaries],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_docs_index(profile_dir, repo, summaries)
        self.build_semantic_dataset(str(repo))
        self._write_state(repo, profile_dir)
        return snapshot

    def estimate_repo_size_bytes(self, repo_path: str) -> int:
        repo = Path(repo_path).resolve()
        if not repo.exists() or not repo.is_dir():
            return 0
        total = 0
        for file_path in self._iter_repo_files(repo):
            rel = file_path.relative_to(repo).as_posix()
            if self._skip_reason(rel, file_path) is not None:
                continue
            try:
                total += int(file_path.stat().st_size)
            except OSError:
                continue
        return total

    def build_semantic_dataset(self, repo_path: str) -> dict:
        repo = Path(repo_path).resolve()
        profile_dir = self._profile_dir(repo)
        docs = self._read_docs_index(profile_dir)
        files_map: dict = docs.get("files", {}) if isinstance(docs, dict) else {}

        facts: list[dict] = []
        source_counts: dict[str, int] = {}
        for rel, data in files_map.items():
            excerpt = str(data.get("excerpt", ""))
            if not excerpt.strip():
                continue

            chunks = self._chunk_text(excerpt, max_chars=1400, overlap=200)
            tags = data.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            ext = Path(rel).suffix.lower() or "<none>"
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.strip()
                if len(chunk_text) < 40:
                    continue
                semantic_hash = hashlib.sha1(f"{rel}:{i}:{chunk_text[:300]}".encode("utf-8", errors="ignore")).hexdigest()
                dynamic_tags = self._derive_tags(rel, chunk_text)
                all_tags = sorted(set([*tags, *dynamic_tags]))
                keywords = self._extract_keywords(chunk_text)
                weight = self._semantic_weight(rel, chunk_text, all_tags)

                facts.append(
                    {
                        "source_file": Path(rel).name,
                        "source_path": rel,
                        "chunk_index": i,
                        "text": chunk_text,
                        "tags": all_tags,
                        "fact_type": "claim",
                        "keywords": keywords,
                        "semantic_hash": semantic_hash,
                        "char_count": len(chunk_text),
                        "tfidf_weight": round(weight, 4),
                        "path_depth": rel.count("/"),
                        "ext": ext,
                    }
                )
            source_counts[ext] = source_counts.get(ext, 0) + 1

        dataset = {
            "_meta": {
                "name": f"repo_watchdog_{repo.name}",
                "repo_path": str(repo),
                "created": datetime.now().isoformat(timespec="seconds"),
                "fact_count": len(facts),
                "file_count": len(files_map),
                "ext_counts": source_counts,
            },
            "facts": facts,
        }
        (profile_dir / "semantic_dataset.json").write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return dataset

    def load_semantic_dataset(self, repo_path: str) -> dict | None:
        repo = Path(repo_path).resolve()
        profile_dir = self._profile_dir(repo)
        ds_file = profile_dir / "semantic_dataset.json"
        if not ds_file.exists():
            return None
        try:
            data = json.loads(ds_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def update_docs_for_changes(
        self,
        repo_path: str,
        changes: RepoChangeSet,
        *,
        max_touched_files: int = 80,
    ) -> RepoUpdateReport:
        repo = Path(repo_path).resolve()
        profile_dir = self._profile_dir(repo)
        profile_dir.mkdir(parents=True, exist_ok=True)

        docs = self._read_docs_index(profile_dir)
        files_map = docs.get("files", {})

        touched_raw = changes.added + changes.modified + changes.deleted
        touched = sorted(set(touched_raw))[:max_touched_files]

        updated = 0
        deleted = 0
        sampled = 0
        full = 0
        unreadable = 0

        for rel in touched:
            if rel in changes.deleted:
                if rel in files_map:
                    files_map.pop(rel, None)
                    deleted += 1
                continue

            file_path = repo / rel
            if not file_path.exists() or not file_path.is_file():
                continue

            reason = self._skip_reason(rel, file_path)
            if reason is not None:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            summary = self._summarize_file(repo, file_path, rel, size)
            files_map[rel] = asdict(summary)
            updated += 1
            if summary.mode == "sampled":
                sampled += 1
            elif summary.mode == "full":
                full += 1
            else:
                unreadable += 1

        docs["repo_path"] = str(repo)
        docs["updated_at"] = datetime.now().isoformat(timespec="seconds")
        docs["files"] = files_map
        (profile_dir / "docs_index.json").write_text(
            json.dumps(docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.build_semantic_dataset(str(repo))

        return RepoUpdateReport(
            updated_files=updated,
            deleted_files=deleted,
            sampled_files=sampled,
            full_files=full,
            unreadable_files=unreadable,
            touched_paths=touched,
        )

    def build_change_brief(self, changes: RepoChangeSet, report: RepoUpdateReport) -> str:
        parts: list[str] = []
        if changes.added:
            parts.append(f"added {len(changes.added)}")
        if changes.modified:
            parts.append(f"modified {len(changes.modified)}")
        if changes.deleted:
            parts.append(f"deleted {len(changes.deleted)}")
        delta = ", ".join(parts) if parts else "no changes"

        touched_preview = ", ".join(report.touched_paths[:5])
        if len(report.touched_paths) > 5:
            touched_preview += f" (+{len(report.touched_paths) - 5} more)"

        return (
            f"Repo Watchdog update: {delta}. "
            f"Docs refreshed for {report.updated_files} file(s), removed {report.deleted_files}, "
            f"sampled={report.sampled_files}, full={report.full_files}. "
            f"Touched: {touched_preview or '(none)'}"
        )

    def check_for_changes(self, repo_path: str) -> RepoChangeSet:
        repo = Path(repo_path).resolve()
        profile_dir = self._profile_dir(repo)
        state_file = profile_dir / "state.json"
        if not state_file.exists():
            self._write_state(repo, profile_dir)
            return RepoChangeSet([], [], [])

        old = json.loads(state_file.read_text(encoding="utf-8"))
        new = self._collect_state(repo)
        old_files = old.get("files", {})
        new_files = new.get("files", {})

        old_set = set(old_files.keys())
        new_set = set(new_files.keys())

        added = sorted(new_set - old_set)
        deleted = sorted(old_set - new_set)

        modified: list[str] = []
        for path in sorted(old_set & new_set):
            if old_files[path] != new_files[path]:
                modified.append(path)

        state_file.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        return RepoChangeSet(added=added, modified=modified, deleted=deleted)

    def build_context_brief(self, snapshot: RepoSnapshot) -> str:
        ext_sorted = sorted(snapshot.extension_counts.items(), key=lambda kv: kv[1], reverse=True)
        ext_lines = [f"  • {ext}: {count} file(s)" for ext, count in ext_sorted[:10]]

        top_dirs = self._top_directories(snapshot.summaries)
        dir_lines = [f"  • {d}: {c} file(s)" for d, c in top_dirs[:10]]

        prioritized = self._prioritized_paths(snapshot.summaries)
        focus_lines = [f"  • {p}" for p in prioritized[:10]]

        return (
            "REPO WATCHDOG SNAPSHOT\n"
            f"Repository: {snapshot.repo_path}\n"
            f"Generated: {snapshot.generated_at}\n"
            f"Included files: {snapshot.included_files}\n"
            f"Skipped files: {snapshot.skipped_files}\n\n"
            "Dominant file types:\n"
            + ("\n".join(ext_lines) if ext_lines else "  • (none)")
            + "\n\n"
            "Repository structure (top folders):\n"
            + ("\n".join(dir_lines) if dir_lines else "  • (none)")
            + "\n\n"
            "Representative system files (code-first, low-noise):\n"
            + ("\n".join(focus_lines) if focus_lines else "  • (none)")
            + "\n\n"
            "Use this snapshot as entry context. Prioritize executable scripts and their cross-file interactions first, then deep-dive into specific scripts and loops where failures or inconsistencies appear."
        )

    def _profile_dir(self, repo: Path) -> Path:
        slug = hashlib.sha1(str(repo).encode("utf-8")).hexdigest()[:12]
        return self._state_root / slug

    def _read_docs_index(self, profile_dir: Path) -> dict:
        docs_file = profile_dir / "docs_index.json"
        if not docs_file.exists():
            return {"repo_path": "", "updated_at": "", "files": {}}
        try:
            data = json.loads(docs_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"repo_path": "", "updated_at": "", "files": {}}

    def _write_docs_index(self, profile_dir: Path, repo: Path, summaries: list[RepoFileSummary]) -> None:
        payload = {
            "repo_path": str(repo),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "files": {s.relative_path: asdict(s) for s in summaries},
        }
        (profile_dir / "docs_index.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _iter_repo_files(self, repo: Path):
        for file_path in repo.rglob("*"):
            if file_path.is_dir():
                continue
            yield file_path

    def _skip_reason(self, rel: str, file_path: Path) -> str | None:
        rel_lower = rel.lower()
        parts = set(rel_lower.split("/"))

        if parts & _SKIP_DIRS:
            return "skip_dir"

        suffix = file_path.suffix.lower()
        if suffix in _SKIP_FILE_SUFFIXES:
            return "skip_binary"

        if suffix and suffix not in _INCLUDE_FILE_SUFFIXES:
            return "unsupported_suffix"

        return None

    def _summarize_file(self, repo: Path, file_path: Path, rel: str, size: int) -> RepoFileSummary:
        text = ""
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

        if not text:
            return RepoFileSummary(
                relative_path=rel,
                size=size,
                line_count=0,
                char_count=0,
                mode="unreadable",
                semantic_hash="",
                tags=[],
                excerpt="",
            )

        lines = text.splitlines()
        line_count = len(lines)
        char_count = len(text)
        semantic_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        tags = self._derive_tags(rel, text)

        if size <= _MAX_FILE_BYTES and len(text) <= 4 * _MAX_SNIPPET_CHARS:
            excerpt = text[: _MAX_SNIPPET_CHARS * 2]
            mode = "full"
        else:
            head = "\n".join(lines[:90])
            tail = "\n".join(lines[-60:]) if line_count > 120 else ""
            excerpt = (head + "\n\n... [SNIPPED] ...\n\n" + tail).strip()
            excerpt = excerpt[: _MAX_SNIPPET_CHARS * 2]
            mode = "sampled"

        return RepoFileSummary(
            relative_path=rel,
            size=size,
            line_count=line_count,
            char_count=char_count,
            mode=mode,
            semantic_hash=semantic_hash,
            tags=tags,
            excerpt=excerpt,
        )

    def _derive_tags(self, rel: str, text: str) -> list[str]:
        tags: set[str] = set()
        lower_rel = rel.lower()
        lower_text = text.lower()
        ext = Path(rel).suffix.lower()

        if ext in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            tags.add("code")
        if ext in {".md", ".txt", ".rst"}:
            tags.add("docs")
        if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
            tags.add("config")
        if any(k in lower_rel for k in ("test", "spec", "pytest")):
            tags.add("tests")
        if "docker" in lower_rel or "compose" in lower_rel:
            tags.add("infra")
        if any(k in lower_rel for k in ("conversation", "transcript", "history", "chatlog", "export")):
            tags.add("low_signal_log")
        if any(k in lower_text for k in ("class ", "def ", "function ", "interface ", "import ")):
            tags.add("implementation")
        return sorted(tags)

    def _extract_keywords(self, text: str) -> list[str]:
        stop = {
            "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
            "have", "has", "had", "not", "but", "you", "your", "their", "there", "into",
            "will", "would", "could", "should", "while", "where", "which", "what",
        }
        tokens = re.findall(r"[A-Za-z_]{4,}", text.lower())
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in stop or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= 24:
                break
        return out

    def _semantic_weight(self, rel: str, text: str, tags: list[str]) -> float:
        score = 0.4
        rel_lower = rel.lower()
        score += min(0.3, len(text) / 6000.0)
        if any(tag in tags for tag in ("code", "implementation")):
            score += 0.25
        if "config" in tags:
            score += 0.1
        if "tests" in tags:
            score -= 0.05
        if "low_signal_log" in tags:
            score -= 0.25
        if any(k in rel_lower for k in ("main.py", "orchestrator", "core/", "ui/", "agents/", "runtime")):
            score += 0.15
        return max(0.05, min(1.25, score))

    def _chunk_text(self, text: str, max_chars: int = 1200, overlap: int = 180) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            j = min(n, i + max_chars)
            chunks.append(text[i:j])
            if j >= n:
                break
            i = max(i + 1, j - overlap)
        return chunks

    def _top_directories(self, summaries: list[RepoFileSummary]) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for s in summaries:
            parts = s.relative_path.split("/")
            top = parts[0] if len(parts) > 1 else "<root>"
            counts[top] = counts.get(top, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    def _prioritized_paths(self, summaries: list[RepoFileSummary]) -> list[str]:
        def score(item: RepoFileSummary) -> float:
            rel = item.relative_path.lower()
            ext = Path(rel).suffix.lower()
            val = 0.0
            if ext in {".py", ".ts", ".tsx", ".js", ".jsx"}:
                val += 3.0
            if ext in {".json", ".yaml", ".yml", ".toml"}:
                val += 1.2
            if ext in {".md", ".txt"}:
                val += 0.4
            if any(k in rel for k in ("main", "orchestrator", "runtime", "agent", "core", "ui")):
                val += 1.8
            if any(k in rel for k in ("conversation", "transcript", "history", "chatlog", "export", "workspace_conversations")):
                val -= 2.5
            if item.mode == "sampled":
                val += 0.5
            return val

        ranked = sorted(summaries, key=score, reverse=True)
        paths: list[str] = []
        seen: set[str] = set()
        for s in ranked:
            rel = s.relative_path
            if rel in seen:
                continue
            seen.add(rel)
            paths.append(rel)
        return paths

    def _collect_state(self, repo: Path) -> dict:
        files: dict[str, dict] = {}
        for file_path in self._iter_repo_files(repo):
            rel = file_path.relative_to(repo).as_posix()
            if self._skip_reason(rel, file_path) is not None:
                continue
            try:
                stat = file_path.stat()
            except OSError:
                continue
            char_count = 0
            content_hash = ""
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                char_count = len(text)
                content_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                try:
                    raw = file_path.read_bytes()
                    char_count = len(raw)
                    content_hash = hashlib.sha1(raw).hexdigest()
                except Exception:
                    content_hash = ""
            files[rel] = {
                "size": stat.st_size,
                "mtime": int(stat.st_mtime),
                "char_count": char_count,
                "content_hash": content_hash,
            }
        return {
            "repo": str(repo),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "files": files,
        }

    def _write_state(self, repo: Path, profile_dir: Path) -> None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        state = self._collect_state(repo)
        (profile_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
