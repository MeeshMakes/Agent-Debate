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
    ast_signature: str
    ast_symbols: dict[str, list[str]]
    tags: list[str]
    excerpt: str


@dataclass
class RepoSnapshot:
    snapshot_id: str
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
        generated_at = datetime.now().isoformat(timespec="seconds")
        snapshot_id = hashlib.sha1(
            f"{repo}:{generated_at}:{included}:{len(summaries)}".encode("utf-8", errors="ignore")
        ).hexdigest()[:12]
        snapshot = RepoSnapshot(
            snapshot_id=snapshot_id,
            repo_path=str(repo),
            generated_at=generated_at,
            included_files=included,
            skipped_files=skipped,
            skipped_reasons=skipped_reasons,
            extension_counts=extension_counts,
            top_large_files=top_large_files[:20],
            summaries=summaries,
        )

        profile_dir = self._profile_dir(repo)
        profile_dir.mkdir(parents=True, exist_ok=True)
        snapshot_payload = {
            **asdict(snapshot),
            "summaries": [asdict(item) for item in snapshot.summaries],
        }
        (profile_dir / "snapshot.json").write_text(
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._archive_snapshot_payload(profile_dir, snapshot_payload)
        self._write_docs_index(profile_dir, repo, summaries)
        self.build_semantic_dataset(str(repo))
        self._write_state(repo, profile_dir)
        return snapshot

    def build_repo_delta_intelligence(
        self,
        repo_path: str,
        *,
        baseline_generated_at: str = "",
        max_files: int = 50,
        refresh_snapshot: bool = True,
    ) -> dict:
        repo = Path(repo_path).resolve()
        if not repo.exists() or not repo.is_dir():
            return {
                "repo_path": str(repo),
                "error": "repo_not_found",
                "message": "Repository path does not exist.",
            }

        profile_dir = self._profile_dir(repo)
        profile_dir.mkdir(parents=True, exist_ok=True)

        current_snapshot = self.build_snapshot(str(repo)) if refresh_snapshot else self._load_latest_snapshot(profile_dir)
        if current_snapshot is None:
            return {
                "repo_path": str(repo),
                "error": "no_current_snapshot",
                "message": "No current repository snapshot is available.",
            }

        baseline_snapshot = self._load_snapshot_by_generated_at(profile_dir, baseline_generated_at)
        if baseline_snapshot is None:
            baseline_snapshot = self._load_previous_snapshot(profile_dir, current_snapshot.generated_at)

        if baseline_snapshot is None:
            nav = self.build_repo_navigation_map(str(repo), max_files=max_files)
            return {
                "repo_path": str(repo),
                "baseline_generated_at": "",
                "current_generated_at": current_snapshot.generated_at,
                "baseline_snapshot_id": "",
                "current_snapshot_id": current_snapshot.snapshot_id,
                "change_counts": {"added": 0, "modified": 0, "deleted": 0, "touched": 0},
                "added": [],
                "modified": [],
                "deleted": [],
                "ast_changes": [],
                "module_change_summaries": [],
                "navigation_map": nav,
                "message": "No historical baseline snapshot was available. Current repository map refreshed.",
            }

        old_map = {s.relative_path: s for s in baseline_snapshot.summaries}
        new_map = {s.relative_path: s for s in current_snapshot.summaries}

        old_paths = set(old_map.keys())
        new_paths = set(new_map.keys())
        added = sorted(new_paths - old_paths)
        deleted = sorted(old_paths - new_paths)

        modified: list[str] = []
        for rel in sorted(old_paths & new_paths):
            old_item = old_map[rel]
            new_item = new_map[rel]
            if old_item.semantic_hash != new_item.semantic_hash:
                modified.append(rel)

        touched = (added + modified + deleted)[:max_files]

        ast_changes: list[dict] = []
        for rel in (added + modified)[:max_files]:
            old_symbols = old_map[rel].ast_symbols if rel in old_map else {}
            new_symbols = new_map[rel].ast_symbols if rel in new_map else {}
            symbol_delta = self._build_symbol_delta(rel, old_symbols, new_symbols)
            if symbol_delta:
                ast_changes.append(symbol_delta)

        module_summaries: list[str] = []
        for rel in touched[:24]:
            if rel in added:
                kind = "added"
            elif rel in modified:
                kind = "modified"
            else:
                kind = "deleted"
            delta = next((d for d in ast_changes if d.get("path") == rel), None)
            if isinstance(delta, dict):
                adds = ", ".join(delta.get("added", [])[:4])
                rems = ", ".join(delta.get("removed", [])[:4])
                extra = ""
                if adds:
                    extra += f" | + {adds}"
                if rems:
                    extra += f" | - {rems}"
                module_summaries.append(f"{kind.upper()}: {rel}{extra}")
            else:
                module_summaries.append(f"{kind.upper()}: {rel}")

        nav = self.build_repo_navigation_map(str(repo), max_files=max_files)

        return {
            "repo_path": str(repo),
            "baseline_generated_at": baseline_snapshot.generated_at,
            "current_generated_at": current_snapshot.generated_at,
            "baseline_snapshot_id": baseline_snapshot.snapshot_id,
            "current_snapshot_id": current_snapshot.snapshot_id,
            "change_counts": {
                "added": len(added),
                "modified": len(modified),
                "deleted": len(deleted),
                "touched": len(touched),
            },
            "added": added[:max_files],
            "modified": modified[:max_files],
            "deleted": deleted[:max_files],
            "ast_changes": ast_changes[:max_files],
            "module_change_summaries": module_summaries,
            "navigation_map": nav,
            "message": (
                f"Compared repo snapshot {baseline_snapshot.generated_at} → {current_snapshot.generated_at}: "
                f"added={len(added)}, modified={len(modified)}, deleted={len(deleted)}"
            ),
        }

    def build_repo_navigation_map(self, repo_path: str, *, max_files: int = 80) -> dict:
        repo = Path(repo_path).resolve()
        profile_dir = self._profile_dir(repo)
        snapshot = self._load_latest_snapshot(profile_dir)
        if snapshot is None:
            snapshot = self.build_snapshot(str(repo))

        summaries = snapshot.summaries if snapshot is not None else []
        prioritized = self._prioritized_paths(summaries)
        summary_by_path = {s.relative_path: s for s in summaries}

        files: list[dict] = []
        for rel in prioritized[:max_files]:
            s = summary_by_path.get(rel)
            if s is None:
                continue
            files.append(
                {
                    "path": rel,
                    "tags": list(s.tags),
                    "line_count": int(s.line_count),
                    "size": int(s.size),
                    "ast_signature": s.ast_signature,
                }
            )

        return {
            "generated_at": snapshot.generated_at if snapshot is not None else "",
            "snapshot_id": snapshot.snapshot_id if snapshot is not None else "",
            "files": files,
            "total_files": len(summaries),
        }

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
        focus_lines = [f"  • {p}" for p in prioritized[:20]]

        # Build a lookup for fast excerpt access
        excerpt_map = {s.relative_path: s.excerpt for s in snapshot.summaries}

        # Include actual code excerpts from the top-priority files so the LLM
        # can ground the brief in real method names, constants, and log signals
        _MAX_EXCERPT_PER_FILE = 1_200   # chars per file in this brief
        _MAX_EXCERPT_FILES    = 14      # how many files to include
        _MAX_TOTAL_EXCERPT    = 18_000  # total excerpt budget

        excerpt_sections: list[str] = []
        total_excerpt_chars = 0
        for rel in prioritized[:_MAX_EXCERPT_FILES]:
            if total_excerpt_chars >= _MAX_TOTAL_EXCERPT:
                break
            raw = excerpt_map.get(rel, "").strip()
            if not raw:
                continue
            # Skip low-signal files
            if any(k in rel.lower() for k in (
                "conversation", "transcript", "chatlog", "history", "export",
                "workspace_conversations", "__pycache__",
            )):
                continue
            snip = raw[:_MAX_EXCERPT_PER_FILE]
            excerpt_sections.append(f"### {rel}\n```\n{snip}\n```")
            total_excerpt_chars += len(snip)

        excerpts_block = (
            "\n\n".join(excerpt_sections)
            if excerpt_sections
            else "(no excerpts available)"
        )

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
            "--- KEY FILE EXCERPTS (actual source code) ---\n"
            "Use these to extract real class names, method names, constants, log strings, "
            "and failure paths for the debate brief.\n\n"
            + excerpts_block
            + "\n\n"
            "Use this snapshot as entry context. Prioritize executable scripts and their "
            "cross-file interactions first, then deep-dive into specific scripts and loops "
            "where failures or inconsistencies appear."
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
                ast_signature="",
                ast_symbols={},
                tags=[],
                excerpt="",
            )

        lines = text.splitlines()
        line_count = len(lines)
        char_count = len(text)
        semantic_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        tags = self._derive_tags(rel, text)
        ast_symbols = self._extract_ast_symbols(rel, text)
        ast_signature = self._build_ast_signature(ast_symbols)

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
            ast_signature=ast_signature,
            ast_symbols=ast_symbols,
            tags=tags,
            excerpt=excerpt,
        )

    def _extract_ast_symbols(self, rel: str, text: str) -> dict[str, list[str]]:
        ext = Path(rel).suffix.lower()
        classes: set[str] = set()
        functions: set[str] = set()
        methods: set[str] = set()
        exports: set[str] = set()

        if ext == ".py":
            for m in re.finditer(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
                classes.add(m.group(1))
            for m in re.finditer(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
                indent = m.group(1)
                name = m.group(2)
                if indent:
                    methods.add(name)
                else:
                    functions.add(name)

        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            for m in re.finditer(r"\bclass\s+([A-Za-z_$][A-Za-z0-9_$]*)", text):
                classes.add(m.group(1))
            for m in re.finditer(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", text):
                functions.add(m.group(1))
            for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*\([^)]*\)\s*=>", text):
                functions.add(m.group(1))
            for m in re.finditer(r"\bexport\s+(?:default\s+)?(?:class|function|const|let|var)?\s*([A-Za-z_$][A-Za-z0-9_$]*)", text):
                if m.group(1):
                    exports.add(m.group(1))

        elif ext in {".go"}:
            for m in re.finditer(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b", text):
                classes.add(m.group(1))
            for m in re.finditer(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
                functions.add(m.group(1))

        out: dict[str, list[str]] = {}
        if classes:
            out["classes"] = sorted(classes)
        if functions:
            out["functions"] = sorted(functions)
        if methods:
            out["methods"] = sorted(methods)
        if exports:
            out["exports"] = sorted(exports)
        return out

    def _build_ast_signature(self, symbols: dict[str, list[str]]) -> str:
        if not symbols:
            return ""
        parts: list[str] = []
        for kind in sorted(symbols.keys()):
            names = symbols.get(kind, [])
            if not isinstance(names, list):
                continue
            for name in names:
                n = str(name).strip()
                if n:
                    parts.append(f"{kind}:{n}")
        if not parts:
            return ""
        return hashlib.sha1("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _build_symbol_delta(self, path: str, old_symbols: dict, new_symbols: dict) -> dict | None:
        old_map = old_symbols if isinstance(old_symbols, dict) else {}
        new_map = new_symbols if isinstance(new_symbols, dict) else {}

        old_flat: set[str] = set()
        new_flat: set[str] = set()
        for kind, names in old_map.items():
            if isinstance(names, list):
                old_flat.update(f"{kind}:{str(n).strip()}" for n in names if str(n).strip())
        for kind, names in new_map.items():
            if isinstance(names, list):
                new_flat.update(f"{kind}:{str(n).strip()}" for n in names if str(n).strip())

        added = sorted(new_flat - old_flat)
        removed = sorted(old_flat - new_flat)
        if not added and not removed:
            return None

        return {
            "path": path,
            "added": added[:20],
            "removed": removed[:20],
        }

    def _archive_snapshot_payload(self, profile_dir: Path, payload: dict) -> None:
        archive = profile_dir / "snapshots"
        archive.mkdir(parents=True, exist_ok=True)
        generated = str(payload.get("generated_at", "") or datetime.now().isoformat(timespec="seconds"))
        safe_ts = generated.replace(":", "-").replace("/", "-")
        file_path = archive / f"snapshot_{safe_ts}.json"
        if file_path.exists():
            sid = str(payload.get("snapshot_id", "") or hashlib.sha1(generated.encode("utf-8", errors="ignore")).hexdigest()[:6])
            file_path = archive / f"snapshot_{safe_ts}_{sid}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        archived = sorted(archive.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in archived[60:]:
            stale.unlink(missing_ok=True)

    def _snapshot_from_dict(self, data: dict) -> RepoSnapshot | None:
        if not isinstance(data, dict):
            return None
        summaries_raw = data.get("summaries", [])
        summaries: list[RepoFileSummary] = []
        if isinstance(summaries_raw, list):
            for item in summaries_raw:
                if not isinstance(item, dict):
                    continue
                summaries.append(
                    RepoFileSummary(
                        relative_path=str(item.get("relative_path", "") or ""),
                        size=int(item.get("size", 0) or 0),
                        line_count=int(item.get("line_count", 0) or 0),
                        char_count=int(item.get("char_count", 0) or 0),
                        mode=str(item.get("mode", "") or ""),
                        semantic_hash=str(item.get("semantic_hash", "") or ""),
                        ast_signature=str(item.get("ast_signature", "") or ""),
                        ast_symbols=item.get("ast_symbols", {}) if isinstance(item.get("ast_symbols", {}), dict) else {},
                        tags=[str(t).strip() for t in item.get("tags", []) if str(t).strip()] if isinstance(item.get("tags", []), list) else [],
                        excerpt=str(item.get("excerpt", "") or ""),
                    )
                )

        snapshot_id = str(data.get("snapshot_id", "") or "").strip()
        generated_at = str(data.get("generated_at", "") or "").strip()
        repo_path = str(data.get("repo_path", "") or "").strip()
        if not snapshot_id and generated_at:
            snapshot_id = hashlib.sha1(f"{repo_path}:{generated_at}".encode("utf-8", errors="ignore")).hexdigest()[:12]

        return RepoSnapshot(
            snapshot_id=snapshot_id,
            repo_path=repo_path,
            generated_at=generated_at,
            included_files=int(data.get("included_files", len(summaries)) or len(summaries)),
            skipped_files=int(data.get("skipped_files", 0) or 0),
            skipped_reasons=data.get("skipped_reasons", {}) if isinstance(data.get("skipped_reasons", {}), dict) else {},
            extension_counts=data.get("extension_counts", {}) if isinstance(data.get("extension_counts", {}), dict) else {},
            top_large_files=data.get("top_large_files", []) if isinstance(data.get("top_large_files", []), list) else [],
            summaries=summaries,
        )

    def _load_latest_snapshot(self, profile_dir: Path) -> RepoSnapshot | None:
        snapshot_file = profile_dir / "snapshot.json"
        if snapshot_file.exists():
            try:
                data = json.loads(snapshot_file.read_text(encoding="utf-8"))
                snap = self._snapshot_from_dict(data)
                if snap is not None:
                    return snap
            except Exception:
                pass

        archive = profile_dir / "snapshots"
        if not archive.exists():
            return None
        files = sorted(archive.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                snap = self._snapshot_from_dict(data)
                if snap is not None:
                    return snap
            except Exception:
                continue
        return None

    def _load_previous_snapshot(self, profile_dir: Path, current_generated_at: str) -> RepoSnapshot | None:
        archive = profile_dir / "snapshots"
        if not archive.exists():
            return None
        current_ts = self._to_epoch(current_generated_at)
        entries: list[tuple[float, RepoSnapshot]] = []
        for fp in archive.glob("snapshot_*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            snap = self._snapshot_from_dict(data)
            if snap is None:
                continue
            ts = self._to_epoch(snap.generated_at)
            entries.append((ts, snap))
        if not entries:
            return None
        entries.sort(key=lambda item: item[0], reverse=True)
        for ts, snap in entries:
            if current_ts <= 0 or ts < current_ts:
                return snap
        return None

    def _load_snapshot_by_generated_at(self, profile_dir: Path, generated_at: str) -> RepoSnapshot | None:
        target = (generated_at or "").strip()
        if not target:
            return None
        target_ts = self._to_epoch(target)
        if target_ts <= 0:
            return None

        archive = profile_dir / "snapshots"
        if not archive.exists():
            return None

        best: tuple[float, RepoSnapshot] | None = None
        for fp in archive.glob("snapshot_*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            snap = self._snapshot_from_dict(data)
            if snap is None:
                continue
            ts = self._to_epoch(snap.generated_at)
            if ts <= 0:
                continue
            if ts <= target_ts and (best is None or ts > best[0]):
                best = (ts, snap)

        return best[1] if best is not None else None

    def _to_epoch(self, value: str) -> float:
        ts = (value or "").strip()
        if not ts:
            return 0.0
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0

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
