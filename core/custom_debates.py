from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CustomDebate:
    id: str
    title: str
    description: str
    talking_points: list[str]
    mode: str = "static_ingestion"  # static_ingestion | repo_watchdog
    repo_path: str = ""
    created_at: str = ""
    updated_at: str = ""


class CustomDebateStore:
    def __init__(self, sessions_root: Path) -> None:
        self._dir = sessions_root / "_custom_debates"
        self._dir.mkdir(parents=True, exist_ok=True)

    def list_all(self) -> list[CustomDebate]:
        debates: list[CustomDebate] = []
        for p in sorted(self._dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                debates.append(
                    CustomDebate(
                        id=str(data.get("id", p.stem)),
                        title=str(data.get("title", "")).strip(),
                        description=str(data.get("description", "")).strip(),
                        talking_points=[str(x).strip() for x in data.get("talking_points", []) if str(x).strip()],
                        mode=str(data.get("mode", "static_ingestion")).strip() or "static_ingestion",
                        repo_path=str(data.get("repo_path", "")).strip(),
                        created_at=str(data.get("created_at", "")),
                        updated_at=str(data.get("updated_at", "")),
                    )
                )
            except Exception:
                continue
        return debates

    def upsert(
        self,
        *,
        debate_id: str | None,
        title: str,
        description: str,
        talking_points: list[str],
        mode: str,
        repo_path: str,
    ) -> CustomDebate:
        now = datetime.now().isoformat(timespec="seconds")
        did = debate_id or uuid.uuid4().hex
        existing = self.get(did)
        created_at = existing.created_at if existing is not None else now

        debate = CustomDebate(
            id=did,
            title=title.strip(),
            description=description.strip(),
            talking_points=[tp.strip() for tp in talking_points if tp.strip()],
            mode=(mode or "static_ingestion").strip(),
            repo_path=repo_path.strip(),
            created_at=created_at,
            updated_at=now,
        )
        path = self._dir / f"{did}.json"
        path.write_text(json.dumps(asdict(debate), ensure_ascii=False, indent=2), encoding="utf-8")
        return debate

    def get(self, debate_id: str) -> CustomDebate | None:
        p = self._dir / f"{debate_id}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return CustomDebate(
                id=str(data.get("id", debate_id)),
                title=str(data.get("title", "")).strip(),
                description=str(data.get("description", "")).strip(),
                talking_points=[str(x).strip() for x in data.get("talking_points", []) if str(x).strip()],
                mode=str(data.get("mode", "static_ingestion")).strip() or "static_ingestion",
                repo_path=str(data.get("repo_path", "")).strip(),
                created_at=str(data.get("created_at", "")),
                updated_at=str(data.get("updated_at", "")),
            )
        except Exception:
            return None

    def delete(self, debate_id: str) -> bool:
        p = self._dir / f"{debate_id}.json"
        if not p.exists():
            return False
        try:
            p.unlink()
            return True
        except Exception:
            return False


_store: CustomDebateStore | None = None


def get_custom_debate_store(sessions_root: Path) -> CustomDebateStore:
    global _store
    if _store is None:
        _store = CustomDebateStore(sessions_root)
    return _store
