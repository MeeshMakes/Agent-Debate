from __future__ import annotations

import json
from pathlib import Path

from agents.persona_agent import PersonaAgent, PersonaConfig
from config.model_prefs import DEFAULT_LEFT_MODEL, DEFAULT_RIGHT_MODEL
from core.arbiter_engine import ArbiterEngine
from core.orchestrator import DebateOrchestrator
from core.session_manager import SessionManager
from debate_graph.manager import DebateGraphManager
from evidence.retriever import CorpusEvidenceRetriever
from providers.local_provider import LocalProvider
from providers.router import ProviderRouter
from providers.vscode_provider import VSCodeProvider
from runtime_logging.event_logger import EventLogger


def _make_provider(model: str, vscode_model_ids: set[str]) -> LocalProvider | VSCodeProvider:
    """Return the correct provider for the given model name."""
    if model in vscode_model_ids:
        return VSCodeProvider(model=model)
    return LocalProvider(model=model)


def _load_debate_config(config_path: Path) -> dict[str, int]:
    config: dict[str, int] = {
        "max_turns": 10,
        "drift_threshold": 2,
        "synthesis_interval": 6,
    }
    if not config_path.exists():
        return config

    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in config:
            config[key] = int(value)
    return config


def build_orchestrator(
    project_root: Path,
    left_model: str | None = None,
    right_model: str | None = None,
    session_manager: SessionManager | None = None,
    vscode_model_ids: set[str] | None = None,
) -> tuple[DebateOrchestrator, dict]:
    profiles_path = project_root / "profiles" / "profiles.json"
    config_path = project_root / "config" / "debate_config.yaml"
    log_path = project_root / "logs" / "debate_events.jsonl"

    profile_data = json.loads(profiles_path.read_text(encoding="utf-8"))
    debate_config = _load_debate_config(config_path)

    left_profile = dict(profile_data["left_agent"])
    right_profile = dict(profile_data["right_agent"])
    left_profile.pop("provider", None)
    right_profile.pop("provider", None)

    left_cfg = PersonaConfig(**left_profile)
    right_cfg = PersonaConfig(**right_profile)

    corpus_root = project_root.parent / "Past Debate Systems"
    evidence_retriever = CorpusEvidenceRetriever(corpus_root=corpus_root)

    # Each agent gets its own provider — LocalProvider (Ollama) or VSCodeProvider
    _vscode_ids: set[str] = vscode_model_ids or set()
    lm = left_model or DEFAULT_LEFT_MODEL
    rm = right_model or DEFAULT_RIGHT_MODEL
    left_provider  = _make_provider(lm, _vscode_ids)
    right_provider = _make_provider(rm, _vscode_ids)

    orchestrator = DebateOrchestrator(
        left_agent=PersonaAgent(left_cfg, provider=left_provider),
        right_agent=PersonaAgent(right_cfg, provider=right_provider),
        arbiter=ArbiterEngine(
            drift_threshold=debate_config["drift_threshold"],
            synthesis_interval=debate_config["synthesis_interval"],
        ),
        graph=DebateGraphManager(),
        logger=EventLogger(output_path=str(log_path)),
        evidence_retriever=evidence_retriever,
        session_manager=session_manager,
    )
    return orchestrator, debate_config
