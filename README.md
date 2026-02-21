# Agent Debate NextGen (PyQt6)

A next-generation dual-agent debate system with:
- center-stage live debate conversation,
- private inner-monologue channels per agent,
- arbiter facilitation,
- debate graph tracking,
- async-safe turn scheduling.

## Quick start
1. Create a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run:
   - `python -m app.main`

## Notes
- Runtime is **local-only** (no OpenAI API usage).
- Provider uses Ollama endpoint (`OLLAMA_HOST` preferred, fallback `LOCAL_MODEL_ENDPOINT`, default `http://localhost:11434`).
- Default local model is `qwen2.5-coder:7b` unless `LOCAL_MODEL_NAME` is set.
- Profiles are in `profiles/profiles.json`.
- Debate config is in `config/debate_config.yaml`.
- Runtime events are logged to `logs/debate_events.jsonl`.
- Evidence retrieval automatically pulls summaries from sibling folder `Past Debate Systems/*/content/conversation.dataset.json` and injects top matches into each turn.
- Every public debate turn now includes citation metadata (source path + relevance score) and displays those sources in the center debate stream.
- Citation source links in the center stream are clickable and open an in-app source viewer dialog.
- Local model setup and recommended model packs are in `LOCAL_MODELS.md`.
