# Local Models and Ollama Setup (Local-Only)

This app is local-first and runs against Ollama on your machine.

## 1) Start Ollama
- Install Ollama if needed.
- Run daemon:
  - `ollama serve`

## 2) Verify endpoint
- Default endpoint: `http://localhost:11434`
- Override with env var:
  - `OLLAMA_HOST=http://localhost:11434`

## 3) Pull recommended models
Based on your Morph/Fathom local stacks, these are strong defaults:

### Core debate + reasoning
- `ollama pull qwen2.5-coder:7b`
- `ollama pull qwen3:14b`
- `ollama pull deepseek-r1:14b`

### Fast/utility turns
- `ollama pull qwen2.5:1.5b`
- `ollama pull phi3:mini`

### Alternative generalists
- `ollama pull llama3.3:latest`
- `ollama pull mistral:latest`

## 4) Configure app model
- Default model env var:
  - `LOCAL_MODEL_NAME=qwen2.5-coder:7b`

## 5) Local-only enforcement in this project
- Provider routing is local-only (`local`/`ollama` aliases map to local provider).
- Profile values are normalized to local provider at bootstrap.
- If Ollama is offline, status bar shows offline state.

## 6) Health checks
The UI runtime badge reports:
- `Local Runtime: Ollama online (N models)`
- or `Local Runtime: Ollama offline`
