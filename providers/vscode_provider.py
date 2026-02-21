"""
VS Code Bridge Provider
=======================
Routes generation requests through the VS Code language-model bridge,
which exposes an OpenAI-compatible HTTP API at (by default) http://127.0.0.1:8765/v1.

The bridge is started by VS Code itself when the "GitHub Copilot" or similar
extension is active. It accepts standard /v1/chat/completions and /v1/models
requests and forwards them to whatever model the user has selected in VS Code.

Usage is identical to LocalProvider — just pass a VSCodeProvider instance
to PersonaAgent instead of LocalProvider.
"""
from __future__ import annotations

import os

import httpx

from providers.base_provider import BaseProvider

_DEFAULT_BRIDGE_URL = "http://127.0.0.1:8765/v1"
_DEFAULT_API_KEY = "vscode"          # bridge ignores this; any non-empty value works
_DEFAULT_TIMEOUT = 120.0             # VS Code models can be slower on first call


class VSCodeProvider(BaseProvider):
    """OpenAI-compatible provider that routes through the local VS Code bridge."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model or os.getenv("VSCODE_BRIDGE_MODEL", "gpt-4o")
        self.base_url = (
            (base_url or os.getenv("VSCODE_BRIDGE_URL") or _DEFAULT_BRIDGE_URL).rstrip("/")
        )
        self.api_key = api_key or os.getenv("VSCODE_BRIDGE_API_KEY") or _DEFAULT_API_KEY
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def is_running(self) -> bool:
        """Return True if the bridge responds to /models."""
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return model IDs from the bridge's /v1/models endpoint."""
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    str(m.get("id", ""))
                    for m in data.get("data", [])
                    if isinstance(m, dict) and m.get("id")
                ]
        except Exception:
            return []

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a chat completion request through the bridge and return the reply."""
        url = f"{self.base_url}/chat/completions"

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, headers=self._headers()
                ) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                # Standard OpenAI shape
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if content:
                    return content

                # Fallback: some bridges put it at top-level "text"
                text = str(data.get("text", "")).strip()
                if text:
                    return text

            except httpx.TimeoutException:
                if attempt == 0:
                    continue
                return f"[VS Code bridge timeout — model {self.model} did not respond in {self.timeout_seconds}s]"
            except Exception as exc:
                if attempt == 1:
                    return f"[VS Code bridge error] {exc}"

        return "[VS Code bridge error] Empty response."
