from __future__ import annotations

import os

import httpx

from providers.base_provider import BaseProvider


class LocalProvider(BaseProvider):
    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 300.0,   # 5 min — large models (27b/30b) need time
    ) -> None:
        self.endpoint = endpoint
        self.model = model or os.getenv("LOCAL_MODEL_NAME", "qwen2.5-coder:7b")
        self.timeout_seconds = timeout_seconds

    def _base_endpoint(self) -> str:
        return (
            self.endpoint
            or os.getenv("OLLAMA_HOST")
            or os.getenv("LOCAL_MODEL_ENDPOINT")
            or "http://localhost:11434"
        ).rstrip("/")

    async def is_running(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self._base_endpoint()}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_endpoint()}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [str(item.get("name", "")) for item in data.get("models", []) if item.get("name")]
        except Exception:
            return []

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        endpoint = self._base_endpoint()
        # Use /api/chat for richer, more humanistic outputs
        url = f"{endpoint}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.85,      # alive and expressive
                "top_p": 0.92,
                "repeat_penalty": 1.1,    # discourages repetitive loops
                "num_predict": 3000,      # allow rich 5-8 paragraph responses
            },
        }

        for attempt in range(2):          # retry once on timeout
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                text = str(data.get("message", {}).get("content", "")).strip()
                if text:
                    return text
                # Fallback: legacy generate endpoint
                text = str(data.get("response", "")).strip()
                if text:
                    return text
            except httpx.TimeoutException:
                if attempt == 0:
                    # On first timeout, try legacy /api/generate as fallback
                    try:
                        gen_url = f"{endpoint}/api/generate"
                        full_prompt = prompt if not system_prompt else f"System:\n{system_prompt}\n\nUser:\n{prompt}"
                        gen_payload = {
                            "model": self.model,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {"temperature": 0.85, "num_predict": 3000},
                        }
                        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                            response = await client.post(gen_url, json=gen_payload)
                            response.raise_for_status()
                            data = response.json()
                        text = str(data.get("response", "")).strip()
                        if text:
                            return text
                    except Exception as exc2:
                        return f"[Local provider timeout — model {self.model} too slow: {exc2}]"
                return f"[Local provider timeout — model {self.model} did not respond in {self.timeout_seconds}s]"
            except Exception as exc:
                if attempt == 1:
                    return f"[Local provider error] {exc}"

        return "[Local provider error] Empty response from local model endpoint."
