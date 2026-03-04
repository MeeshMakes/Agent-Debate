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

    @staticmethod
    def _is_cloud_model_name(name: str) -> bool:
        lower = (name or "").lower().strip()
        if ":" in lower:
            tag = lower.split(":", 1)[1]
            if "cloud" in tag:
                return True
        return lower.endswith("-cloud")

    @staticmethod
    def _strict_local_only() -> bool:
        val = os.getenv("AGENT_DEBATE_STRICT_LOCAL_ONLY", "1").strip().lower()
        return val not in {"0", "false", "no"}

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

    @staticmethod
    def _fallback_preference_order() -> list[str]:
        raw = os.getenv("LOCAL_MODEL_FALLBACKS", "").strip()
        if raw:
            custom = [m.strip() for m in raw.split(",") if m.strip()]
            if custom:
                return custom
        return [
            "qwen2.5-coder:7b",
            "qwen3:8b",
            "mistral:7b",
            "phi4:latest",
            "llama3.1:8b",
            "qwen3:14b",
            "qwen3:30b",
        ]

    async def _retry_with_fallback_model(
        self,
        *,
        endpoint: str,
        messages: list[dict],
        failed_model: str,
    ) -> str | None:
        available = await self.list_models()
        if not available:
            return None

        available_clean = [m.strip() for m in available if m and m.strip()]
        if not available_clean:
            return None

        pref = self._fallback_preference_order()
        ordered: list[str] = []

        for name in pref:
            if name in available_clean and name != failed_model and not self._is_cloud_model_name(name):
                ordered.append(name)
        for name in available_clean:
            if name == failed_model or self._is_cloud_model_name(name):
                continue
            if name not in ordered:
                ordered.append(name)

        if not ordered:
            return None

        url = f"{endpoint}/api/chat"
        for candidate in ordered[:4]:
            payload = {
                "model": candidate,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.85,
                    "top_p": 0.92,
                    "repeat_penalty": 1.1,
                    "num_predict": 3000,
                },
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                text = str(data.get("message", {}).get("content", "")).strip() or str(data.get("response", "")).strip()
                if text:
                    self.model = candidate
                    return text
            except Exception:
                continue

        return None

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        endpoint = self._base_endpoint()
        if self._strict_local_only() and self._is_cloud_model_name(self.model):
            return (
                f"[Local provider blocked] Model '{self.model}' is cloud-tagged. "
                "Strict local mode is enabled; choose a non-cloud Ollama model."
            )

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
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                if status == 500:
                    if attempt == 0:
                        fallback_text = await self._retry_with_fallback_model(
                            endpoint=endpoint,
                            messages=messages,
                            failed_model=self.model,
                        )
                        if fallback_text:
                            return fallback_text
                    return (
                        f"[Local provider error] Ollama local server returned HTTP 500 for model '{self.model}' at "
                        f"{url}. This is local runtime failure (model load/OOM/crash), not an online API call."
                    )
                if attempt == 1:
                    return f"[Local provider error] HTTP {status} from local Ollama endpoint {url}"
            except Exception as exc:
                if attempt == 1:
                    return f"[Local provider error] {exc}"

        return "[Local provider error] Empty response from local model endpoint."
