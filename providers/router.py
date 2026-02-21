from __future__ import annotations

from providers.base_provider import BaseProvider
from providers.local_provider import LocalProvider


class ProviderRouter:
    """All routes resolve to LocalProvider (Ollama). No cloud APIs."""

    def __init__(self) -> None:
        self._local = LocalProvider()

    def get(self, provider_name: str) -> BaseProvider:
        return self._local

    def get_local(self) -> LocalProvider:
        return self._local
