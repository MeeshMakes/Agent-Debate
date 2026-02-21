from __future__ import annotations

from debate_graph.manager import DebateGraphManager


class BranchSelector:
    def __init__(self, manager: DebateGraphManager) -> None:
        self._manager = manager

    def mark_resolved(self, node_id: str) -> None:
        self._manager.set_status(node_id=node_id, status="resolved")
