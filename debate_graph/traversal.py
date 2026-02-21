from __future__ import annotations

from debate_graph.manager import DebateGraphManager


class DebateTraversal:
    def __init__(self, manager: DebateGraphManager) -> None:
        self._manager = manager

    def active_talking_point(self) -> str:
        active_id = self._manager.state.active_talking_point_id
        if not active_id:
            return ""
        return self._manager.state.nodes[active_id].label
