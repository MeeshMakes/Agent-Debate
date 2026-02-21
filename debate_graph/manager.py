from __future__ import annotations

from uuid import uuid4

from debate_graph.models import DebateGraphState, DebateNode


class DebateGraphManager:
    def __init__(self) -> None:
        self.state = DebateGraphState()
        self._order = 0

    def add_talking_point(self, label: str) -> DebateNode:
        self._order += 1
        node = DebateNode(
            node_id=str(uuid4()),
            node_type="talking_point",
            label=label,
            depth=0,
            creation_order=self._order,
        )
        self.state.nodes[node.node_id] = node
        self.state.active_talking_point_id = node.node_id
        return node

    def add_child(self, parent_id: str, node_type: str, label: str) -> DebateNode:
        self._order += 1
        parent = self.state.nodes.get(parent_id)
        depth = (parent.depth + 1) if parent else 1
        node = DebateNode(
            node_id=str(uuid4()),
            node_type=node_type,
            label=label,
            parent_id=parent_id,
            depth=depth,
            creation_order=self._order,
        )
        self.state.nodes[node.node_id] = node
        if parent is not None:
            parent.links.append(node.node_id)
        return node

    def set_status(self, node_id: str, status: str) -> None:
        if node_id in self.state.nodes:
            self.state.nodes[node_id].status = status

    def as_rows(self) -> list[tuple[str, str, str]]:
        """Return simple (type, label, status) tuples in creation order."""
        nodes = sorted(self.state.nodes.values(), key=lambda n: n.creation_order)
        return [(n.node_type, n.label, n.status) for n in nodes]

    def as_tree_nodes(self) -> list[DebateNode]:
        """Return all nodes sorted by creation order — includes depth/parent info."""
        return sorted(self.state.nodes.values(), key=lambda n: n.creation_order)
