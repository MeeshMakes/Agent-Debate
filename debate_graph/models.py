from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DebateNode:
    node_id: str
    node_type: str
    label: str
    status: str = "active"
    links: list[str] = field(default_factory=list)
    parent_id: str = ""
    depth: int = 0
    creation_order: int = 0


@dataclass
class DebateEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str = "elaborates"   # supports | contradicts | refutes | synthesizes | elaborates
    weight: float = 0.5
    directed: bool = True
    evidence: str = ""
    turn: int = 0
    creation_order: int = 0


@dataclass
class DebateGraphState:
    nodes: dict[str, DebateNode] = field(default_factory=dict)
    edges: dict[str, DebateEdge] = field(default_factory=dict)
    active_talking_point_id: str | None = None
    _counter: int = 0
