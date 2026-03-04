"""Multi-lane node graph panel — QGraphicsScene / QGraphicsView.

Layout
------
    Semantic columns, one per node family:
    Col 0 – Root Topic   (#00e5ff)
    Col 1 – Sub-topic    (#ffd740)
        Col 2 – Evidence     (#69f0ae)
        Col 3 – Conflict     (#ef5350)
        Col 4 – Synthesis    (#ce93d8)

  Every node occupies a card positioned in its type's column.
  Creation order drives vertical position within that column.
    Bézier curves connect semantic edges (supports / contradicts / refutes / synthesizes).

Interaction
-----------
  Click-drag to pan.  Ctrl + scroll wheel to zoom.
"""
from __future__ import annotations

from PyQt6.QtCore   import Qt, QPointF, QRectF
from PyQt6.QtGui    import (
    QBrush, QColor, QFont, QPainter, QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

# ------------------------------------------------------------------ layout constants

_TYPE_ALIAS: dict[str, str] = {
    "branch": "sub-topic",
    "falsehood": "contradiction",
}

_COL_TYPES  = ["talking_point", "sub-topic", "conclusion", "contradiction", "synthesis"]
_COL_LABELS = ["◈  Root Topic", "◉  Sub-Topic", "✔  Evidence", "⚠  Conflict", "◆  Synthesis"]
_COL_COLORS = ["#00e5ff",       "#ffd740",      "#69f0ae",     "#ef5350",     "#ce93d8"]
_COL_BG     = ["#0d2a3a",       "#2a1f00",      "#082a14",     "#2a0c12",     "#1a0a2a"]

_EDGE_RELATION_STYLE: dict[str, tuple[str, Qt.PenStyle]] = {
    "supports": ("#69f0ae", Qt.PenStyle.SolidLine),
    "elaborates": ("#4dd0e1", Qt.PenStyle.SolidLine),
    "contradicts": ("#ffd740", Qt.PenStyle.DashLine),
    "refutes": ("#ef5350", Qt.PenStyle.DashLine),
    "synthesizes": ("#ce93d8", Qt.PenStyle.DotLine),
}

CARD_W    = 230
CARD_H    = 72
COL_GAP   = 280   # x distance between column left edges
ROW_H     = 94    # y spacing between cards in the same column
HEADER_H  = 48    # height reserved for column header row
SCENE_PAD = 24    # extra right/bottom padding


# ------------------------------------------------------------------ node card

class _NodeCard(QGraphicsItem):
    """A single rounded-rect node card drawn purely with QPainter."""

    def __init__(self, node_id: str, node_type: str, label: str,
                 status: str, order: int) -> None:
        super().__init__()
        self.node_id   = node_id
        self.node_type = node_type
        self.status    = status
        self.order     = order

        canon_type    = _canonical_type(node_type)
        col_idx       = _COL_TYPES.index(canon_type) if canon_type in _COL_TYPES else 0
        self.col_idx  = col_idx
        self.color    = _COL_COLORS[col_idx]
        self.bg_color = _COL_BG[col_idx]

        # Wrap label to two display lines
        if len(label) > 40:
            self.line1 = label[:40]
            self.line2 = label[40:80] + ("…" if len(label) > 80 else "")
        else:
            self.line1 = label
            self.line2 = ""

        # Dim resolved/concluded cards
        self._alpha = 110 if status in ("concluded", "done", "resolved") else 255

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, CARD_W, CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        c  = QColor(self.color)
        c.setAlpha(self._alpha)
        bg = QColor(self.bg_color)
        r  = QRectF(1, 1, CARD_W - 2, CARD_H - 2)

        # Card background
        card_path = QPainterPath()
        card_path.addRoundedRect(r, 11, 11)
        painter.fillPath(card_path, QBrush(bg))

        # Card border
        painter.setPen(QPen(c, 1.4))
        painter.drawPath(card_path)

        # Left accent bar
        accent = QPainterPath()
        accent.addRoundedRect(QRectF(1, 1, 4, CARD_H - 2), 2, 2)
        painter.fillPath(accent, QBrush(c))

        # Order badge (top-right corner)
        badge_rect = QRectF(CARD_W - 26, 5, 22, 17)
        bp = QPainterPath()
        bp.addRoundedRect(badge_rect, 5, 5)
        badge_bg = QColor(c)
        badge_bg.setAlpha(38)
        painter.fillPath(bp, QBrush(badge_bg))
        painter.setPen(QPen(c))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(self.order))

        # First line (main label)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(c))
        painter.drawText(
            QRectF(12, 8, CARD_W - 44, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.line1,
        )

        # Second line (continuation)
        if self.line2:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor("#90a4ae")))
            painter.drawText(
                QRectF(12, 30, CARD_W - 20, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.line2,
            )

        # Status chip (bottom-right, if not active)
        if self.status not in ("active", ""):
            painter.setFont(QFont("Segoe UI", 7))
            fm  = painter.fontMetrics()
            sw  = fm.horizontalAdvance(self.status) + 10
            cp  = QPainterPath()
            chip_r = QRectF(CARD_W - sw - 4, CARD_H - 18, sw, 13)
            cp.addRoundedRect(chip_r, 4, 4)
            painter.fillPath(cp, QBrush(QColor("#263238")))
            painter.setPen(QPen(QColor("#78909c")))
            painter.drawText(chip_r, Qt.AlignmentFlag.AlignCenter, self.status)

    # Convenience anchor points used by _EdgeItem
    def center_right(self)  -> QPointF: return self.pos() + QPointF(CARD_W, CARD_H / 2)
    def center_left(self)   -> QPointF: return self.pos() + QPointF(0,      CARD_H / 2)
    def center_bottom(self) -> QPointF: return self.pos() + QPointF(CARD_W / 2, CARD_H)
    def center_top(self)    -> QPointF: return self.pos() + QPointF(CARD_W / 2, 0)


# ------------------------------------------------------------------ edge

class _EdgeItem(QGraphicsPathItem):
    """Smooth Bézier curve connecting two node cards."""

    def __init__(self, src: QPointF, dst: QPointF, relation: str, weight: float = 0.5) -> None:
        path = QPainterPath(src)
        cx   = (src.x() + dst.x()) / 2
        path.cubicTo(QPointF(cx, src.y()), QPointF(cx, dst.y()), dst)
        super().__init__(path)
        rel = (relation or "elaborates").strip().lower()
        color, style = _EDGE_RELATION_STYLE.get(rel, ("#607d8b", Qt.PenStyle.SolidLine))
        c = QColor(color)
        c.setAlpha(130)
        width = 1.2 + (max(0.0, min(1.0, float(weight))) * 1.6)
        self.setPen(QPen(c, width, style,
                         Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setZValue(-1)


# ------------------------------------------------------------------ panel

class GraphPanel(QWidget):
    """Multi-lane debate relationship graph using QGraphicsScene."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("graphPanel")
        self._node_data: list = []
        self._build_ui()

    # ---------------------------------------------------------------- setup

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        header = QLabel("⬡  DEBATE MAP  —  Node Relationship Graph")
        header.setStyleSheet(
            "font-size: 9pt; font-weight: 700; color: #69f0ae; letter-spacing: 1px;"
        )
        root.addWidget(header)

        # Legend
        legend_row = QHBoxLayout()
        legend_row.setSpacing(18)
        for label, color in zip(_COL_LABELS, _COL_COLORS):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {color}; font-size: 8pt; font-weight: 600;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        root.addLayout(legend_row)

        # QGraphicsScene + QGraphicsView
        self._scene = QGraphicsScene()
        self._scene.setBackgroundBrush(QBrush(QColor("#080d14")))
        self._view  = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setStyleSheet(
            "QGraphicsView { border: none; background: #080d14; }"
            "QScrollBar:vertical   { background: #0d1521; width: 8px; }"
            "QScrollBar::handle:vertical   { background: #1e3050; border-radius: 4px; }"
            "QScrollBar:horizontal { background: #0d1521; height: 8px; }"
            "QScrollBar::handle:horizontal { background: #1e3050; border-radius: 4px; }"
        )
        root.addWidget(self._view)

    def _draw_column_headers(self) -> None:
        for i, (label, color) in enumerate(zip(_COL_LABELS, _COL_COLORS)):
            x = i * COL_GAP
            pill = QPainterPath()
            pill.addRoundedRect(QRectF(x, 4, CARD_W, 32), 10, 10)
            pill_item = self._scene.addPath(pill)
            bg = QColor(color)
            bg.setAlpha(28)
            pill_item.setBrush(QBrush(bg))
            pill_item.setPen(QPen(QColor(color), 1))
            text_item = self._scene.addText(label)
            text_item.setDefaultTextColor(QColor(color))
            text_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            text_item.setPos(x + 10, 7)

    # ---------------------------------------------------------------- public API

    def set_rows(self, rows: list[tuple[str, str, str]]) -> None:
        """Accepts simple (type, label, status) tuples — no parent info."""
        self._scene.clear()
        self._draw_column_headers()
        col_counts = [0 for _ in _COL_TYPES]
        for i, (ntype, label, status) in enumerate(rows):
            canon = _canonical_type(ntype)
            col = _COL_TYPES.index(canon) if canon in _COL_TYPES else 0
            card = _NodeCard("", canon, label, status, i + 1)
            card.setPos(col * COL_GAP, HEADER_H + col_counts[col] * ROW_H)
            col_counts[col] += 1
            self._scene.addItem(card)
        self._fit_scene()

    def set_tree_nodes(self, nodes) -> None:
        """Accepts DebateNode list from manager.as_tree_nodes()."""
        tree = [
            {
                "node_id": n.node_id,
                "type": n.node_type,
                "label": n.label,
                "status": n.status,
                "parent_id": n.parent_id,
                "order": n.creation_order,
            }
            for n in nodes
        ]
        self.set_graph(tree, edges=None)

    def set_graph(self, tree: list[dict], edges: list[dict] | None = None) -> None:
        """Render graph from rich node list + optional explicit semantic edges."""
        self._scene.clear()
        self._draw_column_headers()
        id_to_card: dict[str, _NodeCard] = {}
        col_counts = [0 for _ in _COL_TYPES]

        sorted_nodes = sorted(tree, key=lambda d: int(d.get("order", 0) or 0))
        for node in sorted_nodes:
            ntype = str(node.get("type", "talking_point"))
            canon = _canonical_type(ntype)
            col = _COL_TYPES.index(canon) if canon in _COL_TYPES else 0
            card = _NodeCard(
                str(node.get("node_id", "")),
                canon,
                str(node.get("label", "")),
                str(node.get("status", "active")),
                int(node.get("order", 0) or 0),
            )
            card.setPos(col * COL_GAP, HEADER_H + col_counts[col] * ROW_H)
            col_counts[col] += 1
            self._scene.addItem(card)
            node_id = str(node.get("node_id", ""))
            if node_id:
                id_to_card[node_id] = card

        if edges:
            sorted_edges = sorted(edges, key=lambda e: int(e.get("order", 0) or 0))
            for edge in sorted_edges:
                src_id = str(edge.get("source_id", ""))
                dst_id = str(edge.get("target_id", ""))
                if src_id not in id_to_card or dst_id not in id_to_card:
                    continue
                p = id_to_card[src_id]
                c = id_to_card[dst_id]
                if p.col_idx < c.col_idx:
                    src, dst = p.center_right(), c.center_left()
                else:
                    src, dst = p.center_bottom(), c.center_top()
                self._scene.addItem(
                    _EdgeItem(
                        src,
                        dst,
                        relation=str(edge.get("relation", "elaborates")),
                        weight=float(edge.get("weight", 0.5) or 0.5),
                    )
                )
        else:
            for node in sorted_nodes:
                parent_id = str(node.get("parent_id", ""))
                node_id = str(node.get("node_id", ""))
                if not parent_id or parent_id not in id_to_card or node_id not in id_to_card:
                    continue
                p = id_to_card[parent_id]
                c = id_to_card[node_id]
                if p.col_idx < c.col_idx:
                    src, dst = p.center_right(), c.center_left()
                else:
                    src, dst = p.center_bottom(), c.center_top()
                self._scene.addItem(_EdgeItem(src, dst, relation="elaborates", weight=0.5))

        self._fit_scene()

    def set_rows_rich(self, tree: list[dict], edges: list[dict] | None = None) -> None:
        """Accepts rich node dicts and optional edge dicts from orchestrator payload."""
        if tree and isinstance(tree[0], dict) and "node_id" in tree[0]:
            self.set_graph(tree, edges=edges)
            return
        self.set_rows([(d["type"], d["label"], d["status"]) for d in tree])

    def clear_rows(self) -> None:
        self._node_data = []
        self._scene.clear()

    # ---------------------------------------------------------------- helpers

    def _fit_scene(self) -> None:
        r = self._scene.itemsBoundingRect()
        r.adjust(-SCENE_PAD, -SCENE_PAD, SCENE_PAD, SCENE_PAD)
        self._scene.setSceneRect(r)
        # Scroll to bottom so newest nodes are visible
        self._view.ensureVisible(0.0, float(r.height()), 1.0, 1.0)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """Ctrl + scroll = zoom; plain scroll = pan (handled by QGraphicsView)."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._view.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)


def _canonical_type(node_type: str) -> str:
    key = (node_type or "").strip().lower()
    return _TYPE_ALIAS.get(key, key)
