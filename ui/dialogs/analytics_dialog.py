"""AnalyticsDialog — full-screen analytics panel for all past debates.

Layout
------
  ┌──────────────────────────────────────────────────────────────────┐
  │ [📊 Analytics] — Session History & Knowledge Bank              X │
  ├──────────────────────────────────────────────────────────────────┤
  │ [Stat badges — Debates | Turns | Facts | Truths | Problems]      │
  ├──────────────┬───────────────────────────────────────────────────┤
  │              │                                                   │
  │  Session     │  Detail — Topic / Winner / Scores / Summary       │
  │  List        │                                                   │
  │  (left)      │  Score Timeline (simple ASCII/HTML bar chart)     │
  │              │                                                   │
  └──────────────┴───────────────────────────────────────────────────┘
  │  [Win Rates bar]     [📂 Open Session]  [🔄 Refresh]            │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
    QFrame, QScrollArea, QTabWidget,
)

from analytics.analytics_store import AnalyticsRow, get_analytics_store
from core.segue_buffer import get_segue_buffer, SegueEntry

_ASTRA_COLOR = "#29b6f6"
_NOVA_COLOR  = "#ff8a65"
_WIN_COLORS  = {
    "Astra": _ASTRA_COLOR,
    "Nova":  _NOVA_COLOR,
    "Draw":  "#aaaaaa",
    "—":     "#555555",
}

_DIALOG_BG      = "#0e1117"
_PANEL_BG       = "#141920"
_CARD_BG        = "#1c2430"
_BORDER_COLOR   = "#2a3545"
_TEXT_MAIN      = "#dce8f5"
_TEXT_SUB       = "#7a9ab5"


class AnalyticsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("📊 Analytics — Debate History")
        self.setMinimumSize(960, 640)
        self.resize(1140, 720)
        self.setStyleSheet(f"""
            QDialog {{ background: {_DIALOG_BG}; color: {_TEXT_MAIN}; font-family: "Segoe UI"; }}
            QSplitter::handle {{ background: {_BORDER_COLOR}; }}
            QListWidget {{
                background: {_PANEL_BG};
                border: 1px solid {_BORDER_COLOR};
                border-radius: 8px;
                color: {_TEXT_MAIN};
                font-size: 11pt;
                outline: 0;
            }}
            QListWidget::item:selected {{
                background: #1e3a5f;
                border-radius: 6px;
            }}
            QListWidget::item {{ padding: 6px 10px; border-radius: 4px; }}
            QListWidget::item:hover {{ background: #1a2d45; }}
            QTextBrowser {{
                background: {_PANEL_BG};
                border: 1px solid {_BORDER_COLOR};
                border-radius: 8px;
                color: {_TEXT_MAIN};
                font-size: 10pt;
                padding: 12px;
            }}
            QPushButton {{
                background: #1c2d3f;
                color: {_TEXT_MAIN};
                border: 1px solid {_BORDER_COLOR};
                border-radius: 8px;
                padding: 6px 16px;
                font-size: 10pt;
            }}
            QPushButton:hover {{ background: #263d54; }}
            QPushButton:pressed {{ background: #1a2d45; }}
        """)

        self._rows: List[AnalyticsRow] = []
        self._selected_row: AnalyticsRow | None = None

        self._build()
        self._load()

    # ─────────────────────────── build layout ───────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Title bar
        title = QLabel("📊 Debate Analytics & History")
        title.setStyleSheet(f"color: {_TEXT_MAIN}; font-size: 15pt; font-weight: 700;")
        root.addWidget(title)

        # ── Stat badges row ──
        self._stat_bar = QWidget()
        self._stat_bar.setStyleSheet("background: transparent;")
        stat_hl = QHBoxLayout(self._stat_bar)
        stat_hl.setContentsMargins(0, 0, 0, 0)
        stat_hl.setSpacing(8)
        self._badges: dict[str, QLabel] = {}
        for key in ("Debates", "Turns", "Facts (Astra)", "Facts (Nova)", "Truths", "Problems"):
            lbl = QLabel(f"{key}\n—")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"""
                QLabel {{
                    background: {_CARD_BG};
                    color: {_TEXT_MAIN};
                    border: 1px solid {_BORDER_COLOR};
                    border-radius: 10px;
                    padding: 8px 14px;
                    font-size: 9pt;
                    line-height: 1.6;
                }}
            """)
            stat_hl.addWidget(lbl)
            self._badges[key] = lbl
        stat_hl.addStretch()
        root.addWidget(self._stat_bar)

        # ── Tab widget: Sessions | Knowledge Gaps ──
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {_DIALOG_BG};
                border: none;
            }}
            QTabBar::tab {{
                background: {_CARD_BG};
                color: {_TEXT_SUB};
                border: 1px solid {_BORDER_COLOR};
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                padding: 8px 18px;
                font-size: 10pt;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {_PANEL_BG};
                color: {_TEXT_MAIN};
            }}
            QTabBar::tab:hover {{
                background: #1a2d45;
            }}
        """)

        # ── Tab 1: Sessions ──
        sessions_tab = QWidget()
        st_lay = QVBoxLayout(sessions_tab)
        st_lay.setContentsMargins(0, 6, 0, 0)
        st_lay.setSpacing(6)

        # ── Main splitter ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # Left: session list
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        list_lbl = QLabel("Session History")
        list_lbl.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 10pt; font-weight: 600;")
        lv.addWidget(list_lbl)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)
        lv.addWidget(self._list)
        splitter.addWidget(left)

        # Right: detail
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        detail_lbl = QLabel("Session Detail")
        detail_lbl.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 10pt; font-weight: 600;")
        rv.addWidget(detail_lbl)
        self._detail = QTextBrowser()
        self._detail.setOpenExternalLinks(False)
        rv.addWidget(self._detail)
        splitter.addWidget(right)

        splitter.setSizes([320, 740])
        st_lay.addWidget(splitter)
        self._tabs.addTab(sessions_tab, "📊 Sessions")

        # ── Tab 2: Knowledge Gaps ──
        gaps_tab = QWidget()
        gt_lay = QVBoxLayout(gaps_tab)
        gt_lay.setContentsMargins(12, 12, 12, 12)
        gt_lay.setSpacing(10)

        gaps_header = QLabel("Knowledge Gaps — Unresolved Segue Concepts")
        gaps_header.setStyleSheet(
            f"color: #ce93d8; font-size: 12pt; font-weight: 700;"
        )
        gt_lay.addWidget(gaps_header)

        gaps_sub = QLabel(
            "These concepts appeared repeatedly in debates but were never fully explored. "
            "Promote them to create new talking points, or dismiss them."
        )
        gaps_sub.setWordWrap(True)
        gaps_sub.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 9pt;")
        gt_lay.addWidget(gaps_sub)

        gaps_scroll = QScrollArea()
        gaps_scroll.setWidgetResizable(True)
        gaps_scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: {_DIALOG_BG}; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: #2a3a55; border-radius: 3px; }}"
        )
        self._gaps_container = QWidget()
        self._gaps_container.setStyleSheet("background: transparent;")
        self._gaps_layout = QVBoxLayout(self._gaps_container)
        self._gaps_layout.setContentsMargins(0, 0, 0, 0)
        self._gaps_layout.setSpacing(8)
        self._gaps_layout.addStretch()
        gaps_scroll.setWidget(self._gaps_container)
        gt_lay.addWidget(gaps_scroll, stretch=1)

        # Refresh button
        gaps_btn_row = QHBoxLayout()
        gaps_refresh = QPushButton("🔄 Refresh Gaps")
        gaps_refresh.clicked.connect(self._load_gaps)
        gaps_btn_row.addStretch()
        gaps_btn_row.addWidget(gaps_refresh)
        gt_lay.addLayout(gaps_btn_row)

        self._tabs.addTab(gaps_tab, "🔍 Knowledge Gaps")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        root.addWidget(self._tabs)

        # ── Button row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._win_bar = QLabel()
        self._win_bar.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 10pt;")
        btn_row.addWidget(self._win_bar)
        btn_row.addStretch()

        self._open_btn = QPushButton("📂 Open Session Folder")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_session_folder)
        btn_row.addWidget(self._open_btn)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._load)
        btn_row.addWidget(refresh_btn)

        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        root.addLayout(btn_row)

    # ─────────────────────────── data loading ───────────────────────────

    def _load(self) -> None:
        store = get_analytics_store()
        self._rows = store.all_sessions()
        totals = store.knowledge_totals()
        wins   = store.win_counts()

        # Update badges
        self._badges["Debates"].setText(f"Debates\n{totals['debates']}")
        self._badges["Turns"].setText(f"Turns\n{totals['turns']}")
        self._badges["Facts (Astra)"].setText(f"Facts (Astra)\n{totals['left_facts']}")
        self._badges["Facts (Nova)"].setText(f"Facts (Nova)\n{totals['right_facts']}")
        self._badges["Truths"].setText(f"Truths\n{totals['truths']}")
        self._badges["Problems"].setText(f"Problems\n{totals['problems']}")

        # Win bar text
        parts = [
            f"<span style='color:{_ASTRA_COLOR};font-weight:700;'>Astra {wins.get('Astra', 0)}</span>",
            f"<span style='color:{_NOVA_COLOR};font-weight:700;'>Nova {wins.get('Nova', 0)}</span>",
            f"<span style='color:#aaa;'>Draw {wins.get('Draw', 0)}</span>",
        ]
        self._win_bar.setText("Win rates — " + " &nbsp;·&nbsp; ".join(parts))

        # Populate list
        self._list.clear()
        for row in self._rows:
            ts_str = _fmt_ts(row.ts)
            winner_color = _WIN_COLORS.get(row.winner, "#aaa")
            display = f"{row.topic[:48]}…" if len(row.topic) > 48 else row.topic
            item = QListWidgetItem(f"[{ts_str}]  {display}")
            item.setForeground(
                __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(_TEXT_MAIN)
            )
            item.setToolTip(f"Winner: {row.winner}  |  {row.astra_avg:.2f} vs {row.nova_avg:.2f}")
            self._list.addItem(item)

        if self._rows:
            self._list.setCurrentRow(0)

    # ─────────────────────────── detail panel ───────────────────────────

    def _on_row_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._rows):
            self._detail.clear()
            self._open_btn.setEnabled(False)
            return
        row = self._rows[index]
        self._selected_row = row
        self._open_btn.setEnabled(bool(row.session_path and os.path.isdir(row.session_path)))
        self._detail.setHtml(self._render_detail(row))

    def _render_detail(self, r: AnalyticsRow) -> str:
        winner_color = _WIN_COLORS.get(r.winner, "#aaa")
        ts_str = _fmt_ts(r.ts)
        status_color = {"complete": "#66bb6a", "stopped": "#ffa726", "running": "#29b6f6"}.get(
            r.status, "#888"
        )

        # Build sub-topics list
        sub_topics_html = ""
        if r.sub_topics:
            items = "".join(
                f"<li style='color:{_TEXT_SUB};'>{st}</li>" for st in r.sub_topics[:8]
            )
            sub_topics_html = f"<ul style='margin:4px 0 0 20px;padding:0;'>{items}</ul>"

        # Score bars (simple proportional width)
        astra_pct = min(int(r.astra_avg * 100), 100)
        nova_pct  = min(int(r.nova_avg  * 100), 100)

        bar_astra = (
            f"<div style='background:{_CARD_BG};border-radius:4px;height:10px;width:100%;'>"
            f"<div style='background:{_ASTRA_COLOR};width:{astra_pct}%;height:10px;border-radius:4px;'></div></div>"
        )
        bar_nova = (
            f"<div style='background:{_CARD_BG};border-radius:4px;height:10px;width:100%;'>"
            f"<div style='background:{_NOVA_COLOR};width:{nova_pct}%;height:10px;border-radius:4px;'></div></div>"
        )

        summary_html = ""
        if r.summary:
            summary_html = f"""
            <div style='background:{_CARD_BG};border-left:3px solid #4a6b8a;border-radius:6px;
                        padding:10px 14px;margin-top:12px;'>
              <div style='color:{_TEXT_SUB};font-size:9pt;font-weight:600;margin-bottom:6px;'>
                AI SUMMARY
              </div>
              <div style='color:{_TEXT_MAIN};font-size:10pt;line-height:1.6;'>
                {r.summary}
              </div>
            </div>"""

        reason_html = ""
        if r.reason:
            reason_html = f"""
            <div style='color:{_TEXT_SUB};font-size:9pt;margin-top:8px;font-style:italic;'>
              Verdict reason: {r.reason}
            </div>"""

        html = f"""
        <html><body style='font-family:"Segoe UI";background:{_DIALOG_BG};color:{_TEXT_MAIN};
                           margin:0;padding:0;'>
        <div style='padding:4px 2px 16px 2px;'>

          <!-- Header -->
          <div style='font-size:14pt;font-weight:700;color:{_TEXT_MAIN};margin-bottom:4px;'>
            {r.topic}
          </div>
          <div style='font-size:9pt;color:{_TEXT_SUB};margin-bottom:14px;'>
            {ts_str} &nbsp;·&nbsp;
            <span style='color:{status_color};'>{r.status}</span> &nbsp;·&nbsp;
            {r.turns} turns &nbsp;·&nbsp; sid: {r.session_id}
          </div>

          <!-- Winner banner -->
          <div style='background:{_CARD_BG};border-radius:10px;padding:14px 18px;
                      border:1px solid {_BORDER_COLOR};margin-bottom:12px;'>
            <span style='font-size:12pt;font-weight:700;color:{winner_color};'>
              🏆 Winner: {r.winner}
            </span>
            <span style='color:{_TEXT_SUB};font-size:10pt;margin-left:12px;'>margin: {r.margin}</span>
          </div>

          <!-- Score bars -->
          <div style='background:{_CARD_BG};border-radius:10px;padding:14px 18px;
                      border:1px solid {_BORDER_COLOR};margin-bottom:12px;'>
            <div style='color:{_TEXT_SUB};font-size:9pt;font-weight:600;margin-bottom:10px;'>
              AVERAGE SCORES
            </div>
            <div style='margin-bottom:8px;'>
              <span style='color:{_ASTRA_COLOR};font-weight:700;'>Astra</span>
              <span style='color:{_TEXT_SUB};float:right;'>{r.astra_avg:.3f}</span>
            </div>
            {bar_astra}
            <div style='margin-top:10px;margin-bottom:8px;'>
              <span style='color:{_NOVA_COLOR};font-weight:700;'>Nova</span>
              <span style='color:{_TEXT_SUB};float:right;'>{r.nova_avg:.3f}</span>
            </div>
            {bar_nova}
          </div>

          <!-- Facts / sub-topics -->
          <div style='background:{_CARD_BG};border-radius:10px;padding:12px 18px;
                      border:1px solid {_BORDER_COLOR};margin-bottom:12px;display:flex;
                      flex-wrap:wrap;gap:8px;'>
            <div style='color:{_TEXT_SUB};font-size:9pt;font-weight:600;margin-bottom:8px;'>
              SESSION STATS
            </div>
            <div style='font-size:10pt;color:{_TEXT_MAIN};'>
              Astra facts: <b>{r.left_facts}</b> &nbsp;&nbsp;
              Nova facts: <b>{r.right_facts}</b> &nbsp;&nbsp;
              Truths: <b style='color:#66bb6a;'>{r.truths}</b> &nbsp;&nbsp;
              Problems: <b style='color:#ef9a9a;'>{r.problems}</b>
            </div>
            {sub_topics_html if r.sub_topics else ""}
          </div>

          {summary_html}
          {reason_html}
        </div>
        </body></html>
        """
        return html

    # ─────────────────────────── actions ────────────────────────────────

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:  # Knowledge Gaps tab
            self._load_gaps()

    def _load_gaps(self) -> None:
        """Populate the Knowledge Gaps tab with segue buffer entries."""
        # Clear existing
        while self._gaps_layout.count() > 1:  # keep the stretch
            item = self._gaps_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        try:
            buf = get_segue_buffer()
            entries = buf.unresolved_entries()
        except Exception:
            entries = []

        if not entries:
            empty = QLabel("No unresolved knowledge gaps found.\n\n"
                          "Run some debates first — the system will detect "
                          "recurring lateral references automatically.")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 10pt; padding: 40px;")
            self._gaps_layout.insertWidget(0, empty)
            return

        for entry in entries[:50]:
            card = self._make_gap_card(entry)
            self._gaps_layout.insertWidget(self._gaps_layout.count() - 1, card)

    def _make_gap_card(self, entry: SegueEntry) -> QFrame:
        """Create a card widget for a segue buffer entry."""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {_CARD_BG}; border: 1px solid {_BORDER_COLOR};"
            f" border-radius: 8px; }}"
        )
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        # Info column
        info_col = QVBoxLayout()
        info_col.setSpacing(2)

        concept_lbl = QLabel(entry.concept)
        concept_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        concept_lbl.setStyleSheet("color: #ce93d8; background: transparent;")
        info_col.addWidget(concept_lbl)

        meta_parts = [f"Mentioned {entry.mention_count}x"]
        if entry.source_agents:
            meta_parts.append(f"by {', '.join(list(entry.source_agents)[:3])}")
        if entry.source_sessions:
            meta_parts.append(f"in {len(entry.source_sessions)} session(s)")
        meta_lbl = QLabel(" · ".join(meta_parts))
        meta_lbl.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 9pt; background: transparent;")
        info_col.addWidget(meta_lbl)

        if entry.surrounding_context:
            ctx_lbl = QLabel(entry.surrounding_context[:120] + ("…" if len(entry.surrounding_context) > 120 else ""))
            ctx_lbl.setWordWrap(True)
            ctx_lbl.setStyleSheet(f"color: {_TEXT_SUB}; font-size: 8pt; font-style: italic; background: transparent;")
            info_col.addWidget(ctx_lbl)

        lay.addLayout(info_col, stretch=1)

        # Action buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        promote_btn = QPushButton("⬆ Promote")
        promote_btn.setStyleSheet(
            "QPushButton { background: #1a0d2a; color: #ce93d8; border: 1px solid #7e57c2;"
            " border-radius: 6px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #311b92; color: #fff; }"
        )
        _concept = entry.concept  # capture for closure
        promote_btn.clicked.connect(lambda _, c=_concept: self._promote_gap(c))
        btn_col.addWidget(promote_btn)

        dismiss_btn = QPushButton("✕ Dismiss")
        dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #546e7a; border: none;"
            " padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { color: #ef5350; }"
        )
        dismiss_btn.clicked.connect(lambda _, c=_concept: self._dismiss_gap(c))
        btn_col.addWidget(dismiss_btn)

        lay.addLayout(btn_col)
        return card

    def _promote_gap(self, concept: str) -> None:
        """Promote a segue concept to an autonomous talking point."""
        try:
            buf = get_segue_buffer()
            # Find the entry
            for e in buf.all_entries():
                if e.concept == concept:
                    from config.autonomous_topics import get_autonomous_store
                    store = get_autonomous_store()
                    result = {
                        "title": f"Exploring: {concept}",
                        "description": (
                            f"This concept appeared {e.mention_count} times across debates. "
                            f"Context: {e.surrounding_context[:200]}"
                        ),
                        "talking_points": [
                            f"What are the key implications of {concept}?",
                            f"How does {concept} relate to our previous debates?",
                            f"What evidence supports or challenges {concept}?",
                        ],
                        "segue_concept": concept,
                        "segue_mentions": e.mention_count,
                        "segue_sessions": list(e.source_sessions),
                    }
                    store.save(result)
                    buf.mark_promoted(concept)
                    break
        except Exception:
            pass
        self._load_gaps()

    def _dismiss_gap(self, concept: str) -> None:
        """Dismiss a segue concept."""
        try:
            buf = get_segue_buffer()
            buf.mark_dismissed(concept)
        except Exception:
            pass
        self._load_gaps()

    def _open_session_folder(self) -> None:
        if self._selected_row and self._selected_row.session_path:
            path = self._selected_row.session_path
            if os.path.isdir(path):
                try:
                    os.startfile(path)  # Windows
                except AttributeError:
                    import subprocess
                    subprocess.Popen(["xdg-open", path])


# ─────────────────── helpers ────────────────────────────

def _fmt_ts(ts: float) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"
