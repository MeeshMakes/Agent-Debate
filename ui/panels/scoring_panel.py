"""Scoring Panel — live per-turn score log + final debate verdict.

Layout
------
  Top section  : scrollable live score feed (turn rows + arbiter interventions)
  Bottom strip : winner banner + AI-written summary (appears after debate ends)
  Button row   : Copy Results | Open Session Folder
"""
from __future__ import annotations

import os
import subprocess
from html import escape

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


# Composite score → colour
def _score_color(score: float) -> str:
    if score >= 0.65:
        return "#69f0ae"   # green
    if score >= 0.45:
        return "#ffd740"   # amber
    return "#ef9a9a"       # red


def _score_dot(score: float) -> str:
    if score >= 0.65:
        return "🟢"
    if score >= 0.45:
        return "🟡"
    return "🔴"


_AGENT_COLORS = {
    "astra": "#00e5ff",
    "nova":  "#ff6e40",
}


class ScoringPanel(QWidget):
    """Debate scoring tab — live feed + final verdict."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("scoringPanel")

        self._session_path: str = ""
        self._result_text: str = ""    # plain text for copy

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Header row
        hdr_row = QHBoxLayout()
        hdr = QLabel("⬥  SCORING  &  VERDICT")
        hdr.setStyleSheet(
            "font-size: 10pt; font-weight: 700; color: #ffd740; letter-spacing: 2px;"
        )
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        # Legend
        for label, color in (("🟢 ≥0.65", "#69f0ae"), ("🟡 ≥0.45", "#ffd740"), ("🔴 <0.45", "#ef9a9a")):
            l = QLabel(label)
            l.setStyleSheet(f"color: {color}; font-size: 8pt;")
            hdr_row.addWidget(l)
        root.addLayout(hdr_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Live score feed (left) ---
        feed_container = QWidget()
        feed_lay = QVBoxLayout(feed_container)
        feed_lay.setContentsMargins(0, 0, 0, 0)
        feed_lay.setSpacing(3)
        feed_hdr = QLabel("\u25b8  SCORE LOG")
        feed_hdr.setStyleSheet(
            "font-size: 8.5pt; font-weight: 700; color: #546e7a;"
            " letter-spacing: 1.5px; padding: 2px 0;"
        )
        feed_lay.addWidget(feed_hdr)

        self._feed = QTextBrowser()
        self._feed.setObjectName("scoreFeed")
        self._feed.setReadOnly(True)
        self._feed.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self._feed.setStyleSheet(
            "QTextBrowser { border: 1px solid #2a3a55; border-radius: 8px;"
            " background: #080d14; color: #cfd8dc; font-size: 9pt; }"
        )
        self._feed.setPlaceholderText("Score log will appear here during the debate\u2026")
        feed_lay.addWidget(self._feed)
        splitter.addWidget(feed_container)

        # --- Verdict / Summary area (right) ---
        verdict_container = QWidget()
        verdict_lay = QVBoxLayout(verdict_container)
        verdict_lay.setContentsMargins(0, 0, 0, 0)
        verdict_lay.setSpacing(3)
        verdict_hdr = QLabel("\u25b8  VERDICT")
        verdict_hdr.setStyleSheet(
            "font-size: 8.5pt; font-weight: 700; color: #546e7a;"
            " letter-spacing: 1.5px; padding: 2px 0;"
        )
        verdict_lay.addWidget(verdict_hdr)

        self._verdict = QTextBrowser()
        self._verdict.setObjectName("verdictArea")
        self._verdict.setReadOnly(True)
        self._verdict.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self._verdict.setStyleSheet(
            "QTextBrowser { border: 1px solid #37474f; border-radius: 8px;"
            " background: #0a0f1a; color: #e8eef9; font-size: 9.5pt; }"
        )
        self._verdict.setPlaceholderText("Final verdict will appear here after the debate ends\u2026")
        verdict_lay.addWidget(self._verdict)
        splitter.addWidget(verdict_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStyleSheet("QSplitter::handle { background: #1a3a55; width: 3px; }")
        root.addWidget(splitter, stretch=1)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._copy_btn = QPushButton("📋 Copy Results")
        self._copy_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #80cbc4; border-radius: 6px;"
            " padding: 5px 12px; border: 1px solid #2a3a55; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        self._copy_btn.clicked.connect(self._copy_results)
        btn_row.addWidget(self._copy_btn)

        self._folder_btn = QPushButton("📂 Open Session Folder")
        self._folder_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #80deea; border-radius: 6px;"
            " padding: 5px 12px; border: 1px solid #2a3a55; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        self._folder_btn.clicked.connect(self._open_folder)
        btn_row.addWidget(self._folder_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

    # ---------------------------------------------------------------- public API

    def new_session(self, session_path: str = "") -> None:
        """Reset for a new debate session."""
        self._session_path = session_path
        self._result_text = ""
        self._feed.clear()
        self._verdict.clear()
        self._verdict.setPlaceholderText("Final verdict will appear here after the debate ends…")

    def set_session_path(self, path: str) -> None:
        self._session_path = path

    def add_turn_score(
        self,
        turn: int,
        agent: str,
        composite: float,
        relevance: float,
        novelty: float,
        evidence: float,
    ) -> None:
        """Append one score row to the live feed."""
        agent_color = _AGENT_COLORS.get(agent.lower(), "#e0e0e0")
        dot = _score_dot(composite)
        sc = _score_color(composite)

        html = (
            f"<div style='margin:3px 0; padding:3px 8px; "
            f"border-left:3px solid {agent_color}; background:#0c1622; border-radius:0 6px 6px 0;'>"
            f"<span style='color:#546e7a; font-size:8pt; margin-right:6px;'>T{turn}</span>"
            f"<span style='color:{agent_color}; font-weight:700; font-size:8.5pt;'>{escape(agent)}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{sc}; font-weight:700;'>{dot} {composite:.3f}</span>"
            f"<span style='color:#546e7a; font-size:7.5pt; margin-left:8px;'>"
            f"rel={relevance:.2f} nov={novelty:.2f} evi={evidence:.2f}</span>"
            f"</div>"
        )
        self._feed.append(html)
        # Accumulate plain text
        self._result_text += (
            f"T{turn} {agent}: {composite:.3f}  "
            f"(rel={relevance:.2f} nov={novelty:.2f} evi={evidence:.2f})\n"
        )

    def add_arbiter_event(self, turn: int, message: str, is_echo: bool = False) -> None:
        """Append an arbiter intervention note to the live feed."""
        icon = "⚠" if is_echo else "⬥"
        color = "#ef9a9a" if is_echo else "#ffd740"
        html = (
            f"<div style='margin:3px 0; padding:3px 8px; "
            f"border-left:3px solid {color}; background:#14110a; border-radius:0 6px 6px 0;'>"
            f"<span style='color:#546e7a; font-size:8pt; margin-right:4px;'>T{turn}</span>"
            f"<span style='color:{color}; font-size:8.5pt;'>{icon} Arbiter: "
            f"{escape(message[:200])}</span>"
            f"</div>"
        )
        self._feed.append(html)
        self._result_text += f"T{turn} [ARBITER]: {message}\n"

    def set_verdict(
        self,
        winner: str,
        margin: str,
        reason: str,
        summary: str,
        astra_avg: float,
        nova_avg: float,
    ) -> None:
        """Display the final debate verdict."""
        winner_color = _AGENT_COLORS.get(winner.lower(), "#ffd740")
        margin_display = {"clear": "Clear Victory", "close": "Close Victory", "draw": "Draw"}.get(
            margin.lower(), margin
        )

        # Score comparison bar
        a_bar = int(astra_avg * 100)
        n_bar = int(nova_avg * 100)

        html = (
            f"<div style='margin:6px 0;'>"
            # Winner banner
            f"<div style='background:{winner_color}22; border:2px solid {winner_color};"
            f"border-radius:10px; padding:8px 16px; margin-bottom:8px;'>"
            f"<span style='color:{winner_color}; font-size:14pt; font-weight:800;'>"
            f"🏆 {escape(winner)} wins</span>"
            f"<span style='color:#b0bec5; font-size:9pt; margin-left:12px;'>{margin_display}</span>"
            f"</div>"
            # Score comparison
            f"<div style='margin:4px 0;'>"
            f"<span style='color:#00e5ff; font-size:9pt;'>Astra avg: {astra_avg:.3f}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#ff6e40; font-size:9pt;'>Nova avg: {nova_avg:.3f}</span>"
            f"</div>"
            # Reason
            f"<div style='color:#e0e0e0; font-size:9.5pt; margin:6px 0; font-style:italic;'>"
            f"{escape(reason)}</div>"
            # Summary
            f"<div style='color:#b0bec5; font-size:9pt; line-height:1.5; margin-top:6px; "
            f"border-top:1px solid #2a3a55; padding-top:6px;'>{escape(summary)}</div>"
            f"</div>"
        )
        self._verdict.setHtml(html)

        # Append to plain text export
        self._result_text += (
            f"\n{'='*60}\n"
            f"WINNER: {winner} ({margin_display})\n"
            f"Astra avg score: {astra_avg:.3f}   Nova avg score: {nova_avg:.3f}\n"
            f"Reason: {reason}\n\n"
            f"Summary:\n{summary}\n"
        )

    def set_summary_generating(self) -> None:
        """Show a 'generating…' placeholder in the verdict area."""
        self._verdict.setHtml(
            "<div style='color:#546e7a; font-style:italic; padding:12px;'>"
            "⏳ Generating final verdict — Arbiter is reviewing the debate…</div>"
        )

    # ---------------------------------------------------------------- internal

    def _copy_results(self) -> None:
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._result_text)

    def _open_folder(self) -> None:
        if self._session_path and os.path.isdir(self._session_path):
            try:
                os.startfile(self._session_path)  # Windows
            except AttributeError:
                subprocess.Popen(["xdg-open", self._session_path])
