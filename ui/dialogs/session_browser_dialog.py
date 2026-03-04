"""Session Browser Dialog.

Pop-out window showing all past debate sessions with full details.
Supports: view transcript, delete session, replay session transcript.
"""
from __future__ import annotations

import json
from html import escape
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.session_manager import SessionManager, SessionMeta


class SessionBrowserDialog(QDialog):
    """Browse, preview, and manage past debate sessions."""

    # Emitted when user wants to replay a session transcript
    replay_requested = pyqtSignal(str)   # session_id

    def __init__(self, session_manager: SessionManager, parent=None) -> None:
        super().__init__(parent)
        self._sm = session_manager
        self.setWindowTitle("\u25B8 Past Debate Sessions")
        self.resize(780, 600)
        self.setMinimumSize(600, 400)
        self.setMaximumWidth(900)
        self.setModal(False)
        self._apply_style()
        self._build_ui()
        try:
            self._sm.backfill_all_session_briefs()
        except Exception:
            pass
        try:
            self._sm.backfill_all_session_diagnostics()
        except Exception:
            pass
        self._refresh_list()

    # ------------------------------------------------------------------
    # Build UI

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)

        header = QLabel("\u25B8  Debate Sessions")
        header.setStyleSheet(
            "color: #c0e0ff; font-size: 13pt; font-weight: 900; "
            "letter-spacing: 2px; padding: 4px 0 6px 0;"
        )
        sub = QLabel("click to preview  \u00b7  double-click to replay")
        sub.setStyleSheet("color: #546e7a; font-size: 9pt; padding: 0 0 4px 0;")
        main.addWidget(header)
        main.addWidget(sub)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter, stretch=1)

        # Left: session list
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        self._list = QListWidget()
        self._list.setObjectName("sessionList")
        self._list.setMinimumWidth(240)
        self._list.setMaximumWidth(320)
        self._list.setStyleSheet("""
            QListWidget { background: #0b1420; border: 1px solid #1a2a44;
                          border-radius: 8px; color: #cfd8dc; font-size: 10pt; }
            QListWidget::item { padding: 8px 10px; border-bottom: 1px solid #162030; }
            QListWidget::item:selected { background: #0e3a6e; color: #e0f0ff; }
            QListWidget::item:hover { background: #12243a; }
        """)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        lv.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._del_btn = QPushButton("🗑  Delete Session")
        self._del_btn.setStyleSheet(
            "QPushButton { background: #6a0000; color: #fff; border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:hover { background: #b71c1c; }"
        )
        self._del_btn.clicked.connect(self._on_delete)
        self._replay_btn = QPushButton("▶  Load & Replay")
        self._replay_btn.setStyleSheet(
            "QPushButton { background: #1a5276; color: #fff; border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:hover { background: #1565c0; }"
        )
        self._replay_btn.clicked.connect(self._on_replay)
        btn_row.addWidget(self._del_btn)
        btn_row.addWidget(self._replay_btn)
        lv.addLayout(btn_row)

        # Right: session detail preview
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self._detail_title = QLabel("Select a session to preview")
        self._detail_title.setStyleSheet(
            "color: #4dd0e1; font-size: 11pt; font-weight: 700; padding: 4px 0;"
        )
        self._detail_title.setWordWrap(True)
        rv.addWidget(self._detail_title)

        self._detail_browser = QTextBrowser()
        self._detail_browser.setStyleSheet(
            "QTextBrowser { background: #080e18; color: #b0bec5; border: 1px solid #1a2a44;"
            " border-radius: 8px; font-size: 10pt; padding: 10px; }"
        )
        self._detail_browser.setOpenExternalLinks(False)
        rv.addWidget(self._detail_browser)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([280, 480])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Bottom bar
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background: #263238; color: #90a4ae; border-radius: 6px; padding: 6px 16px; }"
            "QPushButton:hover { background: #37474f; color: #fff; }"
        )
        close_btn.clicked.connect(self.accept)
        bot = QHBoxLayout()
        bot.addStretch()
        bot.addWidget(close_btn)
        main.addLayout(bot)

    # ------------------------------------------------------------------
    # Data

    def _refresh_list(self) -> None:
        self._list.clear()
        self._sessions = self._sm.list_sessions()
        for meta in self._sessions:
            item = QListWidgetItem(self._format_list_item(meta))
            item.setData(Qt.ItemDataRole.UserRole, meta.session_id)
            color = {
                "complete": "#69f0ae",
                "stopped": "#ffd740",
                "running": "#4dd0e1",
            }.get(meta.status, "#90a4ae")
            item.setForeground(QColor(color))
            self._list.addItem(item)

        if not self._sessions:
            placeholder = QListWidgetItem("  No sessions found yet — start a debate!")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setForeground(QColor("#546e7a"))
            self._list.addItem(placeholder)

    def _format_list_item(self, meta: SessionMeta) -> str:
        status_icon = {"complete": "\u2713", "stopped": "\u25A0", "running": "\u25B6"}.get(meta.status, "\u00b7")
        ts = datetime.fromtimestamp(meta.start_ts).strftime("%b %d  %H:%M")
        turns = f"{meta.turn_count}t"
        facts = meta.left_facts + meta.right_facts
        return (
            f"{status_icon}  {_short_topic(meta.topic, 38)}\n"
            f"   {ts}  \u00b7  {turns}  \u00b7  {facts} facts"
        )

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sessions):
            return
        meta = self._sessions[row]
        self._detail_title.setText(f"\u25C6  {meta.topic}")
        self._detail_browser.setHtml(self._build_detail_html(meta))

    def _build_detail_html(self, meta: SessionMeta) -> str:
        start = datetime.fromtimestamp(meta.start_ts).strftime("%Y-%m-%d  %H:%M:%S")
        end = datetime.fromtimestamp(meta.end_ts).strftime("%H:%M:%S") if meta.end_ts else "—"
        duration = ""
        if meta.end_ts:
            secs = int(meta.end_ts - meta.start_ts)
            duration = f"  ({secs // 60}m {secs % 60}s)"
        subs_html = "".join(f"<li><span style='color:#80cbc4'>→</span> {s}</li>"
                            for s in meta.sub_topics[:10])
        subs_block = f"<ul style='margin:0;padding-left:16px'>{subs_html}</ul>" if subs_html else "<em style='color:#546e7a'>none</em>"

        score = self._sm.load_scoring_report(meta.session_id) or {}
        winner = escape(str(score.get("winner", "—")))
        margin = escape(str(score.get("margin", "—")))
        reason = escape(str(score.get("reason", "")))
        astra_avg = _fmt_score(score.get("astra_avg"))
        nova_avg = _fmt_score(score.get("nova_avg"))

        diagnostics = self._sm.load_session_diagnostics(meta.session_id)
        if not isinstance(diagnostics, dict):
            diagnostics = self._sm.build_session_diagnostics(meta.session_id, meta)
        diag_generated = escape(str(diagnostics.get("generated_at", ""))) if isinstance(diagnostics, dict) else ""
        diag_trigger = escape(str(diagnostics.get("trigger", ""))) if isinstance(diagnostics, dict) else ""
        flow_line = (
            f"public: {diagnostics.get('public_message_count', 0)}  ·  "
            f"private: {diagnostics.get('private_thought_count', 0)}  ·  "
            f"arbiter: {diagnostics.get('arbiter_count', 0)}  ·  "
            f"graph nodes: {diagnostics.get('graph_nodes', 0)}"
        )
        left_fact_type_html = _fact_type_block(diagnostics.get("left_fact_types", {}))
        right_fact_type_html = _fact_type_block(diagnostics.get("right_fact_types", {}))
        issues_html = _issues_block(diagnostics.get("issues", []))

        artifact_status = self._sm.get_session_artifact_status(meta.session_id)
        core_artifacts = [
            "session_meta.json",
            "transcript.jsonl",
            "astra_memory.json",
            "nova_memory.json",
            "graph.json",
            "scoring_report.json",
        ]
        optional_artifacts = [
            "session_diagnostics.json",
            "session_brief.json",
            "scoring_report.md",
            "adaptive_prompts.json",
            "ingested_dataset.json",
            "uploads/",
            "captures/",
        ]
        missing_core = [name for name in core_artifacts if not artifact_status.get(name, False)]
        health_color = "#69f0ae" if not missing_core else "#ff5252"
        health_text = "Complete" if not missing_core else f"Missing {len(missing_core)} core artifact(s)"
        core_artifacts_html = "".join(_artifact_line(name, artifact_status.get(name, False)) for name in core_artifacts)
        optional_artifacts_html = "".join(_artifact_line(name, artifact_status.get(name, False)) for name in optional_artifacts)

        # Load a few transcript lines for preview
        transcript_preview = escape(self._load_transcript_preview(meta.session_id))

        return f"""
<html><body style='background:#0a1018; color:#b0bec5; font-family:Segoe UI,sans-serif; font-size:12px; padding:6px'>
<table width='100%' cellspacing='4'>
  <tr><td style='color:#4dd0e1;width:30%'>Session ID</td><td style='color:#eceff1'>{escape(meta.session_id)}</td></tr>
  <tr><td style='color:#4dd0e1'>Status</td><td style='color:{_status_color(meta.status)}'>{meta.status.upper()}</td></tr>
  <tr><td style='color:#4dd0e1'>Started</td><td>{start}</td></tr>
  <tr><td style='color:#4dd0e1'>Ended</td><td>{end}{duration}</td></tr>
  <tr><td style='color:#4dd0e1'>Models</td>
      <td><span style='color:#00e5ff'>Astra: {escape(meta.left_model)}</span>  &nbsp;
          <span style='color:#ff6e40'>Nova: {escape(meta.right_model)}</span></td></tr>
  <tr><td style='color:#4dd0e1'>Turns</td><td>{meta.turn_count}</td></tr>
  <tr><td style='color:#4dd0e1'>Facts</td>
      <td><span style='color:#00e5ff'>{escape(meta.left_agent)}: {meta.left_facts}</span>  ·  
          <span style='color:#ff6e40'>{escape(meta.right_agent)}: {meta.right_facts}</span></td></tr>
  <tr><td style='color:#4dd0e1'>Truths</td><td style='color:#69f0ae'>✓ {meta.truths_count}</td></tr>
  <tr><td style='color:#4dd0e1'>Problems</td><td style='color:#ff5252'>✗ {meta.problems_count}</td></tr>
  <tr><td style='color:#4dd0e1'>Winner</td><td style='color:#eceff1'>{winner} <span style='color:#546e7a'>({margin})</span></td></tr>
  <tr><td style='color:#4dd0e1'>Score Avg</td><td><span style='color:#00e5ff'>Astra: {astra_avg}</span>  ·  <span style='color:#ff6e40'>Nova: {nova_avg}</span></td></tr>
  <tr><td style='color:#4dd0e1'>Artifacts</td><td style='color:{health_color}'>{health_text}</td></tr>
</table>
<div style='margin:2px 0 8px 0; color:#90a4ae; font-size:11px;'>{reason}</div>
<hr style='border:1px solid #1e2d42; margin:8px 0'>
<p style='color:#4dd0e1; margin:4px 0'>Debate flow diagnostics</p>
<div style='color:#90a4ae; font-size:11px; margin-bottom:6px;'>{flow_line}</div>
<div style='color:#607d8b; font-size:10px; margin-bottom:6px;'>generated: {diag_generated or '—'}  ·  trigger: {diag_trigger or '—'}</div>
<div style='display:flex; gap:24px; margin-bottom:4px;'>
    <div><div style='color:#80cbc4; font-size:11px; margin-bottom:2px;'>Astra fact types</div>{left_fact_type_html}</div>
    <div><div style='color:#ffab91; font-size:11px; margin-bottom:2px;'>Nova fact types</div>{right_fact_type_html}</div>
</div>
<p style='color:#4dd0e1; margin:4px 0'>Integrity checks</p>
{issues_html}
<hr style='border:1px solid #1e2d42; margin:8px 0'>
<p style='color:#4dd0e1; margin:4px 0'>Artifact checklist</p>
<div style='display:flex; gap:24px'>
  <div><div style='color:#80cbc4; font-size:11px; margin-bottom:2px;'>Core</div>{core_artifacts_html}</div>
  <div><div style='color:#607d8b; font-size:11px; margin-bottom:2px;'>Optional</div>{optional_artifacts_html}</div>
</div>
<hr style='border:1px solid #1e2d42; margin:8px 0'>
<p style='color:#4dd0e1; margin:4px 0'>Sub-topics explored</p>
{subs_block}
<hr style='border:1px solid #1e2d42; margin:8px 0'>
<p style='color:#4dd0e1; margin:4px 0'>Transcript preview</p>
<div style='background:#080e16; border-radius:4px; padding:6px; color:#90a4ae; white-space:pre-wrap'>{transcript_preview}</div>
</body></html>
"""

    def _load_transcript_preview(self, session_id: str, max_chars: int = 800) -> str:
        events = self._sm.load_session_transcript(session_id)
        lines: list[str] = []
        total = 0
        for ev in events:
            etype = ev.get("event_type")
            if etype == "public_message":
                p = ev.get("payload", {})
                agent = p.get("agent", "?")
                msg = str(p.get("message", ""))[:300]
                line = f"[{agent}] {msg}"
                lines.append(line)
                total += len(line)
                if total > max_chars:
                    break
            elif etype == "turn":
                p = ev.get("payload", {})
                agent = p.get("speaker", "?")
                msg = str(p.get("text", ""))[:300]
                line = f"[{agent}] {msg}"
                lines.append(line)
                total += len(line)
                if total > max_chars:
                    break
        return "\n\n".join(lines)[:max_chars] or "(no transcript yet)"

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._on_replay()

    def _on_replay(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._sessions):
            return
        meta = self._sessions[row]
        self.replay_requested.emit(meta.session_id)
        self.accept()

    def _on_delete(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._sessions):
            return
        meta = self._sessions[row]
        confirm = QMessageBox.question(
            self,
            "Delete Session",
            f"Permanently delete session:\n{meta.session_id}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._sm.delete_session(meta.session_id)
            self._refresh_list()
            self._detail_browser.clear()
            self._detail_title.setText("Session deleted")

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog { background: #080d14; }
            QLabel  { color: #b0bec5; }
            QSplitter::handle { background: #1a2a44; width: 2px; }
            QPushButton { font-size: 9pt; }
        """)


# ------------------------------------------------------------------
# Helpers

def _short_topic(topic: str, max_len: int = 55) -> str:
    return topic if len(topic) <= max_len else topic[:max_len - 1] + "…"


def _status_color(status: str) -> str:
    return {"complete": "#69f0ae", "stopped": "#ffd740", "running": "#4dd0e1"}.get(status, "#90a4ae")


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "—"


def _artifact_line(name: str, exists: bool) -> str:
    icon = "✓" if exists else "✗"
    color = "#69f0ae" if exists else "#ef5350"
    return f"<div style='color:{color}; font-size:11px;'>{icon} {escape(name)}</div>"


def _fact_type_block(fact_types: object) -> str:
    if not isinstance(fact_types, dict) or not fact_types:
        return "<div style='color:#546e7a; font-size:11px;'>none</div>"

    parts: list[str] = []
    for key in sorted(fact_types):
        try:
            val = int(fact_types[key])
        except Exception:
            continue
        parts.append(f"<div style='color:#90a4ae; font-size:11px;'>• {escape(str(key))}: {val}</div>")
    return "".join(parts) or "<div style='color:#546e7a; font-size:11px;'>none</div>"


def _issues_block(issues: object) -> str:
    if not isinstance(issues, list) or not issues:
        return "<div style='color:#69f0ae; font-size:11px;'>✓ No integrity issues detected</div>"

    rendered = "".join(
        f"<div style='color:#ef5350; font-size:11px;'>✗ {escape(str(msg))}</div>"
        for msg in issues
    )
    return rendered or "<div style='color:#69f0ae; font-size:11px;'>✓ No integrity issues detected</div>"
