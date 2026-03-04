"""Topic Picker Dialog — full rich topic browser with inline Custom editor.

Replaces the dropdown. Opens as a large pop-out:
  Left column  — clickable topic cards (title + 2-line teaser)
  Right panel  — full description + talking points for selected topic
                 Editable when "Custom Debate" is selected; read-only for presets
    Bottom row   — file ingestion toggle + "Go to session" confirm

Emits: topic_confirmed(title: str, full_context: str, file_paths: list[str])
  full_context is the complete brief fed to agents.
"""
from __future__ import annotations

import json
import hashlib
from html import escape
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QDesktopServices

from config.starter_topics import STARTER_TOPICS, StarterTopic
from agents.topic_refine_worker import TopicRefineWorker
from core.custom_debates import CustomDebate, get_custom_debate_store
from core.repo_watchdog import RepoWatchdog
from core.session_manager import get_session_manager, SessionMeta
from ingestion.ingestion_agent import get_datasets_dir, list_global_datasets


_CUSTOM_TOPIC_TITLES = {"custom topic", "custom debate"}


def _is_custom_topic_title(title: str) -> bool:
    return title.strip().lower() in _CUSTOM_TOPIC_TITLES


# ─────────────────────────────────────────────
def _tp_key(topic_title: str, tp_text: str) -> str:
    """Return a stable 12-char hex key for a topic+talking-point pair."""
    raw = (topic_title + tp_text[:60]).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def _custom_debate_tp_key(debate_id: str) -> str:
    """Return a stable talking-point key for a persisted custom debate widget."""
    return f"custom_{debate_id[:12]}"


# ────────────────────────────────────────────────────────────────────────────────
#  AI Rewrite Worker
# ────────────────────────────────────────────────────────────────────────────────

class _RewriteWorker(QThread):
    """Background thread that calls the local LLM to rewrite talking points."""
    finished = pyqtSignal(str)   # improved talking points text
    failed   = pyqtSignal(str)   # error message

    def __init__(
        self,
        title: str,
        description: str,
        talking_points: str,
        model: str = "qwen3:30b",
        dataset_context: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._desc = description
        self._tp = talking_points
        self._model = model
        self._dataset_context = dataset_context

    def run(self) -> None:
        try:
            import httpx, json

            dataset_block = ""
            if self._dataset_context:
                dataset_block = (
                    "\n\nINGESTED KNOWLEDGE DATASET (use this to ground the talking points):\n"
                    f"{self._dataset_context[:3000]}"
                )

            prompt = (
                f"You are an expert debate architect and intellectual strategist.\n\n"
                f"DEBATE TOPIC: {self._title}\n"
                f"TOPIC DESCRIPTION:\n{self._desc[:800]}\n"
                f"CURRENT TALKING POINTS:\n{self._tp}"
                f"{dataset_block}\n\n"
                "YOUR TASK: Rewrite and dramatically improve the talking points above.\n"
                "Each talking point must be:\n"
                "  • A sharp, specific, contestable claim (not a question)\n"
                "  • Something debaters can argue for or against with evidence\n"
                "  • Different enough from the others to open a distinct branch\n"
                "  • Grounded in the ingestion data where relevant\n"
                "  • Written as one powerful complete sentence\n\n"
                "RETURN ONLY the improved talking points, one per line.\n"
                "No bullets, no numbers, no commentary, no introduction. Just the lines."
            )

            endpoint = "http://localhost:11434"
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            response = httpx.post(
                f"{endpoint}/api/chat",
                json=payload,
                timeout=180.0,
            )
            response.raise_for_status()
            data = response.json()
            # Extract text from chat completion format
            text = (
                data.get("message", {}).get("content", "")
                or data.get("response", "")
            ).strip()
            if not text:
                self.failed.emit("LLM returned an empty response.")
                return
            self.finished.emit(text)

        except Exception as exc:
            self.failed.emit(f"Rewrite failed: {exc}")


class _RepoSchemaWorker(QThread):
    """Background worker that prepares a repo-specific debate schema."""

    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        repo_path: str,
        repo_brief: str,
        user_intent: str = "",
        model: str = "qwen3:30b",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._repo_path = repo_path
        self._repo_brief = repo_brief
        self._user_intent = user_intent
        self._model = model

    def run(self) -> None:
        try:
            import httpx
            import re

            prompt = (
                "You are a Repository Debate Prep Agent. Your job is to read actual source code "
                "excerpts from a repository and produce a high-quality, repo-specific debate brief.\n\n"
                "Return JSON only, wrapped in ```json ... ``` with this exact shape:\n"
                "{\n"
                '  "title": "...",\n'
                '  "description": "...",\n'
                '  "talking_points": ["..."],\n'
                '  "prep_schema": {\n'
                '    "modules_to_probe": ["..."],\n'
                '    "risky_areas": ["..."],\n'
                '    "evidence_checklist": ["..."],\n'
                '    "watch_signals": ["..."]\n'
                "  }\n"
                "}\n\n"
                "CRITICAL QUALITY RULES — the brief MUST be grounded in the actual code provided:\n"
                "- title: concise, specific to this codebase — name the real system\n"
                "- description: 3-5 paragraphs. Describe the real architecture, real components, "
                "real data flows, real initialization sequences you can see in the excerpts. "
                "Name actual files, classes, and how they connect.\n"
                "- talking_points: 6-10 sharp contestable claims. Each must reference a REAL "
                "specific concern from the code (a specific method, class, data flow, or pattern "
                "you can actually see in the excerpts). Not generic questions — concrete assertions.\n"
                "- modules_to_probe: For each file you want to investigate, list the REAL method "
                "names you saw in the excerpts that are worth examining (e.g. "
                "'trainer.py — _on_task_failed(), _attempt_recovery()'). Use actual names from the code.\n"
                "- risky_areas: Describe specific failure modes using REAL code patterns you observed "
                "(e.g. 'main.py sequential init: if any single component raises an exception, all "
                "downstream references are set to None — silent degradation'). Cite actual code mechanics.\n"
                "- evidence_checklist: Each item must be a concrete task anchored to a REAL file + "
                "method + thing to verify (e.g. 'Read trainer.py _CROSS_SESSION_MAX_ATTEMPTS — "
                "verify fingerprint is written to RecoveryRecord before the count query'). "
                "Use actual names from the excerpts.\n"
                "- watch_signals: Real log messages, event names, exception types, or method calls "
                "from the code that indicate success or failure. Quote actual strings where you can.\n\n"
                "PRIORITY ORDER for understanding:\n"
                "  1) Identify real entry points and how execution begins (main.py, launch scripts, __init__ calls)\n"
                "  2) Map how modules call each other — trace the real import graph where visible\n"
                "  3) Identify real error paths — exception handling, fallback logic, None-checks\n"
                "  4) Note any constants, config flags, or thresholds that control behavior\n"
                "  5) Include at least one talking point about what currently works well\n\n"
                "DO NOT invent method names, file paths, or constants. Only use names you can "
                "actually see in the provided excerpts. If a section has nothing concrete from the "
                "code, mark it as 'Investigate: <file> — no excerpt available yet'.\n\n"
                "CONTEXT PRESERVATION (MANDATORY):\n"
                "- The user's intent and constraints are authoritative and must be preserved.\n"
                "- Reframe intelligently, but do not drop or replace the user's core concerns.\n"
                "- Description must include a section titled 'Preserved User Intent' with bullets.\n"
                "- At least 3 talking_points must directly reflect those preserved concerns while staying contestable.\n\n"
                "No markdown outside the JSON block.\n\n"
                f"User Intent and Existing Draft (must preserve):\n{self._user_intent[:9000]}\n\n"
                f"Repository Path:\n{self._repo_path}\n\n"
                f"Repository Snapshot Brief (includes source code excerpts):\n{self._repo_brief[:22000]}"
            )

            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            response = httpx.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=240.0,
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("message", {}).get("content", "")
                or data.get("response", "")
            ).strip()
            if not text:
                self.failed.emit("Repo prep agent returned empty output.")
                return

            match = re.search(r"```json\s*\n?(.*?)\n?```", text, re.DOTALL)
            raw = match.group(1).strip() if match else text
            result = json.loads(raw)

            title = str(result.get("title", "")).strip()
            description = str(result.get("description", "")).strip()
            talking_points = result.get("talking_points", [])
            prep_schema = result.get("prep_schema", {})

            if not title or not description or not isinstance(talking_points, list) or not talking_points:
                self.failed.emit("Repo prep agent returned incomplete schema data.")
                return

            self.finished.emit(
                {
                    "title": title,
                    "description": description,
                    "talking_points": [str(tp).strip() for tp in talking_points if str(tp).strip()],
                    "prep_schema": prep_schema if isinstance(prep_schema, dict) else {},
                }
            )
        except Exception as exc:
            self.failed.emit(f"Repo prep failed: {exc}")


class _RebuildDatasetWorker(QThread):
    """Rebuilds the repo watchdog semantic dataset from scratch."""
    finished = pyqtSignal(int)   # fact_count
    failed   = pyqtSignal(str)

    def __init__(self, *, repo_path: str, session_root, parent=None) -> None:
        super().__init__(parent)
        self._repo_path   = repo_path
        self._session_root = session_root

    def run(self) -> None:
        try:
            from core.repo_watchdog import RepoWatchdog
            wd = RepoWatchdog(self._session_root)
            wd.build_snapshot(self._repo_path)
            ds = wd.load_semantic_dataset(self._repo_path)
            fact_count = len((ds or {}).get("facts", []))
            self.finished.emit(fact_count)
        except Exception as exc:
            self.failed.emit(f"Rebuild failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    "bg":         "#080d14",
    "panel":      "#0a1018",
    "card":       "#0d1520",
    "card_sel":   "#0d2030",
    "border":     "#1e2d42",
    "border_sel": "#00acc1",
    "text":       "#cfd8dc",
    "dim":        "#546e7a",
    "accent":     "#00e5ff",
    "accent2":    "#ff6e40",
    "ok":         "#00695c",
    "ok_hover":   "#00796b",
}

_CARD_W = 290


# ──────────────────────────────────────────────────────────────────────────────
#  Topic card widget (left column)
# ──────────────────────────────────────────────────────────────────────────────

class _TopicCard(QFrame):
    clicked = pyqtSignal(int)   # emits index
    delete_requested = pyqtSignal(str)

    def __init__(
        self,
        index: int,
        *,
        title: str,
        description: str,
        talking_points_count: int = 0,
        repo_mode: bool = False,
        deletable: bool = False,
        debate_id: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._repo_mode = repo_mode
        self._deletable = deletable
        self._debate_id = debate_id
        self.setFixedWidth(_CARD_W)
        self.setObjectName("topicCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setWordWrap(True)
        title_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_color = "#ce93d8" if repo_mode else _PALETTE["accent"]
        title_lbl.setStyleSheet(f"color: {title_color}; background: transparent;")
        lay.addWidget(title_lbl)

        # Two-line teaser from description
        teaser = (description[:160] + "…") if len(description) > 160 else description
        teaser_lbl = QLabel(teaser)
        teaser_lbl.setWordWrap(True)
        teaser_lbl.setFont(QFont("Segoe UI", 9))
        teaser_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        lay.addWidget(teaser_lbl)

        if talking_points_count:
            suffix = "  ·  repo" if repo_mode else ""
            tp_count = QLabel(f"  {talking_points_count} talking points{suffix}")
            tp_count.setFont(QFont("Segoe UI", 8))
            tp_count_color = "#7e57c2" if repo_mode else "#2a7a8a"
            tp_count.setStyleSheet(f"color: {tp_count_color}; background: transparent;")
            lay.addWidget(tp_count)

        self._refresh_style()

    def _refresh_style(self) -> None:
        border_sel = "#9c27b0" if self._repo_mode else _PALETTE["border_sel"]
        border = "#2a1a3e" if self._repo_mode else _PALETTE["border"]
        card_bg = "#120d1e" if self._repo_mode else _PALETTE["card"]
        card_sel = "#1a0d2a" if self._repo_mode else _PALETTE["card_sel"]
        if self._selected:
            self.setStyleSheet(
                f"QFrame#topicCard {{ background: {card_sel};"
                f" border: 2px solid {border_sel}; border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#topicCard {{ background: {card_bg};"
                f" border: 1px solid {border}; border-radius: 10px; }}"
            )

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        if not self._deletable or not self._debate_id:
            return super().contextMenuEvent(event)

        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: #0d1520; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; font-size: 11px; }}"
            "QMenu::item:selected { background: #0d2030; }"
        )
        del_action = menu.addAction("🗑  Delete Debate Widget")
        chosen = menu.exec(event.globalPos())
        if chosen == del_action:
            self.delete_requested.emit(self._debate_id)


# ──────────────────────────────────────────────────────────────────────────────
#  Knowledge Library panel  (persistent ingested datasets)
# ──────────────────────────────────────────────────────────────────────────────

class _DatasetLibraryPanel(QWidget):
    """Small scrollable panel listing every dataset in sessions/_datasets/.

    Datasets are persistent user knowledge assets — independent of any session
    or talking-point thread.  They are NOT cleared by Reset Thread / Clear All.

    Controls
    --------
    📂 Open Folder   — reveals sessions/_datasets/ in Explorer
    🔄 Refresh       — re-scans disk
    Right-click row  — context menu with 🗑 Delete this dataset
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        # Top bar: count label + action buttons
        top = QHBoxLayout()
        top.setSpacing(6)
        self._count_lbl = QLabel("Knowledge Library — 0 datasets")
        self._count_lbl.setStyleSheet(
            f"color: {_PALETTE['accent']}; font-size: 10px; font-weight: 600;"
            " background: transparent;"
        )
        top.addWidget(self._count_lbl)
        top.addStretch()

        open_btn = QPushButton("📂  Open Folder")
        open_btn.setToolTip("Open the sessions/_datasets/ folder in Explorer")
        open_btn.setStyleSheet(
            f"QPushButton {{ background: #111a28; color: #78909c; border-radius: 5px;"
            f" padding: 3px 10px; font-size: 10px;"
            f" border: 1px solid {_PALETTE['border']}; }}"
            "QPushButton:hover { background: #1a2a40; color: #b0bec5; }"
        )
        open_btn.clicked.connect(self._on_open_folder)
        top.addWidget(open_btn)

        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Refresh dataset list from disk")
        refresh_btn.setFixedWidth(30)
        refresh_btn.setStyleSheet(
            f"QPushButton {{ background: #111a28; color: #546e7a; border-radius: 5px;"
            f" font-size: 11px; padding: 3px;"
            f" border: 1px solid {_PALETTE['border']}; }}"
            "QPushButton:hover { color: #00e5ff; background: #0d1e30; }"
        )
        refresh_btn.clicked.connect(self.refresh)
        top.addWidget(refresh_btn)
        lay.addLayout(top)

        # Scrollable dataset list
        self._list = QListWidget()
        self._list.setMaximumHeight(110)
        self._list.setMinimumHeight(52)
        self._list.setStyleSheet(
            f"QListWidget {{ background: {_PALETTE['bg']}; color: #b0bec5;"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 5px;"
            f" font-size: 10px; }}"
            "QListWidget::item { padding: 4px 6px; border-bottom: 1px solid #0d1828; }"
            "QListWidget::item:selected { background: #0d2030; color: #00e5ff; }"
            "QListWidget::item:hover { background: #0a1a28; }"
        )
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        lay.addWidget(self._list)

        self._no_data_lbl = QLabel(
            "No ingested datasets yet.  Use Browse above to add files or folders "
            "then start a debate — datasets are saved automatically."
        )
        self._no_data_lbl.setWordWrap(True)
        self._no_data_lbl.setStyleSheet(
            f"color: {_PALETTE['dim']}; font-size: 10px; font-style: italic;"
            " background: transparent;"
        )
        lay.addWidget(self._no_data_lbl)

        self.refresh()

    # ── public ──────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read sessions/_datasets/ and repopulate the list."""
        self._list.clear()
        sm = get_session_manager()
        datasets = list_global_datasets(sm.root)
        # sort newest first
        datasets.sort(key=lambda d: d.get("created", ""), reverse=True)

        for ds in datasets:
            name = ds["name"]
            created = ds.get("created", "")[:10]  # YYYY-MM-DD
            fact_count = ds.get("fact_count", 0)
            kws = ", ".join(ds.get("keywords", [])[:5]) or "—"
            label = f"  {name}"
            sub   = f"  {created}  ·  {fact_count} chunks  ·  {kws}"

            item = QListWidgetItem()
            item.setText(f"{label}\n{sub}")
            item.setToolTip(
                f"Dataset: {name}\n"
                f"Created: {ds.get('created', 'unknown')}\n"
                f"Chunks: {fact_count}\n"
                f"Keywords: {kws}\n"
                f"Path: {ds['path']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, ds["path"])
            self._list.addItem(item)

        count = len(datasets)
        self._count_lbl.setText(
            f"Knowledge Library — {count} dataset{'s' if count != 1 else ''}"
        )
        self._list.setVisible(count > 0)
        self._no_data_lbl.setVisible(count == 0)

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_open_folder(self) -> None:
        sm = get_session_manager()
        folder = get_datasets_dir(sm.root)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        name = item.text().split("\n")[0].strip()
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: #0d1520; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; font-size: 11px; }}"
            "QMenu::item:selected { background: #0d2030; }"
        )
        del_action = menu.addAction(f"🗑  Delete  '{name}'")
        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen == del_action:
            self._delete_dataset(path, name)

    def _delete_dataset(self, path: str, name: str) -> None:
        from pathlib import Path as _Path
        reply = QMessageBox.warning(
            self,
            "Delete Dataset",
            f"Permanently delete the dataset '{name}'?\n\n"
            "This removes the stored knowledge chunks from disk.\n"
            "The original source files you ingested are NOT affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            _Path(path).unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Delete Failed", str(exc))
        self.refresh()


# ──────────────────────────────────────────────────────────────────────────────
#  Drop zone for ingestion
# ──────────────────────────────────────────────────────────────────────────────

class _DropZone(QWidget):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(64)
        self._idle_style = (
            f"background: {_PALETTE['bg']}; border: 2px dashed {_PALETTE['border']};"
            " border-radius: 8px; color: #546e7a;"
        )
        self._hover_style = (
            f"background: #0d2030; border: 2px dashed {_PALETTE['border_sel']}; border-radius: 8px;"
        )
        self.setStyleSheet(self._idle_style)
        lay = QVBoxLayout(self)
        lbl = QLabel("⬇  Drag files or folders here to ingest as knowledge for this debate")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {_PALETTE['dim']}; font-size: 12px; border: none;")
        lay.addWidget(lbl)

    def dragEnterEvent(self, ev: QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, ev) -> None:
        self.setStyleSheet(self._idle_style)

    def dropEvent(self, ev: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        self.setStyleSheet(self._idle_style)
        if paths:
            self.files_dropped.emit(paths)


# ──────────────────────────────────────────────────────────────────────────────
#  Session history helper: individual session card
# ──────────────────────────────────────────────────────────────────────────────

class _SessionCard(QFrame):
    """Compact card representing one completed/stopped session."""

    clicked = pyqtSignal(str)  # emits session_id

    _STATUS_ICON = {"complete": "🟢", "stopped": "🟡", "running": "🔵"}

    def __init__(self, meta: SessionMeta, session_number: int, parent=None) -> None:
        super().__init__(parent)
        self._session_id = meta.session_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("sessionCard")
        self.setStyleSheet(
            f"QFrame#sessionCard {{ background: #0a1320; border: 1px solid #1e2d42;"
            " border-radius: 8px; }}"
            "QFrame#sessionCard:hover { border-color: #00acc1; background: #0d1a2a; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        status_icon = self._STATUS_ICON.get(meta.status, "⚫")
        ts_str = datetime.fromtimestamp(meta.start_ts).strftime("%d %b %Y  %H:%M") if meta.start_ts else ""

        # Top row: session # + status + timestamp
        top = QHBoxLayout()
        num_lbl = QLabel(f"Session {session_number}")
        num_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        num_lbl.setStyleSheet("color: #00e5ff; background: transparent;")
        status_lbl = QLabel(f"{status_icon} {meta.status}")
        status_lbl.setStyleSheet("color: #546e7a; font-size: 9px; background: transparent;")
        ts_lbl = QLabel(ts_str)
        ts_lbl.setStyleSheet("color: #3a5060; font-size: 9px; background: transparent;")
        top.addWidget(num_lbl)
        top.addStretch()
        top.addWidget(status_lbl)
        top.addSpacing(8)
        top.addWidget(ts_lbl)
        lay.addLayout(top)

        # Stats row: turns, truths, problems
        stats = QLabel(
            f"  {meta.turn_count} turns  ·  ✓ {meta.truths_count} truths  ·  ✗ {meta.problems_count} open"
        )
        stats.setStyleSheet("color: #546e7a; font-size: 9px; background: transparent;")
        lay.addWidget(stats)

        # Agents row
        agents_lbl = QLabel(f"  {meta.left_agent} vs {meta.right_agent}")
        agents_lbl.setStyleSheet("color: #2a6070; font-size: 9px; background: transparent;")
        lay.addWidget(agents_lbl)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._session_id)
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
#  Session history panel (right 3rd panel)
# ──────────────────────────────────────────────────────────────────────────────

class _SessionHistoryPanel(QWidget):
    """Scrollable list of past sessions for one talking point, plus Start button.

    Supports two thread-management actions:
      Reset Thread  — keeps all old session data on disk but starts counting
                      from Session 1 again (increments an epoch counter).
      Clear All Data — deletes every session folder and refinement file for
                      ALL epochs of this talking point, confirmation required.
    """

    session_clicked  = pyqtSignal(str)   # session_id
    start_requested  = pyqtSignal()      # user navigates back to session view
    refine_requested = pyqtSignal()      # manually update topic from past sessions
    thread_reset     = pyqtSignal()      # user successfully reset thread
    data_cleared     = pyqtSignal()      # user successfully cleared all data

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._base_tp_key: str = ""
        self._session_count: int = 0
        self._selected_generation: int | None = None
        self.setStyleSheet("background: #060e18;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        # ── Header ──────────────────────────────────────────────────────────
        self._header = QLabel("Session History")
        self._header.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._header.setStyleSheet("color: #00e5ff; background: transparent;")
        lay.addWidget(self._header)

        self._thread_lbl = QLabel("")
        self._thread_lbl.setStyleSheet("color: #2a6070; font-size: 9px; background: transparent;")
        lay.addWidget(self._thread_lbl)

        thread_row = QHBoxLayout()
        thread_row.setSpacing(6)
        self._thread_sel_lbl = QLabel("Thread:")
        self._thread_sel_lbl.setStyleSheet("color: #78909c; font-size: 10px; background: transparent;")
        self._thread_combo = QComboBox()
        self._thread_combo.setEnabled(False)
        self._thread_combo.setMinimumWidth(170)
        self._thread_combo.setStyleSheet(
            "QComboBox { background: #0a1520; color: #dce8f5; border: 1px solid #2a3a55;"
            " border-radius: 5px; padding: 3px 8px; font-size: 10px; }"
            "QComboBox:disabled { color: #546e7a; border-color: #1a2030; }"
            "QComboBox QAbstractItemView { background: #0d1520; color: #dce8f5;"
            " border: 1px solid #2a3a55; selection-background-color: #1e3a5f; }"
        )
        self._thread_combo.currentIndexChanged.connect(self._on_thread_changed)
        thread_row.addWidget(self._thread_sel_lbl)
        thread_row.addWidget(self._thread_combo)
        thread_row.addStretch()
        lay.addLayout(thread_row)

        self._sub = QLabel("No sessions yet for this talking point.")
        self._sub.setStyleSheet("color: #546e7a; font-size: 10px; background: transparent;")
        self._sub.setWordWrap(True)
        lay.addWidget(self._sub)

        # ── Start / Continue button ──────────────────────────────────────────
        self._start_btn = QPushButton("▶  Go to Session 1")
        self._start_btn.setStyleSheet(
            "QPushButton { background: #00695c; color: #fff; font-weight: 700;"
            " border-radius: 8px; padding: 8px 18px; font-size: 12px; border: 1px solid #00897b; }"
            "QPushButton:hover { background: #00796b; }"
        )
        self._start_btn.clicked.connect(self.start_requested)
        lay.addWidget(self._start_btn)

        self._refine_btn = QPushButton("🧠  Update from Past Sessions")
        self._refine_btn.setToolTip(
            "Manually regenerate this debate's description and talking points\n"
            "using all past sessions in the selected thread."
        )
        self._refine_btn.setEnabled(False)
        self._refine_btn.setStyleSheet(
            "QPushButton { background: #12304a; color: #b2ebf2; font-weight: 700;"
            " border-radius: 7px; padding: 6px 12px; font-size: 10px; border: 1px solid #1e4a70; }"
            "QPushButton:hover { background: #1a3a60; color: #ffffff; }"
            "QPushButton:disabled { color: #546e7a; border-color: #1a2030; background: #0a1520; }"
        )
        self._refine_btn.clicked.connect(self.refine_requested)
        lay.addWidget(self._refine_btn)

        # ── Thread management row ────────────────────────────────────────────
        mgmt_row = QHBoxLayout()
        mgmt_row.setSpacing(6)

        self._reset_btn = QPushButton("↺  New Thread")
        self._reset_btn.setToolTip(
            "Keep all existing session data, but start this talking point\n"
            "from Session 1 again in a clean thread.\n"
            "Old sessions stay on disk and are still viewable."
        )
        self._reset_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #80cbc4; border-radius: 6px;"
            " padding: 4px 10px; font-size: 10px; border: 1px solid #1e3a5f; }"
            "QPushButton:hover { background: #0d2a40; color: #00e5ff; border-color: #00acc1; }"
            "QPushButton:disabled { color: #2a3a55; border-color: #1a2030; }"
        )
        self._reset_btn.setEnabled(False)
        self._reset_btn.clicked.connect(self._on_reset_thread)

        self._clear_btn = QPushButton("🗑  Clear All")
        self._clear_btn.setToolTip(
            "Permanently delete ALL session folders and refinement data\n"
            "for EVERY thread of this talking point.\n"
            "This cannot be undone."
        )
        self._clear_btn.setStyleSheet(
            "QPushButton { background: #1a0a0a; color: #ef9a9a; border-radius: 6px;"
            " padding: 4px 10px; font-size: 10px; border: 1px solid #4a1010; }"
            "QPushButton:hover { background: #3a1010; color: #ef5350; border-color: #ef5350; }"
            "QPushButton:disabled { color: #2a1a1a; border-color: #1a0a0a; }"
        )
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._on_clear_data)

        mgmt_row.addWidget(self._reset_btn)
        mgmt_row.addWidget(self._clear_btn)
        mgmt_row.addStretch()
        lay.addLayout(mgmt_row)

        # ── Scroll area for session cards ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: #080d14; width: 5px; border-radius: 2px; }"
            "QScrollBar::handle:vertical { background: #1e2d42; border-radius: 2px; }"
        )
        self._card_widget = QWidget()
        self._card_widget.setStyleSheet("background: transparent;")
        self._card_col = QVBoxLayout(self._card_widget)
        self._card_col.setContentsMargins(0, 0, 0, 0)
        self._card_col.setSpacing(6)
        self._card_col.addStretch()
        scroll.setWidget(self._card_widget)
        lay.addWidget(scroll, stretch=1)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_sessions(self, base_tp_key: str, selected_generation: int | None = None) -> None:
        """Load sessions for base_tp_key and allow switching between all thread generations."""
        self._base_tp_key = base_tp_key

        sm = get_session_manager()

        # Reset thread selector when no talking-point key is active.
        if not base_tp_key:
            self._thread_combo.blockSignals(True)
            self._thread_combo.clear()
            self._thread_combo.blockSignals(False)
            self._thread_combo.setEnabled(False)
            self._selected_generation = None

            while self._card_col.count() > 1:
                item = self._card_col.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            self._session_count = 0
            self._thread_lbl.setText("")
            self._sub.setText("No sessions yet in this thread.")
            self._start_btn.setText("▶  Go to Session 1")
            self._refine_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)
            self._clear_btn.setEnabled(False)
            return

        latest_gen = sm.get_tp_generation(base_tp_key)

        # Populate thread selector (Thread 1 .. Thread N)
        self._thread_combo.blockSignals(True)
        self._thread_combo.clear()
        for g in range(latest_gen + 1):
            eff = base_tp_key if g == 0 else f"{base_tp_key}_g{g}"
            count = len(sm.list_sessions_for_tp(eff))
            suffix = "  (Current)" if g == latest_gen else ""
            self._thread_combo.addItem(f"Thread {g + 1} · {count} sessions{suffix}", g)
        self._thread_combo.blockSignals(False)
        self._thread_combo.setEnabled(self._thread_combo.count() > 0)

        if selected_generation is not None:
            target_gen = max(0, min(latest_gen, int(selected_generation)))
        elif self._selected_generation is not None and self._selected_generation <= latest_gen:
            target_gen = self._selected_generation
        else:
            target_gen = latest_gen

        idx = self._thread_combo.findData(target_gen)
        if idx < 0 and self._thread_combo.count() > 0:
            idx = self._thread_combo.count() - 1
        if idx >= 0:
            self._thread_combo.blockSignals(True)
            self._thread_combo.setCurrentIndex(idx)
            self._thread_combo.blockSignals(False)
        self._selected_generation = target_gen

        self._render_selected_thread_sessions()

    def _render_selected_thread_sessions(self) -> None:
        # Remove old cards (all items except the stretch at end)
        while self._card_col.count() > 1:
            item = self._card_col.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sm = get_session_manager()
        base_tp_key = self._base_tp_key
        if not base_tp_key:
            self._session_count = 0
            self._thread_lbl.setText("")
            self._sub.setText("No sessions yet in this thread.")
            self._start_btn.setText("▶  Go to Session 1")
            self._refine_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)
            self._clear_btn.setEnabled(False)
            return

        latest_gen = sm.get_tp_generation(base_tp_key)
        selected_gen = self._selected_generation if self._selected_generation is not None else latest_gen
        eff_key = base_tp_key if selected_gen == 0 else f"{base_tp_key}_g{selected_gen}"
        sessions = sm.list_sessions_for_tp(eff_key)
        self._session_count = len(sessions)

        if selected_gen == latest_gen:
            self._thread_lbl.setText(f"Viewing current thread (Thread {selected_gen + 1})")
        else:
            self._thread_lbl.setText(
                f"Viewing archived thread {selected_gen + 1}  ·  current is Thread {latest_gen + 1}"
            )

        if sessions:
            self._sub.setText(
                f"{self._session_count} session(s) in Thread {selected_gen + 1} — click to view transcript."
            )
        else:
            self._sub.setText(f"No sessions yet in Thread {selected_gen + 1}.")

        for idx, meta in enumerate(reversed(sessions)):
            card = _SessionCard(meta, idx + 1)
            card.clicked.connect(self.session_clicked)
            self._card_col.insertWidget(self._card_col.count() - 1, card)

        next_num = self._session_count + 1
        self._start_btn.setText(
            f"▶  Go to Session 1" if self._session_count == 0
            else f"▶  Go to Session {next_num}"
        )

        has_key = bool(base_tp_key)
        self._refine_btn.setEnabled(has_key and self._session_count > 0)
        self._reset_btn.setEnabled(has_key and selected_gen == latest_gen and self._session_count > 0)
        self._clear_btn.setEnabled(has_key)

    def set_refine_busy(self, busy: bool) -> None:
        if busy:
            self._refine_btn.setEnabled(False)
            self._refine_btn.setText("🧠  Updating from Sessions…")
            return
        self._refine_btn.setText("🧠  Update from Past Sessions")
        self._refine_btn.setEnabled(bool(self._base_tp_key) and self._session_count > 0)

    def _on_thread_changed(self, _index: int) -> None:
        if not self._base_tp_key:
            return
        data = self._thread_combo.currentData()
        if data is None:
            return
        try:
            self._selected_generation = int(data)
        except Exception:
            return
        self._render_selected_thread_sessions()

    def selected_effective_tp_key(self) -> str:
        if not self._base_tp_key:
            return ""
        sm = get_session_manager()
        latest_gen = sm.get_tp_generation(self._base_tp_key)
        selected_gen = self._selected_generation if self._selected_generation is not None else latest_gen
        selected_gen = max(0, min(latest_gen, selected_gen))
        return self._base_tp_key if selected_gen == 0 else f"{self._base_tp_key}_g{selected_gen}"

    def next_session_number(self) -> int:
        return self._session_count + 1

    # ── Slot handlers ────────────────────────────────────────────────────────

    def _on_reset_thread(self) -> None:
        if not self._base_tp_key:
            return
        reply = QMessageBox.question(
            self,
            "New Thread — Keep Data",
            "Start a fresh thread for this talking point?\n\n"
            "Your existing session data is kept on disk and"
            " will always be accessible. The session counter"
            " resets to Session 1 and the AI refinement is cleared"
            " so the topic presents cleanly again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        sm = get_session_manager()
        sm.reset_tp_thread(self._base_tp_key)
        self.load_sessions(self._base_tp_key, selected_generation=sm.get_tp_generation(self._base_tp_key))
        self.thread_reset.emit()

    def _on_clear_data(self) -> None:
        if not self._base_tp_key:
            return
        sm = get_session_manager()
        gen = sm.get_tp_generation(self._base_tp_key)
        total_sessions = sum(
            len(sm.list_sessions_for_tp(
                self._base_tp_key if g == 0 else f"{self._base_tp_key}_g{g}"
            ))
            for g in range(gen + 1)
        )
        reply = QMessageBox.warning(
            self,
            "Clear All Data — Permanent",
            f"This will permanently delete {total_sessions} session folder(s) across "
            f"{gen + 1} thread(s) for this talking point, along with all refinement "
            f"and generation tracking files.\n\nThis CANNOT be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        sm.clear_tp_data(self._base_tp_key)
        self.load_sessions(self._base_tp_key, selected_generation=0)
        self.data_cleared.emit()


# ──────────────────────────────────────────────────────────────────────────────
#  Transcript viewer panel (far-right 4th panel)
# ──────────────────────────────────────────────────────────────────────────────

class _TranscriptViewerPanel(QWidget):
    """Renders a session transcript as styled HTML inside a QTextBrowser."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #050a10;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        # Top bar
        top = QHBoxLayout()
        self._title_lbl = QLabel("Transcript")
        self._title_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._title_lbl.setStyleSheet("color: #00e5ff; background: transparent;")
        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #546e7a; font-size: 11px;"
            " padding: 3px 8px; border-radius: 4px; }"
            "QPushButton:hover { color: #ef5350; background: #1a1025; }"
        )
        close_btn.clicked.connect(self._on_close)
        top.addWidget(self._title_lbl)
        top.addStretch()
        top.addWidget(close_btn)
        lay.addLayout(top)

        self._browser = QTextBrowser()
        self._browser.setStyleSheet(
            "QTextBrowser { background: #050a10; color: #cfd8dc;"
            " border: 1px solid #1e2d42; border-radius: 6px; font-size: 11px; }"
            "QScrollBar:vertical { background: #080d14; width: 5px; border-radius: 2px; }"
            "QScrollBar::handle:vertical { background: #1e2d42; border-radius: 2px; }"
        )
        self._browser.setOpenExternalLinks(False)
        lay.addWidget(self._browser, stretch=1)

        self._splitter_ref = None  # set by parent when panel is created

    def set_splitter(self, splitter: QSplitter, panel_index: int) -> None:
        self._splitter_ref = splitter
        self._panel_index = panel_index

    def load_session(self, session_id: str) -> None:
        sm = get_session_manager()
        transcript = sm.load_session_transcript(session_id)
        sessions = sm.list_sessions()
        meta = next((m for m in sessions if m.session_id == session_id), None)

        ts_str = ""
        if meta and meta.start_ts:
            ts_str = datetime.fromtimestamp(meta.start_ts).strftime("%d %b %Y  %H:%M")
        self._title_lbl.setText(f"Transcript  ·  {ts_str}")

        html = self._render_transcript(transcript, meta)
        self._browser.setHtml(html)

        # Expand the panel if currently collapsed
        if self._splitter_ref is not None:
            sizes = self._splitter_ref.sizes()
            if len(sizes) > self._panel_index and sizes[self._panel_index] < 50:
                sizes[self._panel_index] = 460
                self._splitter_ref.setSizes(sizes)

    def _on_close(self) -> None:
        if self._splitter_ref is not None:
            sizes = self._splitter_ref.sizes()
            if len(sizes) > self._panel_index:
                sizes[self._panel_index] = 0
                self._splitter_ref.setSizes(sizes)

    def _render_transcript(self, transcript: list, meta: "SessionMeta | None") -> str:
        _AGENT_COLORS = {
            "Astra": "#4fc3f7",
            "Nova": "#ff8a65",
        }
        _DEFAULT_L = "#4fc3f7"
        _DEFAULT_R = "#ff8a65"
        left_name = meta.left_agent if meta else "Left"
        right_name = meta.right_agent if meta else "Right"

        lines = [
            "<html><body style='background:#050a10; color:#cfd8dc; font-family:Segoe UI,sans-serif;"
            " font-size:11px; line-height:1.6; margin:0; padding:8px;'>"
        ]

        if meta:
            lines.append(
                f"<div style='color:#546e7a; font-size:10px; margin-bottom:12px;'>"
                f"Topic: <b style='color:#00e5ff'>{meta.topic}</b> &nbsp;·&nbsp;"
                f"{meta.turn_count} turns &nbsp;·&nbsp; "
                f"✓ {meta.truths_count} &nbsp;✗ {meta.problems_count}</div>"
            )

        for evt in transcript:
            etype = evt.get("event_type", "")
            payload = evt.get("payload", {})

            if etype in {"turn", "public_message"}:
                if etype == "turn":
                    speaker = str(payload.get("speaker", "")).strip()
                    text = str(payload.get("text", "")).strip()
                    turn_n = payload.get("turn_number", "")
                    talking_point = ""
                else:
                    speaker = str(payload.get("agent", "")).strip()
                    text = str(payload.get("message", "")).strip()
                    turn_n = payload.get("turn", "")
                    talking_point = str(payload.get("talking_point", "")).strip()

                speaker_l = speaker.lower()
                if speaker_l == left_name.lower():
                    color = _AGENT_COLORS.get(left_name, _DEFAULT_L)
                elif speaker_l == right_name.lower():
                    color = _AGENT_COLORS.get(right_name, _DEFAULT_R)
                else:
                    color = _AGENT_COLORS.get(speaker, _DEFAULT_L)

                tp_line = (
                    f"<div style='margin-top:2px; color:#607d8b; font-size:9px;'>"
                    f"↳ {escape(talking_point)}</div>"
                    if talking_point else ""
                )
                lines.append(
                    f"<div style='margin-bottom:10px;'>"
                    f"<span style='color:{color}; font-weight:700; font-size:10px;'>"
                    f"  {escape(speaker)}  </span>"
                    f"<span style='color:#3a5060; font-size:9px;'>turn {turn_n}</span>"
                    f"<div style='margin-top:3px; color:#b0bec5;'>{escape(text)}</div>"
                    f"{tp_line}"
                    f"</div>"
                )

            elif etype == "private_thought":
                speaker = str(payload.get("agent", "Agent")).strip()
                thought = str(payload.get("thought", "")).strip()
                turn_n = payload.get("turn", "")
                if thought:
                    lines.append(
                        f"<div style='background:#0b121b; border-left:3px solid #455a64;"
                        f" padding:6px 10px; margin:8px 0; color:#90a4ae; font-size:10px;'>"
                        f"[{escape(speaker)} private · T{turn_n}] {escape(thought)}</div>"
                    )

            elif etype == "resolution":
                truths = payload.get("truths", [])
                problems = payload.get("problems", [])
                if truths or problems:
                    lines.append(
                        "<div style='background:#071a14; border:1px solid #00695c;"
                        " border-radius:6px; padding:8px; margin:10px 0;'>"
                        "<b style='color:#00e5ff; font-size:10px;'>RESOLUTION</b><br>"
                    )
                    for t in truths:
                        lines.append(f"<div style='color:#80cbc4;'>  ✓ {t}</div>")
                    for p in problems:
                        lines.append(f"<div style='color:#ef9a9a;'>  ✗ {p}</div>")
                    lines.append("</div>")

            elif etype == "arbiter":
                msg = payload.get("message", "")
                lines.append(
                    f"<div style='background:#1a0a00; border-left:3px solid #ff6e40;"
                    f" padding:6px 10px; margin:8px 0; color:#ffab40; font-size:10px;'>"
                    f"[Arbiter] {msg}</div>"
                )

        lines.append("</body></html>")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Main dialog
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  Debate Assistant — chatbot panel
# ──────────────────────────────────────────────────────────────────────────────

class _AssistantChatPanel(QWidget):
    """Lightweight agent chatbot for helping create / edit debates."""

    topic_created = pyqtSignal(dict)   # emitted when assistant generates a TP

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker = None
        self._generation_context: str = ""
        self._assistant_cloud_enabled: bool = False
        self.setStyleSheet(f"background: {_PALETTE['panel']};")
        self._build()

    def set_generation_context(self, context: str) -> None:
        self._generation_context = (context or "").strip()

    class _InputBox(QTextEdit):
        send_requested = pyqtSignal()

        def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
            is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if is_enter and not shift:
                event.accept()
                self.send_requested.emit()
                return
            super().keyPressEvent(event)

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        header = QLabel("Debate Assistant")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {_PALETTE['accent']}; background: transparent;")
        lay.addWidget(header)

        sub = QLabel("Chat with the assistant to create or refine debates")
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        lay.addWidget(sub)

        # Model selector row
        model_row = QHBoxLayout()
        model_row.setSpacing(4)
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; font-size: 10px; background: transparent;")
        model_row.addWidget(model_lbl)

        self._model_combo = QComboBox()
        self._model_combo.setEditable(False)
        self._model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._model_combo.setStyleSheet(
            f"QComboBox {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 4px;"
            f" padding: 2px 6px; font-size: 10px; min-width: 120px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {_PALETTE['bg']};"
            f" color: {_PALETTE['text']}; selection-background-color: {_PALETTE['accent']}; }}"
        )
        self._populate_model_combo()
        model_row.addWidget(self._model_combo, stretch=1)

        self._cloud_toggle_btn = QPushButton("☁ Cloud OFF")
        self._cloud_toggle_btn.setCheckable(True)
        self._cloud_toggle_btn.setChecked(False)
        self._cloud_toggle_btn.setToolTip("Toggle cloud-tagged models in this assistant dropdown")
        self._cloud_toggle_btn.setStyleSheet(
            f"QPushButton {{ background: {_PALETTE['bg']}; color: {_PALETTE['dim']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 4px;"
            f" padding: 2px 8px; font-size: 10px; font-weight: 700; }}"
            f"QPushButton:hover {{ color: {_PALETTE['text']}; border-color: {_PALETTE['accent']}; }}"
            f"QPushButton:checked {{ color: #e1bee7; border-color: #ce93d8; background: #3a2455; }}"
        )
        self._cloud_toggle_btn.toggled.connect(self._on_cloud_toggle)
        model_row.addWidget(self._cloud_toggle_btn)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(22, 22)
        refresh_btn.setToolTip("Refresh Ollama model list")
        refresh_btn.setStyleSheet(
            f"QPushButton {{ background: {_PALETTE['bg']}; color: {_PALETTE['dim']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 4px; font-size: 11px; }}"
            f"QPushButton:hover {{ color: {_PALETTE['text']}; border-color: {_PALETTE['accent']}; }}"
        )
        refresh_btn.clicked.connect(self._populate_model_combo)
        model_row.addWidget(refresh_btn)
        lay.addLayout(model_row)

        # Chat history
        self._chat = QTextBrowser()
        self._chat.setOpenExternalLinks(False)
        self._chat.setStyleSheet(
            f"QTextBrowser {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 8px;"
            f" padding: 8px; font-size: 10pt; }}"
        )
        lay.addWidget(self._chat, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self._input = self._InputBox()
        self._input.setPlaceholderText(
            "e.g. Create a debate about submarine pressure physics..."
        )
        self._input.setAcceptRichText(False)
        self._input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._input.setStyleSheet(
            f"QTextEdit {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 6px;"
            f" padding: 8px 12px; font-size: 11px; }}"
            f"QTextEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        self._input.send_requested.connect(self._on_send)
        self._input.textChanged.connect(self._resize_input_for_text)
        self._resize_input_for_text()
        input_row.addWidget(self._input, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setStyleSheet(
            f"QPushButton {{ background: {_PALETTE['ok']}; color: #fff;"
            " border-radius: 6px; padding: 8px 16px; font-weight: 700; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {_PALETTE['ok_hover']}; }}"
            "QPushButton:disabled { background: #263238; color: #546e7a; }"
        )
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)
        lay.addLayout(input_row)

        # Status row
        status_row = QHBoxLayout()
        status_row.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {_PALETTE['dim']}; font-size: 9px; background: transparent;"
        )
        status_row.addWidget(self._status)
        lay.addLayout(status_row)

        # Initial welcome message
        self._append_system(
            "Welcome to the Debate Assistant! I can help you create new debate "
            "topics or refine existing ones. Describe what you want to explore "
            "and I'll generate a full debate with title, description, and talking points."
        )

    def _append_user(self, text: str) -> None:
        self._chat.append(
            f'<div style="color:#80cbc4;font-weight:600;margin-top:8px;">You:</div>'
            f'<div style="color:{_PALETTE["text"]};margin-bottom:6px;">{text}</div>'
        )

    def _append_system(self, text: str) -> None:
        self._chat.append(
            f'<div style="color:#ce93d8;font-weight:600;margin-top:8px;">Assistant:</div>'
            f'<div style="color:{_PALETTE["text"]};margin-bottom:6px;">{text}</div>'
        )

    def _populate_model_combo(self) -> None:
        """Fetch Ollama models and populate the combo; cloud models optional via toggle."""
        import urllib.request as _ur
        import json as _json

        previous = self._model_combo.currentText().strip() if self._model_combo.count() > 0 else ""

        # Fall back to pref default
        try:
            from config.model_prefs import load_model_prefs
            pref_model = load_model_prefs().get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            pref_model = "qwen3:30b"

        # Try to fetch live list from Ollama
        models: list[str] = []
        try:
            with _ur.urlopen("http://localhost:11434/api/tags", timeout=4) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
        except Exception:
            pass

        local_models = sorted([m for m in models if not self._is_cloud_model_name(m)])
        cloud_models = sorted([m for m in models if self._is_cloud_model_name(m)])

        # Include cloud-tagged models only when toggle is ON
        pool = list(local_models)
        if self._assistant_cloud_enabled:
            pool.extend(cloud_models)

        # Ensure fallback selection exists even if Ollama is unavailable
        if not pool:
            if self._assistant_cloud_enabled or not self._is_cloud_model_name(pref_model):
                pool = [pref_model]
            elif local_models:
                pool = local_models
            else:
                pool = ["qwen3:30b"]

        # Ensure pref model is selectable when valid for current toggle mode
        if pref_model not in pool:
            if self._assistant_cloud_enabled or not self._is_cloud_model_name(pref_model):
                pool.insert(0, pref_model)

        # Deduplicate while preserving order
        deduped: list[str] = []
        seen: set[str] = set()
        for m in pool:
            key = m.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(m)

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItems(deduped)

        # Restore previous selection or default to pref
        target = previous if (previous and previous in deduped) else pref_model
        idx = self._model_combo.findText(target)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        elif deduped:
            self._model_combo.setCurrentIndex(0)
        self._model_combo.blockSignals(False)

    @staticmethod
    def _is_cloud_model_name(name: str) -> bool:
        lower = (name or "").lower().strip()
        if ":" in lower:
            tag = lower.split(":", 1)[1]
            if "cloud" in tag:
                return True
        return lower.endswith("-cloud")

    def _on_cloud_toggle(self, checked: bool) -> None:
        self._assistant_cloud_enabled = bool(checked)
        self._cloud_toggle_btn.setText("☁ Cloud ON" if self._assistant_cloud_enabled else "☁ Cloud OFF")
        self._populate_model_combo()

    def _resize_input_for_text(self) -> None:
        line_h = max(16, self._input.fontMetrics().lineSpacing())
        min_h = line_h + 14
        max_h = (line_h * 5) + 14
        doc = self._input.document()
        doc_h = int(doc.size().height()) + 8 if doc is not None else min_h
        target_h = max(min_h, min(max_h, doc_h))
        self._input.setFixedHeight(target_h)

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._input.clear()
        self._resize_input_for_text()
        self._append_user(text)
        self._send_btn.setEnabled(False)
        self._status.setText("Generating...")

        # Gather context
        from config.starter_topics import get_topic_titles
        from analytics.analytics_store import get_analytics_store

        existing = get_topic_titles()

        # Analytics summary
        try:
            store = get_analytics_store()
            totals = store.knowledge_totals()
            analytics_text = (
                f"Past debates: {totals['debates']}, Total turns: {totals['turns']}, "
                f"Facts (Astra): {totals['left_facts']}, Facts (Nova): {totals['right_facts']}, "
                f"Truths: {totals['truths']}, Problems: {totals['problems']}"
            )
        except Exception:
            analytics_text = ""

        # Model — use dropdown selection
        model = self._model_combo.currentText().strip() or "qwen3:30b"

        from agents.topic_assistant_worker import TopicAssistantWorker
        self._worker = TopicAssistantWorker(
            text,
            model=model,
            existing_titles=existing,
            analytics_summary=analytics_text,
            segue_context="",
            workspace_context=self._generation_context,
            origin="user",
            parent=self,
        )
        self._worker.chat_reply.connect(self._on_chat_reply)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_chat_reply(self, text: str) -> None:
        self._append_system(text)

    def _on_worker_done(self, result: dict) -> None:
        self._send_btn.setEnabled(True)
        title = result.get("title", "New Debate")
        tp_count = len(result.get("talking_points", []))
        self._status.setText(f"✓ Created: {title}")
        self._append_system(
            f"✅ Created debate: <b>{title}</b> with {tp_count} talking points. "
            "It's been added to your list."
        )
        self.topic_created.emit(result)

    def _on_worker_failed(self, error: str) -> None:
        self._send_btn.setEnabled(True)
        self._status.setText(f"✗ Failed")
        self._append_system(f"Sorry, I ran into an error: {error[:120]}")

    def selected_model(self) -> str:
        return self._model_combo.currentText().strip() if hasattr(self, "_model_combo") else ""


class TopicPickerDialog(QDialog):
    """Debate Studio — full topic browser with assistant, autonomous learning,
    session history, and transcript viewer."""

    topic_confirmed = pyqtSignal(str, str, list)   # title, full_context, file_paths

    def __init__(self, current_title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._initial_title = current_title.strip()
        session_root = get_session_manager().root
        try:
            get_session_manager().backfill_all_session_briefs()
        except Exception:
            pass
        self._custom_debate_store = get_custom_debate_store(session_root)
        self._custom_debates: list[CustomDebate] = []
        self._card_entries: list[dict] = []
        self._active_custom_id: str | None = None
        self.setWindowTitle("Debate Studio")
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setSizeGripEnabled(True)
        self.resize(1280, 760)
        self.setMinimumSize(1120, 600)
        self.setModal(True)
        self._queued_files: list[Path] = []
        self._selected_index: int = 0
        self._cards: list[_TopicCard] = []
        self._card_col: QVBoxLayout | None = None
        self._rewrite_worker: _RewriteWorker | None = None
        self._manual_refine_worker: TopicRefineWorker | None = None
        self._manual_refine_tp_key: str = ""
        self._manual_refine_session_id: str = ""
        self._repo_schema_worker: _RepoSchemaWorker | None = None
        self._repo_schema_generated_once: bool = False
        self._repo_all_models: list[str] = []
        self._seed_source_session_id: str = ""
        self._skip_next_auto_seed_from_session: bool = False
        self._session_autosave_lbl: QLabel | None = None
        self._active_session_brief_id: str = ""
        self._suspend_session_autosave: bool = False
        self._session_autosave_timer = QTimer(self)
        self._session_autosave_timer.setSingleShot(True)
        self._session_autosave_timer.setInterval(650)
        self._session_autosave_timer.timeout.connect(self._autosave_active_session_brief)
        # Per-talking-point key for session history
        self._current_tp_key: str = ""
        # References to new panels (set in _build_ui)
        self._history_panel: _SessionHistoryPanel | None = None
        self._transcript_panel: _TranscriptViewerPanel | None = None
        self._dataset_library: _DatasetLibraryPanel | None = None
        self._assistant_panel: _AssistantChatPanel | None = None
        self._main_splitter: QSplitter | None = None
        self._repo_watchdog = RepoWatchdog(session_root)
        self._load_persisted_custom_debates()
        self._apply_style()
        self._build_ui(current_title)

    @property
    def tp_key(self) -> str:
        """Generation-aware key that will be written to the new session."""
        if not self._current_tp_key:
            return ""
        if self._history_panel is not None:
            selected = self._history_panel.selected_effective_tp_key()
            if selected:
                return selected
        return get_session_manager().effective_tp_key(self._current_tp_key)

    def refresh_library(self) -> None:
        """Re-scan disk and repopulate the Knowledge Library dataset list."""
        if self._dataset_library is not None:
            self._dataset_library.refresh()

    def _set_repo_busy(self, busy: bool, status_text: str = "") -> None:
        self._repo_progress.setVisible(busy)
        self._repo_progress.setRange(0, 0 if busy else 1)
        if not busy:
            self._repo_progress.setValue(1)

        for w in (self._repo_browse_btn, self._repo_prep_btn, self._repo_rebuild_btn, self._repo_model_combo):
            w.setEnabled(not busy)

        if status_text:
            self._repo_prep_status.setText(status_text)
        self._repo_prep_status.setVisible(self._mode_repo_rb.isChecked() and self._custom_title_edit.isVisible())
        QApplication.processEvents()

    def _derive_repo_topic_title(self, repo_path: Path) -> str:
        label = repo_path.name.strip().replace("_", " ").replace("-", " ").replace(".", " ")
        label = " ".join(label.split()) or "Repository"

        for name in ("README.md", "README.rst", "README.txt", "readme.md"):
            p = repo_path / name
            if not p.exists() or not p.is_file():
                continue
            try:
                for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = raw.strip()
                    if not s:
                        continue
                    if s.startswith("#"):
                        s = s.lstrip("#").strip()
                    if len(s) >= 4:
                        label = s
                        raise StopIteration
            except StopIteration:
                break
            except Exception:
                continue

        if "debate" in label.lower():
            return label[:90]
        return f"{label[:72]} System Debate"

    @staticmethod
    def _is_generic_custom_title(title: str) -> bool:
        t = title.strip().lower()
        return not t or t in {"custom", "custom topic", "custom debate"}

    # ── Build ────────────────────────────────────────────────────────────────

    def _build_ui(self, current_title: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background: #060b12; border-bottom: 1px solid {_PALETTE['border']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        h_title = QLabel("Debate Studio")
        h_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        h_title.setStyleSheet(f"color: {_PALETTE['accent']}; background: transparent;")
        h_sub = QLabel("Select a curated brief, create/edit debates with the assistant, or link a repository with Repo Watchdog")
        h_sub.setFont(QFont("Segoe UI", 9))
        h_sub.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        hl.addWidget(h_title)
        hl.addSpacing(20)
        hl.addWidget(h_sub)
        hl.addStretch()
        root.addWidget(header)

        # Splitter: main wall | assistant (2 windows only)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(0)
        splitter.setStyleSheet("QSplitter::handle { width: 0px; background: transparent; }")

        # ── Left: card list ──────────────────────────────────────────────────
        left_container = QWidget()
        left_container.setStyleSheet(f"background: #060b12;")
        left_container.setFixedWidth(_CARD_W + 32)
        left_lay = QVBoxLayout(left_container)
        left_lay.setContentsMargins(12, 12, 12, 12)
        left_lay.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: #0a1018; width: 6px; border-radius: 3px; }}"
            f"QScrollBar::handle:vertical {{ background: #2a3a55; border-radius: 3px; }}"
        )
        card_widget = QWidget()
        card_widget.setStyleSheet("background: transparent;")
        card_col = QVBoxLayout(card_widget)
        card_col.setContentsMargins(0, 0, 0, 0)
        card_col.setSpacing(8)
        self._card_col = card_col
        self._rebuild_left_cards()
        scroll.setWidget(card_widget)
        left_lay.addWidget(scroll)

        # ── Right: detail / edit panel (vertically scrollable) ──────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setStyleSheet(
            f"QScrollArea {{ background: {_PALETTE['panel']}; border: none; }}"
            f"QScrollBar:vertical {{ background: #0a1018; width: 6px; border-radius: 3px; }}"
            f"QScrollBar::handle:vertical {{ background: #2a3a55; border-radius: 3px; }}"
        )

        right_container = QWidget()
        right_container.setStyleSheet(f"background: {_PALETTE['panel']};")
        right_lay = QVBoxLayout(right_container)
        right_lay.setContentsMargins(24, 20, 24, 16)
        right_lay.setSpacing(14)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self._preset_title_lbl = QLabel("")
        self._preset_title_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self._preset_title_lbl.setWordWrap(True)
        self._preset_title_lbl.setStyleSheet(f"color: {_PALETTE['accent']}; background: transparent;")

        self._custom_title_edit = QLineEdit()
        self._custom_title_edit.setPlaceholderText("Enter a clear, specific debate topic…")
        self._custom_title_edit.setStyleSheet(
            f"QLineEdit {{ background: {_PALETTE['bg']}; color: #e8eaf6; border: 1px solid {_PALETTE['border']};"
            " border-radius: 6px; padding: 8px 12px; font-size: 14px; font-weight: 700; }}"
            f"QLineEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        self._custom_title_edit.hide()
        self._custom_title_edit.textChanged.connect(self._schedule_active_session_autosave)
        self._custom_title_edit.textChanged.connect(lambda _t: self._refresh_assistant_context())

        title_row.addWidget(self._preset_title_lbl, stretch=1)
        title_row.addWidget(self._custom_title_edit, stretch=1)
        right_lay.addLayout(title_row)

        # Custom Debate mode selector
        mode_grp = QGroupBox("Custom Debate Mode")
        mode_grp.setObjectName("detailGroup")
        mode_lay = QVBoxLayout(mode_grp)
        mode_lay.setSpacing(6)
        self._custom_mode_group = mode_grp

        self._mode_static_rb = QRadioButton("Static Ingestion (manual brief)")
        self._mode_repo_rb = QRadioButton("Repo Watchdog (link to repository)")
        self._mode_static_rb.setChecked(True)
        self._mode_static_rb.setStyleSheet(
            "QRadioButton { background: #111a29; color: #9fc1d5; border: 1px solid #2a3a55;"
            " border-radius: 8px; padding: 6px 10px; font-size: 11px; font-weight: 600; }"
            "QRadioButton:checked { background: #2e1a47; color: #f3e5f5; border: 1px solid #7e57c2; }"
            "QRadioButton:hover { border: 1px solid #4b6483; }"
            "QRadioButton::indicator { width: 0px; height: 0px; }"
        )
        self._mode_repo_rb.setStyleSheet(
            "QRadioButton { background: #111a29; color: #9fc1d5; border: 1px solid #2a3a55;"
            " border-radius: 8px; padding: 6px 10px; font-size: 11px; font-weight: 600; }"
            "QRadioButton:checked { background: #2e1a47; color: #f3e5f5; border: 1px solid #7e57c2; }"
            "QRadioButton:hover { border: 1px solid #4b6483; }"
            "QRadioButton::indicator { width: 0px; height: 0px; }"
        )

        self._mode_static_rb.toggled.connect(self._on_custom_mode_changed)
        self._mode_repo_rb.toggled.connect(self._on_custom_mode_changed)
        self._mode_static_rb.toggled.connect(lambda _v: self._refresh_assistant_context())
        self._mode_repo_rb.toggled.connect(lambda _v: self._refresh_assistant_context())

        mode_lay.addWidget(self._mode_static_rb)
        mode_lay.addWidget(self._mode_repo_rb)

        repo_row = QHBoxLayout()
        self._repo_path_edit = QLineEdit()
        self._repo_path_edit.setPlaceholderText("Select a repository folder to link to this debate…")
        self._repo_path_edit.setStyleSheet(
            f"QLineEdit {{ background: {_PALETTE['bg']}; color: #e8eaf6; border: 1px solid {_PALETTE['border']};"
            " border-radius: 6px; padding: 6px 10px; font-size: 11px; }}"
            f"QLineEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        self._repo_path_edit.setVisible(False)
        self._repo_path_edit.textChanged.connect(lambda _t: self._refresh_assistant_context())

        self._repo_browse_btn = QPushButton("📂  Link Repo")
        self._repo_browse_btn.setVisible(False)
        self._repo_browse_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #90a4ae; border-radius: 6px; padding: 5px 14px;"
            " font-size: 11px; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        self._repo_browse_btn.clicked.connect(self._browse_repo)

        repo_row.addWidget(self._repo_path_edit, stretch=1)
        repo_row.addWidget(self._repo_browse_btn)
        mode_lay.addLayout(repo_row)

        self._repo_hint_lbl = QLabel(
            "Repo Watchdog builds a codebase snapshot, skips noisy folders, and can detect changed files in later debates."
        )
        self._repo_hint_lbl.setWordWrap(True)
        self._repo_hint_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; font-size: 10px;")
        self._repo_hint_lbl.setVisible(False)
        mode_lay.addWidget(self._repo_hint_lbl)

        self._repo_progress = QProgressBar()
        self._repo_progress.setVisible(False)
        self._repo_progress.setTextVisible(False)
        self._repo_progress.setMaximumHeight(8)
        self._repo_progress.setRange(0, 0)
        self._repo_progress.setStyleSheet(
            "QProgressBar { background: #0a1520; border: 1px solid #1e2d42; border-radius: 4px; }"
            "QProgressBar::chunk { background: #00acc1; border-radius: 3px; }"
        )
        mode_lay.addWidget(self._repo_progress)

        prep_top_row = QHBoxLayout()
        self._repo_prep_btn = QPushButton("🧠  Analyze Repo & Fill Debate Schema")
        self._repo_prep_btn.setVisible(False)
        self._repo_prep_btn.setStyleSheet(
            "QPushButton { background: #1a1040; color: #ce93d8; border-radius: 6px;"
            " padding: 5px 12px; font-size: 11px; border: 1px solid #7e57c2; }"
            "QPushButton:hover { background: #311b92; color: #fff; }"
            "QPushButton:disabled { color: #546e7a; border-color: #2a3a55; }"
        )
        self._repo_prep_btn.clicked.connect(self._on_repo_prep_clicked)

        self._repo_allow_cloud_cb = QCheckBox("Allow Cloud")
        self._repo_allow_cloud_cb.setVisible(False)
        self._repo_allow_cloud_cb.setChecked(False)
        self._repo_allow_cloud_cb.setStyleSheet(
            f"QCheckBox {{ color: {_PALETTE['dim']}; font-size: 10px; }}"
            f"QCheckBox:checked {{ color: {_PALETTE['accent']}; }}"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
        )
        self._repo_allow_cloud_cb.toggled.connect(self._on_repo_cloud_toggled)

        self._repo_model_combo = QComboBox()
        self._repo_model_combo.setVisible(False)
        self._repo_model_combo.setMinimumWidth(150)
        self._repo_model_combo.setStyleSheet(
            f"QComboBox {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 6px; padding: 4px 8px;"
            " font-size: 10px; }}"
            f"QComboBox:focus {{ border-color: {_PALETTE['accent']}; }}"
            "QComboBox QAbstractItemView { background: #0d1520; color: #dce8f5;"
            " border: 1px solid #2a3a55; selection-background-color: #1e3a5f; }"
        )

        self._repo_prep_status = QLabel("")
        self._repo_prep_status.setVisible(False)
        self._repo_prep_status.setStyleSheet("color: #7e57c2; font-size: 10px; font-style: italic;")
        prep_top_row.addWidget(self._repo_prep_btn)
        prep_top_row.addStretch()

        self._repo_rebuild_btn = QPushButton("\U0001f504  Rebuild Dataset")
        self._repo_rebuild_btn.setVisible(False)
        self._repo_rebuild_btn.setToolTip(
            "Re-scan the linked repository and rebuild the semantic dataset from scratch.\n"
            "Run this after large changes to keep the agent context up to date."
        )
        self._repo_rebuild_btn.setStyleSheet(
            "QPushButton { background: #0d2030; color: #80cbc4; border-radius: 6px;"
            " padding: 5px 12px; font-size: 11px; border: 1px solid #2a6a7a; }"
            "QPushButton:hover { background: #1a3a50; color: #fff; }"
            "QPushButton:disabled { color: #546e7a; border-color: #2a3a55; }"
        )
        self._repo_rebuild_btn.clicked.connect(self._on_rebuild_dataset_clicked)

        prep_bottom_row = QHBoxLayout()
        prep_bottom_row.addWidget(self._repo_allow_cloud_cb)
        prep_bottom_row.addWidget(self._repo_model_combo)
        prep_bottom_row.addWidget(self._repo_prep_status)
        prep_bottom_row.addWidget(self._repo_rebuild_btn)
        prep_bottom_row.addStretch()

        mode_lay.addLayout(prep_top_row)
        mode_lay.addLayout(prep_bottom_row)

        right_lay.addWidget(mode_grp)

        # ── Session history panel (embedded in middle wall) ────────────────
        self._history_panel = _SessionHistoryPanel()
        self._history_panel.start_requested.connect(self._on_confirm)
        self._history_panel.session_clicked.connect(self._on_session_card_clicked)
        self._history_panel.refine_requested.connect(self._on_manual_refine_requested)
        self._history_panel.thread_reset.connect(lambda: self._select_topic(self._selected_index))
        self._history_panel.data_cleared.connect(lambda: self._select_topic(self._selected_index))
        right_lay.addWidget(self._history_panel)

        # Description block
        desc_grp = QGroupBox("Topic Description & Context")
        desc_grp.setObjectName("detailGroup")
        desc_gl = QVBoxLayout(desc_grp)
        self._desc_edit = QTextEdit()
        self._desc_edit.setFont(QFont("Segoe UI", 11))
        self._desc_edit.setMinimumHeight(130)
        self._desc_edit.setMaximumHeight(200)
        self._desc_edit.setStyleSheet(
            f"QTextEdit {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 6px; padding: 8px;"
            " font-size: 11px; line-height: 1.6; }"
            f"QTextEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        desc_gl.addWidget(self._desc_edit)
        self._desc_edit.textChanged.connect(self._schedule_active_session_autosave)
        self._desc_edit.textChanged.connect(lambda: self._refresh_assistant_context())
        right_lay.addWidget(desc_grp)

        # Talking points block
        tp_grp = QGroupBox("Talking Points  (one per line — agents use these as focal anchors)")
        tp_grp.setObjectName("detailGroup")
        tp_gl = QVBoxLayout(tp_grp)
        self._tp_edit = QTextEdit()
        self._tp_edit.setFont(QFont("Segoe UI", 10))
        self._tp_edit.setStyleSheet(
            f"QTextEdit {{ background: {_PALETTE['bg']}; color: #b0bec5;"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 6px; padding: 8px;"
            " font-size: 10px; line-height: 1.5; }"
            f"QTextEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        self._tp_edit.setMinimumHeight(120)
        tp_gl.addWidget(self._tp_edit)
        self._tp_edit.textChanged.connect(self._schedule_active_session_autosave)
        self._tp_edit.textChanged.connect(lambda: self._refresh_assistant_context())

        # Rewrite button row
        rewrite_row = QHBoxLayout()
        self._rewrite_btn = QPushButton("✨  Rewrite with AI")
        self._rewrite_btn.setToolTip(
            "Use the selected LLM to rewrite the talking points — "
            "it will read any ingested dataset for extra context."
        )
        self._rewrite_btn.setStyleSheet(
            f"QPushButton {{ background: #1a1040; color: #ce93d8;"
            f" border-radius: 6px; padding: 5px 14px; font-size: 11px;"
            f" border: 1px solid #7e57c2; }}"
            "QPushButton:hover { background: #311b92; color: #fff; }"
            "QPushButton:disabled { color: #546e7a; border-color: #2a3a55; }"
        )
        self._rewrite_btn.clicked.connect(self._on_rewrite_clicked)
        self._rewrite_status = QLabel("")
        self._rewrite_status.setStyleSheet("color: #7e57c2; font-size: 10px; font-style: italic;")
        rewrite_row.addWidget(self._rewrite_btn)
        rewrite_row.addWidget(self._rewrite_status)
        rewrite_row.addStretch()
        tp_gl.addLayout(rewrite_row)

        right_lay.addWidget(tp_grp, stretch=1)

        # Ingestion section — NOT checkable so children are always enabled
        ingest_grp = QGroupBox("Knowledge Ingestion  ·  optional")
        ingest_grp.setObjectName("detailGroup")
        # NOTE: deliberately NOT setCheckable — a checkable+unchecked QGroupBox
        # disables ALL children in PyQt6, making Browse and drag-and-drop unusable.
        self._ingest_grp = ingest_grp
        ig_lay = QVBoxLayout(ingest_grp)
        ig_lay.setSpacing(6)

        # Toggle: include files when starting debate
        self._use_ingest_cb = QCheckBox("Include ingested files when starting debate")
        self._use_ingest_cb.setChecked(False)
        self._use_ingest_cb.setStyleSheet(
            f"QCheckBox {{ color: {_PALETTE['dim']}; font-size: 11px; }}"
            f"QCheckBox:checked {{ color: {_PALETTE['accent']}; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        ig_lay.addWidget(self._use_ingest_cb)

        self._drop_zone = _DropZone()
        self._drop_zone.files_dropped.connect(self._add_files)
        ig_lay.addWidget(self._drop_zone)

        file_row = QHBoxLayout()
        browse_btn = QPushButton("📂  Browse")
        browse_btn.setToolTip(
            "Open the knowledge folder.  Select any files or folders to queue for ingestion."
        )
        browse_btn.setStyleSheet(
            f"QPushButton {{ background: #1a2840; color: #90a4ae; border-radius: 6px; padding: 5px 14px;"
            " font-size: 11px; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        browse_btn.clicked.connect(self._browse)
        clear_btn = QPushButton("✕  Clear List")
        clear_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #546e7a; font-size: 11px; padding: 5px 10px; }"
            "QPushButton:hover { color: #ef5350; }"
        )
        clear_btn.clicked.connect(self._clear_files)
        file_row.addWidget(browse_btn)
        file_row.addStretch()
        file_row.addWidget(clear_btn)
        ig_lay.addLayout(file_row)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(80)
        self._file_list.setStyleSheet(
            f"QListWidget {{ background: {_PALETTE['bg']}; color: #80cbc4;"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 4px; font-size: 11px; }}"
            "QListWidget::item { padding: 3px 6px; }"
        )
        ig_lay.addWidget(self._file_list)

        # ── Knowledge Library: persistent dataset list ───────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {_PALETTE['border']};")
        ig_lay.addWidget(sep)

        self._dataset_library = _DatasetLibraryPanel()
        ig_lay.addWidget(self._dataset_library)

        right_lay.addWidget(ingest_grp)

        # Confirm row
        confirm_row = QHBoxLayout()
        self._topic_badge = QLabel("")
        self._topic_badge.setStyleSheet(f"color: {_PALETTE['dim']}; font-size: 11px;")
        self._session_autosave_lbl = QLabel("")
        self._session_autosave_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; font-size: 10px;")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #263238; color: #90a4ae; border-radius: 6px;"
            " padding: 9px 22px; font-size: 13px; }"
            "QPushButton:hover { background: #37474f; color: #fff; }"
        )
        cancel_btn.clicked.connect(self.reject)

        self._confirm_btn = QPushButton("▶  Go to Session")
        self._confirm_btn.setStyleSheet(
            f"QPushButton {{ background: {_PALETTE['ok']}; color: #fff; font-weight: 700;"
            f" border-radius: 8px; padding: 9px 26px; font-size: 13px;"
            f" border: 1px solid #00897b; }}"
            f"QPushButton:hover {{ background: {_PALETTE['ok_hover']}; }}"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)

        confirm_row.addWidget(self._topic_badge)
        confirm_row.addSpacing(10)
        confirm_row.addWidget(self._session_autosave_lbl)
        confirm_row.addStretch()
        confirm_row.addWidget(cancel_btn)
        confirm_row.addSpacing(8)
        confirm_row.addWidget(self._confirm_btn)
        right_lay.addLayout(confirm_row)

        right_scroll.setWidget(right_container)

        # Main wall combines topic cards + middle scrollable editor/history wall
        main_wall = QWidget()
        main_wall.setStyleSheet("background: transparent;")
        main_wall_lay = QHBoxLayout(main_wall)
        main_wall_lay.setContentsMargins(0, 0, 0, 0)
        main_wall_lay.setSpacing(0)
        main_wall_lay.addWidget(left_container)
        main_wall_lay.addWidget(right_scroll, stretch=1)
        splitter.addWidget(main_wall)

        # ── Debate Assistant panel (5th pane) ────────────────────────────────
        self._assistant_panel = _AssistantChatPanel()
        self._assistant_panel.setMinimumWidth(260)
        self._assistant_panel.topic_created.connect(self._on_assistant_topic_created)
        splitter.addWidget(self._assistant_panel)

        total_w = max(self.width(), 960)
        assistant_w = min(360, max(260, int(total_w * 0.22)))
        main_w = max(640, total_w - assistant_w - 16)
        splitter.setSizes([main_w, assistant_w])
        splitter.setCollapsible(1, False)
        try:
            handle = splitter.handle(1)
            if handle is not None:
                handle.setEnabled(False)
                handle.setCursor(Qt.CursorShape.ArrowCursor)
        except Exception:
            pass
        self._main_splitter = splitter
        root.addWidget(splitter, stretch=1)

        # Select initial topic
        starter_choice = next(
            (i for i, t in enumerate(STARTER_TOPICS) if t.title == current_title),
            -1,
        )
        start_idx = -1
        if starter_choice >= 0:
            start_idx = next(
                (
                    i for i, entry in enumerate(self._card_entries)
                    if entry.get("kind") == "starter" and int(entry.get("index", -1)) == starter_choice
                ),
                -1,
            )
        if start_idx < 0 and self._initial_title:
            target = self._initial_title.lower()
            for idx, entry in enumerate(self._card_entries):
                if entry.get("kind") != "custom":
                    continue
                debate = self._find_custom_debate(str(entry.get("id", "")))
                if debate is None:
                    continue
                if debate.title.strip().lower() == target:
                    start_idx = idx
                    break
        if start_idx < 0:
            start_idx = 0
        self._select_topic(start_idx)

    # ── Topic selection ──────────────────────────────────────────────────────

    def _load_persisted_custom_debates(self) -> None:
        self._custom_debates = self._custom_debate_store.list_all()

    def _rebuild_left_cards(self) -> None:
        if self._card_col is None:
            return

        while self._card_col.count():
            item = self._card_col.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._cards.clear()
        self._card_entries.clear()

        index = 0
        custom_starter_index = next(
            (i for i, topic in enumerate(STARTER_TOPICS) if _is_custom_topic_title(topic.title)),
            -1,
        )

        if custom_starter_index >= 0:
            topic = STARTER_TOPICS[custom_starter_index]
            card = _TopicCard(
                index,
                title=topic.title,
                description=topic.description,
                talking_points_count=len(topic.talking_points),
            )
            card.clicked.connect(self._select_topic)
            self._cards.append(card)
            self._card_entries.append({"kind": "starter", "index": custom_starter_index})
            self._card_col.addWidget(card)
            index += 1

        for starter_index, topic in enumerate(STARTER_TOPICS):
            if starter_index == custom_starter_index:
                continue
            card = _TopicCard(
                index,
                title=topic.title,
                description=topic.description,
                talking_points_count=len(topic.talking_points),
            )
            card.clicked.connect(self._select_topic)
            self._cards.append(card)
            self._card_entries.append({"kind": "starter", "index": starter_index})
            self._card_col.addWidget(card)
            index += 1

        for debate in self._custom_debates:
            is_repo = debate.mode == "repo_watchdog" and bool((debate.repo_path or "").strip())
            card = _TopicCard(
                index,
                title=debate.title,
                description=debate.description,
                talking_points_count=len(debate.talking_points),
                repo_mode=is_repo,
                deletable=True,
                debate_id=debate.id,
            )
            card.clicked.connect(self._select_topic)
            card.delete_requested.connect(self._delete_custom_debate)
            self._cards.append(card)
            self._card_entries.append({"kind": "custom", "id": debate.id})
            self._card_col.addWidget(card)
            index += 1

        self._card_col.addStretch()

    def _find_custom_debate(self, debate_id: str) -> CustomDebate | None:
        for debate in self._custom_debates:
            if debate.id == debate_id:
                return debate
        return None

    def _delete_custom_debate(self, debate_id: str) -> None:
        debate = self._find_custom_debate(debate_id)
        title = debate.title if debate is not None else "this debate"
        reply = QMessageBox.warning(
            self,
            "Delete Debate Widget",
            f"Delete '{title}' from your saved debate widgets?\n\n"
            "This removes the saved custom debate entry.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted = self._custom_debate_store.delete(debate_id)
        if not deleted:
            QMessageBox.warning(
                self,
                "Delete Debate Widget",
                "Could not delete the selected debate widget.",
            )
            return

        if self._active_custom_id == debate_id:
            self._active_custom_id = None

        self._load_persisted_custom_debates()
        self._rebuild_left_cards()

        fallback_idx = next(
            (i for i, entry in enumerate(self._card_entries)
             if entry.get("kind") == "starter" and _is_custom_topic_title(STARTER_TOPICS[int(entry.get("index", 0))].title)),
            0,
        )
        self._select_topic(fallback_idx)

    def _select_custom_card_by_id(self, debate_id: str) -> None:
        for idx, entry in enumerate(self._card_entries):
            if entry.get("kind") == "custom" and entry.get("id") == debate_id:
                self._select_topic(idx)
                return

    def _upsert_current_custom_debate(self) -> CustomDebate | None:
        title = self._custom_title_edit.text().strip()
        if not title:
            return None

        description = self._desc_edit.toPlainText().strip()
        tps = [ln.strip() for ln in self._tp_edit.toPlainText().split("\n") if ln.strip()]
        repo_mode = self._mode_repo_rb.isChecked()
        repo_path = self._repo_path_edit.text().strip() if repo_mode else ""

        payload = CustomDebate(
            id=self._active_custom_id or "",
            title=title,
            description=description,
            talking_points=tps,
            mode="repo_watchdog" if repo_mode else "static_ingestion",
            repo_path=repo_path,
        )
        saved = self._custom_debate_store.upsert(
            debate_id=payload.id,
            title=payload.title,
            description=payload.description,
            talking_points=payload.talking_points,
            mode=payload.mode,
            repo_path=payload.repo_path,
        )
        self._active_custom_id = saved.id
        self._load_persisted_custom_debates()
        self._rebuild_left_cards()
        self._select_custom_card_by_id(saved.id)
        return saved

    def _select_topic(self, index: int) -> None:
        if index < 0 or index >= len(self._card_entries):
            return

        self._active_session_brief_id = ""
        self._session_autosave_timer.stop()
        self._set_session_autosave_status("")

        for i, card in enumerate(self._cards):
            card.set_selected(i == index)
        self._selected_index = index

        entry = self._card_entries[index]
        if entry.get("kind") == "custom":
            debate = self._find_custom_debate(entry.get("id", ""))
            if debate is None:
                return

            self._seed_source_session_id = ""
            self._active_custom_id = debate.id
            self._preset_title_lbl.setVisible(False)
            self._custom_mode_group.setVisible(True)
            self._custom_title_edit.setVisible(True)
            self._custom_title_edit.setText(debate.title)
            self._desc_edit.setReadOnly(False)
            self._tp_edit.setReadOnly(False)
            self._desc_edit.setPlainText(debate.description)
            self._tp_edit.setPlainText("\n".join(debate.talking_points))
            self._repo_path_edit.setText(debate.repo_path or "")
            self._repo_schema_generated_once = bool(debate.description or debate.talking_points)

            if debate.mode == "repo_watchdog":
                self._mode_repo_rb.setChecked(True)
            else:
                self._mode_static_rb.setChecked(True)
            self._on_custom_mode_changed()

            self._current_tp_key = self._resolve_custom_tp_key(debate)

            sm = get_session_manager()
            refinement = sm.load_tp_refinement(sm.effective_tp_key(self._current_tp_key)) if self._current_tp_key else None
            if isinstance(refinement, dict):
                refined_desc = str(refinement.get("description", "") or "").strip()
                refined_tps_raw = refinement.get("talking_points", [])
                refined_tps = [
                    str(tp).strip() for tp in refined_tps_raw
                    if str(tp).strip()
                ] if isinstance(refined_tps_raw, list) else []
                brief = str(refinement.get("session_brief", "") or "").strip()
                if refined_desc:
                    badge = f"✨ Refined after session — last: \"{brief[:80]}\"\n\n" if brief else "✨ Refined\n\n"
                    self._desc_edit.setPlainText(badge + refined_desc)
                if refined_tps:
                    self._tp_edit.setPlainText("\n".join(refined_tps))

            self._confirm_btn.setText("▶  Go to Custom Session")
            pts = len([ln for ln in self._tp_edit.toPlainText().splitlines() if ln.strip()])
            suffix = "  ·  repo linked" if debate.mode == "repo_watchdog" else ""
            self._topic_badge.setText(f"  {pts} talking points{suffix}" if pts else suffix)

            if self._history_panel is not None and self._current_tp_key:
                self._history_panel.load_sessions(self._current_tp_key)
            elif self._history_panel is not None:
                self._history_panel.load_sessions("")

            # Auto-seed the next session form from the latest saved session config.
            eff_key = sm.effective_tp_key(self._current_tp_key) if self._current_tp_key else ""
            latest = sm.list_sessions_for_tp(eff_key) if eff_key else []
            has_widget_draft = bool(
                (debate.description or "").strip()
                or any(str(tp).strip() for tp in (debate.talking_points or []))
            )
            should_skip_seed = self._skip_next_auto_seed_from_session
            self._skip_next_auto_seed_from_session = False
            if latest and not has_widget_draft and not should_skip_seed:
                self._load_session_brief_into_editor(latest[0].session_id)
                if not self._seed_source_session_id:
                    self._seed_source_session_id = latest[0].session_id
            return

        starter_index = int(entry.get("index", 0))
        topic = STARTER_TOPICS[starter_index]
        is_custom = _is_custom_topic_title(topic.title)

        self._preset_title_lbl.setVisible(not is_custom)
        self._custom_mode_group.setVisible(is_custom)
        # Reset title label color to default accent
        self._preset_title_lbl.setStyleSheet(
            f"color: {_PALETTE['accent']}; background: transparent;"
        )
        self._custom_title_edit.setVisible(is_custom)

        # Compute default tp_key from the first talking point (or topic title for custom)
        if is_custom:
            self._active_custom_id = None
            self._seed_source_session_id = ""
            self._repo_schema_generated_once = False
            self._current_tp_key = ""
            self._custom_title_edit.setText("")
            self._repo_path_edit.setText("")
            self._desc_edit.setReadOnly(False)
            self._tp_edit.setReadOnly(False)
            self._desc_edit.setPlainText("")
            self._tp_edit.setPlainText("")
            self._desc_edit.setPlaceholderText(
                "Describe your topic in detail — multi-paragraph encouraged.\n\n"
                "Include: background, the core disagreement, any constraints, desired depth…"
            )
            self._tp_edit.setPlaceholderText(
                "Write talking points one per line.\n"
                "These are the sharp rhetorical positions the agents will argue from.\n\n"
                "Example:\n"
                "Is consciousness purely physical or is there something irreducibly subjective?\n"
                "If determinism is true, can moral responsibility exist?"
            )
            self._confirm_btn.setText("▶  Go to Custom Session")
            self._on_custom_mode_changed()
        else:
            self._active_custom_id = None
            self._seed_source_session_id = ""
            self._preset_title_lbl.setText(topic.title)
            self._desc_edit.setReadOnly(False)
            self._tp_edit.setReadOnly(False)

            # Check for a refinement and use it if available
            first_tp = topic.talking_points[0] if topic.talking_points else ""
            self._current_tp_key = _tp_key(topic.title, first_tp)
            sm = get_session_manager()
            refinement = sm.load_tp_refinement(sm.effective_tp_key(self._current_tp_key))

            if refinement:
                # Show refined description + talking points with ✨ badge
                refined_desc = refinement.get("description", topic.description)
                refined_tps = refinement.get("talking_points", topic.talking_points)
                brief = refinement.get("session_brief", "")
                badge = f"✨ Refined after session — last: \"{brief[:80]}\"\n\n" if brief else "✨ Refined\n\n"
                self._desc_edit.setPlainText(badge + refined_desc)
                tp_text = "\n\n".join(refined_tps)
            else:
                self._desc_edit.setPlainText(topic.description)
                tp_text = "\n\n".join(topic.talking_points)

            self._tp_edit.setPlainText(tp_text)
            self._confirm_btn.setText("▶  Go to Session")
            self._mode_static_rb.setChecked(True)
            self._on_custom_mode_changed()

        pts = len(topic.talking_points) if not is_custom else 0
        self._topic_badge.setText(f"  {pts} talking points" if pts else "")

        # Load session history for this talking point (pass the BASE key;
        # the panel resolves the effective generation internally)
        if self._history_panel is not None and self._current_tp_key:
            self._history_panel.load_sessions(self._current_tp_key)
        elif self._history_panel is not None:
            self._history_panel.load_sessions("")

        self._refresh_assistant_context()

    def _on_session_card_clicked(self, session_id: str) -> None:
        """Show the transcript for a clicked session card."""
        if self._transcript_panel is not None:
            self._transcript_panel.load_session(session_id)
        self._load_session_brief_into_editor(session_id)
        self._refresh_assistant_context()

    def _refresh_assistant_context(self) -> None:
        if self._assistant_panel is None:
            return

        title = (
            self._custom_title_edit.text().strip()
            if self._custom_title_edit.isVisible()
            else self._preset_title_lbl.text().strip()
        )
        desc = self._desc_edit.toPlainText().strip()
        tps = [ln.strip() for ln in self._tp_edit.toPlainText().splitlines() if ln.strip()]
        repo_mode = bool(self._custom_title_edit.isVisible() and self._mode_repo_rb.isChecked())
        repo_path = self._repo_path_edit.text().strip() if repo_mode else ""

        tp_block = "\n".join(f"- {tp}" for tp in tps[:12])
        context = (
            f"Active title: {title}\n"
            f"Repo mode: {'yes' if repo_mode else 'no'}\n"
            f"Repo path: {repo_path}\n\n"
            f"Current description draft:\n{desc[:2500]}\n\n"
            f"Current talking points draft:\n{tp_block}"
        )
        self._assistant_panel.set_generation_context(context)

    def _on_manual_refine_requested(self) -> None:
        effective_tp_key = self.tp_key
        if not effective_tp_key:
            QMessageBox.information(
                self,
                "Update from Past Sessions",
                "No talking-point thread is selected for this debate yet.",
            )
            return
        if self._manual_refine_worker is not None and self._manual_refine_worker.isRunning():
            return

        sm = get_session_manager()
        sessions = sm.list_sessions_for_tp(effective_tp_key)
        if not sessions:
            QMessageBox.information(
                self,
                "Update from Past Sessions",
                "No past sessions exist in this thread yet.",
            )
            return

        ordered = sorted(sessions, key=lambda m: m.start_ts)
        latest = sessions[0]

        title = self._current_editor_title()
        current_desc = self._strip_refinement_badge(self._desc_edit.toPlainText())
        current_tps = [ln.strip() for ln in self._tp_edit.toPlainText().splitlines() if ln.strip()]

        baseline_brief = sm.load_session_brief(ordered[0].session_id)
        if isinstance(baseline_brief, dict):
            title = str(baseline_brief.get("title", "") or title).strip() or title
            base_desc = str(baseline_brief.get("description", "") or "").strip()
            if base_desc:
                current_desc = base_desc
            base_tps_raw = baseline_brief.get("talking_points", [])
            if isinstance(base_tps_raw, list):
                base_tps = [str(tp).strip() for tp in base_tps_raw if str(tp).strip()]
                if base_tps:
                    current_tps = base_tps

        continuation_context = self._build_refine_continuation_context(
            tp_key=effective_tp_key,
            sessions=sessions,
            current_title=title,
            current_description=current_desc,
            current_talking_points=current_tps,
        )
        resolution_data = self._build_refine_resolution_data_for_sessions(sessions)
        scoring_data = sm.load_scoring_report(latest.session_id) or {}
        diagnostics_data = sm.load_session_diagnostics(latest.session_id) or sm.build_session_diagnostics(latest.session_id, latest)
        graph_rows = sm.load_session_graph(latest.session_id)
        session_brief_data = sm.load_session_brief(latest.session_id) or {}

        model = ""
        if self._assistant_panel is not None:
            model = self._assistant_panel.selected_model()
        if not model:
            try:
                from config.model_prefs import load_model_prefs
                model = load_model_prefs().get("right_model", "qwen3:30b") or "qwen3:30b"
            except Exception:
                model = "qwen3:30b"

        self._manual_refine_tp_key = effective_tp_key
        self._manual_refine_session_id = latest.session_id
        self._manual_refine_worker = TopicRefineWorker(
            title=title,
            description=current_desc,
            talking_points=current_tps,
            resolution_data=resolution_data,
            scoring_data=scoring_data if isinstance(scoring_data, dict) else {},
            diagnostics_data=diagnostics_data if isinstance(diagnostics_data, dict) else {},
            graph_rows=graph_rows if isinstance(graph_rows, list) else [],
            session_brief_data=session_brief_data if isinstance(session_brief_data, dict) else {},
            continuation_context=continuation_context,
            model=model,
            tp_key=effective_tp_key,
            session_manager=sm,
            parent=self,
        )
        self._manual_refine_worker.finished.connect(self._on_manual_refine_done)
        self._manual_refine_worker.failed.connect(self._on_manual_refine_failed)
        if self._history_panel is not None:
            self._history_panel.set_refine_busy(True)
        self._manual_refine_worker.start()

    def _on_manual_refine_done(self, result: dict) -> None:
        if self._history_panel is not None:
            self._history_panel.set_refine_busy(False)

        description = str(result.get("description", "") or "").strip()
        talking_points_raw = result.get("talking_points", [])
        talking_points = [str(tp).strip() for tp in talking_points_raw if str(tp).strip()] if isinstance(talking_points_raw, list) else []
        brief = str(result.get("session_brief", "") or "").strip()

        if description:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            badge = f"✨ Manual thread refinement ({stamp})"
            if brief:
                badge += f" — {brief[:120]}"
            self._desc_edit.setPlainText(f"{badge}\n\n{description}")
        if talking_points:
            self._tp_edit.setPlainText("\n".join(talking_points))

        if self._custom_title_edit.isVisible() and self._custom_title_edit.text().strip():
            self._upsert_current_custom_debate()

        if self._history_panel is not None and self._current_tp_key:
            self._history_panel.load_sessions(self._current_tp_key)

        self._refresh_assistant_context()
        self._set_session_autosave_status("Manual refinement applied", tone="ok")
        QMessageBox.information(
            self,
            "Update from Past Sessions",
            "Debate description and talking points were updated from thread history.",
        )
        self._manual_refine_worker = None

    def _on_manual_refine_failed(self, error: str) -> None:
        if self._history_panel is not None:
            self._history_panel.set_refine_busy(False)
        self._set_session_autosave_status("Manual refinement failed", tone="err")
        QMessageBox.warning(
            self,
            "Update from Past Sessions",
            f"Manual refinement failed:\n{str(error)[:260]}",
        )
        self._manual_refine_worker = None

    def _build_refine_resolution_data_for_sessions(self, sessions: list[SessionMeta]) -> dict:
        sm = get_session_manager()
        truths: list[str] = []
        problems: list[str] = []
        sub_topics: list[str] = []
        total_memory_facts = 0

        for meta in sessions:
            for ev in sm.load_session_transcript(meta.session_id):
                if not isinstance(ev, dict):
                    continue
                etype = str(ev.get("event_type", "") or "").strip()
                payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}

                if etype == "resolution":
                    truths.extend(str(x).strip() for x in payload.get("truths_discovered", []) if str(x).strip())
                    problems.extend(str(x).strip() for x in payload.get("problems_found", []) if str(x).strip())
                    sub_topics.extend(str(x).strip() for x in payload.get("sub_topics_explored", []) if str(x).strip())
                    try:
                        total_memory_facts = max(total_memory_facts, int(payload.get("total_memory_facts", 0) or 0))
                    except Exception:
                        pass
                    continue

                if etype not in {"public_message", "turn"}:
                    continue
                text = str(payload.get("message", "") or payload.get("text", "") or "")
                if not text:
                    continue
                truths.extend(self._extract_signal_lines(text, ("TRUTH:", "VERIFIED:", "CONCLUDE:"), max_items=8))
                problems.extend(self._extract_signal_lines(text, ("PROBLEM:", "QUESTION:", "HYPOTHETICAL:", "UNRESOLVED:"), max_items=8))
                sub_topics.extend(self._extract_signal_lines(text, ("EXPAND-TOPIC:",), max_items=5))

        return {
            "truths": self._dedupe_trim(truths, max_items=80),
            "problems": self._dedupe_trim(problems, max_items=120),
            "sub_topics": self._dedupe_trim(sub_topics, max_items=80),
            "total_memory_facts": total_memory_facts,
        }

    def _build_refine_continuation_context(
        self,
        *,
        tp_key: str,
        sessions: list[SessionMeta],
        current_title: str,
        current_description: str,
        current_talking_points: list[str],
    ) -> dict:
        sm = get_session_manager()
        ordered = sorted(sessions, key=lambda m: m.start_ts)

        original_payload = {
            "title": current_title,
            "description": current_description,
            "talking_points": list(current_talking_points),
            "session_id": ordered[0].session_id if ordered else "",
        }

        if ordered:
            first_meta = ordered[0]
            first_brief = sm.load_session_brief(first_meta.session_id)
            if isinstance(first_brief, dict):
                original_payload["title"] = str(first_brief.get("title", "") or original_payload["title"]).strip()
                original_payload["description"] = str(first_brief.get("description", "") or original_payload["description"]).strip()
                raw = first_brief.get("talking_points", [])
                if isinstance(raw, list):
                    tps = [str(tp).strip() for tp in raw if str(tp).strip()]
                    if tps:
                        original_payload["talking_points"] = tps

        agreements: list[str] = []
        disagreements: list[str] = []
        open_problems: list[str] = []
        unresolved: list[str] = []
        covered_ground: list[str] = []
        transcript_signals: list[str] = []
        scoring_trend: list[str] = []

        seen_agree: set[str] = set()
        seen_disagree: set[str] = set()
        seen_problem: set[str] = set()
        seen_unresolved: set[str] = set()
        seen_covered: set[str] = set()
        seen_signal: set[str] = set()

        for meta in ordered:
            sid = meta.session_id
            brief = sm.load_session_brief(sid)
            if isinstance(brief, dict):
                raw_tps = brief.get("talking_points", [])
                if isinstance(raw_tps, list):
                    self._append_unique(
                        covered_ground,
                        [str(tp).strip() for tp in raw_tps if str(tp).strip()],
                        seen_covered,
                        max_items=100,
                    )

            sig = self._collect_refine_session_signals(sid)
            self._append_unique(agreements, sig.get("agreements", []), seen_agree, max_items=80)
            self._append_unique(disagreements, sig.get("disagreements", []), seen_disagree, max_items=80)
            self._append_unique(open_problems, sig.get("problems", []), seen_problem, max_items=120)
            self._append_unique(unresolved, sig.get("unresolved", []), seen_unresolved, max_items=120)
            self._append_unique(transcript_signals, sig.get("signals", []), seen_signal, max_items=120)

            score = sm.load_scoring_report(sid)
            if isinstance(score, dict):
                winner = str(score.get("winner", "Draw") or "Draw").strip()
                margin = str(score.get("margin", "") or "").strip()
                turns = int(score.get("turns", 0) or 0)
                summary = str(score.get("summary", "") or "").strip()
                scoring_trend.append(
                    f"{sid}: winner={winner} margin={margin or 'n/a'} turns={turns} summary={summary[:160]}"
                )

        return {
            "thread_session_count": len(ordered),
            "thread_session_ids": [m.session_id for m in ordered],
            "tp_key": tp_key,
            "original": original_payload,
            "current": {
                "title": current_title,
                "description": current_description,
                "talking_points": list(current_talking_points),
                "session_id": sessions[0].session_id if sessions else "",
            },
            "agreements": agreements,
            "disagreements": disagreements,
            "open_problems": open_problems,
            "unresolved": unresolved,
            "covered_ground": covered_ground,
            "transcript_signals": transcript_signals,
            "scoring_trend": scoring_trend[:24],
        }

    def _collect_refine_session_signals(self, session_id: str) -> dict:
        sm = get_session_manager()
        events = sm.load_session_transcript(session_id)

        agreements: list[str] = []
        disagreements: list[str] = []
        problems: list[str] = []
        unresolved: list[str] = []
        signals: list[str] = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            etype = str(ev.get("event_type", "") or "").strip()
            payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}

            if etype == "resolution":
                agreements.extend(str(x).strip() for x in payload.get("truths_discovered", []) if str(x).strip())
                problems.extend(str(x).strip() for x in payload.get("problems_found", []) if str(x).strip())
                continue

            if etype not in {"public_message", "turn"}:
                continue

            text = str(payload.get("message", "") or payload.get("text", "") or "")
            if not text:
                continue

            agreements.extend(self._extract_signal_lines(text, ("TRUTH:", "VERIFIED:", "CONCLUDE:"), max_items=8))
            disagreements.extend(self._extract_signal_lines(text, ("CONTRADICT:", "FALSE:"), max_items=8))
            problems.extend(self._extract_signal_lines(text, ("PROBLEM:",), max_items=8))
            unresolved.extend(self._extract_signal_lines(text, ("QUESTION:", "HYPOTHETICAL:", "UNRESOLVED:"), max_items=8))

            for line in text.splitlines():
                s = " ".join(line.split()).strip()
                if not s:
                    continue
                if s.endswith("?") and len(s) > 30:
                    unresolved.append(s)
                if any(tok in s.upper() for tok in ("PROBLEM:", "QUESTION:", "UNRESOLVED:", "HYPOTHETICAL:")):
                    signals.append(s[:260])

        return {
            "agreements": self._dedupe_trim(agreements, max_items=60),
            "disagreements": self._dedupe_trim(disagreements, max_items=60),
            "problems": self._dedupe_trim(problems, max_items=100),
            "unresolved": self._dedupe_trim(unresolved, max_items=100),
            "signals": self._dedupe_trim(signals, max_items=100),
        }

    @staticmethod
    def _extract_signal_lines(text: str, prefixes: tuple[str, ...], max_items: int = 8) -> list[str]:
        out: list[str] = []
        upper_prefixes = tuple(p.upper() for p in prefixes)
        for line in text.splitlines():
            s = " ".join(line.split()).strip()
            if not s:
                continue
            us = s.upper()
            for pref in upper_prefixes:
                if us.startswith(pref):
                    item = s[len(pref):].strip(" -:\t")
                    if item:
                        out.append(item[:260])
                    break
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _dedupe_trim(items: list[str], max_items: int = 40) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in items:
            s = " ".join(str(raw).split()).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _append_unique(target: list[str], additions: list[str], seen: set[str], *, max_items: int) -> None:
        for raw in additions:
            s = " ".join(str(raw).split()).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            target.append(s)
            if len(target) >= max_items:
                break

    @staticmethod
    def _strip_refinement_badge(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("✨"):
            lines = lines[1:]
            if lines and not lines[0].strip():
                lines = lines[1:]
        return "\n".join(lines).strip()

    def _current_editor_title(self) -> str:
        if self._custom_title_edit.isVisible():
            return self._custom_title_edit.text().strip() or self._preset_title_lbl.text().strip()
        return self._preset_title_lbl.text().strip()

    # ── File ingestion ───────────────────────────────────────────────────────

    def _add_files(self, paths: list[str]) -> None:
        for p_str in paths:
            p = Path(p_str)
            if p not in self._queued_files:
                self._queued_files.append(p)
                icon = "📁" if p.is_dir() else "📄"
                self._file_list.addItem(QListWidgetItem(f"{icon}  {p.name}   ({p})"))
        # Auto-check the toggle as soon as files are queued (browse or drop)
        if paths:
            self._use_ingest_cb.setChecked(True)

    def _browse(self) -> None:
        """Show a menu to pick files or a folder for ingestion."""
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #0d1520; color: #dce8f5;"
            " border: 1px solid #2a3a55; font-size: 11px; }"
            "QMenu::item:selected { background: #1e3a5f; }"
        )
        files_action = menu.addAction("📄  Select Files…")
        folder_action = menu.addAction("📁  Select Folder (all files inside)…")

        btn = self.sender()
        pos = btn.mapToGlobal(btn.rect().bottomLeft()) if btn else self.cursor().pos()
        chosen = menu.exec(pos)

        sm = get_session_manager()
        knowledge_dir = sm.root.parent / "knowledge_base"
        knowledge_dir.mkdir(parents=True, exist_ok=True)

        if chosen == files_action:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select files to ingest", str(knowledge_dir), "All files (*.*)"
            )
            if files:
                self._add_files(files)
                self._use_ingest_cb.setChecked(True)

        elif chosen == folder_action:
            folder = QFileDialog.getExistingDirectory(
                self, "Select folder to ingest", str(knowledge_dir),
            )
            if folder:
                folder_path = Path(folder)
                # Recursively collect all files in the folder
                all_files = [
                    str(f) for f in folder_path.rglob("*") if f.is_file()
                ]
                if all_files:
                    self._add_files(all_files)
                    self._use_ingest_cb.setChecked(True)

    def _clear_files(self) -> None:
        self._queued_files.clear()
        self._file_list.clear()

    def _browse_repo(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select repository folder")
        if folder:
            self._set_repo_busy(True, "Linking repository…")
            self._repo_path_edit.setText(folder)
            try:
                repo_path = Path(folder)
                if self._is_generic_custom_title(self._custom_title_edit.text()):
                    self._custom_title_edit.setText(self._derive_repo_topic_title(repo_path))

                if not self._desc_edit.toPlainText().strip():
                    repo_name = repo_path.name.strip() or "repository"
                    self._desc_edit.setPlainText(
                        f"Debate the architecture, reliability, and trade-offs in the {repo_name} codebase. "
                        "Focus on concrete modules, data flows, and operational risks based on repository evidence."
                    )

                saved = self._upsert_current_custom_debate()
                if saved is not None:
                    self._repo_prep_status.setText(f"✓ Repo linked — widget updated: {saved.title}")
                else:
                    self._repo_prep_status.setText("✓ Repo linked")
            finally:
                self._set_repo_busy(False)

    def _on_custom_mode_changed(self) -> None:
        is_repo_mode = self._mode_repo_rb.isChecked() and self._custom_title_edit.isVisible()
        self._repo_path_edit.setVisible(is_repo_mode)
        self._repo_browse_btn.setVisible(is_repo_mode)
        self._repo_hint_lbl.setVisible(is_repo_mode)
        self._repo_progress.setVisible(is_repo_mode and self._repo_progress.isVisible())
        self._repo_prep_btn.setVisible(is_repo_mode)
        self._repo_allow_cloud_cb.setChecked(False)
        self._repo_allow_cloud_cb.setVisible(False)
        self._repo_model_combo.setVisible(is_repo_mode)
        self._repo_prep_status.setVisible(is_repo_mode)
        self._repo_rebuild_btn.setVisible(is_repo_mode)
        if is_repo_mode:
            self._load_repo_prep_models(fetch=True)

    def _on_repo_cloud_toggled(self, _checked: bool) -> None:
        self._repo_allow_cloud_cb.setChecked(False)
        self._load_repo_prep_models(fetch=False)

    @staticmethod
    def _is_cloud_model_name(name: str) -> bool:
        lower = name.lower().strip()
        if ":" in lower:
            tag = lower.split(":", 1)[1]
            if "cloud" in tag:
                return True
        return lower.endswith("-cloud")

    def _load_repo_prep_models(self, *, fetch: bool = False) -> None:
        previous = self._repo_model_combo.currentData()
        if fetch or not self._repo_all_models:
            try:
                import httpx
                resp = httpx.get("http://localhost:11434/api/tags", timeout=12.0)
                resp.raise_for_status()
                data = resp.json()
                models = [
                    str(m.get("name", "")).strip()
                    for m in data.get("models", [])
                    if str(m.get("name", "")).strip()
                ]
                self._repo_all_models = sorted(set(models))
            except Exception:
                self._repo_all_models = []

        try:
            from config.model_prefs import load_model_prefs
            fallback_model = load_model_prefs().get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            fallback_model = "qwen3:30b"

        if not self._repo_all_models:
            self._repo_all_models = [fallback_model]

        local_models = [m for m in self._repo_all_models if not self._is_cloud_model_name(m)]
        pool = local_models
        if not pool:
            pool = [fallback_model]

        self._repo_model_combo.blockSignals(True)
        self._repo_model_combo.clear()
        for model in pool:
            prefix = "☁ " if self._is_cloud_model_name(model) else "💻 "
            self._repo_model_combo.addItem(f"{prefix}{model}", model)

        target = previous if previous in pool else fallback_model
        idx = self._repo_model_combo.findData(target)
        if idx < 0 and self._repo_model_combo.count() > 0:
            idx = 0
        if idx >= 0:
            self._repo_model_combo.setCurrentIndex(idx)
        self._repo_model_combo.blockSignals(False)

    def _on_repo_prep_clicked(self) -> None:
        if self._repo_schema_worker is not None and self._repo_schema_worker.isRunning():
            return

        if self._repo_schema_generated_once:
            reply = QMessageBox.question(
                self,
                "Rewrite Repo Prep Data",
                "Are you sure you want to rewrite all generated repo prep data?\n\n"
                "This will replace the current title, description, and talking points.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        repo_path = Path(self._repo_path_edit.text().strip())
        if not repo_path.exists() or not repo_path.is_dir():
            QMessageBox.warning(
                self,
                "Repo Watchdog",
                "Choose a valid repository folder before running repo prep.",
            )
            return

        try:
            size_bytes = self._repo_watchdog.estimate_repo_size_bytes(str(repo_path))
        except Exception:
            size_bytes = 0
        if size_bytes >= 3 * 1024 * 1024 * 1024:
            size_gb = size_bytes / (1024 ** 3)
            reply = QMessageBox.warning(
                self,
                "Large Repository",
                f"This repository is approximately {size_gb:.2f} GB of included files.\n\n"
                "Building the semantic dataset may be memory and time intensive.\n"
                "Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._set_repo_busy(True, "Building repository snapshot…")
        user_intent = (
            f"Title draft:\n{self._custom_title_edit.text().strip()}\n\n"
            f"Description draft:\n{self._desc_edit.toPlainText().strip()}\n\n"
            f"Talking points draft:\n{self._tp_edit.toPlainText().strip()}"
        )
        try:
            snapshot = self._repo_watchdog.build_snapshot(str(repo_path))
            brief = self._repo_watchdog.build_context_brief(snapshot)
        except Exception as exc:
            self._set_repo_busy(False)
            QMessageBox.critical(
                self,
                "Repo Watchdog",
                f"Failed to build repository snapshot for prep:\n{exc}",
            )
            return

        selected_model = self._repo_model_combo.currentData()
        model = str(selected_model).strip() if selected_model else "qwen3:30b"

        self._repo_prep_btn.setEnabled(False)
        self._repo_prep_status.setText(f"Preparing debate schema with {model}…")

        self._repo_schema_worker = _RepoSchemaWorker(
            repo_path=str(repo_path.resolve()),
            repo_brief=brief,
            user_intent=user_intent,
            model=model,
            parent=self,
        )
        self._repo_schema_worker.finished.connect(self._on_repo_prep_done)
        self._repo_schema_worker.failed.connect(self._on_repo_prep_failed)
        self._repo_schema_worker.start()

    def _on_repo_prep_done(self, result: dict) -> None:
        self._set_repo_busy(False)
        self._repo_prep_btn.setEnabled(True)
        self._repo_schema_generated_once = True

        existing_desc = self._desc_edit.toPlainText().strip()
        existing_tps = [ln.strip() for ln in self._tp_edit.toPlainText().splitlines() if ln.strip()]

        title = result.get("title", "").strip()
        description = result.get("description", "").strip()
        tps = result.get("talking_points", [])
        prep_schema = result.get("prep_schema", {})

        schema_lines: list[str] = []
        if isinstance(prep_schema, dict):
            labels = [
                ("modules_to_probe", "Modules to Probe"),
                ("risky_areas", "Risky Areas"),
                ("evidence_checklist", "Evidence Checklist"),
                ("watch_signals", "Watch Signals"),
            ]
            for key, label in labels:
                values = prep_schema.get(key, [])
                if isinstance(values, list) and values:
                    schema_lines.append(f"{label}:")
                    schema_lines.extend(f"- {str(v).strip()}" for v in values[:8] if str(v).strip())
                    schema_lines.append("")

        if title:
            self._custom_title_edit.setText(title)
        if description:
            final_desc = self._merge_preserved_user_context(description, existing_desc)
            if schema_lines:
                final_desc += "\n\nDEBATE PREP SCHEMA:\n" + "\n".join(schema_lines).strip()
            self._desc_edit.setPlainText(final_desc)
        if tps:
            merged_tps = self._merge_preserved_talking_points(
                [str(tp).strip() for tp in tps if str(tp).strip()],
                existing_tps,
            )
            self._tp_edit.setPlainText("\n".join(merged_tps))

        self._upsert_current_custom_debate()
        self._repo_prep_status.setText("✓ Repo schema ready — review and start debate")
        self._refresh_assistant_context()

    @staticmethod
    def _merge_preserved_user_context(generated_desc: str, existing_desc: str) -> str:
        g = (generated_desc or "").strip()
        e = (existing_desc or "").strip()
        if not e:
            return g

        g_lower = g.lower()
        if "preserved user intent" in g_lower:
            return g

        preserve_lines = TopicPickerDialog._extract_preserve_lines(e, max_lines=8)
        if not preserve_lines:
            return g

        block = "\n".join(f"- {ln}" for ln in preserve_lines)
        return (
            f"{g}\n\n"
            "Preserved User Intent:\n"
            f"{block}"
        ).strip()

    @staticmethod
    def _merge_preserved_talking_points(generated_tps: list[str], existing_tps: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        def _add(item: str) -> None:
            s = " ".join((item or "").split()).strip()
            if not s:
                return
            key = s.lower()
            if key in seen:
                return
            seen.add(key)
            merged.append(s)

        for tp in existing_tps[:4]:
            _add(tp)
        for tp in generated_tps:
            _add(tp)

        return merged[:12]

    @staticmethod
    def _extract_preserve_lines(text: str, max_lines: int = 8) -> list[str]:
        import re

        src = (text or "").strip()
        if not src:
            return []

        lines: list[str] = []
        for part in re.split(r"\n+|(?<=[.!?])\s+", src):
            s = " ".join(part.split()).strip("-• ")
            if len(s) < 24:
                continue
            if len(s) > 210:
                s = s[:207].rstrip() + "..."
            lines.append(s)
            if len(lines) >= max_lines:
                break
        return lines

    def _on_repo_prep_failed(self, error: str) -> None:
        self._set_repo_busy(False)
        self._repo_prep_btn.setEnabled(True)
        self._repo_prep_status.setText(f"✗ {error[:100]}")

    def _on_rebuild_dataset_clicked(self) -> None:
        if hasattr(self, "_rebuild_worker") and self._rebuild_worker is not None:
            if self._rebuild_worker.isRunning():
                return
        repo_path = self._repo_path_edit.text().strip()
        if not repo_path:
            QMessageBox.warning(self, "Rebuild Dataset", "Link a repository folder first.")
            return
        from pathlib import Path
        if not Path(repo_path).is_dir():
            QMessageBox.warning(self, "Rebuild Dataset", f"Folder not found:\n{repo_path}")
            return
        self._set_repo_busy(True, "🔄 Rebuilding dataset…")
        self._repo_rebuild_btn.setEnabled(False)
        self._repo_prep_status.setText("🔄 Rebuilding dataset…")
        self._repo_prep_status.setVisible(True)
        from core.session_manager import get_session_manager
        session_root = get_session_manager().root
        self._rebuild_worker = _RebuildDatasetWorker(
            repo_path=repo_path,
            session_root=session_root,
            parent=self,
        )
        self._rebuild_worker.finished.connect(self._on_rebuild_done)
        self._rebuild_worker.failed.connect(self._on_rebuild_failed)
        self._rebuild_worker.start()

    def _on_rebuild_done(self, fact_count: int) -> None:
        self._set_repo_busy(False)
        self._repo_rebuild_btn.setEnabled(True)
        self._repo_prep_status.setText(f"✓ Dataset rebuilt — {fact_count} chunks")

    def _on_rebuild_failed(self, error: str) -> None:
        self._set_repo_busy(False)
        self._repo_rebuild_btn.setEnabled(True)
        self._repo_prep_status.setText(f"✗ {error[:100]}")

    # ── AI Rewrite ───────────────────────────────────────────────────────────

    def _on_rewrite_clicked(self) -> None:
        if self._rewrite_worker is not None and self._rewrite_worker.isRunning():
            return  # already running

        title = (
            self._custom_title_edit.text().strip()
            if self._custom_title_edit.isVisible()
            else self._preset_title_lbl.text()
        )
        description = self._desc_edit.toPlainText().strip()
        talking_points = self._tp_edit.toPlainText().strip()

        if not (title or talking_points):
            self._rewrite_status.setText("Add a title or talking points first.")
            return

        # Load dataset context from global _datasets/ if available
        dataset_context = self._load_dataset_context()

        # Determine model — use left model preference (default qwen3:30b)
        try:
            from config.model_prefs import load_model_prefs
            prefs = load_model_prefs()
            model = prefs.get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            model = "qwen3:30b"

        self._rewrite_btn.setEnabled(False)
        self._rewrite_status.setText(f"Rewriting with {model}…")

        self._rewrite_worker = _RewriteWorker(
            title=title,
            description=description,
            talking_points=talking_points or "(no talking points yet — generate from scratch)",
            model=model,
            dataset_context=dataset_context,
            parent=self,
        )
        self._rewrite_worker.finished.connect(self._on_rewrite_done)
        self._rewrite_worker.failed.connect(self._on_rewrite_failed)
        self._rewrite_worker.start()

    def _load_dataset_context(self) -> str:
        """Read the most recently created global dataset for context."""
        try:
            from ingestion.ingestion_agent import list_global_datasets
            from core.session_manager import get_session_manager
            import json
            sm = get_session_manager()
            datasets = list_global_datasets(sm.root)
            if not datasets:
                return ""
            # Sort by creation time desc — take newest
            datasets.sort(key=lambda d: d.get("created", ""), reverse=True)
            newest = datasets[0]
            import json as _json
            raw = _json.loads(Path(newest["path"]).read_text(encoding="utf-8"))
            facts = raw.get("facts", [])
            # Extract top 30 highest-weighted fact texts for context
            facts_sorted = sorted(
                facts, key=lambda f: f.get("tfidf_weight", 0), reverse=True
            )[:30]
            lines = [f.get("text", "")[:200] for f in facts_sorted if f.get("text")]
            context = f"[Dataset: {newest['name']}]\n" + "\n---\n".join(lines)
            return context
        except Exception:
            return ""

    def _on_rewrite_done(self, improved_text: str) -> None:
        self._tp_edit.setPlainText(improved_text)
        self._rewrite_btn.setEnabled(True)
        self._rewrite_status.setText("✓ Rewritten — review and edit as needed")

    def _on_rewrite_failed(self, error: str) -> None:
        self._rewrite_btn.setEnabled(True)
        self._rewrite_status.setText(f"✗ {error[:80]}")

    def _on_assistant_topic_created(self, result: dict) -> None:
        title = str(result.get("title", "")).strip()
        if not title:
            return
        description = str(result.get("description", "")).strip()
        tps = [str(tp).strip() for tp in result.get("talking_points", []) if str(tp).strip()]

        mode = "repo_watchdog" if self._mode_repo_rb.isChecked() else "static_ingestion"
        repo_path = self._repo_path_edit.text().strip() if mode == "repo_watchdog" else ""

        # If we are currently editing a custom debate, update it in place (no duplicate widget).
        target_id = (self._active_custom_id or "").strip()
        if not target_id and 0 <= self._selected_index < len(self._card_entries):
            selected = self._card_entries[self._selected_index]
            if selected.get("kind") == "custom":
                target_id = str(selected.get("id", "")).strip()
        if not target_id:
            current_title = self._custom_title_edit.text().strip().lower()
            if current_title:
                match = next(
                    (d for d in self._custom_debates if d.title.strip().lower() == current_title),
                    None,
                )
                if match is not None:
                    target_id = match.id

        saved = self._custom_debate_store.upsert(
            debate_id=target_id,
            title=title,
            description=description,
            talking_points=tps,
            mode=mode,
            repo_path=repo_path,
        )
        self._skip_next_auto_seed_from_session = True
        self._active_custom_id = saved.id
        self._load_persisted_custom_debates()
        self._rebuild_left_cards()
        self._select_custom_card_by_id(saved.id)

    # ── Confirm ──────────────────────────────────────────────────────────────

    def _on_confirm(self) -> None:
        entry = self._card_entries[self._selected_index] if self._card_entries else {"kind": "starter", "index": 0}
        topic = None
        is_custom = False
        if entry.get("kind") == "starter":
            topic = STARTER_TOPICS[int(entry.get("index", 0))]
            is_custom = _is_custom_topic_title(topic.title)
        else:
            is_custom = True

        if is_custom:
            title = self._custom_title_edit.text().strip()
            if not title:
                self._custom_title_edit.setStyleSheet(
                    f"QLineEdit {{ border: 1px solid #ef5350; background: {_PALETTE['bg']};"
                    " color: #e8eaf6; border-radius: 6px; padding: 8px 12px; font-size: 14px; }}"
                )
                return
            saved = self._upsert_current_custom_debate()
            if saved is None:
                return
            if not self._current_tp_key:
                self._current_tp_key = _custom_debate_tp_key(saved.id)
        else:
            assert topic is not None
            title = topic.title

        desc = self._desc_edit.toPlainText().strip()
        if desc.startswith("✨"):
            newline_idx = desc.find("\n\n")
            if newline_idx != -1:
                desc = desc[newline_idx + 2:].strip()

        tp_raw = self._tp_edit.toPlainText().strip()
        tp_lines = [ln.strip() for ln in tp_raw.split("\n") if ln.strip()]

        selected_tp_key = self.tp_key

        # Persist the exact brief config so the session is reproducible and replayable.
        get_session_manager().pending_session_brief = {
            "title": title,
            "description": desc,
            "talking_points": tp_lines,
            "mode": "repo_watchdog" if (is_custom and self._mode_repo_rb.isChecked()) else "static_ingestion",
            "repo_path": self._repo_path_edit.text().strip() if (is_custom and self._mode_repo_rb.isChecked()) else "",
            "debate_id": self._active_custom_id or "",
            "talking_point_key": selected_tp_key,
            "source_session_id": self._seed_source_session_id,
        }

        if tp_lines:
            tp_block = (
                "\n\nKEY TALKING POINTS — anchor every argument to these:\n"
                + "\n".join(f"  • {ln}" for ln in tp_lines)
            )
        else:
            tp_block = ""

        full_context = f"{title}\n\n{desc}{tp_block}".strip()

        if is_custom and self._mode_repo_rb.isChecked():
            repo_path = Path(self._repo_path_edit.text().strip())
            if not repo_path.exists() or not repo_path.is_dir():
                QMessageBox.warning(
                    self,
                    "Repo Watchdog",
                    "Choose a valid repository folder before starting this custom debate.",
                )
                return

            try:
                size_bytes = self._repo_watchdog.estimate_repo_size_bytes(str(repo_path))
            except Exception:
                size_bytes = 0
            if size_bytes >= 3 * 1024 * 1024 * 1024:
                size_gb = size_bytes / (1024 ** 3)
                reply = QMessageBox.warning(
                    self,
                    "Large Repository",
                    f"This repository is approximately {size_gb:.2f} GB of included files.\n\n"
                    "Starting a repo-linked debate will build/refresh a large semantic dataset.\n"
                    "Continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            try:
                self._set_repo_busy(True, "Building repository snapshot for debate start…")
                snapshot = self._repo_watchdog.build_snapshot(str(repo_path))
                brief = self._repo_watchdog.build_context_brief(snapshot)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Repo Watchdog",
                    f"Failed to build repository snapshot:\n{exc}",
                )
                return
            finally:
                self._set_repo_busy(False)

            full_context = (
                f"{full_context}\n\n"
                "REPO MODE: This debate is linked to a live repository.\n"
                "Use the repository snapshot for global understanding, then inspect real files for final claims.\n\n"
                f"{brief}"
            )

            marker = {
                "mode": "repo_watchdog",
                "repo_path": str(repo_path.resolve()),
                "snapshot_generated_at": snapshot.generated_at,
            }
            full_context += f"\n\n[REPO_WATCHDOG_META]{json.dumps(marker)}[/REPO_WATCHDOG_META]"

        files = (
            [str(f) for f in self._queued_files]
            if self._use_ingest_cb.isChecked()
            else []
        )
        self.topic_confirmed.emit(title, full_context, files)
        self.accept()

    def _resolve_custom_tp_key(self, debate: CustomDebate) -> str:
        """Resolve custom debate key with backward compatibility for legacy threads."""
        sm = get_session_manager()
        stable = _custom_debate_tp_key(debate.id)
        stable_sessions = sm.list_sessions_for_tp(sm.effective_tp_key(stable))
        if stable_sessions:
            return stable

        legacy_first_tp = (debate.talking_points[0] if debate.talking_points else "").strip()
        legacy = _tp_key(debate.title, legacy_first_tp)
        legacy_sessions = sm.list_sessions_for_tp(sm.effective_tp_key(legacy))
        return legacy if legacy_sessions else stable

    def _load_session_brief_into_editor(self, session_id: str) -> None:
        data = get_session_manager().load_session_brief(session_id)
        if not isinstance(data, dict):
            self._active_session_brief_id = ""
            self._set_session_autosave_status("", tone="dim")
            return

        title = str(data.get("title", "")).strip()
        desc = str(data.get("description", "")).strip()
        tps = data.get("talking_points", [])
        mode = str(data.get("mode", "")).strip().lower()
        repo_path = str(data.get("repo_path", "")).strip()

        self._suspend_session_autosave = True
        try:
            if self._custom_title_edit.isVisible() and title:
                self._custom_title_edit.setText(title)
            if desc:
                self._desc_edit.setPlainText(desc)
            if isinstance(tps, list):
                clean_tps = [str(tp).strip() for tp in tps if str(tp).strip()]
                if clean_tps:
                    self._tp_edit.setPlainText("\n".join(clean_tps))

            if self._custom_title_edit.isVisible():
                if mode == "repo_watchdog":
                    self._mode_repo_rb.setChecked(True)
                elif mode == "static_ingestion":
                    self._mode_static_rb.setChecked(True)
                if repo_path:
                    self._repo_path_edit.setText(repo_path)
        finally:
            self._suspend_session_autosave = False

        self._seed_source_session_id = session_id
        self._active_session_brief_id = session_id
        self._set_session_autosave_status(
            f"Editing {session_id} · autosave ON",
            tone="info",
        )

    def _schedule_active_session_autosave(self) -> None:
        if self._suspend_session_autosave:
            return
        if not self._active_session_brief_id:
            return
        self._set_session_autosave_status("Saving…", tone="dim")
        self._session_autosave_timer.start()

    def _autosave_active_session_brief(self) -> None:
        session_id = self._active_session_brief_id.strip()
        if not session_id:
            return

        title = (
            self._custom_title_edit.text().strip()
            if self._custom_title_edit.isVisible()
            else self._preset_title_lbl.text().strip()
        )
        description = self._desc_edit.toPlainText().strip()
        talking_points = [ln.strip() for ln in self._tp_edit.toPlainText().split("\n") if ln.strip()]

        mode = ""
        repo_path = ""
        if self._custom_title_edit.isVisible():
            mode = "repo_watchdog" if self._mode_repo_rb.isChecked() else "static_ingestion"
            repo_path = self._repo_path_edit.text().strip() if mode == "repo_watchdog" else ""

        ok = get_session_manager().save_session_brief(
            session_id,
            {
                "title": title,
                "description": description,
                "talking_points": talking_points,
                "mode": mode,
                "repo_path": repo_path,
            },
            merge_existing=True,
        )
        if ok:
            self._seed_source_session_id = session_id
            stamp = datetime.now().strftime("%H:%M:%S")
            self._set_session_autosave_status(f"Saved to {session_id} at {stamp}", tone="ok")
        else:
            self._set_session_autosave_status(f"Autosave failed for {session_id}", tone="err")

    def _set_session_autosave_status(self, text: str, tone: str = "dim") -> None:
        if self._session_autosave_lbl is None:
            return
        palette = {
            "dim": _PALETTE["dim"],
            "info": "#80cbc4",
            "ok": "#66bb6a",
            "err": "#ef9a9a",
        }
        color = palette.get(tone, _PALETTE["dim"])
        self._session_autosave_lbl.setStyleSheet(f"color: {color}; font-size: 10px;")
        self._session_autosave_lbl.setText(text)

    # ── Style ────────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QDialog   {{ background: {_PALETTE['bg']}; }}
            QGroupBox#detailGroup {{
                color: {_PALETTE['accent']}; font-weight: 600; font-size: 10px;
                border: 1px solid {_PALETTE['border']}; border-radius: 8px;
                margin-top: 8px; padding: 10px 10px 8px 10px;
            }}
            QGroupBox#detailGroup::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 4px;
            }}
            QGroupBox#detailGroup::indicator {{
                width: 14px; height: 14px;
            }}
            QSplitter {{ background: {_PALETTE['bg']}; }}
            QScrollBar:vertical {{
                background: {_PALETTE['bg']}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: #2a3a55; border-radius: 3px;
            }}
        """)
