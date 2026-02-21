"""Topic Picker Dialog — full rich topic browser with inline Custom editor.

Replaces the dropdown. Opens as a large pop-out:
  Left column  — clickable topic cards (title + 2-line teaser)
  Right panel  — full description + talking points for selected topic
                 Editable when "Custom Topic" is selected; read-only for presets
  Bottom row   — file ingestion toggle + "Start this topic" confirm

Emits: topic_confirmed(title: str, full_context: str, file_paths: list[str])
  full_context is the complete brief fed to agents.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
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
    QPushButton,
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
from config.autonomous_topics import (
    AutonomousTopic, get_autonomous_store,
)
from core.session_manager import get_session_manager, SessionMeta
from core.segue_buffer import get_segue_buffer, SegueEntry
from ingestion.ingestion_agent import get_datasets_dir, list_global_datasets


# ─────────────────────────────────────────────
def _tp_key(topic_title: str, tp_text: str) -> str:
    """Return a stable 12-char hex key for a topic+talking-point pair."""
    raw = (topic_title + tp_text[:60]).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


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

    def __init__(self, index: int, topic: StarterTopic, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self._selected = False
        self.setFixedWidth(_CARD_W)
        self.setObjectName("topicCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        title_lbl = QLabel(topic.title)
        title_lbl.setWordWrap(True)
        title_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {_PALETTE['accent']}; background: transparent;")
        lay.addWidget(title_lbl)

        # Two-line teaser from description
        teaser = (topic.description[:160] + "…") if len(topic.description) > 160 else topic.description
        teaser_lbl = QLabel(teaser)
        teaser_lbl.setWordWrap(True)
        teaser_lbl.setFont(QFont("Segoe UI", 9))
        teaser_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        lay.addWidget(teaser_lbl)

        if topic.talking_points:
            tp_count = QLabel(f"  {len(topic.talking_points)} talking points")
            tp_count.setFont(QFont("Segoe UI", 8))
            tp_count.setStyleSheet(f"color: #2a7a8a; background: transparent;")
            lay.addWidget(tp_count)

        self._refresh_style()

    def _refresh_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                f"QFrame#topicCard {{ background: {_PALETTE['card_sel']};"
                f" border: 2px solid {_PALETTE['border_sel']}; border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#topicCard {{ background: {_PALETTE['card']};"
                f" border: 1px solid {_PALETTE['border']}; border-radius: 10px; }}"
            )

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


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
    start_requested  = pyqtSignal()      # user hit Start / Continue
    thread_reset     = pyqtSignal()      # user successfully reset thread
    data_cleared     = pyqtSignal()      # user successfully cleared all data

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._base_tp_key: str = ""
        self._session_count: int = 0
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

        self._sub = QLabel("No sessions yet for this talking point.")
        self._sub.setStyleSheet("color: #546e7a; font-size: 10px; background: transparent;")
        self._sub.setWordWrap(True)
        lay.addWidget(self._sub)

        # ── Start / Continue button ──────────────────────────────────────────
        self._start_btn = QPushButton("▶  Start Session 1")
        self._start_btn.setStyleSheet(
            "QPushButton { background: #00695c; color: #fff; font-weight: 700;"
            " border-radius: 8px; padding: 8px 18px; font-size: 12px; border: 1px solid #00897b; }"
            "QPushButton:hover { background: #00796b; }"
        )
        self._start_btn.clicked.connect(self.start_requested)
        lay.addWidget(self._start_btn)

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

    def load_sessions(self, base_tp_key: str) -> None:
        """Load sessions for the current generation of base_tp_key."""
        self._base_tp_key = base_tp_key

        # Remove old cards (all items except the stretch at end)
        while self._card_col.count() > 1:
            item = self._card_col.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        sm = get_session_manager()
        eff_key = sm.effective_tp_key(base_tp_key) if base_tp_key else ""
        sessions = sm.list_sessions_for_tp(eff_key) if eff_key else []
        self._session_count = len(sessions)

        # Thread / generation badge
        gen = sm.get_tp_generation(base_tp_key) if base_tp_key else 0
        if gen > 0:
            self._thread_lbl.setText(f"Thread {gen + 1}  ·  previous threads archived")
        else:
            self._thread_lbl.setText("")

        if sessions:
            self._sub.setText(
                f"{self._session_count} session(s) in this thread — click to view transcript."
            )
        else:
            self._sub.setText("No sessions yet in this thread.")

        # Number oldest=1 … newest=N
        for idx, meta in enumerate(reversed(sessions)):
            card = _SessionCard(meta, idx + 1)
            card.clicked.connect(self.session_clicked)
            self._card_col.insertWidget(self._card_col.count() - 1, card)

        next_num = self._session_count + 1
        self._start_btn.setText(
            f"▶  Start Session 1" if self._session_count == 0
            else f"▶  Continue — Start Session {next_num}"
        )

        # Enable management buttons only when there is a real tp_key
        has_key = bool(base_tp_key)
        self._reset_btn.setEnabled(has_key and self._session_count > 0)
        self._clear_btn.setEnabled(has_key)

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
        self.load_sessions(self._base_tp_key)
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
        self.load_sessions(self._base_tp_key)
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

            if etype == "turn":
                speaker = payload.get("speaker", "")
                text = payload.get("text", "")
                turn_n = payload.get("turn_number", "")
                if speaker == left_name:
                    color = _AGENT_COLORS.get(left_name, _DEFAULT_L)
                else:
                    color = _AGENT_COLORS.get(right_name, _DEFAULT_R)
                lines.append(
                    f"<div style='margin-bottom:10px;'>"
                    f"<span style='color:{color}; font-weight:700; font-size:10px;'>"
                    f"  {speaker}  </span>"
                    f"<span style='color:#3a5060; font-size:9px;'>turn {turn_n}</span>"
                    f"<div style='margin-top:3px; color:#b0bec5;'>{text}</div>"
                    f"</div>"
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
#  Autonomous topic card (purple)
# ──────────────────────────────────────────────────────────────────────────────

class _AutonomousTopicCard(QFrame):
    """Clickable card for an autonomous talking point — purple accent."""
    clicked = pyqtSignal(str)   # emits filename
    delete_requested = pyqtSignal(str)   # emits filename

    def __init__(self, at: AutonomousTopic, parent=None) -> None:
        super().__init__(parent)
        self._filename = at.filename
        self._selected = False
        self.setFixedWidth(_CARD_W)
        self.setObjectName("autoTopicCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        title_lbl = QLabel(at.title if len(at.title) <= 48 else at.title[:48] + "…")
        title_lbl.setWordWrap(True)
        title_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #ce93d8; background: transparent;")
        title_lbl.setToolTip(at.title)
        lay.addWidget(title_lbl)

        teaser = (at.description[:140] + "…") if len(at.description) > 140 else at.description
        teaser_lbl = QLabel(teaser)
        teaser_lbl.setWordWrap(True)
        teaser_lbl.setFont(QFont("Segoe UI", 9))
        teaser_lbl.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        lay.addWidget(teaser_lbl)

        meta_row = QHBoxLayout()
        tp_count = QLabel(f"  {len(at.talking_points)} talking points")
        tp_count.setFont(QFont("Segoe UI", 8))
        tp_count.setStyleSheet("color: #7e57c2; background: transparent;")
        meta_row.addWidget(tp_count)

        if at.segue_concept:
            origin_lbl = QLabel(f"  segue: {at.segue_concept[:30]}")
            origin_lbl.setFont(QFont("Segoe UI", 7))
            origin_lbl.setStyleSheet("color: #546e7a; background: transparent;")
            meta_row.addWidget(origin_lbl)

        meta_row.addStretch()
        lay.addLayout(meta_row)

        self._refresh_style()

    def _refresh_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "QFrame#autoTopicCard { background: #1a0d2a;"
                " border: 2px solid #9c27b0; border-radius: 10px; }"
            )
        else:
            self.setStyleSheet(
                f"QFrame#autoTopicCard {{ background: #120d1e;"
                f" border: 1px solid #2a1a3e; border-radius: 10px; }}"
            )

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._filename)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: #0d1520; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; font-size: 11px; }}"
            "QMenu::item:selected { background: #0d2030; }"
        )
        del_action = menu.addAction("🗑  Delete this talking point")
        promote_action = menu.addAction("⬆  Promote to Custom")
        chosen = menu.exec(event.globalPos())
        if chosen == del_action:
            self.delete_requested.emit(self._filename)
        elif chosen == promote_action:
            store = get_autonomous_store()
            store.promote_to_custom(self._filename)


# ──────────────────────────────────────────────────────────────────────────────
#  Topic Assistant — chatbot panel
# ──────────────────────────────────────────────────────────────────────────────

class _AssistantChatPanel(QWidget):
    """Lightweight agent chatbot for creating talking points conversationally."""

    topic_created = pyqtSignal(dict)   # emitted when assistant generates a TP

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker = None
        self.setStyleSheet(f"background: {_PALETTE['panel']};")
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        header = QLabel("Topic Assistant")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {_PALETTE['accent']}; background: transparent;")
        lay.addWidget(header)

        sub = QLabel("Chat with the assistant to create new talking points")
        sub.setFont(QFont("Segoe UI", 9))
        sub.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        lay.addWidget(sub)

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
        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "e.g. Make a talking point about submarine pressure physics..."
        )
        self._input.setStyleSheet(
            f"QLineEdit {{ background: {_PALETTE['bg']}; color: {_PALETTE['text']};"
            f" border: 1px solid {_PALETTE['border']}; border-radius: 6px;"
            f" padding: 8px 12px; font-size: 11px; }}"
            f"QLineEdit:focus {{ border-color: {_PALETTE['accent']}; }}"
        )
        self._input.returnPressed.connect(self._on_send)
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

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        new_tp_btn = QPushButton("📁 New Talking Point")
        new_tp_btn.setStyleSheet(
            f"QPushButton {{ background: #1a2840; color: #90a4ae; border-radius: 6px;"
            f" padding: 5px 14px; font-size: 10px; border: 1px solid {_PALETTE['border']}; }}"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        new_tp_btn.clicked.connect(self._on_new_tp)
        btn_row.addWidget(new_tp_btn)
        btn_row.addStretch()

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color: {_PALETTE['dim']}; font-size: 9px; background: transparent;"
        )
        btn_row.addWidget(self._status)
        lay.addLayout(btn_row)

        # Initial welcome message
        self._append_system(
            "Welcome to the Topic Assistant! I can help you create new debate "
            "talking points. Just describe what you want to explore and I'll "
            "generate a full talking point with title, description, and focal anchors."
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

    def _on_new_tp(self) -> None:
        self._append_system(
            "What would you like to explore? Describe the topic you want to create "
            "a talking point for — I'll generate the title, description, and focal "
            "anchors automatically."
        )
        self._input.setFocus()

    def _on_send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self._input.clear()
        self._append_user(text)
        self._send_btn.setEnabled(False)
        self._status.setText("Generating...")

        # Gather context
        from config.autonomous_topics import get_autonomous_store
        from config.starter_topics import get_topic_titles
        from analytics.analytics_store import get_analytics_store

        existing = get_topic_titles() + get_autonomous_store().titles()

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

        # Segue buffer summary
        try:
            buf = get_segue_buffer()
            unresolved = buf.unresolved_entries()
            if unresolved:
                segue_text = "Unresolved segue concepts:\n" + "\n".join(
                    f"  - {e.concept} (mentioned {e.mention_count}x)"
                    for e in unresolved[:10]
                )
            else:
                segue_text = ""
        except Exception:
            segue_text = ""

        # Model
        try:
            from config.model_prefs import load_model_prefs
            model = load_model_prefs().get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            model = "qwen3:30b"

        from agents.topic_assistant_worker import TopicAssistantWorker
        self._worker = TopicAssistantWorker(
            text,
            model=model,
            existing_titles=existing,
            analytics_summary=analytics_text,
            segue_context=segue_text,
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
        title = result.get("title", "New Topic")
        tp_count = len(result.get("talking_points", []))
        self._status.setText(f"✓ Created: {title}")
        self._append_system(
            f"✅ Created talking point: <b>{title}</b> with {tp_count} focal anchors. "
            "It's been added to your list."
        )
        self.topic_created.emit(result)

    def _on_worker_failed(self, error: str) -> None:
        self._send_btn.setEnabled(True)
        self._status.setText(f"✗ Failed")
        self._append_system(f"Sorry, I ran into an error: {error[:120]}")


class TopicPickerDialog(QDialog):
    """Debate Studio — full topic browser with assistant, autonomous learning,
    session history, and transcript viewer."""

    topic_confirmed = pyqtSignal(str, str, list)   # title, full_context, file_paths

    def __init__(self, current_title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Debate Studio")
        self.resize(1620, 860)
        self.setModal(True)
        self._queued_files: list[Path] = []
        self._selected_index: int = 0
        self._cards: list[_TopicCard] = []
        self._auto_cards: list[_AutonomousTopicCard] = []
        self._selected_auto_filename: str = ""
        self._rewrite_worker: _RewriteWorker | None = None
        self._promotion_worker = None
        # Per-talking-point key for session history
        self._current_tp_key: str = ""
        # References to new panels (set in _build_ui)
        self._history_panel: _SessionHistoryPanel | None = None
        self._transcript_panel: _TranscriptViewerPanel | None = None
        self._dataset_library: _DatasetLibraryPanel | None = None
        self._assistant_panel: _AssistantChatPanel | None = None
        self._main_splitter: QSplitter | None = None
        self._autonomous_learning_enabled: bool = False
        self._apply_style()
        self._build_ui(current_title)

    @property
    def tp_key(self) -> str:
        """Generation-aware key that will be written to the new session."""
        if not self._current_tp_key:
            return ""
        return get_session_manager().effective_tp_key(self._current_tp_key)

    def refresh_library(self) -> None:
        """Re-scan disk and repopulate the Knowledge Library dataset list."""
        if self._dataset_library is not None:
            self._dataset_library.refresh()

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
        h_sub = QLabel("Select a curated brief, create new talking points with the assistant, or let autonomous learning generate them")
        h_sub.setFont(QFont("Segoe UI", 9))
        h_sub.setStyleSheet(f"color: {_PALETTE['dim']}; background: transparent;")
        hl.addWidget(h_title)
        hl.addSpacing(20)
        hl.addWidget(h_sub)
        hl.addStretch()
        root.addWidget(header)

        # Splitter: left cards | right detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #1e2d42; width: 1px; }")

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

        for i, topic in enumerate(STARTER_TOPICS):
            card = _TopicCard(i, topic)
            card.clicked.connect(self._select_topic)
            self._cards.append(card)
            card_col.addWidget(card)

        # ── Autonomous topics divider + cards ────────────────────────────────
        self._auto_divider = QLabel("── AUTONOMOUS ──")
        self._auto_divider.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._auto_divider.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._auto_divider.setStyleSheet(
            "color: #7e57c2; background: transparent; margin-top: 12px;"
        )
        card_col.addWidget(self._auto_divider)

        self._auto_card_container = QVBoxLayout()
        self._auto_card_container.setSpacing(8)
        card_col.addLayout(self._auto_card_container)

        # Load existing autonomous topics
        self._refresh_auto_cards()

        card_col.addStretch()
        scroll.setWidget(card_widget)
        left_lay.addWidget(scroll)
        splitter.addWidget(left_container)

        # ── Right: detail / edit panel ───────────────────────────────────────
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

        title_row.addWidget(self._preset_title_lbl, stretch=1)
        title_row.addWidget(self._custom_title_edit, stretch=1)
        right_lay.addLayout(title_row)

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
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #263238; color: #90a4ae; border-radius: 6px;"
            " padding: 9px 22px; font-size: 13px; }"
            "QPushButton:hover { background: #37474f; color: #fff; }"
        )
        cancel_btn.clicked.connect(self.reject)

        self._confirm_btn = QPushButton("▶  Start This Topic")
        self._confirm_btn.setStyleSheet(
            f"QPushButton {{ background: {_PALETTE['ok']}; color: #fff; font-weight: 700;"
            f" border-radius: 8px; padding: 9px 26px; font-size: 13px;"
            f" border: 1px solid #00897b; }}"
            f"QPushButton:hover {{ background: {_PALETTE['ok_hover']}; }}"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)

        confirm_row.addWidget(self._topic_badge)
        confirm_row.addStretch()
        confirm_row.addWidget(cancel_btn)
        confirm_row.addSpacing(8)
        confirm_row.addWidget(self._confirm_btn)
        right_lay.addLayout(confirm_row)

        splitter.addWidget(right_container)

        # ── Session history panel ────────────────────────────────────────────
        self._history_panel = _SessionHistoryPanel()
        self._history_panel.start_requested.connect(self._on_confirm)
        self._history_panel.session_clicked.connect(self._on_session_card_clicked)
        # When user resets or clears, re-select the topic so the detail panel
        # also refreshes (strips stale refinement badge, etc.)
        self._history_panel.thread_reset.connect(lambda: self._select_topic(self._selected_index))
        self._history_panel.data_cleared.connect(lambda: self._select_topic(self._selected_index))
        splitter.addWidget(self._history_panel)

        # ── Transcript viewer panel (starts collapsed) ───────────────────────
        self._transcript_panel = _TranscriptViewerPanel()
        splitter.addWidget(self._transcript_panel)
        self._transcript_panel.set_splitter(splitter, 3)

        # ── Topic Assistant panel (5th pane) ─────────────────────────────────
        self._assistant_panel = _AssistantChatPanel()
        self._assistant_panel.topic_created.connect(self._on_assistant_topic_created)
        splitter.addWidget(self._assistant_panel)

        splitter.setSizes([_CARD_W + 32, 480, 280, 0, 320])
        self._main_splitter = splitter
        root.addWidget(splitter, stretch=1)

        # ── Bottom bar: autonomous learning controls ─────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(44)
        bottom.setStyleSheet(
            f"background: #060b12; border-top: 1px solid {_PALETTE['border']};"
        )
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(16, 0, 16, 0)
        bl.setSpacing(12)

        self._auto_cb = QCheckBox("Autonomous Learning")
        self._auto_cb.setStyleSheet(
            f"QCheckBox {{ color: #7e57c2; font-size: 11px; font-weight: 700; }}"
            f"QCheckBox:checked {{ color: #ce93d8; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        self._auto_cb.setToolTip(
            "When enabled, the system automatically promotes recurring "
            "lateral references (segues) into new talking points."
        )
        self._auto_cb.toggled.connect(self._on_auto_learning_toggled)
        bl.addWidget(self._auto_cb)

        self._auto_start_btn = QPushButton("▶ Start")
        self._auto_start_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #7e57c2; border-radius: 6px;"
            " padding: 4px 14px; font-size: 10px; border: 1px solid #2a1a3e; }"
            "QPushButton:hover { background: #311b92; color: #fff; }"
            "QPushButton:disabled { background: #0d1520; color: #37474f; }"
        )
        self._auto_start_btn.setEnabled(False)
        self._auto_start_btn.clicked.connect(self._start_autonomous_learning)
        bl.addWidget(self._auto_start_btn)

        self._auto_stop_btn = QPushButton("■ Stop")
        self._auto_stop_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #ef5350; border-radius: 6px;"
            " padding: 4px 14px; font-size: 10px; border: 1px solid #3e1a1a; }"
            "QPushButton:hover { background: #b71c1c; color: #fff; }"
            "QPushButton:disabled { background: #0d1520; color: #37474f; }"
        )
        self._auto_stop_btn.setEnabled(False)
        self._auto_stop_btn.clicked.connect(self._stop_autonomous_learning)
        bl.addWidget(self._auto_stop_btn)

        thresh_lbl = QLabel("Threshold:")
        thresh_lbl.setStyleSheet(
            f"color: {_PALETTE['dim']}; font-size: 10px; background: transparent;"
        )
        bl.addWidget(thresh_lbl)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(2, 10)
        self._threshold_slider.setValue(3)
        self._threshold_slider.setFixedWidth(100)
        self._threshold_slider.setToolTip("Minimum mention count before a segue is promoted")
        self._threshold_slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #1a2840; height: 4px; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #7e57c2; width: 12px; height: 12px;"
            " margin: -4px 0; border-radius: 6px; }"
        )
        bl.addWidget(self._threshold_slider)

        self._threshold_val = QLabel("3")
        self._threshold_val.setFixedWidth(18)
        self._threshold_val.setStyleSheet(
            "color: #7e57c2; font-size: 10px; font-weight: 700; background: transparent;"
        )
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_val.setText(str(v))
        )
        bl.addWidget(self._threshold_val)

        bl.addStretch()

        self._auto_status = QLabel("")
        self._auto_status.setStyleSheet(
            f"color: {_PALETTE['dim']}; font-size: 9px; background: transparent;"
        )
        bl.addWidget(self._auto_status)

        root.addWidget(bottom)

        # Select initial topic
        start_idx = next(
            (i for i, t in enumerate(STARTER_TOPICS) if t.title == current_title), 0
        )
        self._select_topic(start_idx)

    # ── Topic selection ──────────────────────────────────────────────────────

    def _select_topic(self, index: int) -> None:
        # Clear autonomous selection
        self._selected_auto_filename = ""
        for c in self._auto_cards:
            c.set_selected(False)

        for i, card in enumerate(self._cards):
            card.set_selected(i == index)
        self._selected_index = index
        topic = STARTER_TOPICS[index]
        is_custom = topic.title == "Custom Topic"

        self._preset_title_lbl.setVisible(not is_custom)
        # Reset title label color to default accent
        self._preset_title_lbl.setStyleSheet(
            f"color: {_PALETTE['accent']}; background: transparent;"
        )
        self._custom_title_edit.setVisible(is_custom)

        # Compute default tp_key from the first talking point (or topic title for custom)
        if is_custom:
            self._current_tp_key = ""
            self._custom_title_edit.setText("")
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
            self._confirm_btn.setText("▶  Start Custom Topic")
        else:
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
            self._confirm_btn.setText("▶  Start This Topic")

        pts = len(topic.talking_points) if not is_custom else 0
        self._topic_badge.setText(f"  {pts} talking points" if pts else "")

        # Load session history for this talking point (pass the BASE key;
        # the panel resolves the effective generation internally)
        if self._history_panel is not None and self._current_tp_key:
            self._history_panel.load_sessions(self._current_tp_key)
        elif self._history_panel is not None:
            self._history_panel.load_sessions("")

    def _on_session_card_clicked(self, session_id: str) -> None:
        """Show the transcript for a clicked session card."""
        if self._transcript_panel is not None:
            self._transcript_panel.load_session(session_id)

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

    # ── Autonomous learning helpers ────────────────────────────────────────

    def _refresh_auto_cards(self) -> None:
        """Rebuild the autonomous topic card list in the left column."""
        # Remove old cards
        for c in self._auto_cards:
            c.setParent(None)
            c.deleteLater()
        self._auto_cards.clear()

        store = get_autonomous_store()
        topics = store.load_all(include_deleted=False)

        self._auto_divider.setVisible(bool(topics))

        for at in topics:
            card = _AutonomousTopicCard(at)
            card.clicked.connect(self._select_auto_topic)
            card.delete_requested.connect(self._delete_auto_topic)
            self._auto_cards.append(card)
            self._auto_card_container.addWidget(card)

    def _select_auto_topic(self, filename: str) -> None:
        """Select an autonomous topic and populate the detail panel."""
        # Deselect all starter cards
        for c in self._cards:
            c.set_selected(False)
        # Select this auto card
        self._selected_auto_filename = filename
        for c in self._auto_cards:
            c.set_selected(c._filename == filename)

        store = get_autonomous_store()
        at = store.get_by_filename(filename)
        if at is None:
            return

        # Switch to autonomous topic view
        self._preset_title_lbl.setVisible(True)
        self._custom_title_edit.setVisible(False)
        self._preset_title_lbl.setText(f"✦ {at.title}")
        self._preset_title_lbl.setStyleSheet(
            "color: #ce93d8; background: transparent;"
        )
        self._desc_edit.setReadOnly(False)
        self._tp_edit.setReadOnly(False)
        self._desc_edit.setPlainText(at.description)
        tp_text = "\n\n".join(at.talking_points)
        self._tp_edit.setPlainText(tp_text)
        self._confirm_btn.setText("▶  Start Autonomous Topic")

        pts = len(at.talking_points)
        origin_text = f"  (segue: {at.segue_concept})" if at.segue_concept else ""
        self._topic_badge.setText(
            f"  {pts} talking points  ·  autonomous{origin_text}"
        )

        # Compute tp_key for session history
        first_tp = at.talking_points[0] if at.talking_points else ""
        self._current_tp_key = _tp_key(at.title, first_tp)
        if self._history_panel is not None and self._current_tp_key:
            self._history_panel.load_sessions(self._current_tp_key)

    def _delete_auto_topic(self, filename: str) -> None:
        """Soft-delete an autonomous topic and refresh cards."""
        store = get_autonomous_store()
        store.delete(filename)
        self._refresh_auto_cards()

    def _on_assistant_topic_created(self, result: dict) -> None:
        """Handle a Topic created by the assistant chatbot."""
        store = get_autonomous_store()
        store.save(result)
        self._refresh_auto_cards()

    def _on_auto_learning_toggled(self, checked: bool) -> None:
        self._autonomous_learning_enabled = checked
        self._auto_start_btn.setEnabled(checked)
        if not checked:
            self._stop_autonomous_learning()

    def _start_autonomous_learning(self) -> None:
        """Kick off autonomous promotion: scan segue buffer and create TPs."""
        if self._promotion_worker is not None and self._promotion_worker.isRunning():
            return

        from agents.topic_assistant_worker import AutonomousPromotionWorker
        from config.model_prefs import load_model_prefs
        from config.starter_topics import get_topic_titles

        try:
            model = load_model_prefs().get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            model = "qwen3:30b"

        existing = get_topic_titles() + get_autonomous_store().titles()
        threshold = self._threshold_slider.value()

        self._promotion_worker = AutonomousPromotionWorker(
            model=model,
            existing_titles=existing,
            threshold=threshold,
            parent=self,
        )
        self._promotion_worker.promoted.connect(self._on_auto_promoted)
        self._promotion_worker.progress.connect(
            lambda msg: self._auto_status.setText(msg)
        )
        self._promotion_worker.all_done.connect(self._on_auto_all_done)
        self._promotion_worker.failed.connect(
            lambda err: self._auto_status.setText(f"✗ {err[:60]}")
        )
        self._promotion_worker.start()

        self._auto_start_btn.setEnabled(False)
        self._auto_stop_btn.setEnabled(True)
        self._auto_status.setText("Scanning segue buffer…")

    def _stop_autonomous_learning(self) -> None:
        if self._promotion_worker is not None and self._promotion_worker.isRunning():
            self._promotion_worker.request_stop()
        self._auto_start_btn.setEnabled(self._autonomous_learning_enabled)
        self._auto_stop_btn.setEnabled(False)
        self._auto_status.setText("Stopped")

    def _on_auto_promoted(self, result: dict) -> None:
        store = get_autonomous_store()
        store.save(result)
        self._refresh_auto_cards()
        title = result.get("title", "?")
        self._auto_status.setText(f"Created: {title[:40]}")

    def _on_auto_all_done(self, count: int) -> None:
        self._auto_start_btn.setEnabled(self._autonomous_learning_enabled)
        self._auto_stop_btn.setEnabled(False)
        self._auto_status.setText(
            f"✓ Done — {count} topic{'s' if count != 1 else ''} created"
        )

    # ── Confirm (updated for autonomous topics) ─────────────────────────────

    def _on_confirm(self) -> None:
        # Check if an autonomous topic is selected
        if self._selected_auto_filename:
            store = get_autonomous_store()
            at = store.get_by_filename(self._selected_auto_filename)
            if at is not None:
                title = at.title
                desc = self._desc_edit.toPlainText().strip()
                tp_raw = self._tp_edit.toPlainText().strip()
                tp_lines = [ln.strip() for ln in tp_raw.split("\n") if ln.strip()]
                if tp_lines:
                    tp_block = (
                        "\n\nKEY TALKING POINTS — anchor every argument to these:\n"
                        + "\n".join(f"  • {ln}" for ln in tp_lines)
                    )
                else:
                    tp_block = ""
                full_context = f"{title}\n\n{desc}{tp_block}".strip()
                files = (
                    [str(f) for f in self._queued_files]
                    if self._use_ingest_cb.isChecked()
                    else []
                )
                self.topic_confirmed.emit(title, full_context, files)
                self.accept()
                return

        # Original logic for starter topics
        topic = STARTER_TOPICS[self._selected_index]
        is_custom = topic.title == "Custom Topic"

        if is_custom:
            title = self._custom_title_edit.text().strip()
            if not title:
                self._custom_title_edit.setStyleSheet(
                    f"QLineEdit {{ border: 1px solid #ef5350; background: {_PALETTE['bg']};"
                    " color: #e8eaf6; border-radius: 6px; padding: 8px 12px; font-size: 14px; }}"
                )
                return
            first_tp_line = (self._tp_edit.toPlainText().strip().split("\n")[0]).strip()
            self._current_tp_key = _tp_key(title, first_tp_line)
        else:
            title = topic.title

        desc = self._desc_edit.toPlainText().strip()
        if desc.startswith("✨"):
            newline_idx = desc.find("\n\n")
            if newline_idx != -1:
                desc = desc[newline_idx + 2:].strip()

        tp_raw = self._tp_edit.toPlainText().strip()
        tp_lines = [ln.strip() for ln in tp_raw.split("\n") if ln.strip()]

        if tp_lines:
            tp_block = (
                "\n\nKEY TALKING POINTS — anchor every argument to these:\n"
                + "\n".join(f"  • {ln}" for ln in tp_lines)
            )
        else:
            tp_block = ""

        full_context = f"{title}\n\n{desc}{tp_block}".strip()
        files = (
            [str(f) for f in self._queued_files]
            if self._use_ingest_cb.isChecked()
            else []
        )
        self.topic_confirmed.emit(title, full_context, files)
        self.accept()

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
