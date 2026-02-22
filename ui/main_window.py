from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.bootstrap import build_orchestrator
from config.model_prefs import (
    DEFAULT_LEFT_MODEL,
    DEFAULT_RIGHT_MODEL,
    load_model_prefs,
    save_model_prefs,
)
from agents.topic_refine_worker import TopicRefineWorker
from config.starter_topics import STARTER_TOPICS
from core.orchestrator import DebateEvent, DebateOrchestrator
from core.repo_watchdog import RepoWatchdog
from ingestion.ingestion_agent import IngestionWorker
from memory.cross_session_memory import get_cross_session_memory
from core.session_manager import get_session_manager
from tts.speech_engine import TTSPlaybackWorker
from ui.dialogs.session_browser_dialog import SessionBrowserDialog
from ui.dialogs.topic_picker_dialog import TopicPickerDialog
from ui.panels.arbiter_panel import ArbiterPanel
from ui.panels.center_debate_panel import CenterDebatePanel
from ui.panels.graph_panel import GraphPanel
from ui.panels.left_agent_panel import LeftAgentPanel
from ui.panels.right_agent_panel import RightAgentPanel
from ui.panels.scoring_panel import ScoringPanel
from core.debate_summarizer import DebateSummaryWorker
from ui.dialogs.analytics_dialog import AnalyticsDialog


class _NoWheelComboBox(QComboBox):
    """ComboBox that ignores scroll-wheel — click only."""
    def wheelEvent(self, e: QWheelEvent) -> None:  # type: ignore[override]
        e.ignore()


# ---------------------------------------------------------------------------
# Ollama cloud model detection
# ---------------------------------------------------------------------------
# Cloud models served by Ollama's datacenter appear with ":cloud" or
# "-cloud" in their tag (e.g. "qwen3-coder:480b-cloud", "kimi-k2.5:cloud",
# "deepseek-v3.1:671b-cloud").  All other models run locally.


def _is_cloud_model(name: str) -> bool:
    """Return True if the model name/tag indicates an Ollama cloud model.

    Detection is based on the actual tag containing 'cloud' — the authoritative
    indicator from Ollama's API.  This avoids false positives from local
    variants of the same base model (e.g. qwen3-vl:8b is local, but
    qwen3-vl:235b-cloud would be cloud).
    """
    lower = name.lower().strip()
    # Tag-level: "model:480b-cloud", "model:cloud"
    if ":" in lower:
        tag = lower.split(":", 1)[1]
        if "cloud" in tag:
            return True
    # Suffix-level fallback: "model-cloud"
    if lower.endswith("-cloud"):
        return True
    return False


def _clean_model_name(text: str) -> str:
    """Strip the ☁/💻/◈ display prefixes from combo item text to get the raw model name."""
    return text.replace("☁", "").replace("💻", "").replace("◈", "").strip()


class _NoWheelSlider(QSlider):
    """Slider that ignores scroll-wheel — drag/click only."""
    def wheelEvent(self, e: QWheelEvent) -> None:  # type: ignore[override]
        e.ignore()


class _HScrollArea(QScrollArea):
    """Scroll area that routes wheel events into horizontal scrolling."""
    def wheelEvent(self, e: QWheelEvent) -> None:  # type: ignore[override]
        delta = e.angleDelta().y()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta // 2)
        e.accept()


class _BottomComposer(QWidget):
    """Fixed bottom-of-screen arbiter workflow bar — always visible.

    Left section  : typed arbiter message + Send (enabled only when paused).
    Right section : Capture Segment button + before/after sentence spinners
                    + scrollable staging panel for captured snippets.

    Emits:
        send_requested(str, list)  — (message_text, list[snippet_dict])
        capture_requested()        — pressed while TTS is playing
    """
    send_requested    = pyqtSignal(str, list)
    capture_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("bottomComposer")
        self._staging_snippets: list[dict] = []
        self.setFixedHeight(98)
        self.setStyleSheet(
            "QWidget#bottomComposer { background: #07111c;"
            " border-top: 2px solid #1a3a55; }"
        )
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 6, 12, 6)
        outer.setSpacing(0)

        # ── Left: arbiter text input ─────────────────────────────────────────
        left_w = QWidget()
        left_w.setStyleSheet("background: transparent;")
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        lbl_row = QHBoxLayout()
        lbl_row.setSpacing(6)
        icon_lbl = QLabel("💬")
        icon_lbl.setStyleSheet("font-size: 14px; background: transparent;")
        arb_lbl = QLabel("Arbiter Intervention")
        arb_lbl.setStyleSheet(
            "color: #ff6e40; font-weight: 700; font-size: 11px; background: transparent;"
        )
        self._state_lbl = QLabel("(pause the debate to intervene)")
        self._state_lbl.setStyleSheet(
            "color: #37474f; font-size: 9px; background: transparent;"
        )
        lbl_row.addWidget(icon_lbl)
        lbl_row.addWidget(arb_lbl)
        lbl_row.addSpacing(8)
        lbl_row.addWidget(self._state_lbl)
        lbl_row.addStretch()

        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Type guidance for both agents — requires debate paused…"
        )
        self._input.setStyleSheet(
            "QLineEdit { background: #0a1520; color: #e8eaf6;"
            " border: 1px solid #1a2d40; border-radius: 6px;"
            " padding: 7px 12px; font-size: 12px; }"
            "QLineEdit:enabled:focus { border-color: #ff6e40; }"
            "QLineEdit:disabled { color: #2a3a55; border-color: #0d1520; }"
        )
        self._input.returnPressed.connect(self._send)

        self._send_btn = QPushButton("Send  ▶")
        self._send_btn.setFixedWidth(88)
        self._send_btn.setStyleSheet(
            "QPushButton { background: #bf360c; color: #fff; font-weight: 700;"
            " border-radius: 7px; padding: 7px 14px; font-size: 12px;"
            " border: 1px solid #ff6e40; }"
            "QPushButton:hover { background: #e64a19; }"
            "QPushButton:disabled { background: #111c28; color: #2a3a55;"
            " border-color: #0d1520; }"
        )
        self._send_btn.clicked.connect(self._send)

        input_row.addWidget(self._input, stretch=1)
        input_row.addWidget(self._send_btn)
        left_lay.addLayout(lbl_row)
        left_lay.addLayout(input_row)

        # ── Vertical divider ─────────────────────────────────────────────────
        vline = QFrame()
        vline.setFrameShape(QFrame.Shape.VLine)
        vline.setStyleSheet("color: #1a3a55;")
        vline.setFixedWidth(1)

        # ── Right: capture tools + staging panel ─────────────────────────────
        right_w = QWidget()
        right_w.setFixedWidth(400)
        right_w.setStyleSheet("background: transparent;")
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(10, 0, 0, 0)
        right_lay.setSpacing(4)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(5)

        cap_hdg = QLabel("📎 Capture")
        cap_hdg.setStyleSheet("color: #546e7a; font-size: 9px; font-weight:600; background: transparent;")

        before_lbl = QLabel("Before:")
        before_lbl.setStyleSheet("color: #37474f; font-size: 9px; background: transparent;")
        self._before_spin = QSpinBox()
        self._before_spin.setRange(0, 20)
        self._before_spin.setValue(4)
        self._before_spin.setFixedWidth(44)
        self._before_spin.setToolTip("Sentences to capture BEFORE current TTS word")
        self._before_spin.setStyleSheet(
            "QSpinBox { background: #0a1520; color: #78909c;"
            " border: 1px solid #1a2d40; border-radius: 4px;"
            " padding: 2px 3px; font-size: 10px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 14px; }"
        )

        after_lbl = QLabel("After:")
        after_lbl.setStyleSheet("color: #37474f; font-size: 9px; background: transparent;")
        self._after_spin = QSpinBox()
        self._after_spin.setRange(0, 10)
        self._after_spin.setValue(1)
        self._after_spin.setFixedWidth(44)
        self._after_spin.setToolTip("Sentences to capture AFTER current TTS word")
        self._after_spin.setStyleSheet(
            "QSpinBox { background: #0a1520; color: #78909c;"
            " border: 1px solid #1a2d40; border-radius: 4px;"
            " padding: 2px 3px; font-size: 10px; }"
            "QSpinBox::up-button, QSpinBox::down-button { width: 14px; }"
        )

        self._capture_btn = QPushButton("📌 Capture Segment")
        self._capture_btn.setToolTip(
            "Capture sentences around current TTS read position into staging area"
        )
        self._capture_btn.setStyleSheet(
            "QPushButton { background: #1a1f40; color: #90caf9;"
            " font-size: 10px; border-radius: 5px; padding: 4px 10px;"
            " border: 1px solid #283593; }"
            "QPushButton:hover { background: #283593; color: #fff; }"
            "QPushButton:disabled { background: #0d1018; color: #1e2d42;"
            " border-color: #0d1018; }"
        )
        self._capture_btn.clicked.connect(self.capture_requested.emit)

        ctrl_row.addWidget(cap_hdg)
        ctrl_row.addWidget(before_lbl)
        ctrl_row.addWidget(self._before_spin)
        ctrl_row.addSpacing(2)
        ctrl_row.addWidget(after_lbl)
        ctrl_row.addWidget(self._after_spin)
        ctrl_row.addSpacing(8)
        ctrl_row.addWidget(self._capture_btn)
        ctrl_row.addStretch()

        # Staging list
        self._staging_list = QListWidget()
        self._staging_list.setMaximumHeight(42)
        self._staging_list.setToolTip(
            "Captured segments — all will be sent as context with the next arbiter message"
        )
        self._staging_list.setStyleSheet(
            "QListWidget { background: #050d16; color: #80cbc4;"
            " border: 1px solid #1a3a55; border-radius: 4px; font-size: 9px; }"
            "QListWidget::item { padding: 2px 6px; }"
            "QListWidget::item:selected { background: #0a2030; }"
        )

        right_lay.addLayout(ctrl_row)
        right_lay.addWidget(self._staging_list)

        outer.addWidget(left_w, stretch=1)
        outer.addSpacing(10)
        outer.addWidget(vline)
        outer.addWidget(right_w)

        # Start in disabled state
        self._set_interactive(False)

    # ── State ────────────────────────────────────────────────────────────────

    def set_paused_state(self, paused: bool) -> None:
        """Call when debate pauses or resumes."""
        self._set_interactive(paused)
        if paused:
            self._state_lbl.setText("paused — ready to send")
            self._state_lbl.setStyleSheet(
                "color: #ff8a65; font-size: 9px; background: transparent;"
            )
        else:
            self._state_lbl.setText("(pause the debate to intervene)")
            self._state_lbl.setStyleSheet(
                "color: #37474f; font-size: 9px; background: transparent;"
            )

    def _set_interactive(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._capture_btn.setEnabled(enabled)

    def focus_input(self) -> None:
        self._input.setFocus()

    # ── Staging ───────────────────────────────────────────────────────────────

    def add_snippet(self, snippet: dict) -> None:
        """snippet keys: label, speaker, turn, filepath, text"""
        self._staging_snippets.append(snippet)
        spk   = snippet.get("speaker", "?")
        turn  = snippet.get("turn", "?")
        prev  = snippet.get("text", "")[:55].replace("\n", " ")
        self._staging_list.addItem(
            QListWidgetItem(f"[{spk} t{turn}]  {prev}…")
        )

    def clear_staging(self) -> None:
        self._staging_snippets.clear()
        self._staging_list.clear()

    @property
    def staged_snippets(self) -> list[dict]:
        return list(self._staging_snippets)

    @property
    def before_count(self) -> int:
        return self._before_spin.value()

    @property
    def after_count(self) -> int:
        return self._after_spin.value()

    # ── Send ─────────────────────────────────────────────────────────────────

    def _send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.send_requested.emit(text, list(self._staging_snippets))
        self._input.clear()


class DebateWorker(QThread):
    event_signal = pyqtSignal(object)
    done_signal = pyqtSignal()

    def __init__(
        self,
        orchestrator: DebateOrchestrator,
        topic: str,
        turns: int,
        endless: bool = False,
        continue_run: bool = False,
    ) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.topic = topic
        self.turns = turns
        self.endless = endless
        self.continue_run = continue_run

    def run(self) -> None:
        self.orchestrator.subscribe(lambda event: self.event_signal.emit(event))
        if self.continue_run:
            asyncio.run(self.orchestrator.continue_debate(extra_turns=self.turns))
        else:
            asyncio.run(self.orchestrator.run_debate(topic=self.topic, turns=self.turns, endless=self.endless))
        self.done_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.setWindowTitle("Agent Debate System — Astra vs Nova")
        self.resize(1680, 1000)
        self.project_root = project_root

        # Load saved model preferences
        self._model_prefs = load_model_prefs()
        # VS Code bridge model IDs discovered at last populate (must exist before build_orchestrator)
        self._vscode_model_ids: set[str] = set()

        self.orchestrator, self.debate_config = build_orchestrator(
            project_root,
            left_model=self._model_prefs["left_model"],
            right_model=self._model_prefs["right_model"],
            session_manager=get_session_manager(),
            vscode_model_ids=self._vscode_model_ids,
        )
        self.worker: DebateWorker | None = None
        self._tts_worker: TTSPlaybackWorker | None = None
        self._current_speaking_idx: int = -1
        # Topic state — set by TopicPickerDialog
        self._current_topic_title: str = STARTER_TOPICS[0].title
        self._current_topic_context: str = ""   # full brief fed to agents
        self._pending_ingest_files: list[str] = []
        self._ingest_worker: IngestionWorker | None = None
        self._session_browser: SessionBrowserDialog | None = None
        # Score accumulators for live stats bar
        self._astra_scores: list[float] = []
        self._nova_scores: list[float] = []
        # Pending mid-debate model swaps — applied at next pair boundary
        self._pending_left_model: str | None = None
        self._pending_right_model: str | None = None
        # Cloud model usage counter for current session
        self._cloud_calls: int = 0
        # TTS word-tracking for Capture Segment
        self._current_tts_msg_idx: int = -1
        self._current_tts_char_offset: int = 0
        self._capture_seq: int = 0
        # Talking-point key (set from TopicPickerDialog) — tags each new session
        self._current_tp_key: str = ""
        # Latest resolution payload — used to feed TopicRefineWorker after debate ends
        self._last_resolution_payload: dict = {}
        self._refine_worker: TopicRefineWorker | None = None
        self._repo_watchdog_meta: dict | None = None
        self._repo_watchdog_last_sig: str = ""
        self._repo_watchdog = RepoWatchdog(get_session_manager().root)
        self._repo_watchdog_timer = QTimer(self)
        self._repo_watchdog_timer.setInterval(12000)
        self._repo_watchdog_timer.timeout.connect(self._poll_repo_watchdog)

        # -- panels --
        self.left_panel = LeftAgentPanel()
        self.right_panel = RightAgentPanel()
        self.center_panel = CenterDebatePanel()
        self.arbiter_panel = ArbiterPanel()
        self.graph_panel = GraphPanel()
        self.scoring_panel = ScoringPanel()
        self._turn_scores: list[dict] = []
        self._arbiter_events: list[dict] = []
        self._summary_worker: DebateSummaryWorker | None = None

        self._build_layout()
        self._build_toolbar()
        self._bind_events()
        self._set_debate_transport_state("stopped")
        self.center_panel.set_source_click_handler(self.open_source_dialog)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._runtime_badge = QLabel("Checking Ollama...")
        self._runtime_badge.setStyleSheet("color: #4dd0e1; padding: 0 8px;")
        self._status_bar.addPermanentWidget(self._runtime_badge)
        self._refresh_runtime_badge()
        # Populate model dropdowns from Ollama after badge check
        self._populate_model_combos()
        # Default: semantic awareness ON — agents recall past debate sessions
        self._semantic_btn.setChecked(True)  # fires _on_semantic_toggled(True) via bound signal
        self._status_bar.showMessage("Ready — select a topic and start the debate")

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        # Wrap center_panel and arbiter inject bar into one container
        _central = QWidget()
        _central.setObjectName("centralContainer")
        _cl = QVBoxLayout(_central)
        _cl.setContentsMargins(0, 0, 0, 0)
        _cl.setSpacing(0)
        _cl.addWidget(self.center_panel, stretch=1)
        # Arbiter intervention controls now live inside the Arbiter dock panel
        self.arbiter_panel.send_requested.connect(self._on_arbiter_inject)
        self.arbiter_panel.capture_requested.connect(self._on_capture_pressed)

        # ------ live stats bar ------
        self._stats_bar = QWidget()
        self._stats_bar.setObjectName("statsBar")
        self._stats_bar.setStyleSheet(
            "QWidget#statsBar { background: #07111c; border-top: 1px solid #1a3a55; }"
        )
        stats_lay = QHBoxLayout(self._stats_bar)
        stats_lay.setContentsMargins(14, 3, 14, 3)
        stats_lay.setSpacing(14)

        _score_lbl = QLabel("Quality:")
        _score_lbl.setStyleSheet("color: #455a64; font-size: 9pt;")
        self._stats_astra_lbl = QLabel("Astra: —")
        self._stats_astra_lbl.setStyleSheet("color: #546e7a; font-size: 9pt; font-weight: 600;")
        self._stats_nova_lbl = QLabel("Nova: —")
        self._stats_nova_lbl.setStyleSheet("color: #546e7a; font-size: 9pt; font-weight: 600;")
        _sep_s = QFrame()
        _sep_s.setFrameShape(QFrame.Shape.VLine)
        _sep_s.setStyleSheet("color: #1a3a55;")
        _res_lbl = QLabel("Resolutions:")
        _res_lbl.setStyleSheet("color: #455a64; font-size: 9pt;")
        self._stats_conclusions_lbl = QLabel("✓ 0")
        self._stats_conclusions_lbl.setStyleSheet("color: #4caf50; font-size: 9pt; font-weight: 600;")
        self._stats_conclusions_lbl.setToolTip("Conclusions reached")
        self._stats_contradictions_lbl = QLabel("↔ 0")
        self._stats_contradictions_lbl.setStyleSheet("color: #ffc107; font-size: 9pt; font-weight: 600;")
        self._stats_contradictions_lbl.setToolTip("Contradictions flagged")
        self._stats_falsehoods_lbl = QLabel("✗ 0")
        self._stats_falsehoods_lbl.setStyleSheet("color: #ef5350; font-size: 9pt; font-weight: 600;")
        self._stats_falsehoods_lbl.setToolTip("Falsehoods identified")
        self._stats_subtopics_lbl = QLabel("🌿 0")
        self._stats_subtopics_lbl.setStyleSheet("color: #80cbc4; font-size: 9pt; font-weight: 600;")
        self._stats_subtopics_lbl.setToolTip("Open sub-topics explored")

        stats_lay.addWidget(_score_lbl)
        stats_lay.addWidget(self._stats_astra_lbl)
        stats_lay.addWidget(self._stats_nova_lbl)
        stats_lay.addWidget(_sep_s)
        stats_lay.addWidget(_res_lbl)
        stats_lay.addWidget(self._stats_conclusions_lbl)
        stats_lay.addWidget(self._stats_contradictions_lbl)
        stats_lay.addWidget(self._stats_falsehoods_lbl)
        stats_lay.addWidget(self._stats_subtopics_lbl)
        stats_lay.addStretch()

        # Follow Text button — re-engages TTS word scroll after user breaks it by scrolling
        self._follow_text_btn = QPushButton("Follow Text")
        self._follow_text_btn.setObjectName("followTextBtn")
        self._follow_text_btn.setCheckable(True)
        self._follow_text_btn.setChecked(True)   # on by default
        self._follow_text_btn.setFixedHeight(22)
        self._follow_text_btn.setToolTip(
            "When active, the view scrolls with TTS playback.\n"
            "Scrolling up/down with the mouse wheel breaks it."
        )
        self._follow_text_btn.setStyleSheet(
            "QPushButton#followTextBtn {"
            "  background: #0d2137; color: #546e7a;"
            "  border: 1px solid #1a3a55; border-radius: 3px;"
            "  font-size: 8pt; font-weight: 600; padding: 0 8px;"
            "}"
            "QPushButton#followTextBtn:checked {"
            "  background: #0d3349; color: #00e5ff;"
            "  border: 1px solid #00e5ff;"
            "}"
            "QPushButton#followTextBtn:hover { background: #112640; }"
        )
        self._follow_text_btn.clicked.connect(self._on_follow_text_btn_clicked)
        # When the panel breaks follow-mode (user scrolled), update button appearance
        self.center_panel.follow_mode_changed.connect(self._on_follow_mode_changed)
        stats_lay.addWidget(self._follow_text_btn)

        # Cloud usage counter — increments each time a cloud model speaks
        _cloud_sep = QFrame()
        _cloud_sep.setFrameShape(QFrame.Shape.VLine)
        _cloud_sep.setStyleSheet("color: #1a3a55;")
        self._cloud_calls_lbl = QLabel("☁ 0 cloud calls")
        self._cloud_calls_lbl.setObjectName("cloudCallsLbl")
        self._cloud_calls_lbl.setStyleSheet(
            "color: #546e7a; font-size: 9pt; font-weight: 600;"
        )
        self._cloud_calls_lbl.setToolTip(
            "Number of responses generated by Ollama cloud models this session.\n"
            "Cloud models count against your Ollama plan usage limits.\n"
            "Free tier: light usage  |  Pro $20/mo  |  Max $100/mo"
        )
        stats_lay.addWidget(_cloud_sep)
        stats_lay.addWidget(self._cloud_calls_lbl)

        _cl.addWidget(self._stats_bar)

        self.setCentralWidget(_central)

        left_dock = QDockWidget("Astra — Inner Monologue", self)
        left_dock.setObjectName("leftDock")
        left_dock.setWidget(self.left_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)

        right_dock = QDockWidget("Nova — Inner Monologue", self)
        right_dock.setObjectName("rightDock")
        right_dock.setWidget(self.right_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)

        arbiter_dock = QDockWidget("Arbiter", self)
        arbiter_dock.setObjectName("arbiterDock")
        arbiter_dock.setWidget(self.arbiter_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, arbiter_dock)

        graph_dock = QDockWidget("Debate Graph — Branches & Sub-Topics", self)
        graph_dock.setObjectName("graphDock")
        graph_dock.setWidget(self.graph_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, graph_dock)
        self.tabifyDockWidget(arbiter_dock, graph_dock)

        scoring_dock = QDockWidget("Scoring & Verdict", self)
        scoring_dock.setObjectName("scoringDock")
        scoring_dock.setWidget(self.scoring_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, scoring_dock)
        self.tabifyDockWidget(arbiter_dock, scoring_dock)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Debate Controls")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)

        container = QWidget()
        container.setObjectName("toolbarContainer")
        container.setFixedHeight(44)
        row = QHBoxLayout(container)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(8)

        # Topic picker button — opens full rich pop-out dialog
        self._topic_btn = QPushButton()
        self._topic_btn.setObjectName("topicPickerBtn")
        self._topic_btn.setMinimumWidth(320)
        self._topic_btn.setMaximumWidth(520)
        self._topic_btn.setStyleSheet(
            "QPushButton#topicPickerBtn { background: #0d1e30; color: #00e5ff;"
            " font-weight: 700; font-size: 12px; border-radius: 8px;"
            " padding: 7px 14px; border: 1px solid #1a3a55;"
            " text-align: left; }"
            "QPushButton#topicPickerBtn:hover { background: #112640;"
            " border-color: #00acc1; }"
        )
        self._topic_btn.setToolTip("Click to choose or configure the debate topic")
        self._refresh_topic_button()

        # ---- Per-agent model selectors ----
        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setStyleSheet("color: #2a3a55;")

        astra_model_label = QLabel("Astra:")
        astra_model_label.setStyleSheet("color: #00e5ff; font-weight: 700; padding-left: 4px;")
        self._astra_model_combo = _NoWheelComboBox()
        self._astra_model_combo.setObjectName("astraModelCombo")
        self._astra_model_combo.setMinimumWidth(170)
        self._astra_model_combo.setToolTip("Ollama model used by Astra (left agent)")
        # Seed with saved/default value so it shows immediately before Ollama responds
        self._astra_model_combo.addItem(self._model_prefs["left_model"])

        nova_model_label = QLabel("Nova:")
        nova_model_label.setStyleSheet("color: #ff6e40; font-weight: 700; padding-left: 4px;")
        self._nova_model_combo = _NoWheelComboBox()
        self._nova_model_combo.setObjectName("novaModelCombo")
        self._nova_model_combo.setMinimumWidth(170)
        self._nova_model_combo.setToolTip("Ollama model used by Nova (right agent)")
        self._nova_model_combo.addItem(self._model_prefs["right_model"])

        # Ollama info / pricing button
        self._ollama_info_btn = QPushButton("?")
        self._ollama_info_btn.setObjectName("ollamaInfoBtn")
        self._ollama_info_btn.setFixedSize(22, 22)
        self._ollama_info_btn.setToolTip("Ollama API info, cloud model pricing & session usage")
        self._ollama_info_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #546e7a; border-radius: 11px;"
            " border: 1px solid #2a3a55; font-weight: 700; font-size: 10pt; }"
            "QPushButton:hover { background: #1e3a5f; color: #80cbc4; "
            " border-color: #00acc1; }"
        )
        self._ollama_info_btn.clicked.connect(self._show_ollama_info_dialog)

        model_sep = QFrame()
        model_sep.setFrameShape(QFrame.Shape.VLine)
        model_sep.setStyleSheet("color: #2a3a55;")

        # Turns / Endless
        turns_label = QLabel("Turns:")
        turns_label.setStyleSheet("color: #90a4ae;")
        self._turns_display = QLabel("10")
        self._turns_display.setStyleSheet("color: #e0e0e0; min-width: 24px;")
        self._turns_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._turns_slider.setObjectName("turnsSlider")
        self._turns_slider.setRange(4, 100)
        self._turns_slider.setValue(10)
        self._turns_slider.setMaximumWidth(100)
        self._turns_slider.valueChanged.connect(
            lambda v: self._turns_display.setText(str(v))
        )

        self._endless_check = QCheckBox("Endless")
        self._endless_check.setObjectName("endlessCheck")
        self._endless_check.setStyleSheet("color: #ffd740; font-weight: 600;")
        self._endless_check.setToolTip("Agents debate forever — never stops until you press Stop")
        self._endless_check.toggled.connect(self._on_endless_toggled)

        # Debate control buttons
        self.start_button = QPushButton("▶  Start")
        self.start_button.setObjectName("startButton")
        self.start_button.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #00897b, stop:1 #00695c); color: #fff; font-weight: 700; "
            "border-radius: 8px; padding: 7px 18px; border: 1px solid #26a69a; }"
            "QPushButton:hover { background: #00acc1; }"
            "QPushButton:pressed { background: #004d40; }"
            "QPushButton:disabled { background: #102226; color: #4f6a6f; border: 1px solid #1d3a3f; }"
        )

        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #c62828, stop:1 #8e0000); color: #fff; font-weight: 700; "
            "border-radius: 8px; padding: 7px 18px; border: 1px solid #e53935; }"
            "QPushButton:hover { background: #e53935; }"
            "QPushButton:disabled { background: #2a1212; color: #7a5555; border: 1px solid #4a2020; }"
        )

        self._debate_pause_btn = QPushButton("⏸  Pause")
        self._debate_pause_btn.setObjectName("debatePauseBtn")
        self._debate_pause_btn.setCheckable(True)
        self._debate_pause_btn.setStyleSheet(
            "QPushButton { background: #37474f; color: #eceff1; font-weight:700;"
            " border-radius: 8px; padding: 7px 16px; border: 1px solid #546e7a; }"
            "QPushButton:hover { background: #455a64; }"
            "QPushButton:checked { background: #e65100; color: #fff;"
            " border: 1px solid #ff8a65; }"
            "QPushButton:checked:hover { background: #f4511e; }"
            "QPushButton:disabled { background: #1f2a30; color: #54656e; border: 1px solid #2f4049; }"
        )
        self._debate_pause_btn.setToolTip("Pair-aware pause: waits until both agents finish their current turns")

        self._continue_btn = QPushButton("▶▶  Continue")
        self._continue_btn.setObjectName("continueBtn")
        self._continue_btn.setVisible(False)
        self._continue_btn.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #1565c0, stop:1 #0d47a1); color: #fff; font-weight: 700;"
            " border-radius: 8px; padding: 7px 18px; border: 1px solid #42a5f5; }"
            "QPushButton:hover { background: #1976d2; }"
        )
        self._continue_btn.setToolTip("Continue the debate for the same number of turns again")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #2a3a55;")

        # TTS controls
        tts_label = QLabel("TTS:")
        tts_label.setStyleSheet("color: #90a4ae; font-weight: 600;")

        # Single play/stop toggle button — swaps text & colour when active
        self._tts_play_btn = QPushButton("🔊 Read Debate")
        self._tts_play_btn.setObjectName("ttsPlayBtn")
        self._tts_play_btn.setToolTip("Read debate from top (Astra=Zira, Nova=David) — press again to stop")
        self._tts_play_btn.setCheckable(True)
        self._tts_playing = False
        self._tts_play_btn.setStyleSheet(
            "QPushButton { background: #1a237e; color: #e8eaf6; font-weight:600;"
            "border-radius: 8px; padding: 7px 14px; border: 1px solid #3949ab; }"
            "QPushButton:hover { background: #283593; }"
            "QPushButton:checked { background: #b71c1c; color: #fff;"
            " border: 1px solid #ef5350; }"
            "QPushButton:checked:hover { background: #c62828; }"
        )

        self._tts_pause_btn = QPushButton("⏸ Pause")
        self._tts_pause_btn.setObjectName("ttsPauseBtn")
        self._tts_pause_btn.setStyleSheet(
            "QPushButton { background: #37474f; color: #eceff1; font-weight:600;"
            "border-radius: 8px; padding: 7px 14px; border: 1px solid #546e7a; }"
            "QPushButton:hover { background: #455a64; }"
        )

        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: #90a4ae;")
        self._speed_display = QLabel("260")
        self._speed_display.setStyleSheet("color: #e0e0e0; min-width: 28px;")
        self._speed_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setObjectName("speedSlider")
        self._speed_slider.setRange(80, 350)
        self._speed_slider.setValue(260)
        self._speed_slider.setMaximumWidth(100)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        row.addWidget(self._topic_btn)
        row.addWidget(sep0)
        row.addWidget(astra_model_label)
        row.addWidget(self._astra_model_combo)
        row.addWidget(nova_model_label)
        row.addWidget(self._nova_model_combo)
        row.addWidget(self._ollama_info_btn)
        row.addWidget(model_sep)
        row.addWidget(turns_label)
        row.addWidget(self._turns_slider)
        row.addWidget(self._turns_display)
        row.addWidget(self._endless_check)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self._debate_pause_btn)
        row.addWidget(self._continue_btn)
        row.addWidget(sep)
        row.addWidget(tts_label)
        row.addWidget(self._tts_play_btn)
        row.addWidget(self._tts_pause_btn)
        row.addWidget(speed_label)
        row.addWidget(self._speed_slider)
        row.addWidget(self._speed_display)

        # Separator
        sess_sep = QFrame()
        sess_sep.setFrameShape(QFrame.Shape.VLine)
        sess_sep.setStyleSheet("color: #2a3a55;")
        row.addWidget(sess_sep)

        # Sessions browser button
        self._sessions_btn = QPushButton("📚 Sessions")
        self._sessions_btn.setObjectName("sessionsBtn")
        self._sessions_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #80cbc4; border-radius: 6px;"
            " padding: 6px 12px; border: 1px solid #2a3a55; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        self._sessions_btn.setToolTip("Browse and replay past debates")
        row.addWidget(self._sessions_btn)

        # Open current session folder in Explorer
        self._open_folder_btn = QPushButton("📂 Folder")
        self._open_folder_btn.setObjectName("openFolderBtn")
        self._open_folder_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #80deea; border-radius: 6px;"
            " padding: 6px 12px; border: 1px solid #2a3a55; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        self._open_folder_btn.setToolTip("Open the current session folder in Explorer")
        row.addWidget(self._open_folder_btn)

        # Semantic Awareness toggle
        self._semantic_btn = QPushButton("🌐 Semantic")
        self._semantic_btn.setObjectName("semanticBtn")
        self._semantic_btn.setCheckable(True)
        self._semantic_btn.setChecked(False)
        self._semantic_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #546e7a; border-radius: 6px;"
            " padding: 6px 12px; border: 1px solid #2a3a55; font-weight: 600; }"
            "QPushButton:hover { background: #1e3a5f; color: #90a4ae; }"
            "QPushButton:checked { background: #004d40; color: #00e5ff;"
            " border: 1px solid #00acc1; }"
        )
        self._semantic_btn.setToolTip(
            "Semantic Awareness — agents can recall facts from ALL past sessions"
        )
        row.addWidget(self._semantic_btn)

        self._analytics_btn = QPushButton("📊 Analytics")
        self._analytics_btn.setObjectName("analyticsBtn")
        self._analytics_btn.setToolTip("Open Analytics Manager — view all past debates and scoring history")
        self._analytics_btn.setStyleSheet(
            "QPushButton { background: #1b2a3a; color: #80cbc4; border: 1px solid #2a3d50;"
            " border-radius: 8px; padding: 4px 14px; font-size: 9.5pt; font-weight: 600; }"
            "QPushButton:hover { background: #243648; }"
            "QPushButton:pressed { background: #1a2b3c; }"
        )
        row.addWidget(self._analytics_btn)

        # NO addStretch — let content pack naturally so the scroll area can overflow

        # Wrap in a horizontal scroll area — widgetResizable must be False so the
        # container can be wider than the viewport and left/right scrolling works.
        _scroll_wrap = _HScrollArea()
        _scroll_wrap.setWidget(container)
        _scroll_wrap.setWidgetResizable(False)
        _scroll_wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        _scroll_wrap.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _scroll_wrap.setFrameShape(QFrame.Shape.NoFrame)
        _scroll_wrap.setFixedHeight(56)
        _scroll_wrap.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:horizontal { height: 6px; background: #0d1e30; }"
            "QScrollBar::handle:horizontal { background: #2a3a55; border-radius: 3px; }"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        )
        toolbar.addWidget(_scroll_wrap)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    # ------------------------------------------------------------------ events

    def _bind_events(self) -> None:
        self.start_button.clicked.connect(self.start_debate)
        self.stop_button.clicked.connect(self.stop_debate)
        self._debate_pause_btn.toggled.connect(self._on_debate_pause_toggled)
        self._continue_btn.clicked.connect(self._continue_debate)
        self._tts_play_btn.clicked.connect(self.tts_play_stop_toggle)
        self._tts_pause_btn.clicked.connect(self.tts_toggle_pause)
        self._topic_btn.clicked.connect(self._open_topic_picker)
        # Model combo saves
        self._astra_model_combo.currentTextChanged.connect(self._on_model_changed)
        self._nova_model_combo.currentTextChanged.connect(self._on_model_changed)
        # Session / semantic buttons
        self._sessions_btn.clicked.connect(self._open_session_browser)
        self._open_folder_btn.clicked.connect(self._open_session_folder)
        self._semantic_btn.toggled.connect(self._on_semantic_toggled)
        self._analytics_btn.clicked.connect(self._open_analytics_dialog)

    def _set_debate_transport_state(self, state: str) -> None:
        """Update Start/Stop/Pause button enabled + label state.

        state: stopped | running | pausing | paused
        """
        if state == "stopped":
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self._debate_pause_btn.setEnabled(False)
            self._debate_pause_btn.blockSignals(True)
            self._debate_pause_btn.setChecked(False)
            self._debate_pause_btn.blockSignals(False)
            self._debate_pause_btn.setText("⏸  Pause")
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._debate_pause_btn.setEnabled(True)

        if state == "paused":
            self._debate_pause_btn.blockSignals(True)
            self._debate_pause_btn.setChecked(True)
            self._debate_pause_btn.blockSignals(False)
            self._debate_pause_btn.setText("▶  Resume")
        elif state == "pausing":
            self._debate_pause_btn.blockSignals(True)
            self._debate_pause_btn.setChecked(True)
            self._debate_pause_btn.blockSignals(False)
            self._debate_pause_btn.setText("⏳  Pausing…")
        else:
            self._debate_pause_btn.blockSignals(True)
            self._debate_pause_btn.setChecked(False)
            self._debate_pause_btn.blockSignals(False)
            self._debate_pause_btn.setText("⏸  Pause")

    def _refresh_topic_button(self) -> None:
        """Update the topic button label to show current selection."""
        label = self._current_topic_title
        if len(label) > 62:
            label = label[:59] + "…"
        files_hint = f"  · {len(self._pending_ingest_files)} file(s)" if self._pending_ingest_files else ""
        self._topic_btn.setText(f"🎯  {label}{files_hint}")

    def _open_topic_picker(self) -> None:
        from PyQt6.QtWidgets import QDialog
        dlg = TopicPickerDialog(current_title=self._current_topic_title, parent=self)
        dlg.topic_confirmed.connect(self._on_topic_confirmed)
        dlg.exec()
        # Capture the talking-point key after the dialog closes
        if dlg.result() == QDialog.DialogCode.Accepted:
            self._current_tp_key = dlg.tp_key

    def _on_topic_confirmed(self, title: str, full_context: str, files: list) -> None:
        clean_context, repo_meta = self._extract_repo_watchdog_meta(full_context)
        self._current_topic_title = title
        self._current_topic_context = clean_context
        self._repo_watchdog_meta = repo_meta
        self._repo_watchdog_last_sig = ""
        self._pending_ingest_files = files
        self._refresh_topic_button()
        file_hint = f"  +  {len(files)} file(s) queued for ingestion" if files else ""
        repo_hint = "  · repo watchdog linked" if repo_meta else ""
        self._status_bar.showMessage(f"Topic set: {title[:80]}{file_hint}{repo_hint}")

    def _load_repo_dataset_into_orchestrator(self) -> bool:
        if self.orchestrator is None or not self._repo_watchdog_meta:
            return False
        repo_path = str(self._repo_watchdog_meta.get("repo_path", "") or "").strip()
        if not repo_path:
            return False

        data = self._repo_watchdog.load_semantic_dataset(repo_path)
        if not data:
            try:
                self._repo_watchdog.build_snapshot(repo_path)
                data = self._repo_watchdog.load_semantic_dataset(repo_path)
            except Exception:
                data = None
        if not data:
            return False

        facts = data.get("facts", []) if isinstance(data, dict) else []
        meta = data.get("_meta", {}) if isinstance(data, dict) else {}
        name = str(meta.get("name", "repo_watchdog_dataset"))
        if not isinstance(facts, list) or not facts:
            return False

        self.orchestrator.load_dataset(facts, name=name)
        return True

    def _extract_repo_watchdog_meta(self, context: str) -> tuple[str, dict | None]:
        match = re.search(r"\[REPO_WATCHDOG_META\](.*?)\[/REPO_WATCHDOG_META\]", context, re.DOTALL)
        if not match:
            return context, None

        payload = match.group(1).strip()
        meta = None
        try:
            meta = json.loads(payload)
        except Exception:
            meta = None

        clean = (context[: match.start()] + context[match.end() :]).strip()
        return clean, meta

    def _open_session_browser(self) -> None:
        if self._session_browser is None:
            self._session_browser = SessionBrowserDialog(
                session_manager=get_session_manager(),
                parent=self,
            )
        self._session_browser.show()
        self._session_browser.raise_()

    def _on_semantic_toggled(self, checked: bool) -> None:
        csm = get_cross_session_memory()
        csm.enabled = checked
        if checked:
            csm.refresh_index()
            self._semantic_btn.setText("🌐 Semantic ON")
            self._status_bar.showMessage("Semantic Awareness ON — agents will recall past debates")
        else:
            self._semantic_btn.setText("🌐 Semantic")
            self._status_bar.showMessage("Semantic Awareness OFF — isolated session")

    def _on_model_changed(self, _text: str = "") -> None:
        """Persist model selections whenever either combo changes.
        If a debate is running, queue the swap to apply at the next pair boundary."""
        left = _clean_model_name(self._astra_model_combo.currentText())
        right = _clean_model_name(self._nova_model_combo.currentText())
        if not (left and right):
            return
        self._model_prefs = {"left_model": left, "right_model": right}
        save_model_prefs(left, right)

        if self.orchestrator is not None:
            # Queue model swaps for a running debate
            live_left = getattr(self.orchestrator.left_agent.provider, "model", "")
            live_right = getattr(self.orchestrator.right_agent.provider, "model", "")
            if left != live_left:
                self._pending_left_model = left
            if right != live_right:
                self._pending_right_model = right
            # Build badge with pending indicators
            left_display = f"{left} ⏳" if self._pending_left_model else left
            right_display = f"{right} ⏳" if self._pending_right_model else right
            self._runtime_badge.setText(
                f"✓ Ollama  |  Astra: {left_display}  ·  Nova: {right_display}"
            )
            if self._pending_left_model or self._pending_right_model:
                self._status_bar.showMessage(
                    "Model change queued — will apply after current pair", 4000
                )
            else:
                self._status_bar.showMessage(
                    f"Models saved — Astra: {left}  Nova: {right}", 3000
                )
        else:
            self._runtime_badge.setText(
                f"✓ Ollama  |  Astra: {left}  ·  Nova: {right}"
            )
            self._status_bar.showMessage(f"Models saved — Astra: {left}  Nova: {right}", 3000)

    def _show_ollama_info_dialog(self) -> None:
        """Show Ollama API info, cloud model pricing, and current session cloud usage."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Ollama Cloud Info")
        dialog.setMinimumWidth(480)
        dialog.setStyleSheet(
            "QDialog { background: #07111c; color: #e0e0e0; }"
            "QLabel { color: #e0e0e0; }"
        )
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 14, 18, 14)

        def _section(title: str, color: str) -> QLabel:
            lbl = QLabel(title)
            lbl.setStyleSheet(
                f"color: {color}; font-weight: 700; font-size: 11pt;"
                " border-bottom: 1px solid #1a3a55; padding-bottom: 3px;"
            )
            return lbl

        def _row(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #b0bec5; font-size: 9pt; padding-left: 8px;")
            return lbl

        # --- API Key ---
        layout.addWidget(_section("🔑  API Key", "#80cbc4"))
        api_key_path = Path(self.project_root) / "ollama_api.txt"
        if api_key_path.exists():
            raw = api_key_path.read_text(encoding="utf-8").strip()
            masked = raw[:10] + "…" + raw[-8:] if len(raw) > 20 else raw
            layout.addWidget(_row(f"Key (masked): {masked}"))
            layout.addWidget(_row(f"File: {api_key_path}"))
        else:
            layout.addWidget(_row(
                f"No API key file found at:\n{api_key_path}\n"
                "Create this file and paste your Ollama API key into it."
            ))

        # --- Pricing ---
        layout.addWidget(_section("💰  Cloud Pricing (ollama.com/pricing)", "#ffd740"))
        layout.addWidget(_row(
            "▸  Free  — $0/mo  Light usage: chat, quick questions, trying models"
        ))
        layout.addWidget(_row(
            "▸  Pro   — $20/mo  Day-to-day: RAG, document analysis, coding tasks"
        ))
        layout.addWidget(_row(
            "▸  Max  — $100/mo  Heavy sustained: coding agents, batch processing"
        ))
        layout.addWidget(_row(
            "Pricing is subscription-based (not per-token). \n"
            "Usage limits apply by plan. Local models are always unlimited."
        ))

        # --- Session Usage ---
        layout.addWidget(_section("☁  This Session", "#4dd0e1"))
        cloud_count = self._cloud_calls
        lm = _clean_model_name(self._astra_model_combo.currentText()) or "(none)"
        rm = _clean_model_name(self._nova_model_combo.currentText()) or "(none)"
        layout.addWidget(_row(
            f"Cloud responses generated: {cloud_count}\n"
            f"Astra model: {lm}  (☁ cloud) " if _is_cloud_model(lm) else
            f"Cloud responses generated: {cloud_count}\n"
            f"Astra model: {lm}  (local)"
        ))
        layout.addWidget(_row(
            f"Nova model: {rm}  (☁ cloud)" if _is_cloud_model(rm)
            else f"Nova model: {rm}  (local)"
        ))
        layout.addWidget(_row(
            "View full usage at: ollama.com → Account → Usage"
        ))

        # --- Known cloud models ---
        layout.addWidget(_section("📌  Known Cloud Models", "#90a4ae"))
        cloud_list = ", ".join(sorted(_KNOWN_CLOUD_MODELS))
        cloud_lbl = QLabel(cloud_list)
        cloud_lbl.setWordWrap(True)
        cloud_lbl.setStyleSheet(
            "color: #546e7a; font-size: 8pt; padding-left: 8px; font-style: italic;"
        )
        layout.addWidget(cloud_lbl)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background: #1a3a55; color: #80cbc4; border-radius: 6px;"
            " padding: 6px 24px; border: 1px solid #2a4a65; }"
            "QPushButton:hover { background: #1e4a70; }"
        )
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    def _populate_model_combos(self) -> None:
        """Query Ollama AND the VS Code bridge for available models, then populate both dropdowns."""
        from providers.local_provider import LocalProvider
        from providers.vscode_provider import VSCodeProvider
        provider = LocalProvider()
        vsc = VSCodeProvider()

        async def _fetch() -> tuple[list[str], list[str]]:
            try:
                local = await provider.list_models()
            except Exception:
                local = []
            try:
                vscode = await vsc.list_models()
            except Exception:
                vscode = []
            return local, vscode

        try:
            local_raw, vscode_raw = asyncio.run(_fetch())
        except Exception:
            local_raw, vscode_raw = [], []

        # Update the VS Code model ID set so bootstrap uses the right provider
        self._vscode_model_ids = set(vscode_raw)

        saved_left = self._model_prefs.get("left_model", DEFAULT_LEFT_MODEL)
        saved_right = self._model_prefs.get("right_model", DEFAULT_RIGHT_MODEL)

        for combo, saved in (
            (self._astra_model_combo, saved_left),
            (self._nova_model_combo, saved_right),
        ):
            # Block signals while repopulating
            combo.blockSignals(True)
            combo.clear()

            # Build the full model list from Ollama — these are REAL models
            all_ollama = list(local_raw)

            # Separate cloud-tagged models from true local models
            # Cloud models have ":cloud" or "-cloud" in the Ollama tag
            cloud_models = sorted([m for m in all_ollama if _is_cloud_model(m)])
            local_models = sorted([m for m in all_ollama if not _is_cloud_model(m)])

            # Ensure the saved selection and defaults are always selectable
            # (even if Ollama didn't return them — e.g. model was pulled after last run)
            for must_have in (saved, DEFAULT_LEFT_MODEL, DEFAULT_RIGHT_MODEL):
                if must_have and must_have not in all_ollama and must_have not in self._vscode_model_ids:
                    if _is_cloud_model(must_have):
                        cloud_models.insert(0, must_have)
                    else:
                        local_models.insert(0, must_have)

            from PyQt6.QtGui import QFont, QColor, QBrush

            def _add_header(combo: _NoWheelComboBox, text: str) -> None:
                combo.addItem(text)
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(False)
                f = QFont()
                f.setBold(True)
                f.setPointSize(8)
                item.setFont(f)
                item.setForeground(QBrush(QColor("#455a64")))

            # ── Section 1: VS Code bridge models ─────────────────────────
            if self._vscode_model_ids:
                _add_header(combo, "  ◈  VS Code Bridge")
                for m in sorted(self._vscode_model_ids):
                    combo.addItem(f"  ◈ {m}")
                combo.insertSeparator(combo.count())

            # ── Section 2: Local Ollama models (always first / primary) ──
            if local_models:
                _add_header(combo, "  💻  Local Models")
                for m in local_models:
                    combo.addItem(f"  💻 {m}")

            # ── Section 3: Cloud-tagged Ollama models ────────────────────
            if cloud_models:
                if local_models:
                    combo.insertSeparator(combo.count())
                _add_header(combo, "  ☁  Cloud Models")
                for m in cloud_models:
                    combo.addItem(f"  ☁ {m}")

            # Restore saved selection — try with each prefix
            idx = -1
            for prefix in ("  💻 ", "  ☁ ", "  ◈ ", ""):
                idx = combo.findText(f"{prefix}{saved}")
                if idx >= 0:
                    break
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_endless_toggled(self, checked: bool) -> None:
        self._turns_slider.setEnabled(not checked)
        if checked:
            self._turns_display.setText("∞")
        else:
            self._turns_display.setText(str(self._turns_slider.value()))

    def _on_speed_changed(self, value: int) -> None:
        self._speed_display.setText(str(value))
        if self._tts_worker is not None:
            self._tts_worker.set_rate(value)

    # ---------------------------------------------------- debate control

    def start_debate(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self._status_bar.showMessage("Debate already running")
            return

        # Get topic — full context brief set by TopicPickerDialog
        topic = self._current_topic_context.strip()
        if not topic:
            # Fallback: first preset topic if user never opened the picker
            first = STARTER_TOPICS[0]
            topic = f"{first.title}\n\n{first.description}\n\nKEY TALKING POINTS:\n" + \
                    "\n".join(f"  • {tp}" for tp in first.talking_points)

        endless = self._endless_check.isChecked()
        turns = self._turns_slider.value() if not endless else 10

        # Reset UI
        self.left_panel.private_view.clear()
        self.right_panel.private_view.clear()
        self.center_panel.clear_messages()
        self.arbiter_panel.set_message("Arbiter watching...")
        self._continue_btn.setVisible(False)
        self._set_debate_transport_state("running")
        self.arbiter_panel.set_paused_state(False)
        self.arbiter_panel.clear_staging()
        self.graph_panel.clear_rows()
        # Reset scoring data for new debate
        self._turn_scores = []
        self._arbiter_events = []
        sm_scoring = get_session_manager()
        self.scoring_panel.new_session(str(sm_scoring.current_path) if sm_scoring.current_path else "")
        # Reset live score accumulators and cloud counter
        self._astra_scores = []
        self._nova_scores = []
        self._pending_left_model = None
        self._pending_right_model = None
        self._cloud_calls = 0
        self._cloud_calls_lbl.setText("☁ 0 cloud calls")
        self._cloud_calls_lbl.setStyleSheet("color: #546e7a; font-size: 9pt; font-weight: 600;")
        self._capture_seq = 0
        self._current_tts_msg_idx = -1
        self._current_tts_char_offset = 0
        self._update_stats_bar({}, 0)

        # Tag the next session with the current talking-point key
        get_session_manager().pending_tp_key = self._current_tp_key
        # Reset resolution tracking for the new session
        self._last_resolution_payload = {}

        # Rebuild orchestrator to reset memory for fresh start, with chosen models
        left_model = _clean_model_name(self._astra_model_combo.currentText()) or DEFAULT_LEFT_MODEL
        right_model = _clean_model_name(self._nova_model_combo.currentText()) or DEFAULT_RIGHT_MODEL
        self.orchestrator, self.debate_config = build_orchestrator(
            self.project_root,
            left_model=left_model,
            right_model=right_model,
            session_manager=get_session_manager(),
            vscode_model_ids=self._vscode_model_ids,
        )
        if self._repo_watchdog_meta:
            loaded_repo_ds = self._load_repo_dataset_into_orchestrator()
            if loaded_repo_ds:
                self._status_bar.showMessage("Repo Watchdog semantic dataset loaded for debate context", 4500)

        # Refresh cross-session index before starting (skip running sessions automatically)
        csm = get_cross_session_memory()
        if csm.enabled:
            csm.refresh_index()

        self.worker = DebateWorker(self.orchestrator, topic=topic, turns=turns, endless=endless)
        self.worker.event_signal.connect(self.handle_event)
        self.worker.done_signal.connect(self._on_debate_done)
        self.worker.start()
        self._set_debate_transport_state("running")
        self._start_repo_watchdog()

        # Start ingestion if files were queued — wait 600 ms for session to be created
        if self._pending_ingest_files:
            QTimer.singleShot(600, self._start_ingestion)

        mode_str = "ENDLESS" if endless else f"{turns} turns"
        self._status_bar.showMessage(f"Debate running [{mode_str}] — {topic[:60]}...")

    def stop_debate(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            self._set_debate_transport_state("stopped")
            return
        self.orchestrator.stop()
        self._stop_repo_watchdog()
        self._set_debate_transport_state("pausing")
        self._status_bar.showMessage("Stop requested — finishing current turn...")

    def _start_repo_watchdog(self) -> None:
        if not self._repo_watchdog_meta:
            self._stop_repo_watchdog()
            return
        repo_path = self._repo_watchdog_meta.get("repo_path", "")
        if repo_path:
            try:
                self._repo_watchdog.check_for_changes(repo_path)
            except Exception:
                pass
        self._repo_watchdog_timer.start()

    def _stop_repo_watchdog(self) -> None:
        if self._repo_watchdog_timer.isActive():
            self._repo_watchdog_timer.stop()

    def _poll_repo_watchdog(self) -> None:
        if not self._repo_watchdog_meta:
            return
        if self.worker is None or not self.worker.isRunning():
            return

        repo_path = self._repo_watchdog_meta.get("repo_path", "")
        if not repo_path:
            return

        try:
            changes = self._repo_watchdog.check_for_changes(repo_path)
        except Exception:
            return

        if not changes.has_changes:
            return

        signature = json.dumps(
            {"a": changes.added[:12], "m": changes.modified[:12], "d": changes.deleted[:12]},
            sort_keys=True,
        )
        if signature == self._repo_watchdog_last_sig:
            return
        self._repo_watchdog_last_sig = signature

        try:
            report = self._repo_watchdog.update_docs_for_changes(repo_path, changes)
            change_brief = self._repo_watchdog.build_change_brief(changes, report)
        except Exception:
            report = None
            change_brief = "Repository changes detected, but incremental docs refresh failed."

        if self.orchestrator is not None:
            self._load_repo_dataset_into_orchestrator()

        added = changes.added[:3]
        modified = changes.modified[:3]
        deleted = changes.deleted[:3]
        lines: list[str] = []
        if added:
            lines.append("Added: " + ", ".join(added))
        if modified:
            lines.append("Modified: " + ", ".join(modified))
        if deleted:
            lines.append("Deleted: " + ", ".join(deleted))
        short_msg = " | ".join(lines)

        self.center_panel.append_message(
            "Repo Watchdog",
            f"Repository changes detected during debate. {short_msg}\n\n{change_brief}",
        )
        if self.orchestrator is not None:
            self.orchestrator.inject_arbiter_message(
                "[REPO WATCHDOG UPDATE]\n"
                f"Detected repository changes: {short_msg}\n"
                f"{change_brief}\n"
                "Re-check any claims that depend on the changed files."
            )
        if report is not None:
            self._status_bar.showMessage(
                f"Repo Watchdog: {report.updated_files} docs refreshed, {report.deleted_files} removed",
                5000,
            )
        else:
            self._status_bar.showMessage("Repo Watchdog update injected into debate context", 5000)

    def _on_debate_pause_toggled(self, checked: bool) -> None:
        if not hasattr(self, 'orchestrator') or self.orchestrator is None:
            return
        if checked:
            self.orchestrator.pause()
            self._set_debate_transport_state("pausing")
            self._status_bar.showMessage("Pause requested — waiting for both agents to finish their turn…")
        else:
            # Manual uncheck (resume without injection)
            self.orchestrator.resume()
            self.arbiter_panel.set_paused_state(False)
            self._set_debate_transport_state("running")
            self._status_bar.showMessage("Debate resumed.")

    def _continue_debate(self) -> None:
        if not hasattr(self, 'orchestrator') or self.orchestrator is None:
            return
        extra_turns = self._turns_slider.value()
        self._continue_btn.setVisible(False)
        self._set_debate_transport_state("running")

        seed_topic = (
            self.orchestrator._living_topic.seed
            if self.orchestrator._living_topic is not None
            else self._current_topic_title
        )
        self.worker = DebateWorker(
            self.orchestrator,
            topic=seed_topic,
            turns=extra_turns,
            endless=False,
            continue_run=True,
        )
        self.worker.event_signal.connect(self.handle_event)
        self.worker.done_signal.connect(self._on_debate_done)
        self.worker.start()
        self._start_repo_watchdog()
        self._status_bar.showMessage(f"Continuing debate for {extra_turns} more turns...")

    # ---------------------------------------------------------------- follow-text

    def _on_follow_text_btn_clicked(self, checked: bool) -> None:
        """User clicked the Follow Text toggle button."""
        self.center_panel.set_follow_mode(checked)

    def _on_follow_mode_changed(self, enabled: bool) -> None:
        """Panel emitted follow_mode_changed (e.g. user scrolled away) — sync button."""
        self._follow_text_btn.blockSignals(True)
        self._follow_text_btn.setChecked(enabled)
        self._follow_text_btn.blockSignals(False)

    def _on_arbiter_inject(self, text: str, snippets: list) -> None:
        """Called when user sends arbiter message + staged snippets while paused."""
        if hasattr(self, 'orchestrator') and self.orchestrator is not None:
            combined = text
            if snippets:
                snippet_block = "\n\n".join(
                    f"[CAPTURED SEGMENT — {s.get('speaker', '?')} "
                    f"turn {s.get('turn', '?')}]:\n{s.get('text', '')}"
                    for s in snippets
                )
                combined = (
                    f"{text}\n\n"
                    f"[ARBITER CONTEXT — captured from the live debate as "
                    f"supporting evidence for this intervention]:\n{snippet_block}"
                )
            self.orchestrator.inject_arbiter_message(combined)
        self._cleanup_snippet_files(snippets)
        self.arbiter_panel.clear_staging()
        self.arbiter_panel.set_paused_state(False)
        self._set_debate_transport_state("running")
        self._status_bar.showMessage(f"Injected: “{text[:60]}” — debate resumed.")

    def _on_debate_done(self) -> None:
        self._stop_repo_watchdog()
        self._status_bar.showMessage("Debate complete — press Continue to add more turns")
        left_facts = len(self.orchestrator.left_agent.semantic_memory.facts)
        right_facts = len(self.orchestrator.right_agent.semantic_memory.facts)
        self.left_panel.update_memory_count(left_facts)
        self.right_panel.update_memory_count(right_facts)
        # Show continue button (hide if endless mode)
        if not self._endless_check.isChecked():
            self._continue_btn.setVisible(True)
        # Reset debate controls
        self._set_debate_transport_state("stopped")
        self.arbiter_panel.set_paused_state(False)

        # Trigger debate summary + verdict worker
        self.scoring_panel.set_summary_generating()
        _sm_done = get_session_manager()
        _session_path_done = str(_sm_done.current_path) if _sm_done.current_path else ""
        self.scoring_panel.set_session_path(_session_path_done)
        try:
            from config.model_prefs import load_model_prefs as _lmp
            _verdict_model = _lmp().get("right_model", "qwen3:30b") or "qwen3:30b"
        except Exception:
            _verdict_model = "qwen3:30b"
        self._summary_worker = DebateSummaryWorker(
            topic=self._current_topic_title,
            left_agent="Astra",
            right_agent="Nova",
            turn_scores=list(self._turn_scores),
            arbiter_events=list(self._arbiter_events),
            model=_verdict_model,
            session_path=_session_path_done,
            parent=self,
        )
        self._summary_worker.finished.connect(self._on_summary_done)
        self._summary_worker.failed.connect(
            lambda e: self._status_bar.showMessage(f"Summary failed: {e[:80]}")
        )
        self._summary_worker.start()

        # Spawn topic refinement if we have a talking-point key and resolution data
        if self._current_tp_key and self._last_resolution_payload:
            try:
                from config.model_prefs import load_model_prefs
                prefs = load_model_prefs()
                refine_model = prefs.get("right_model", "qwen3:30b") or "qwen3:30b"
            except Exception:
                refine_model = "qwen3:30b"

            from config.starter_topics import get_topic_by_title
            st = get_topic_by_title(self._current_topic_title)
            desc = st.description if st else ""
            tps = st.talking_points if st else []

            rd = {
                "truths": self._last_resolution_payload.get("truths_discovered", []),
                "problems": self._last_resolution_payload.get("problems_found", []),
                "sub_topics": self._last_resolution_payload.get("sub_topics_explored", []),
            }
            self._refine_worker = TopicRefineWorker(
                title=self._current_topic_title,
                description=desc,
                talking_points=tps,
                resolution_data=rd,
                model=refine_model,
                tp_key=self._current_tp_key,
                session_manager=get_session_manager(),
                parent=self,
            )
            self._refine_worker.finished.connect(
                lambda _d: self._status_bar.showMessage(
                    "✨ Topic refined for next session — reopen the topic picker to see it.", 8000
                )
            )
            self._refine_worker.failed.connect(
                lambda err: self._status_bar.showMessage(f"Topic refinement failed: {err[:80]}")
            )
            self._refine_worker.start()

    # ------------------------------------------------------------------ scoring

    def _on_summary_done(self, result: dict) -> None:
        """Called when DebateSummaryWorker finishes — push verdict to scoring panel."""
        self.scoring_panel.set_verdict(
            winner   = result.get("winner", "—"),
            margin   = result.get("margin", "—"),
            reason   = result.get("reason", ""),
            summary  = result.get("summary", ""),
            astra_avg= result.get("astra_avg", 0.0),
            nova_avg = result.get("nova_avg", 0.0),
        )
        winner = result.get("winner", "—")
        self._status_bar.showMessage(
            f"🏆 Verdict: {winner} wins — see Scoring & Verdict tab", 8000
        )

    # ------------------------------------------------------------------ analytics

    def _open_analytics_dialog(self) -> None:
        """Show the Analytics Manager dialog."""
        dlg = AnalyticsDialog(self)
        dlg.exec()

    def _start_ingestion(self) -> None:
        """Start background file ingestion once the session folder has been created."""
        if not self._pending_ingest_files:
            return
        sm = get_session_manager()
        if sm.current_session is None:
            # Session not ready yet — try again in 500 ms
            QTimer.singleShot(500, self._start_ingestion)
            return
        session_id = sm.current_session.session_id
        from pathlib import Path as _Path
        source_paths = [_Path(p) for p in self._pending_ingest_files]
        self._ingest_worker = IngestionWorker(
            source_paths=source_paths,
            session_manager=sm,
            session_id=session_id,
            parent=self,
        )
        self._ingest_worker.progress.connect(
            lambda msg: self._status_bar.showMessage(f"[Ingestion] {msg}")
        )
        self._ingest_worker.finished.connect(self._on_ingestion_done)
        self._ingest_worker.dataset_saved.connect(
            lambda name: self._status_bar.showMessage(f"Dataset \"{name}\" saved to global store — agents now reading it like a LORA", 6000)
        )
        self._ingest_worker.failed.connect(
            lambda err: self._status_bar.showMessage(f"Ingestion failed: {err}")
        )
        self._ingest_worker.start()
        # Clear the queue so rerunning the same debate doesn't re-ingest
        self._pending_ingest_files = []
        self._refresh_topic_button()

    def _on_ingestion_done(self, num_facts: int) -> None:
        """Called when background ingestion finishes — load dataset into orchestrator."""
        self._status_bar.showMessage(
            f"Ingestion complete — {num_facts} facts indexed & semantically weighted. Agents now dataset-aware."
        )
        # Load the ingested dataset into the orchestrator so agents can reference it
        if hasattr(self, 'orchestrator') and self.orchestrator is not None:
            loaded = self.orchestrator.load_dataset_from_session()
            if loaded:
                self._status_bar.showMessage(
                    f"Ingestion complete — {num_facts} facts loaded as agent LORA context",
                    6000,
                )

    def _open_session_folder(self) -> None:
        """Open the current session directory in Windows Explorer."""
        sm = get_session_manager()
        if sm.current_path is not None:
            try:
                os.startfile(str(sm.current_path))
            except Exception as exc:
                self._status_bar.showMessage(f"Cannot open folder: {exc}")
        else:
            self._status_bar.showMessage("No active session — start a debate first")

    def _update_stats_bar(self, res_stats: dict, sub_topics_count: int = 0) -> None:
        """Refresh the live stats bar with current quality averages and resolution counts."""
        if self._astra_scores:
            astra_avg = round(sum(self._astra_scores) / len(self._astra_scores), 2)
            astra_disp = f"Astra: {astra_avg:.2f}"
        else:
            astra_avg = 0.0
            astra_disp = "Astra: —"

        if self._nova_scores:
            nova_avg = round(sum(self._nova_scores) / len(self._nova_scores), 2)
            nova_disp = f"Nova: {nova_avg:.2f}"
        else:
            nova_avg = 0.0
            nova_disp = "Nova: —"

        self._stats_astra_lbl.setText(astra_disp)
        self._stats_nova_lbl.setText(nova_disp)
        self._stats_conclusions_lbl.setText(f"✓ {res_stats.get('conclusions', 0)}")
        self._stats_contradictions_lbl.setText(f"↔ {res_stats.get('contradictions', 0)}")
        self._stats_falsehoods_lbl.setText(f"✗ {res_stats.get('falsehoods', 0)}")
        self._stats_subtopics_lbl.setText(f"🌿 {sub_topics_count}")

        if self._astra_scores and self._nova_scores:
            if astra_avg > nova_avg:
                self._stats_astra_lbl.setStyleSheet("color: #4caf50; font-size: 9pt; font-weight: 700;")
                self._stats_nova_lbl.setStyleSheet("color: #ef5350; font-size: 9pt; font-weight: 700;")
            elif nova_avg > astra_avg:
                self._stats_astra_lbl.setStyleSheet("color: #ef5350; font-size: 9pt; font-weight: 700;")
                self._stats_nova_lbl.setStyleSheet("color: #4caf50; font-size: 9pt; font-weight: 700;")
            else:
                # Tied — both blue
                self._stats_astra_lbl.setStyleSheet("color: #64b5f6; font-size: 9pt; font-weight: 700;")
                self._stats_nova_lbl.setStyleSheet("color: #64b5f6; font-size: 9pt; font-weight: 700;")
        else:
            self._stats_astra_lbl.setStyleSheet("color: #546e7a; font-size: 9pt; font-weight: 600;")
            self._stats_nova_lbl.setStyleSheet("color: #546e7a; font-size: 9pt; font-weight: 600;")

    # ---------------------------------------------------- TTS

    def _set_tts_button_playing(self, playing: bool) -> None:
        """Swap the play/stop button appearance."""
        self._tts_playing = playing
        self._tts_play_btn.setChecked(playing)
        if playing:
            self._tts_play_btn.setText("⏹  Stop Reading")
        else:
            self._tts_play_btn.setText("🔊 Read Debate")
            self._tts_pause_btn.setText("⏸ Pause")

    def tts_play_stop_toggle(self) -> None:
        """Single button: if TTS is running → stop; otherwise → start."""
        if self._tts_playing:
            self.tts_stop()
        else:
            self._tts_start()

    def _tts_start(self) -> None:
        messages = self.center_panel.get_messages()
        if not messages:
            self._status_bar.showMessage("No messages to read yet — start a debate first")
            self._set_tts_button_playing(False)
            return

        # Kill any lingering worker
        if self._tts_worker is not None:
            self._tts_worker.request_stop()
            self._tts_worker.wait(500)
            self._tts_worker = None

        self._tts_worker = TTSPlaybackWorker(self)
        self._tts_worker.load_messages(messages, start_index=0)
        self._tts_worker.set_rate(self._speed_slider.value())
        self._tts_worker.now_speaking.connect(self._on_tts_now_speaking)
        self._tts_worker.now_speaking.connect(
            lambda idx, _agent: self.center_panel.highlight_message(idx)
        )
        self._tts_worker.word_at.connect(
            lambda idx, offset, length: self.center_panel.highlight_word(idx, offset, length)
        )
        self._tts_worker.word_at.connect(self._on_tts_word_at)
        self._tts_worker.finished_all.connect(self._on_tts_done)
        self._tts_worker.error.connect(self._on_tts_error)
        self._tts_worker.start()
        self._set_tts_button_playing(True)
        self._status_bar.showMessage("TTS reading debate from the top...")

    def tts_toggle_pause(self) -> None:
        if self._tts_worker is None or not self._tts_worker.isRunning():
            return
        self._tts_worker.toggle_pause()
        if self._tts_worker.is_paused:
            self._tts_pause_btn.setText("▶ Resume")
            self._status_bar.showMessage("TTS paused")
        else:
            self._tts_pause_btn.setText("⏸ Pause")
            self._status_bar.showMessage("TTS resumed")

    def tts_stop(self) -> None:
        """Immediately stop TTS and reset button."""
        if self._tts_worker is not None:
            self._tts_worker.request_stop()
            self._tts_worker.wait(1500)
            self._tts_worker = None
        self.center_panel.clear_highlight()
        self._set_tts_button_playing(False)
        self._current_speaking_idx = -1
        self._status_bar.showMessage("TTS stopped")

    def _on_tts_now_speaking(self, idx: int, agent: str) -> None:
        self._current_speaking_idx = idx
        self._status_bar.showMessage(f"🔊 Reading [{idx + 1}] {agent}...")

    def _on_tts_done(self) -> None:
        self.center_panel.clear_highlight()
        self._set_tts_button_playing(False)
        self._status_bar.showMessage("TTS finished reading all messages")

    def _on_tts_error(self, msg: str) -> None:
        self.center_panel.clear_highlight()
        self._set_tts_button_playing(False)
        self._status_bar.showMessage(f"TTS error: {msg}")

    def _on_tts_word_at(self, idx: int, offset: int, length: int) -> None:
        """Track current TTS word position for Capture Segment."""
        self._current_tts_msg_idx = idx
        self._current_tts_char_offset = offset

    def _on_capture_pressed(self) -> None:
        """Capture N sentences before + M sentences after current TTS word."""
        import re as _re
        from datetime import datetime as _dt

        msg_idx    = self._current_tts_msg_idx
        char_offset = self._current_tts_char_offset
        messages   = self.center_panel.get_messages()

        if msg_idx < 0 or msg_idx >= len(messages):
            self._status_bar.showMessage(
                "No TTS position to capture — start TTS playback first"
            )
            return

        speaker, full_text = messages[msg_idx]
        before_n = self.arbiter_panel.before_count
        after_n  = self.arbiter_panel.after_count

        # Split into sentences on .  !  ? followed by space/end
        sent_re   = _re.compile(r'(?<=[.!?])\s+')
        sentences = sent_re.split(full_text)

        # Find which sentence contains char_offset
        pos      = 0
        sent_idx = 0
        for i, s in enumerate(sentences):
            if pos + len(s) >= char_offset:
                sent_idx = i
                break
            pos += len(s) + 1   # +1 for the split space

        start_i  = max(0, sent_idx - before_n)
        end_i    = min(len(sentences), sent_idx + after_n + 1)
        captured = sentences[start_i:end_i]
        snippet_text = " ".join(captured)

        # Metadata
        self._capture_seq += 1
        ts    = _dt.now().strftime("%H%M%S")
        label = f"cap_{self._capture_seq:03d}_{ts}"

        # Save to session captures folder
        filepath = None
        sm = get_session_manager()
        if sm.current_path:
            cap_dir = sm.current_path / "captures"
            cap_dir.mkdir(exist_ok=True)
            filepath = cap_dir / f"{label}.md"
            header = (
                f"<!-- CAPTURED RESPONSE SEGMENT -->\n"
                f"<!-- Speaker: {speaker} | Message index: {msg_idx} "
                f"| Timestamp: {_dt.now().isoformat()} -->\n"
                f"<!-- Capture settings: {before_n} sentences before, "
                f"{after_n} sentence(s) after -->\n\n"
            )
            filepath.write_text(header + snippet_text, encoding="utf-8")

        snippet = {
            "label":     label,
            "speaker":   speaker,
            "turn":      msg_idx,
            "before":    before_n,
            "after":     after_n,
            "timestamp": _dt.now().isoformat(),
            "filepath":  str(filepath) if filepath else None,
            "text":      snippet_text,
        }

        self.arbiter_panel.add_snippet(snippet)
        self._status_bar.showMessage(
            f"Captured {len(captured)} sentence(s) from {speaker} [msg {msg_idx + 1}]"
        )

    def _cleanup_snippet_files(self, snippets: list) -> None:
        """Delete temporary snippet files after they have been sent."""
        for s in snippets:
            fp = s.get("filepath")
            if fp:
                try:
                    Path(fp).unlink(missing_ok=True)
                except Exception:
                    pass

    # ---------------------------------------------------- event handler

    def handle_event(self, event: DebateEvent) -> None:  # noqa: C901
        if event.event_type == "private_thought":
            agent = event.payload["agent"]
            thought = event.payload["thought"]
            turn = event.payload.get("turn", "")
            if agent.lower() == "astra":
                self.left_panel.append_thought(f"[T{turn}] {thought}")
            else:
                self.right_panel.append_thought(f"[T{turn}] {thought}")

        elif event.event_type == "public_message":
            agent   = event.payload["agent"]
            message = event.payload["message"]
            turn    = event.payload.get("turn", 0)
            quality = event.payload.get("quality")
            # Update the turn/speaker indicator in the header
            total_turns = 0 if self._endless_check.isChecked() else self._turns_slider.value()
            self.center_panel.update_turn_indicator(agent, turn, total_turns)
            # Resolve the model name from the live provider at the moment of delivery
            model_name = ""
            if self.orchestrator is not None:
                if agent.lower() == "astra":
                    model_name = getattr(self.orchestrator.left_agent.provider, "model", "")
                else:
                    model_name = getattr(self.orchestrator.right_agent.provider, "model", "")
            self.center_panel.append_message(
                agent,
                message,
                citations=event.payload.get("citations", []),
                evidence_score=event.payload.get("evidence_score"),
                talking_point=event.payload.get("talking_point"),
                quality=quality,
                model_name=model_name,
                reframe=event.payload.get("reframe"),
            )
            # Track cloud model usage
            if _is_cloud_model(model_name):
                self._cloud_calls += 1
                self._cloud_calls_lbl.setText(f"☁ {self._cloud_calls} cloud call{'s' if self._cloud_calls != 1 else ''}")
                self._cloud_calls_lbl.setStyleSheet("color: #4dd0e1; font-size: 9pt; font-weight: 700;")
            # After Nova's turn (end of a full pair), apply any pending model swaps
            if agent.lower() == "nova" and self.orchestrator is not None:
                swapped = False
                if self._pending_left_model:
                    self.orchestrator.left_agent.provider.model = self._pending_left_model
                    self._pending_left_model = None
                    swapped = True
                if self._pending_right_model:
                    self.orchestrator.right_agent.provider.model = self._pending_right_model
                    self._pending_right_model = None
                    swapped = True
                if swapped:
                    lm = getattr(self.orchestrator.left_agent.provider, "model", "?")
                    rm = getattr(self.orchestrator.right_agent.provider, "model", "?")
                    self._runtime_badge.setText(f"✓ Ollama  |  Astra: {lm}  ·  Nova: {rm}")
                    self._status_bar.showMessage(
                        f"Models swapped — Astra: {lm}  Nova: {rm}", 4000
                    )
            # Update live score accumulators
            if quality:
                composite = round(
                    (quality.get("relevance", 0.0) + quality.get("novelty", 0.0) + quality.get("evidence", 0.0)) / 3,
                    3,
                )
                if agent.lower() == "astra":
                    self._astra_scores.append(composite)
                else:
                    self._nova_scores.append(composite)
                # Feed live turn score to scoring panel
                turn_num = event.payload.get("turn", len(self._turn_scores) + 1)
                self._turn_scores.append({
                    "turn":      turn_num,
                    "agent":     agent,
                    "composite": composite,
                    "relevance": quality.get("relevance", 0.0),
                    "novelty":   quality.get("novelty", 0.0),
                    "evidence":  quality.get("evidence", 0.0),
                })
                self.scoring_panel.add_turn_score(
                    turn_num, agent, composite,
                    quality.get("relevance", 0.0),
                    quality.get("novelty", 0.0),
                    quality.get("evidence", 0.0),
                )
            # Refresh stats bar with latest resolution counters
            self._update_stats_bar(
                event.payload.get("resolution_stats", {}),
                event.payload.get("sub_topics_count", 0),
            )
            # Live-feed to TTS if it is running in heartbeat mode
            if self._tts_worker is not None and self._tts_worker.isRunning():
                self._tts_worker.add_message(agent, message)
            # Update memory counts live
            left_facts = len(self.orchestrator.left_agent.semantic_memory.facts)
            right_facts = len(self.orchestrator.right_agent.semantic_memory.facts)
            self.left_panel.update_memory_count(left_facts)
            self.right_panel.update_memory_count(right_facts)

        elif event.event_type == "arbiter":
            self.arbiter_panel.set_message(
                f"[T{event.payload.get('turn','')}] {event.payload['message']}"
            )
            _arb_turn = event.payload.get("turn", "")
            _arb_msg  = event.payload.get("message", "")
            _arb_echo = event.payload.get("echo", False)
            self._arbiter_events.append({"turn": _arb_turn, "message": _arb_msg, "echo": _arb_echo})
            self.scoring_panel.add_arbiter_event(_arb_turn, _arb_msg, _arb_echo)

        elif event.event_type == "branch":
            sub = event.payload.get("sub_topic", "")
            self.arbiter_panel.set_message(
                f"🌿 New branch: {sub}"
            )

        elif event.event_type == "branch_switch":
            np = event.payload.get("new_talking_point", "")
            self.arbiter_panel.set_message(
                f"↪ Now exploring: {np}"
            )

        elif event.event_type == "evidence":
            items = event.payload.get("items", [])
            if items:
                self.arbiter_panel.set_message(f"📚 Evidence: {items[0][:80]}...")

        elif event.event_type == "graph":
            tree = event.payload.get("tree")
            if tree:
                self.graph_panel.set_rows_rich(tree)
            else:
                self.graph_panel.set_rows(event.payload["rows"])

        elif event.event_type == "state":
            t = event.payload["transition"]
            self._status_bar.showMessage(
                f"State: {t['previous']} → {t['current']}  ({t['reason']})"
            )

        elif event.event_type == "paused":
            # Both agents finished — enable bottom composer
            pair = event.payload.get("pair", "")
            self._set_debate_transport_state("paused")
            self.arbiter_panel.set_paused_state(True)
            self.arbiter_panel.focus_input()
            self.arbiter_panel.set_message(
                f"⏸ Debate paused after pair {pair} — type an arbiter intervention below."
            )
            self._status_bar.showMessage(
                "Debate paused — type a message in the bottom composer and press Send"
            )

        elif event.event_type == "arbiter_injection":
            msg = event.payload.get("message", "")
            self.center_panel.append_message(
                "Arbiter",
                f"💬 CROWD INJECTION: {msg}",
            )
            self.arbiter_panel.set_message(f"💬 Injected: {msg[:80]}")

        elif event.event_type == "resolution":
            # Capture for TopicRefineWorker after session ends
            self._last_resolution_payload = event.payload
            truths = event.payload.get("truths_discovered", [])
            problems = event.payload.get("problems_found", [])
            subs = event.payload.get("sub_topics_explored", [])
            facts = event.payload.get("total_memory_facts", 0)
            summary_lines = ["=== RESOLUTION ==="]
            if truths:
                summary_lines.append("TRUTHS DISCOVERED:")
                summary_lines.extend(f"  • {t}" for t in truths[:5])
            if problems:
                summary_lines.append("PROBLEMS FOUND:")
                summary_lines.extend(f"  ✗ {p}" for p in problems[:5])
            if subs:
                summary_lines.append(f"SUB-TOPICS EXPLORED ({len(subs)}):")
                summary_lines.extend(f"  → {s}" for s in subs[:8])
            summary_lines.append(f"Total memory facts accumulated: {facts}")
            self.center_panel.append_message(
                "Resolution", "\n".join(summary_lines)
            )

    # -------------------------------------------- source dialog

    def open_source_dialog(self, source_path: str) -> None:
        decoded = unquote(source_path)
        path = Path(decoded)
        if not path.exists():
            self._status_bar.showMessage(f"Source not found: {decoded}")
            return

        content = path.read_text(encoding="utf-8", errors="replace")

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Source — {path.name}")
        dialog.resize(1000, 700)
        layout = QVBoxLayout(dialog)
        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(content)
        layout.addWidget(viewer)
        dialog.exec()

    # -------------------------------------------- runtime badge

    def _refresh_runtime_badge(self) -> None:
        from providers.local_provider import LocalProvider
        provider = LocalProvider()

        async def _probe() -> tuple[bool, int]:
            try:
                running = await provider.is_running()
                models = await provider.list_models() if running else []
                return running, len(models)
            except Exception:
                return False, 0

        try:
            running, count = asyncio.run(_probe())
        except Exception:
            running, count = False, 0

        left_m = self._model_prefs.get("left_model", DEFAULT_LEFT_MODEL)
        right_m = self._model_prefs.get("right_model", DEFAULT_RIGHT_MODEL)

        if running:
            self._runtime_badge.setText(
                f"✓ Ollama — {count} models  |  Astra: {left_m}  ·  Nova: {right_m}"
            )
            self._runtime_badge.setStyleSheet("color: #69f0ae; padding: 0 8px; font-weight:600;")
        else:
            self._runtime_badge.setText("✗ Ollama offline")
            self._runtime_badge.setStyleSheet("color: #ff5252; padding: 0 8px; font-weight:600;")
