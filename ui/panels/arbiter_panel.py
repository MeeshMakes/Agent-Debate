from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ArbiterPanel(QWidget):
    """Arbiter dock panel — split left/right.

    Left  : arbiter log (responses from the watching arbiter)
    Right : intervention controls (text input, send, capture, staging list)

    Exposes the same public API that main_window.py used from _BottomComposer
    so wiring is a simple rename.
    """

    send_requested    = pyqtSignal(str, list)   # (message_text, list[snippet_dict])
    capture_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("arbiterPanel")
        self._staging_snippets: list[dict] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #1a3a55; }")
        root.addWidget(splitter)

        # ── LEFT: arbiter log ─────────────────────────────────────────────────
        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(6, 6, 4, 6)
        left_lay.setSpacing(4)

        log_hdr = QLabel("ARBITER")
        log_hdr.setStyleSheet(
            "font-size: 10pt; font-weight: 700; color: #ffd740; letter-spacing: 2px;"
        )

        self._view = QTextBrowser()
        self._view.setReadOnly(True)
        self._view.setPlaceholderText("Arbiter watching…")
        self._view.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self._view.setStyleSheet(
            "QTextBrowser { border: 1px solid #5d4037; border-radius: 8px;"
            " background-color: #1a1500; color: #fff9c4; font-size: 9pt; }"
        )

        left_lay.addWidget(log_hdr)
        left_lay.addWidget(self._view)

        # ── RIGHT: intervention panel ─────────────────────────────────────────
        right_w = QWidget()
        right_w.setStyleSheet("background: #07111c; border-left: 2px solid #1a3a55;")
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(12, 10, 12, 10)
        right_lay.setSpacing(8)

        # Header
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(6)
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
        hdr_row.addWidget(icon_lbl)
        hdr_row.addWidget(arb_lbl)
        hdr_row.addSpacing(8)
        hdr_row.addWidget(self._state_lbl)
        hdr_row.addStretch()

        # Text input + Send button
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

        # Capture controls
        cap_row = QHBoxLayout()
        cap_row.setSpacing(5)

        cap_hdg = QLabel("📎 Capture")
        cap_hdg.setStyleSheet(
            "color: #546e7a; font-size: 9px; font-weight:600; background: transparent;"
        )
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

        cap_row.addWidget(cap_hdg)
        cap_row.addWidget(before_lbl)
        cap_row.addWidget(self._before_spin)
        cap_row.addSpacing(2)
        cap_row.addWidget(after_lbl)
        cap_row.addWidget(self._after_spin)
        cap_row.addSpacing(8)
        cap_row.addWidget(self._capture_btn)
        cap_row.addStretch()

        # Staging list
        self._staging_list = QListWidget()
        self._staging_list.setToolTip(
            "Captured segments — all will be sent as context with the next arbiter message"
        )
        self._staging_list.setStyleSheet(
            "QListWidget { background: #050d16; color: #80cbc4;"
            " border: 1px solid #1a3a55; border-radius: 4px; font-size: 9px; }"
            "QListWidget::item { padding: 2px 6px; }"
            "QListWidget::item:selected { background: #0a2030; }"
        )

        right_lay.addLayout(hdr_row)
        right_lay.addLayout(input_row)
        right_lay.addLayout(cap_row)
        right_lay.addWidget(self._staging_list, stretch=1)

        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setSizes([560, 440])

        # Start locked until debate is paused
        self._set_interactive(False)

    # ── Arbiter log ──────────────────────────────────────────────────────────

    def set_message(self, message: str) -> None:
        self._view.append(
            f"<div style='color:#ffd740; margin:2px 0;'>{message}</div>"
        )

    # ── State ────────────────────────────────────────────────────────────────

    def set_paused_state(self, paused: bool) -> None:
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

    # ── Capture / staging ────────────────────────────────────────────────────

    @property
    def before_count(self) -> int:
        return self._before_spin.value()

    @property
    def after_count(self) -> int:
        return self._after_spin.value()

    def add_snippet(self, snippet: dict) -> None:
        self._staging_snippets.append(snippet)
        preview = snippet.get("text", "")[:80].replace("\n", " ")
        self._staging_list.addItem(QListWidgetItem(f"📎 {preview}…"))

    def clear_staging(self) -> None:
        self._staging_snippets.clear()
        self._staging_list.clear()

    def focus_input(self) -> None:
        self._input.setFocus()

    # ── Send ─────────────────────────────────────────────────────────────────

    def _send(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        snippets = list(self._staging_snippets)
        self._input.clear()
        self.send_requested.emit(text, snippets)
