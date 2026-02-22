"""Centre debate stream \u2014 page-per-turn architecture.

Each agent response fills the entire panel as a full-screen "page".  The
visible page is split horizontally:

    Left sidebar  (~20%)  \u2014  talking-point changes, quality scores,
                              citations / sources, model name.
    Right body    (~80%)  \u2014  scrollable QTextBrowser with the response text.
                              This is the ONLY text TTS reads, making
                              cursor-follow trivial.

Navigation between turns is via \u25c0/\u25b6 buttons or auto-advancing when
a new message arrives (follow-mode).
"""
from __future__ import annotations

from html import escape
import re
from urllib.parse import quote

try:
    import markdown as _md_lib
    _MD_AVAILABLE = True
except ImportError:
    _MD_AVAILABLE = False


# \u2500\u2500 Document stylesheet applied inside each page's QTextBrowser \u2500\u2500
_DOC_STYLESHEET = (
    "body { margin: 0; padding: 0; } "
    "p { margin: 4px 0; line-height: 180%; } "
    "code { font-family: 'Cascadia Code', Consolas, 'Courier New', monospace; "
    "background-color: #0d1b2a; color: #80cbc4; font-size: 10pt; "
    "padding: 1px 5px; border-radius: 3px; } "
    "pre { background-color: #0a1520; color: #a8e6cf; "
    "font-family: 'Cascadia Code', Consolas, 'Courier New', monospace; "
    "font-size: 10pt; padding: 10px 14px; margin: 6px 0; "
    "border-left: 3px solid #00bcd4; line-height: 150%; } "
    "blockquote { color: #b0bec5; margin-left: 12px; padding-left: 12px; "
    "border-left: 2px solid #37474f; font-style: italic; } "
    "strong { color: #e8f0ff; } "
    "em { color: #b8c9d9; font-style: italic; } "
    "a { color: #64b5f6; text-decoration: none; } "
    "li { color: #e8eef9; margin: 2px 0; line-height: 170%; } "
    "ul, ol { margin: 4px 0 4px 16px; } "
    "h1 { color: #a0cfff; font-size: 14pt; margin: 8px 0 4px 0; } "
    "h2 { color: #a0cfff; font-size: 13pt; margin: 6px 0 4px 0; } "
    "h3 { color: #90b8e0; font-size: 12pt; margin: 4px 0 2px 0; } "
    "table { border-collapse: collapse; margin: 4px 0; } "
    "td, th { padding: 4px 8px; border: 1px solid #2a3a55; } "
    "th { background-color: #0f1e35; color: #80cbc4; font-weight: 700; } "
)


def _md_to_html(text: str) -> str:
    if _MD_AVAILABLE:
        try:
            return _md_lib.markdown(
                text,
                extensions=["fenced_code", "tables"],
                output_format="html",
            )
        except Exception:
            pass
    return escape(text)


from PyQt6.QtCore import QUrl, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)
from tts.speech_engine import TTSPlaybackWorker


# \u2500\u2500 Per-agent theming \u2500\u2500
AGENT_COLORS = {
    "Astra": "#00e5ff",
    "Nova": "#ff6e40",
    "Arbiter": "#ffd740",
    "Resolution": "#69f0ae",
    "Repo Watchdog": "#b39ddb",
}

AGENT_ICONS = {
    "Astra": "\u25c6",        # diamond
    "Nova": "\u25cf",         # circle
    "Arbiter": "\u25b2",      # triangle
    "Resolution": "\u2726",   # star
    "Repo Watchdog": "\u29d6",  # hourglass
}

AGENT_CARD_BG = {
    "Astra": "#0a1525",
    "Nova": "#150f0a",
    "Arbiter": "#14120a",
    "Resolution": "#0a1510",
    "Repo Watchdog": "#100e18",
}

_TTS_HIGHLIGHT_BG = QColor("#e65100")
_TTS_HIGHLIGHT_FG = QColor("#ffffff")


# =====================================================================
#  TurnPageWidget \u2014 one full-panel page per agent response
# =====================================================================

class TurnPageWidget(QFrame):
    """Full-page widget for a single debate turn.

    Layout:  [ left sidebar ~20% | right body ~80% ]
    """

    anchor_clicked = pyqtSignal(QUrl)

    def __init__(
        self,
        speaker: str,
        text: str,
        msg_num: int,
        *,
        citations: list[dict] | None = None,
        evidence_score: float | None = None,
        talking_point_html: str = "",
        quality: dict | None = None,
        model_name: str | None = None,
        reframe: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        color = AGENT_COLORS.get(speaker, "#e0e0e0")
        icon = AGENT_ICONS.get(speaker, "\u25cf")
        card_bg = AGENT_CARD_BG.get(speaker, "#0c1422")

        self.speaker = speaker
        self.raw_text = text
        self.cleaned_text = TTSPlaybackWorker.clean_text(text)

        # -- Frame styling -- the whole page
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"TurnPageWidget {{ background: {card_bg}; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        page_layout = QHBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        # =============================================================
        #  LEFT SIDEBAR  (~20%)
        # =============================================================
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sidebar_scroll.setStyleSheet(
            f"QScrollArea {{ background: {card_bg}; border: none; "
            f"border-right: 2px solid {color}30; }}"
            "QScrollBar:vertical { width: 4px; background: transparent; }"
            f"QScrollBar::handle:vertical {{ background: {color}40; "
            "border-radius: 2px; }"
        )
        sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Preferred,
                                     QSizePolicy.Policy.Expanding)
        sidebar_scroll.setMinimumWidth(170)
        sidebar_scroll.setMaximumWidth(300)

        sidebar = QWidget()
        sidebar.setStyleSheet(f"background: {card_bg};")
        side_lay = QVBoxLayout(sidebar)
        side_lay.setContentsMargins(14, 16, 14, 16)
        side_lay.setSpacing(10)

        # -- Agent identity block --
        agent_hdr = QLabel()
        agent_hdr.setTextFormat(Qt.TextFormat.RichText)
        agent_hdr.setWordWrap(True)
        agent_hdr.setStyleSheet("background: transparent;")
        agent_hdr.setText(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='color:{color}; font-size:18pt; font-weight:900; "
            f"letter-spacing:1.5px; "
            f"font-family:Segoe UI,system-ui,sans-serif;'>"
            f"{icon} {escape(speaker).upper()}</span></div>"
            f"<span style='color:#3a5065; font-size:9pt; "
            f"font-weight:600;'>Response #{msg_num}</span>"
        )
        side_lay.addWidget(agent_hdr)

        # -- Separator --
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {color}25; border: none;")
        side_lay.addWidget(sep)

        # -- Talking-point badge --
        if talking_point_html:
            tp_section = QLabel()
            tp_section.setTextFormat(Qt.TextFormat.RichText)
            tp_section.setWordWrap(True)
            tp_section.setStyleSheet("background: transparent; padding: 0;")
            tp_section.setText(
                f"<div style='margin-bottom:2px;'>"
                f"<span style='color:#546e7a; font-size:7.5pt; "
                f"font-weight:700; letter-spacing:1.5px;'>TALKING POINT</span></div>"
                f"{talking_point_html}"
            )
            side_lay.addWidget(tp_section)

        # -- Quality scores --
        if quality:
            q_composite = round(
                (quality.get("relevance", 0.0) + quality.get("novelty", 0.0)
                 + quality.get("evidence", 0.0)) / 3, 2,
            )
            if q_composite >= 0.65:
                badge_color, dot = "#69f0ae", "\u25b2"
            elif q_composite >= 0.45:
                badge_color, dot = "#ffd740", "\u25cf"
            else:
                badge_color, dot = "#ef5350", "\u25bc"

            q_lbl = QLabel()
            q_lbl.setTextFormat(Qt.TextFormat.RichText)
            q_lbl.setWordWrap(True)
            q_lbl.setStyleSheet("background: transparent;")
            q_lbl.setText(
                f"<div style='margin-bottom:2px;'>"
                f"<span style='color:#546e7a; font-size:7.5pt; "
                f"font-weight:700; letter-spacing:1.5px;'>QUALITY</span></div>"
                f"<span style='color:{badge_color}; font-size:14pt; "
                f"font-weight:800;'>{dot} {q_composite:.2f}</span>"
                f"<div style='margin-top:4px; color:#607d8b; font-size:8pt;'>"
                f"Relevance: {quality.get('relevance', 0):.2f}<br>"
                f"Novelty: {quality.get('novelty', 0):.2f}<br>"
                f"Evidence: {quality.get('evidence', 0):.2f}"
                f"</div>"
            )
            side_lay.addWidget(q_lbl)

        # -- Evidence score --
        if evidence_score is not None and evidence_score > 0:
            ev_lbl = QLabel()
            ev_lbl.setTextFormat(Qt.TextFormat.RichText)
            ev_lbl.setWordWrap(True)
            ev_lbl.setStyleSheet("background: transparent;")
            ev_lbl.setText(
                f"<div style='margin-bottom:2px;'>"
                f"<span style='color:#546e7a; font-size:7.5pt; "
                f"font-weight:700; letter-spacing:1.5px;'>EVIDENCE</span></div>"
                f"<span style='color:#80cbc4; font-size:11pt; "
                f"font-weight:700;'>{evidence_score:.3f}</span>"
            )
            side_lay.addWidget(ev_lbl)

        # -- Model name --
        if model_name:
            mod_lbl = QLabel()
            mod_lbl.setTextFormat(Qt.TextFormat.RichText)
            mod_lbl.setWordWrap(True)
            mod_lbl.setStyleSheet("background: transparent;")
            mod_lbl.setText(
                f"<div style='margin-bottom:2px;'>"
                f"<span style='color:#546e7a; font-size:7.5pt; "
                f"font-weight:700; letter-spacing:1.5px;'>MODEL</span></div>"
                f"<span style='color:#455a64; font-size:8.5pt; "
                f"font-style:italic;'>{escape(model_name)}</span>"
            )
            side_lay.addWidget(mod_lbl)

        # -- Reframe card --
        if reframe:
            agent_low = speaker.lower()
            if agent_low == "astra":
                rf_border, rf_label = "#80cbc4", "#80cbc4"
            elif agent_low == "nova":
                rf_border, rf_label = "#ffab91", "#ffab91"
            else:
                rf_border, rf_label = "#b39ddb", "#b39ddb"

            rf_lbl = QLabel()
            rf_lbl.setTextFormat(Qt.TextFormat.RichText)
            rf_lbl.setWordWrap(True)
            rf_lbl.setStyleSheet(
                f"background: #10101e; border-left: 3px solid {rf_border}; "
                f"padding: 8px 10px; margin: 0;"
            )
            rf_lbl.setText(
                f"<span style='color:{rf_label}; font-size:7.5pt; "
                f"font-weight:700; letter-spacing:1px;'>"
                f"\u2726 REFRAME</span><br>"
                f"<span style='color:#c5cae9; font-size:9pt; "
                f"font-style:italic; font-family:Georgia,serif; "
                f"line-height:1.7;'>{_md_to_html(reframe)}</span>"
            )
            side_lay.addWidget(rf_lbl)

        # -- Citations / Sources --
        if citations:
            src_lbl = QLabel()
            src_lbl.setTextFormat(Qt.TextFormat.RichText)
            src_lbl.setWordWrap(True)
            src_lbl.setOpenExternalLinks(False)
            src_lbl.setStyleSheet("background: transparent;")
            src_lines = [
                "<div style='margin-bottom:2px;'>"
                "<span style='color:#546e7a; font-size:7.5pt; "
                "font-weight:700; letter-spacing:1.5px;'>SOURCES</span></div>"
            ]
            for i, cit in enumerate(citations, 1):
                sp = str(cit.get("source_path", "")).replace("\\", "/")
                href = f"source:///{quote(sp, safe='/:._-')}"
                title = escape(str(cit.get("title", "Unknown")))
                disp = escape(str(cit.get("source", "")))
                sc = float(cit.get("score", 0.0))
                src_lines.append(
                    f"<div style='margin:2px 0;'>"
                    f"<span style='color:#546e7a; font-size:8pt;'>"
                    f"[{i}] {title}</span><br>"
                    f"<span style='color:#455a64; font-size:7.5pt;'>"
                    f"Score: {sc:.3f}</span> "
                    f"<a href='{href}' style='color:#64b5f6; font-size:7.5pt;'>"
                    f"{disp}</a>"
                    f"</div>"
                )
            src_lbl.setText("".join(src_lines))
            side_lay.addWidget(src_lbl)

        # Push everything to the top
        side_lay.addStretch()
        sidebar_scroll.setWidget(sidebar)

        page_layout.addWidget(sidebar_scroll)

        # =============================================================
        #  RIGHT BODY  (~80%) -- the only readable / TTS text
        # =============================================================
        body_container = QWidget()
        body_container.setStyleSheet(f"background: {card_bg};")
        body_lay = QVBoxLayout(body_container)
        body_lay.setContentsMargins(20, 14, 20, 14)
        body_lay.setSpacing(0)

        self.body = QTextBrowser()
        self.body.setObjectName("turnBody")
        self.body.setOpenExternalLinks(False)
        self.body.document().setDefaultStyleSheet(_DOC_STYLESHEET)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setStyleSheet(
            "QTextBrowser#turnBody { "
            "  background: transparent; "
            "  border: none; "
            "  padding: 0; "
            "  color: #e0e8f8; "
            "  font-size: 11pt; "
            "  font-family: 'Segoe UI', system-ui, sans-serif; "
            "}"
        )
        self.body.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.body.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        body_html = _md_to_html(text)
        self.body.setHtml(
            f"<div style='color:#e0e8f8; line-height:1.9; "
            f"letter-spacing:0.15px;'>{body_html}</div>"
        )
        self.body.anchorClicked.connect(self.anchor_clicked.emit)
        body_lay.addWidget(self.body)

        page_layout.addWidget(body_container)

        # Set sidebar vs body proportions via stretch
        page_layout.setStretch(0, 1)   # sidebar  (~20%)
        page_layout.setStretch(1, 4)   # body     (~80%)

        # -- TTS state --
        self._word_cursor: QTextCursor | None = None
        self._word_cursor_fmt: QTextCharFormat | None = None
        self._clean_to_doc_map: list[int] | None = None

    # -- TTS word highlight -----------------------------------------------

    def highlight_word(self, char_offset: int, word_len: int) -> None:
        """Highlight a single word in this page's body from TTS char offset."""
        if word_len <= 0:
            return
        cleaned = self.cleaned_text
        if not cleaned:
            return

        clean_start = max(0, min(char_offset, len(cleaned) - 1))
        clean_end = max(
            clean_start + 1, min(char_offset + word_len, len(cleaned))
        )

        # Trim whitespace at edges
        while clean_start < clean_end and cleaned[clean_start].isspace():
            clean_start += 1
        while clean_end > clean_start and cleaned[clean_end - 1].isspace():
            clean_end -= 1
        if clean_end <= clean_start:
            return

        mapping = self._get_clean_to_doc_map()
        if mapping is None:
            return

        if clean_start >= len(mapping):
            clean_start = len(mapping) - 1
        if clean_end - 1 >= len(mapping):
            clean_end = len(mapping)

        doc_start = mapping[clean_start]
        doc_end = mapping[clean_end - 1] + 1
        if doc_start < 0 or doc_end <= doc_start:
            return

        # Clamp to document length
        doc_len = self.body.document().characterCount()
        doc_start = min(doc_start, doc_len - 1)
        doc_end = min(doc_end, doc_len)
        if doc_end <= doc_start:
            return

        self.clear_word_highlight()

        cursor = QTextCursor(self.body.document())
        cursor.setPosition(doc_start)
        cursor.setPosition(doc_end, QTextCursor.MoveMode.KeepAnchor)

        self._word_cursor_fmt = cursor.charFormat()
        fmt = QTextCharFormat()
        fmt.setBackground(_TTS_HIGHLIGHT_BG)
        fmt.setForeground(_TTS_HIGHLIGHT_FG)
        cursor.setCharFormat(fmt)
        self._word_cursor = cursor

        # Scroll the body browser so the highlighted word stays visible
        visible_cursor = QTextCursor(self.body.document())
        visible_cursor.setPosition(doc_start)
        self.body.setTextCursor(visible_cursor)
        self.body.ensureCursorVisible()

    def clear_word_highlight(self) -> None:
        if self._word_cursor is not None and not self._word_cursor.isNull():
            if self._word_cursor_fmt is not None:
                self._word_cursor.setCharFormat(self._word_cursor_fmt)
            else:
                self._word_cursor.setCharFormat(QTextCharFormat())
        self._word_cursor = None
        self._word_cursor_fmt = None

    def _get_clean_to_doc_map(self) -> list[int] | None:
        if self._clean_to_doc_map is not None:
            return self._clean_to_doc_map

        cleaned = self.cleaned_text
        if not cleaned:
            return None

        doc_text = self.body.document().toPlainText()
        if not doc_text:
            return None

        mapping: list[int] = [-1] * len(cleaned)
        seg_len = len(doc_text)
        scan_pos = 0

        for ci, ch in enumerate(cleaned):
            if scan_pos >= seg_len:
                break
            if ch.isspace():
                while scan_pos < seg_len and doc_text[scan_pos].isspace():
                    scan_pos += 1
                mapping[ci] = min(scan_pos, seg_len - 1)
                continue
            target = ch.lower()
            probe = scan_pos
            limit = min(seg_len, scan_pos + 220)
            while probe < limit:
                if doc_text[probe].lower() == target:
                    mapping[ci] = probe
                    scan_pos = probe + 1
                    break
                probe += 1

        # Fill gaps -- forward then backward
        last = -1
        for i, v in enumerate(mapping):
            if v >= 0:
                last = v
            else:
                mapping[i] = last
        nxt = -1
        for i in range(len(mapping) - 1, -1, -1):
            if mapping[i] >= 0:
                nxt = mapping[i]
            else:
                mapping[i] = nxt

        if all(v < 0 for v in mapping):
            return None

        self._clean_to_doc_map = mapping
        return mapping


# =====================================================================
#  CenterDebatePanel -- page-based turn container
# =====================================================================

class CenterDebatePanel(QWidget):
    follow_mode_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("centerDebatePanel")

        # -- Header row --
        self._header_widget = QWidget()
        self._header_widget.setObjectName("debateHeader")
        header_row = QHBoxLayout(self._header_widget)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self._header = QLabel("\u25b8 LIVE DEBATE STREAM")
        self._header.setStyleSheet(
            "font-size: 14pt; font-weight: 900; color: #c0e0ff; "
            "padding: 8px 0; letter-spacing: 3px;"
        )
        header_row.addWidget(self._header)
        header_row.addStretch()

        self._turn_indicator = QLabel("")
        self._turn_indicator.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._turn_indicator.setStyleSheet(
            "font-size: 10pt; color: #78909c; padding: 4px 8px;"
        )
        header_row.addWidget(self._turn_indicator)

        # -- Stacked widget -- holds one TurnPageWidget per turn
        self._stack = QStackedWidget()
        self._stack.setObjectName("publicView")
        self._stack.setStyleSheet(
            "QStackedWidget#publicView { "
            "  background-color: #060c18; "
            "  border: 2px solid #0e2240; "
            "  border-radius: 12px; "
            "}"
        )

        # -- Navigation bar --
        self._nav_widget = QWidget()
        nav_row = QHBoxLayout(self._nav_widget)
        nav_row.setContentsMargins(4, 4, 4, 4)
        nav_row.setSpacing(8)

        _nav_btn_style = (
            "QPushButton { background: #0e1a2f; color: #80cbc4; "
            "border: 1px solid #1a3a55; border-radius: 6px; "
            "padding: 4px 14px; font-size: 10pt; font-weight: 700; }"
            "QPushButton:hover { background: #162a45; color: #fff; }"
            "QPushButton:disabled { color: #2a3a50; border-color: #12203a; }"
        )

        self._prev_btn = QPushButton("\u25c0  Prev")
        self._prev_btn.setStyleSheet(_nav_btn_style)
        self._prev_btn.clicked.connect(self._go_prev)
        nav_row.addWidget(self._prev_btn)

        self._page_label = QLabel("0 / 0")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setStyleSheet(
            "color: #607d8b; font-size: 9.5pt; font-weight: 600; "
            "letter-spacing: 1px;"
        )
        self._page_label.setTextFormat(Qt.TextFormat.RichText)
        nav_row.addWidget(self._page_label, stretch=1)

        self._next_btn = QPushButton("Next  \u25b6")
        self._next_btn.setStyleSheet(_nav_btn_style)
        self._next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self._next_btn)

        # -- Main layout --
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._header_widget)
        layout.addWidget(self._stack, stretch=1)
        layout.addWidget(self._nav_widget)

        # Backward-compat: some code may reference public_view
        self.public_view = self._stack

        # -- State --
        self._follow_mode: bool = True
        self._pages: list[TurnPageWidget] = []
        self._messages: list[tuple[str, str]] = []
        self._cleaned_texts: list[str] = []
        self._last_talking_point: str = ""
        self._msg_count: int = 0
        self._highlighted_idx: int = -1
        self._source_click_handler = None

        self._update_nav()

    # -- public API -------------------------------------------------------

    def set_source_click_handler(self, handler) -> None:
        self._source_click_handler = handler

    @property
    def is_follow_mode(self) -> bool:
        return self._follow_mode

    def set_follow_mode(self, enabled: bool) -> None:
        if self._follow_mode != enabled:
            self._follow_mode = enabled
            self.follow_mode_changed.emit(enabled)

    def get_messages(self) -> list[tuple[str, str]]:
        return list(self._messages)

    def clear_messages(self) -> None:
        self._messages.clear()
        self._cleaned_texts.clear()
        while self._stack.count():
            w = self._stack.widget(0)
            self._stack.removeWidget(w)
            w.deleteLater()
        self._pages.clear()
        self._highlighted_idx = -1
        self._msg_count = 0
        self._last_talking_point = ""
        self._turn_indicator.setText("")
        self._update_nav()

    def update_turn_indicator(
        self, speaker: str, turn: int, total_turns: int = 0
    ) -> None:
        color = AGENT_COLORS.get(speaker, "#e0e0e0")
        turn_text = (
            f"Turn {turn}/{total_turns}" if total_turns > 0 else f"Turn {turn}"
        )
        self._turn_indicator.setText(
            f"<span style='color:{color}; font-weight:700;'>{speaker}</span>"
            f"<span style='color:#546e7a;'>  \u00b7  </span>"
            f"<span style='color:#90a4ae; font-weight:600;'>{turn_text}</span>"
        )

    def append_message(
        self,
        speaker: str,
        text: str,
        citations: list[dict] | None = None,
        evidence_score: float | None = None,
        talking_point: str | None = None,
        quality: dict | None = None,
        model_name: str | None = None,
        reframe: str | None = None,
    ) -> None:
        self._messages.append((speaker, text))
        self._cleaned_texts.append(TTSPlaybackWorker.clean_text(text))
        self._msg_count += 1

        # Talking-point HTML (suppress repeats)
        tp_html = ""
        if talking_point and talking_point != self._last_talking_point:
            self._last_talking_point = talking_point
            tp_html = (
                f"<span style='color:#6a9e98; font-size:9pt; "
                f"font-style:italic;'>"
                f"\u25b8 {escape(talking_point)}</span>"
            )

        page = TurnPageWidget(
            speaker,
            text,
            self._msg_count,
            citations=citations,
            evidence_score=evidence_score,
            talking_point_html=tp_html,
            quality=quality,
            model_name=model_name,
            reframe=reframe,
        )
        if self._source_click_handler:
            page.anchor_clicked.connect(self._on_page_anchor)

        self._stack.addWidget(page)
        self._pages.append(page)

        # In follow mode, auto-advance to the newest page
        if self._follow_mode:
            self._stack.setCurrentIndex(len(self._pages) - 1)

        self._update_nav()

    # -- TTS highlight ----------------------------------------------------

    def highlight_message(self, index: int) -> None:
        """TTS started a new message -- flip to that page."""
        if 0 <= self._highlighted_idx < len(self._pages):
            self._pages[self._highlighted_idx].clear_word_highlight()
        self._highlighted_idx = index
        if 0 <= index < len(self._pages):
            self._stack.setCurrentIndex(index)
            self._update_nav()

    def highlight_word(
        self, msg_idx: int, char_offset: int, word_len: int
    ) -> None:
        """Move the orange word highlight inside the active page's body."""
        if not (0 <= msg_idx < len(self._pages)):
            return
        # Clear previous page's highlight if message changed
        if msg_idx != self._highlighted_idx:
            if 0 <= self._highlighted_idx < len(self._pages):
                self._pages[self._highlighted_idx].clear_word_highlight()
            self._highlighted_idx = msg_idx
            self._stack.setCurrentIndex(msg_idx)
            self._update_nav()

        page = self._pages[msg_idx]
        page.highlight_word(char_offset, word_len)

    def clear_highlight(self) -> None:
        if 0 <= self._highlighted_idx < len(self._pages):
            self._pages[self._highlighted_idx].clear_word_highlight()
        self._highlighted_idx = -1

    # -- navigation -------------------------------------------------------

    def _go_prev(self) -> None:
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._update_nav()

    def _go_next(self) -> None:
        idx = self._stack.currentIndex()
        if idx < self._stack.count() - 1:
            self._stack.setCurrentIndex(idx + 1)
            self._update_nav()

    def _update_nav(self) -> None:
        total = self._stack.count()
        cur = self._stack.currentIndex() + 1 if total > 0 else 0
        self._page_label.setText(f"{cur} / {total}")
        self._prev_btn.setEnabled(cur > 1)
        self._next_btn.setEnabled(cur < total)

        # Show speaker color indicator in the nav row
        if total > 0 and 0 <= self._stack.currentIndex() < len(self._pages):
            page = self._pages[self._stack.currentIndex()]
            color = AGENT_COLORS.get(page.speaker, "#e0e0e0")
            self._page_label.setText(
                f"<span style='color:{color}; font-weight:700;'>"
                f"{escape(page.speaker)}</span>"
                f"<span style='color:#546e7a;'>  "
                f"{cur} / {total}</span>"
            )

    # -- internal ---------------------------------------------------------

    def _on_page_anchor(self, url: QUrl) -> None:
        if url.scheme() == "source":
            source_path = url.path().lstrip("/")
            if self._source_click_handler:
                self._source_click_handler(source_path)
