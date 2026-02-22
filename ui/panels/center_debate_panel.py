"""Centre debate stream — widget-per-message card architecture.

Each agent response is rendered as its own ``DebateMessageCard`` widget inside
a vertical ``QScrollArea``.  TTS word-highlighting operates on each card's
private ``QTextBrowser``, making the character mapping trivial and precise.

Follow-text is always active — scrolling the wheel does **not** break it.
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


# ── Document stylesheet applied inside each card's QTextBrowser ──────────
_DOC_STYLESHEET = (
    "body { margin: 0; padding: 0; } "
    "p { margin: 4px 0; line-height: 170%; } "
    "code { font-family: 'Cascadia Code', Consolas, 'Courier New', monospace; "
    "background-color: #0d1b2a; color: #80cbc4; font-size: 9.5pt; "
    "padding: 1px 5px; border-radius: 3px; } "
    "pre { background-color: #0a1520; color: #a8e6cf; "
    "font-family: 'Cascadia Code', Consolas, 'Courier New', monospace; "
    "font-size: 9.5pt; padding: 10px 14px; margin: 6px 0; "
    "border-left: 3px solid #00bcd4; line-height: 150%; } "
    "blockquote { color: #b0bec5; margin-left: 12px; padding-left: 12px; "
    "border-left: 2px solid #37474f; font-style: italic; } "
    "strong { color: #e8f0ff; } "
    "em { color: #b8c9d9; font-style: italic; } "
    "a { color: #64b5f6; text-decoration: none; } "
    "li { color: #e8eef9; margin: 2px 0; line-height: 160%; } "
    "ul, ol { margin: 4px 0 4px 16px; } "
    "h1 { color: #a0cfff; font-size: 13pt; margin: 8px 0 4px 0; } "
    "h2 { color: #a0cfff; font-size: 12pt; margin: 6px 0 4px 0; } "
    "h3 { color: #90b8e0; font-size: 11pt; margin: 4px 0 2px 0; } "
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
    QFrame, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QTextBrowser, QVBoxLayout, QWidget,
)
from tts.speech_engine import TTSPlaybackWorker


# ── Per-agent theming ─────────────────────────────────────────────────────
AGENT_COLORS = {
    "Astra": "#00e5ff",
    "Nova": "#ff6e40",
    "Arbiter": "#ffd740",
    "Resolution": "#69f0ae",
    "Repo Watchdog": "#b39ddb",
}

AGENT_ICONS = {
    "Astra": "\u25C6",        # ◆
    "Nova": "\u25CF",         # ●
    "Arbiter": "\u25B2",      # ▲
    "Resolution": "\u2726",   # ✦
    "Repo Watchdog": "\u29D6",  # ⧖
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


# ═══════════════════════════════════════════════════════════════════════════
#  DebateMessageCard — one per agent response
# ═══════════════════════════════════════════════════════════════════════════

class DebateMessageCard(QFrame):
    """Self-contained widget that renders one debate message."""

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
        icon = AGENT_ICONS.get(speaker, "\u25CF")
        card_bg = AGENT_CARD_BG.get(speaker, "#0c1422")

        self.speaker = speaker
        self.raw_text = text
        self.cleaned_text = TTSPlaybackWorker.clean_text(text)

        # ── Card frame styling ──
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"DebateMessageCard {{ "
            f"  background: {card_bg}; "
            f"  border-left: 4px solid {color}; "
            f"  border-radius: 0px; "
            f"  margin: 0px; "
            f"}}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        card_lay = QVBoxLayout(self)
        card_lay.setContentsMargins(16, 12, 16, 10)
        card_lay.setSpacing(4)

        # ── Talking-point badge (optional) ──
        if talking_point_html:
            tp_label = QLabel(talking_point_html)
            tp_label.setTextFormat(Qt.TextFormat.RichText)
            tp_label.setWordWrap(True)
            tp_label.setStyleSheet(
                "background: transparent; padding: 0; margin: 0 0 2px 0;"
            )
            card_lay.addWidget(tp_label)

        # ── Header row ──
        hdr = QLabel()
        hdr.setTextFormat(Qt.TextFormat.RichText)
        hdr.setStyleSheet("background: transparent; padding: 0;")

        # Quality badge
        score_badge = ""
        if quality:
            q_composite = round(
                (quality.get("relevance", 0.0) + quality.get("novelty", 0.0)
                 + quality.get("evidence", 0.0)) / 3, 2,
            )
            if q_composite >= 0.65:
                badge_color, dot = "#69f0ae", "\u25B2"
            elif q_composite >= 0.45:
                badge_color, dot = "#ffd740", "\u25CF"
            else:
                badge_color, dot = "#ef5350", "\u25BC"
            score_badge = (
                f"<span style='color:{badge_color}; font-size:8.5pt; "
                f"font-weight:700; margin-left:8px;'>"
                f"{dot} {q_composite:.2f}</span>"
            )

        model_tag = ""
        if model_name:
            model_tag = (
                f"<span style='color:#455a64; font-size:8pt; "
                f"font-style:italic; margin-left:8px;'>"
                f"{escape(model_name)}</span>"
            )

        hdr.setText(
            f"<span style='color:{color}; font-size:15pt; font-weight:900; "
            f"letter-spacing:1.2px; "
            f"font-family:Segoe UI,system-ui,sans-serif;'>"
            f"{icon} {escape(speaker).upper()}</span>"
            f"<span style='color:#2a3a55; font-size:9pt; font-weight:600; "
            f"margin-left:10px; letter-spacing:0.5px;'>#{msg_num}</span>"
            f"{model_tag}{score_badge}"
        )
        card_lay.addWidget(hdr)

        # ── Accent separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {color}40; border: none;")
        card_lay.addWidget(sep)

        # ── Body text browser ──
        self.body = QTextBrowser()
        self.body.setObjectName("cardBody")
        self.body.setOpenExternalLinks(False)
        self.body.document().setDefaultStyleSheet(_DOC_STYLESHEET)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setStyleSheet(
            "QTextBrowser#cardBody { "
            "  background: transparent; "
            "  border: none; "
            "  padding: 0; "
            "  color: #e0e8f8; "
            "  font-size: 10.5pt; "
            "  font-family: 'Segoe UI', system-ui, sans-serif; "
            "}"
        )
        self.body.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.body.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.body.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        body_html = _md_to_html(text)
        self.body.setHtml(
            f"<div style='color:#e0e8f8; line-height:1.8; "
            f"letter-spacing:0.15px;'>{body_html}</div>"
        )
        self.body.anchorClicked.connect(self.anchor_clicked.emit)
        # Auto-size the body to its content (no internal scrollbar)
        self.body.document().documentLayout().documentSizeChanged.connect(
            self._resize_body
        )
        card_lay.addWidget(self.body)
        QTimer.singleShot(0, self._resize_body)

        # ── Reframe card (optional) ──
        if reframe:
            agent_low = speaker.lower()
            if agent_low == "astra":
                rf_border, rf_label = "#80cbc4", "#80cbc4"
            elif agent_low == "nova":
                rf_border, rf_label = "#ffab91", "#ffab91"
            else:
                rf_border, rf_label = "#b39ddb", "#b39ddb"

            rf = QLabel()
            rf.setTextFormat(Qt.TextFormat.RichText)
            rf.setWordWrap(True)
            rf.setStyleSheet(
                f"background: #10101e; border-left: 3px solid {rf_border}; "
                f"padding: 8px 14px; margin: 6px 0 0 0;"
            )
            rf.setText(
                f"<span style='color:{rf_label}; font-size:8.5pt; "
                f"font-weight:700; letter-spacing:0.5px;'>"
                f"\u2726 Reframe</span><br>"
                f"<span style='color:#c5cae9; font-size:9.5pt; "
                f"font-style:italic; font-family:Georgia,serif; "
                f"line-height:1.7;'>{_md_to_html(reframe)}</span>"
            )
            card_lay.addWidget(rf)

        # ── Evidence + citations footer (optional) ──
        footer_parts: list[str] = []
        if evidence_score is not None and evidence_score > 0:
            footer_parts.append(
                f"<span style='color:#546e7a; font-size:8pt;'>"
                f"Evidence: {evidence_score:.3f}</span>"
            )
        if citations:
            src_lines = [
                "<span style='color:#546e7a; font-size:8pt; "
                "font-weight:600;'>Sources</span>"
            ]
            for i, cit in enumerate(citations, 1):
                sp = str(cit.get("source_path", "")).replace("\\", "/")
                href = f"source:///{quote(sp, safe='/:._-')}"
                title = escape(str(cit.get("title", "Unknown")))
                disp = escape(str(cit.get("source", "")))
                sc = float(cit.get("score", 0.0))
                src_lines.append(
                    f"<br><span style='color:#546e7a; font-size:8pt;'>"
                    f"[{i}] {title} ({sc:.3f}) "
                    f"<a href='{href}' style='color:#64b5f6;'>{disp}</a>"
                    f"</span>"
                )
            footer_parts.append("".join(src_lines))

        if footer_parts:
            ft = QLabel("<br>".join(footer_parts))
            ft.setTextFormat(Qt.TextFormat.RichText)
            ft.setWordWrap(True)
            ft.setOpenExternalLinks(False)
            ft.setStyleSheet(
                "background: transparent; padding: 2px 0 0 0;"
            )
            card_lay.addWidget(ft)

        # ── TTS state ──
        self._word_cursor: QTextCursor | None = None
        self._word_cursor_fmt: QTextCharFormat | None = None
        self._clean_to_doc_map: list[int] | None = None

    # ── auto-resize body to content ──────────────────────────────────────

    def _resize_body(self) -> None:
        doc_h = int(self.body.document().size().height()) + 4
        self.body.setFixedHeight(max(doc_h, 20))

    # ── TTS word highlight ───────────────────────────────────────────────

    def highlight_word(self, char_offset: int, word_len: int) -> None:
        """Highlight a single word in this card's body from TTS char offset."""
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

        # Fill gaps — forward then backward
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


# ═══════════════════════════════════════════════════════════════════════════
#  CenterDebatePanel — scroll-area container
# ═══════════════════════════════════════════════════════════════════════════

class CenterDebatePanel(QWidget):
    follow_mode_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("centerDebatePanel")

        # ── Header row ──
        self._header_widget = QWidget()
        self._header_widget.setObjectName("debateHeader")
        header_row = QHBoxLayout(self._header_widget)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self._header = QLabel("\u25B8 LIVE DEBATE STREAM")
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

        # ── Scroll area holding the message cards ──
        self._scroll = QScrollArea()
        self._scroll.setObjectName("publicView")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet(
            "QScrollArea#publicView { "
            "  background-color: #060c18; "
            "  border: 2px solid #0e2240; "
            "  border-radius: 12px; "
            "}"
        )

        self._card_container = QWidget()
        self._card_container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(6, 6, 6, 6)
        self._card_layout.setSpacing(12)
        self._card_layout.addStretch()  # bottom spacer
        self._scroll.setWidget(self._card_container)

        # ── Main layout ──
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._header_widget)
        layout.addWidget(self._scroll)

        # Backward-compat: some code may reference public_view
        self.public_view = self._scroll

        # ── State ──
        self._follow_mode: bool = True
        self._cards: list[DebateMessageCard] = []
        self._messages: list[tuple[str, str]] = []
        self._cleaned_texts: list[str] = []
        self._last_talking_point: str = ""
        self._msg_count: int = 0
        self._highlighted_idx: int = -1
        self._source_click_handler = None

    # ── public API ─────────────────────────────────────────────────────────

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
        for card in self._cards:
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._highlighted_idx = -1
        self._msg_count = 0
        self._last_talking_point = ""
        self._turn_indicator.setText("")

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

        card = DebateMessageCard(
            speaker,
            text,
            self._msg_count,
            citations=citations,
            evidence_score=evidence_score,
            talking_point_html=tp_html,
            quality=quality,
            model_name=model_name,
            reframe=reframe,
            parent=self._card_container,
        )
        if self._source_click_handler:
            card.anchor_clicked.connect(self._on_card_anchor)

        # Insert before the bottom stretch spacer
        idx = max(0, self._card_layout.count() - 1)
        self._card_layout.insertWidget(idx, card)
        self._cards.append(card)

        # Always scroll to show the new card
        if self._follow_mode:
            QTimer.singleShot(20, self._scroll_to_bottom)

    # ── TTS highlight ──────────────────────────────────────────────────────

    def highlight_message(self, index: int) -> None:
        """TTS started a new message — scroll that card into view."""
        if 0 <= self._highlighted_idx < len(self._cards):
            self._cards[self._highlighted_idx].clear_word_highlight()
        self._highlighted_idx = index
        if 0 <= index < len(self._cards):
            self._scroll_to_card(index)

    def highlight_word(
        self, msg_idx: int, char_offset: int, word_len: int
    ) -> None:
        """Move the orange word highlight inside the active card."""
        if not (0 <= msg_idx < len(self._cards)):
            return
        # Clear previous card's highlight if message changed
        if msg_idx != self._highlighted_idx:
            if 0 <= self._highlighted_idx < len(self._cards):
                self._cards[self._highlighted_idx].clear_word_highlight()
            self._highlighted_idx = msg_idx

        card = self._cards[msg_idx]
        card.highlight_word(char_offset, word_len)

        # Keep the card visible during TTS
        if self._follow_mode:
            self._scroll.ensureWidgetVisible(card, 50, 80)

    def clear_highlight(self) -> None:
        if 0 <= self._highlighted_idx < len(self._cards):
            self._cards[self._highlighted_idx].clear_word_highlight()
        self._highlighted_idx = -1

    # ── internal ───────────────────────────────────────────────────────────

    def _on_card_anchor(self, url: QUrl) -> None:
        if url.scheme() == "source":
            source_path = url.path().lstrip("/")
            if self._source_click_handler:
                self._source_click_handler(source_path)

    def _scroll_to_bottom(self) -> None:
        vbar = self._scroll.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def _scroll_to_card(self, index: int) -> None:
        if 0 <= index < len(self._cards):
            self._scroll.ensureWidgetVisible(self._cards[index], 50, 80)
