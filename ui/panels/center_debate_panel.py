from __future__ import annotations

from html import escape
import re
from urllib.parse import quote

try:
    import markdown as _md_lib
    _MD_AVAILABLE = True
except ImportError:
    _MD_AVAILABLE = False


_DOC_STYLESHEET = (
    "p { margin: 2px 0; } "
    "code { font-family: Consolas, 'Courier New', monospace; "
    "background-color: #0d1b2a; color: #80cbc4; font-size: 9pt; } "
    "pre { background-color: #0a1520; color: #a8e6cf; "
    "font-family: Consolas, 'Courier New', monospace; "
    "font-size: 9pt; padding: 8px; margin: 4px 0; "
    "border-left: 3px solid #00bcd4; } "
    "blockquote { color: #90a4ae; margin-left: 8px; padding-left: 8px; } "
    "strong { color: #e0e6ff; } "
    "em { color: #b0bec5; } "
    "a { color: #64b5f6; } "
    "li { color: #e8eef9; margin: 1px 0; } "
    "h1, h2, h3 { color: #a0cfff; margin: 4px 0 2px 0; } "
)


def _md_to_html(text: str) -> str:
    """Render markdown text to HTML for QTextBrowser display."""
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

from PyQt6.QtCore import QObject, QEvent, QUrl, pyqtSignal, Qt
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QTextBrowser, QVBoxLayout, QWidget, QLabel
from tts.speech_engine import TTSPlaybackWorker


# Agent color map
AGENT_COLORS = {
    "Astra": "#00e5ff",       # cyan
    "Nova": "#ff6e40",        # deep orange
    "Arbiter": "#ffd740",     # amber
    "Resolution": "#69f0ae",  # green
}

_TTS_HIGHLIGHT_BG = QColor("#e65100")   # deep orange
_TTS_HIGHLIGHT_FG = QColor("#ffffff")
_TTS_PREV_BG      = QColor("#1a1a1a")   # near-invisible: clears previous word


class _WheelBreakFilter(QObject):
    """Event filter that detects mouse-wheel scroll and notifies the panel."""
    scrolled = pyqtSignal()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel:
            self.scrolled.emit()
        return False   # never consume — let Qt handle the actual scroll


class CenterDebatePanel(QWidget):
    # Emitted whenever follow-mode changes so the main window can update its button
    follow_mode_changed = pyqtSignal(bool)  # True = following, False = broken
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("centerDebatePanel")

        # ── Header row: title on left, turn/speaker on right ──
        self._header_widget = QWidget()
        self._header_widget.setObjectName("debateHeader")
        header_row = QHBoxLayout(self._header_widget)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        self._header = QLabel("LIVE DEBATE STREAM")
        self._header.setStyleSheet(
            "font-size: 13pt; font-weight: 700; color: #a0cfff; "
            "padding: 6px 0; letter-spacing: 2px;"
        )
        header_row.addWidget(self._header)
        header_row.addStretch()

        self._turn_indicator = QLabel("")
        self._turn_indicator.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._turn_indicator.setStyleSheet(
            "font-size: 10pt; color: #78909c; padding: 4px 8px;"
        )
        header_row.addWidget(self._turn_indicator)

        self.public_view = QTextBrowser()
        self.public_view.setObjectName("publicView")
        self.public_view.setOpenExternalLinks(False)
        self.public_view.setPlaceholderText("Select a topic and press Start to begin the debate...")
        self.public_view.document().setDefaultStyleSheet(_DOC_STYLESHEET)
        self._source_click_handler = None
        self.public_view.anchorClicked.connect(self._on_anchor_clicked)

        # Follow-mode: when True, the view scrolls to the TTS word being spoken.
        # Wheel scroll breaks it; the main window's Follow Text button restores it.
        self._follow_mode: bool = True
        self._wheel_filter = _WheelBreakFilter(self)
        self._wheel_filter.scrolled.connect(self._on_user_scroll)
        self.public_view.installEventFilter(self._wheel_filter)

        # Last talking point shown — used to suppress repetition in the feed
        self._last_talking_point: str = ""

        # Store messages for TTS playback
        self._messages: list[tuple[str, str]] = []
        # Character-range (start, end) for each message in the document
        self._message_positions: list[tuple[int, int]] = []
        # Cleaned text (TTS version) for each message — used for word search
        self._cleaned_texts: list[str] = []
        # Lazy per-message map from cleaned-text char index -> document-local char index
        self._clean_to_doc_maps: list[list[int] | None] = []
        # Per-message search cursor: where to start the next find() from
        self._word_search_pos: int = 0
        self._word_search_msg: int = -1
        self._last_tts_char_offset: int = -1
        # The QTextCursor covering the currently orange word
        self._word_cursor: QTextCursor | None = None
        self._word_cursor_fmt: QTextCharFormat | None = None  # saved pre-highlight format
        self._highlighted_idx: int = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._header_widget)
        layout.addWidget(self.public_view)

    def set_source_click_handler(self, handler) -> None:
        self._source_click_handler = handler

    # ---------------------------------------------------------------- follow-mode

    @property
    def is_follow_mode(self) -> bool:
        return self._follow_mode

    def set_follow_mode(self, enabled: bool) -> None:
        if self._follow_mode != enabled:
            self._follow_mode = enabled
            self.follow_mode_changed.emit(enabled)

    def _on_user_scroll(self) -> None:
        """Mouse-wheel detected.

        Follow mode is controlled explicitly by the Follow Text toggle button;
        wheel movement should not disable it.
        """
        return

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() != "source":
            return
        source_path = url.path().lstrip("/")
        if self._source_click_handler:
            self._source_click_handler(source_path)

    def get_messages(self) -> list[tuple[str, str]]:
        """Return all (agent, text) pairs for TTS playback."""
        return list(self._messages)

    def clear_messages(self) -> None:
        self._messages.clear()
        self._message_positions.clear()
        self._cleaned_texts.clear()
        self._clean_to_doc_maps.clear()
        self._highlighted_idx = -1
        self._word_cursor = None
        self._word_cursor_fmt = None
        self._last_tts_char_offset = -1
        self._turn_indicator.setText("")

    def update_turn_indicator(self, speaker: str, turn: int, total_turns: int = 0) -> None:
        """Update the top-right turn/speaker badge.

        Parameters
        ----------
        speaker : str
            Current speaker name (e.g. "Astra" or "Nova").
        turn : int
            1-based turn number.
        total_turns : int
            Total turns configured (0 = endless).
        """
        color = AGENT_COLORS.get(speaker, "#e0e0e0")
        if total_turns > 0:
            turn_text = f"Turn {turn}/{total_turns}"
        else:
            turn_text = f"Turn {turn}"
        self._turn_indicator.setText(
            f"<span style='color:{color}; font-weight:700;'>{speaker}</span>"
            f"<span style='color:#546e7a;'>  ·  </span>"
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
        color = AGENT_COLORS.get(speaker, "#e0e0e0")
        self._messages.append((speaker, text))
        self._cleaned_texts.append(TTSPlaybackWorker.clean_text(text))
        self._clean_to_doc_maps.append(None)

        # --- Build full HTML for the message block ---

        # Talking point badge — only render when the talking point changes
        tp_html = ""
        if talking_point and talking_point != self._last_talking_point:
            self._last_talking_point = talking_point
            tp_html = (
                f"<div style='margin-bottom:3px;'>"
                f"<span style='color:rgba(96,160,152,0.8); font-size:9pt; font-style:italic;'>"
                f"▸ {escape(talking_point)}</span></div>"
            )

        # Agent name badge with optional quality score indicator
        score_badge_html = ""
        if quality:
            q_composite = round(
                (quality.get("relevance", 0.0) + quality.get("novelty", 0.0) + quality.get("evidence", 0.0)) / 3,
                2,
            )
            if q_composite >= 0.65:
                badge_bg = "#1b5e20"
                badge_fg = "#a5d6a7"
                dot = "🟢"
            elif q_composite >= 0.45:
                badge_bg = "#e65100"
                badge_fg = "#ffcc80"
                dot = "🟡"
            else:
                badge_bg = "#7f0000"
                badge_fg = "#ef9a9a"
                dot = "🔴"
            score_badge_html = (
                f"<span style='background:{badge_bg}; color:{badge_fg}; "
                f"font-size:8pt; padding:1px 7px; border-radius:3px; margin-left:6px;'>"
                f"{dot} {q_composite:.2f}</span>"
            )
        model_tag_html = ""
        if model_name:
            model_tag_html = (
                f"<span style='color:#546e7a; font-size:8pt; "
                f"font-style:italic; margin-left:6px; "
                f"vertical-align:middle;'>{escape(model_name)}</span>"
            )
        name_html = (
            f"<span style='"
            f"background-color:{color}; color:#0b0f16; "
            f"font-weight:800; padding:3px 18px; border-radius:20px; "
            f"font-size:13pt; letter-spacing:0.3px; "
            f"font-family:'Segoe UI',system-ui,sans-serif;'>{escape(speaker)}</span>"
            f"{model_tag_html}"
            f"{score_badge_html}"
        )

        # Evidence score
        ev_html = ""
        if evidence_score is not None and evidence_score > 0:
            ev_html = (
                f"<div style='margin-top:2px;'>"
                f"<span style='color:#78909c; font-size:8pt; margin-left:12px;'>"
                f"Evidence score: {evidence_score:.3f}</span></div>"
            )

        # Citations
        cit_html = ""
        if citations:
            lines = ["<div style='margin-top:2px;'>"
                     "<span style='color:#78909c; font-size:8pt; margin-left:12px;'>Sources:</span>"]
            for i, citation in enumerate(citations, start=1):
                source_path = str(citation.get("source_path", "")).replace("\\", "/")
                href = f"source:///{quote(source_path, safe='/:._-')}"
                title = escape(str(citation.get("title", "Unknown")))
                display_source = escape(str(citation.get("source", "")))
                score = float(citation.get("score", 0.0))
                lines.append(
                    f"<br><span style='color:#78909c; font-size:8pt; margin-left:16px;'>"
                    f"[{i}] {title} | score={score:.3f} | "
                    f"<a href='{href}' style='color:#64b5f6;'>{display_source}</a></span>"
                )
            lines.append("</div>")
            cit_html = "".join(lines)

        # Reframe card — creative second pass, shown beneath the main response
        reframe_html = ""
        if reframe:
            # Choose a soft accent that complements the agent colour without clashing
            agent_low = speaker.lower()
            if agent_low == "astra":
                reframe_border = "#80cbc4"   # soft teal
                reframe_label_color = "#80cbc4"
            elif agent_low == "nova":
                reframe_border = "#ffab91"   # soft coral
                reframe_label_color = "#ffab91"
            else:
                reframe_border = "#b39ddb"   # lavender default
                reframe_label_color = "#b39ddb"
            reframe_html = (
                f"<div style='"
                f"margin:8px 0 0 10px;"
                f"border-left:3px solid {reframe_border};"
                f"border-radius:0 8px 8px 0;"
                f"background:#12102a;"
                f"padding:6px 12px 8px 14px;"
                f"'>"
                f"<span style='"
                f"color:{reframe_label_color}; font-size:8pt; font-weight:700;"
                f"letter-spacing:0.4px; font-family:'Segoe UI',system-ui,sans-serif;"
                f"'>&#10022; Reframe</span>"
                f"<div style='"
                f"color:#c5cae9; font-size:9.5pt; font-style:italic;"
                f"font-family:'Georgia','Palatino Linotype',serif;"
                f"line-height:1.65; margin-top:4px;"
                f"'>{_md_to_html(reframe)}</div>"
                f"</div>"
            )

        full_html = (
            f"<div style='margin:8px 0 2px 0;'>"
            f"{tp_html}"
            f"{name_html}"
            f"<div style='color:#e8eef9; margin:6px 0 0 10px; "
            f"line-height:1.6; font-size:10pt; "
            f"font-family:\'Segoe UI\',system-ui,sans-serif;'>{_md_to_html(text)}</div>"
            f"{reframe_html}"
            f"{ev_html}"
            f"{cit_html}"
            f"</div>"
            f"<hr style='border:none; border-top:1px solid #2a3a55; margin:10px 0;'/>"
        )

        # --- Record document positions before and after this single append ---
        doc = self.public_view.document()
        start_pos = doc.characterCount()

        # Preserve scroll position so append() doesn't bounce the viewport.
        # QTextBrowser.append() always scrolls to the end; we undo that here.
        vbar = self.public_view.verticalScrollBar()
        saved_scroll = vbar.value()
        self.public_view.append(full_html)
        # Restore position — prevents the jarring bounce when new messages arrive
        vbar.setValue(saved_scroll)

        end_pos = doc.characterCount()
        self._message_positions.append((start_pos, end_pos))

    # ---------------------------------------------------------------- TTS highlight

    def highlight_message(self, index: int) -> None:
        """Called when TTS starts a new message — resets word search, scrolls if following."""
        # Clear the previous orange word
        self._clear_word_cursor()
        self.set_follow_mode(True)
        self._highlighted_idx = index
        self._word_search_msg = index
        self._last_tts_char_offset = -1
        if 0 <= index < len(self._message_positions):
            self._word_search_pos = self._message_positions[index][0]
            if self._follow_mode:
                self._scroll_to_message(index)

    def _find_word_in_message(self, msg_idx: int, word: str, start_pos: int) -> QTextCursor | None:
        if not (0 <= msg_idx < len(self._message_positions)):
            return None

        msg_start, msg_end = self._message_positions[msg_idx]
        if msg_end <= msg_start:
            return None

        from PyQt6.QtGui import QTextDocument
        doc = self.public_view.document()
        flags = (
            QTextDocument.FindFlag.FindCaseSensitively
            | QTextDocument.FindFlag.FindWholeWords
        )

        pos = max(msg_start, min(start_pos, msg_end - 1))
        found = doc.find(word, pos, flags)
        while not found.isNull():
            s = found.selectionStart()
            e = found.selectionEnd()
            if s >= msg_end:
                break
            if s >= msg_start and e <= msg_end:
                return found
            pos = max(pos + 1, e)
            if pos >= msg_end:
                break
            found = doc.find(word, pos, flags)

        return None

    def highlight_word(self, msg_idx: int, char_offset: int, word_len: int) -> None:
        """Called on each SAPI5 word-boundary event — moves the orange cursor."""
        if word_len <= 0:
            return

        # If the message changed, reset search position
        if msg_idx != self._word_search_msg:
            self._word_search_msg = msg_idx
            if 0 <= msg_idx < len(self._message_positions):
                self._word_search_pos = self._message_positions[msg_idx][0]
                self._last_tts_char_offset = -1
            else:
                return

        if 0 <= msg_idx < len(self._message_positions):
            msg_start, _ = self._message_positions[msg_idx]
        else:
            return

        # SAPI offsets should be monotonic within one utterance; if not, reset safely.
        if char_offset < self._last_tts_char_offset:
            self._word_search_pos = msg_start
        self._last_tts_char_offset = char_offset

        # Get the active cleaned utterance (the exact text TTS is reading)
        if msg_idx >= len(self._cleaned_texts):
            return
        cleaned = self._cleaned_texts[msg_idx]
        if not cleaned:
            return

        clean_start = max(0, min(char_offset, len(cleaned) - 1))
        clean_end = max(clean_start + 1, min(char_offset + word_len, len(cleaned)))

        # Trim spaces from both ends of the active span
        while clean_start < clean_end and cleaned[clean_start].isspace():
            clean_start += 1
        while clean_end > clean_start and cleaned[clean_end - 1].isspace():
            clean_end -= 1
        if clean_end <= clean_start:
            return

        map_local = self._get_clean_to_doc_map(msg_idx)
        if map_local is None or not map_local:
            return

        if clean_start >= len(map_local):
            clean_start = len(map_local) - 1
        if clean_end - 1 >= len(map_local):
            clean_end = len(map_local)

        local_start = map_local[clean_start]
        local_end = map_local[clean_end - 1] + 1
        if local_start < 0 or local_end <= local_start:
            return

        msg_start, msg_end = self._message_positions[msg_idx]
        doc_start = msg_start + local_start
        doc_end = msg_start + local_end
        if doc_start < msg_start or doc_start >= msg_end:
            return
        doc_end = max(doc_start + 1, min(doc_end, msg_end))

        # Clear the previous word highlight first
        self._clear_word_cursor()

        doc = self.public_view.document()
        found = QTextCursor(doc)
        found.setPosition(doc_start)
        found.setPosition(doc_end, QTextCursor.MoveMode.KeepAnchor)

        # Advance search position for next word
        self._word_search_pos = found.selectionEnd()

        # Save original format, then apply orange highlight
        self._word_cursor_fmt = found.charFormat()
        fmt = QTextCharFormat()
        fmt.setBackground(_TTS_HIGHLIGHT_BG)
        fmt.setForeground(_TTS_HIGHLIGHT_FG)
        found.setCharFormat(fmt)
        self._word_cursor = found

        # Scroll to keep the word visible — only when follow mode is on
        if self._follow_mode:
            self._stable_scroll_to_cursor(found)

    def _get_clean_to_doc_map(self, msg_idx: int) -> list[int] | None:
        if not (0 <= msg_idx < len(self._cleaned_texts)):
            return None
        cached = self._clean_to_doc_maps[msg_idx] if msg_idx < len(self._clean_to_doc_maps) else None
        if cached is not None:
            return cached

        if not (0 <= msg_idx < len(self._message_positions)):
            return None
        msg_start, msg_end = self._message_positions[msg_idx]
        if msg_end <= msg_start:
            return None

        cleaned = self._cleaned_texts[msg_idx]
        if not cleaned:
            return None

        doc_text = self.public_view.document().toPlainText()
        segment = doc_text[msg_start:msg_end]
        if not segment:
            return None

        # Start alignment near the likely message body to avoid matching speaker/title noise.
        start_hint = 0
        for tok in re.findall(r"\w{4,}", cleaned)[:3]:
            pos = segment.lower().find(tok.lower())
            if pos >= 0:
                start_hint = pos
                break

        mapping: list[int] = [-1] * len(cleaned)
        seg_len = len(segment)
        scan_pos = min(max(0, start_hint), max(0, seg_len - 1))

        for clean_i, ch in enumerate(cleaned):
            if scan_pos >= seg_len:
                break

            if ch.isspace():
                while scan_pos < seg_len and segment[scan_pos].isspace():
                    scan_pos += 1
                mapping[clean_i] = min(scan_pos, seg_len - 1)
                continue

            target = ch.lower()
            found = -1
            hard_limit = min(seg_len, scan_pos + 220)
            probe = scan_pos
            while probe < hard_limit:
                if segment[probe].lower() == target:
                    found = probe
                    break
                probe += 1

            if found < 0:
                continue

            mapping[clean_i] = found
            scan_pos = found + 1

        # Fill gaps to preserve monotonic mapping
        last = -1
        for i, pos in enumerate(mapping):
            if pos >= 0:
                last = pos
            else:
                mapping[i] = last

        nxt = -1
        for i in range(len(mapping) - 1, -1, -1):
            pos = mapping[i]
            if pos >= 0:
                nxt = pos
            else:
                mapping[i] = nxt

        if all(pos < 0 for pos in mapping):
            return None

        if msg_idx < len(self._clean_to_doc_maps):
            self._clean_to_doc_maps[msg_idx] = mapping
        return mapping

    def clear_highlight(self) -> None:
        self._clear_word_cursor()
        self._highlighted_idx = -1
        self._word_search_msg = -1
        self._word_search_pos = 0

    def _clear_word_cursor(self) -> None:
        """Remove the orange highlight from the previously highlighted word."""
        if self._word_cursor is not None and not self._word_cursor.isNull():
            if self._word_cursor_fmt is not None:
                self._word_cursor.setCharFormat(self._word_cursor_fmt)
            else:
                self._word_cursor.setCharFormat(QTextCharFormat())
        self._word_cursor = None
        self._word_cursor_fmt = None

    def _apply_highlight(self, index: int, *, clear: bool) -> None:
        """(Unused externally but kept for scroll-only use.)"""
        pass

    # ---------------------------------------------------------------- stable scroll

    def _stable_scroll_to_cursor(self, cursor: QTextCursor) -> None:
        """Scroll only when needed, placing the cursor line at a stable 30%
        position from the top of the viewport. Never bounces."""
        view = self.public_view
        vbar = view.verticalScrollBar()

        # Temporarily set the cursor so cursorRect() reflects its position,
        # but save/restore the scrollbar so setTextCursor doesn't jolt.
        saved = vbar.value()
        view.setTextCursor(cursor)
        vbar.setValue(saved)

        rect = view.cursorRect(cursor)
        viewport_h = view.viewport().height()
        cursor_y = rect.top()

        # Comfortable band: 20%-80% of the viewport height
        band_top = int(viewport_h * 0.20)
        band_bot = int(viewport_h * 0.80)

        if band_top <= cursor_y <= band_bot:
            # Word is comfortably visible — don't scroll at all
            return

        # Scroll so the word sits at ~30% from the top
        target_y = int(viewport_h * 0.30)
        delta = cursor_y - target_y
        new_val = max(0, min(vbar.maximum(), saved + delta))
        vbar.setValue(new_val)

    def _scroll_to_message(self, index: int) -> None:
        if index >= len(self._message_positions):
            return
        start, _ = self._message_positions[index]
        doc = self.public_view.document()
        cursor = QTextCursor(doc)
        cursor.setPosition(min(start, doc.characterCount() - 1))
        self._stable_scroll_to_cursor(cursor)