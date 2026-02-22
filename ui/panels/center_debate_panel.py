"""center_debate_panel.py — Live debate stream, one page per turn.

TTS Word-Highlight Design
─────────────────────────
Problem the old code had:
  • engine.say(clean_text) — SAPI5 reads the *cleaned* string
  • word_at fires (msg_idx, location, length) where location is a char
    offset into that *cleaned* string
  • The body QTextBrowser was rendering markdown → HTML, a completely
    different string.  The old _get_clean_to_doc_map() tried to align
    them but drifted and broke within seconds.

Fix:
  • body.setPlainText(cleaned_text)   ← same string SAPI5 is reading
  • char_offset == document position  ← 1:1, zero drift, guaranteed
  • No mapping algorithm at all.

Architecture
────────────
CenterDebatePanel (QWidget)
  └─ QStackedWidget
       └─ TurnPageWidget (QFrame)  ← one per agent turn
            ├─ left sidebar: talking point / quality / model / citations
            └─ right body:   LockedTextBrowser (plain-text, frozen viewport)
                              ▼ more bar (arrow indicator when overflowing)
"""

from __future__ import annotations

from html import escape
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# ── TTS highlight colours ──────────────────────────────────────────────────
_TTS_BG = QColor("#e65100")
_TTS_FG = QColor("#ffffff")

# ── Visual theme ───────────────────────────────────────────────────────────
_AGENT_COLOR: dict[str, str] = {}   # populated at runtime via append_message
_PANEL_BG = "#070d18"

# ═══════════════════════════════════════════════════════════════════════════
#  LockedTextBrowser
#  Viewport is frozen during TTS.  The ONLY movement allowed is our own
#  TTS-triggered page advance when the highlighted word has scrolled off-screen.
# ═══════════════════════════════════════════════════════════════════════════

class LockedTextBrowser(QTextBrowser):
    """Plain-text QTextBrowser with a frozen viewport and TTS page-advance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lock_scroll   = True      # viewport frozen by default
        self._paging        = False     # True only during our own page jump
        self._last_cur: Optional[QTextCursor] = None
        self._last_fmt: Optional[QTextCharFormat] = None
        # called with (value, maximum) when the scrollbar position changes
        self.on_scroll_changed: Optional[Callable[[int, int], None]] = None

    # ── Suppress Qt-internal auto-scroll ──────────────────────────────────
    def scrollContentsBy(self, dx: int, dy: int) -> None:
        if self._lock_scroll and not self._paging:
            return
        super().scrollContentsBy(dx, dy)
        self._notify_scroll()

    def ensureCursorVisible(self) -> None:
        if self._lock_scroll:
            return
        super().ensureCursorVisible()

    def _notify_scroll(self) -> None:
        if self.on_scroll_changed:
            vb = self.verticalScrollBar()
            self.on_scroll_changed(vb.value(), vb.maximum())

    # ── TTS-triggered page advance ─────────────────────────────────────────
    def _try_page_down(self, doc_pos: int) -> None:
        """Advance viewport by one page if doc_pos is below the visible area."""
        cursor = QTextCursor(self.document())
        cursor.setPosition(max(0, min(doc_pos, self.document().characterCount() - 1)))
        rect = self.cursorRect(cursor)           # coords relative to viewport
        vh   = self.viewport().height()
        if rect.top() < vh:
            return                               # still on screen — nothing to do
        vb   = self.verticalScrollBar()
        step = max(vh - 40, 60)                  # one page minus a small overlap
        self._paging = True
        vb.setValue(vb.value() + step)
        self._paging = False
        self._notify_scroll()

    # ── Word highlight ─────────────────────────────────────────────────────
    def highlight_word(self, char_offset: int, word_len: int) -> None:
        """Highlight characters [char_offset, char_offset+word_len) in the doc.

        Because body.setPlainText(cleaned_text) was used and the TTS engine
        also reads cleaned_text, char_offset == document position.  No mapping.
        """
        if word_len <= 0:
            return
        doc     = self.document()
        doc_len = doc.characterCount()
        start   = max(0, char_offset)
        end     = min(char_offset + word_len, doc_len - 1)
        if end <= start:
            return

        # Page-advance before applying highlight so the word is visible
        self._try_page_down(start)

        self._clear_last_highlight()

        cur = QTextCursor(doc)
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.MoveMode.KeepAnchor)

        self._last_fmt = cur.charFormat()
        self._last_cur = cur

        fmt = QTextCharFormat()
        fmt.setBackground(_TTS_BG)
        fmt.setForeground(_TTS_FG)
        cur.setCharFormat(fmt)

    def clear_highlight(self) -> None:
        self._clear_last_highlight()

    def _clear_last_highlight(self) -> None:
        if self._last_cur and not self._last_cur.isNull():
            self._last_cur.setCharFormat(
                self._last_fmt if self._last_fmt is not None else QTextCharFormat()
            )
        self._last_cur = None
        self._last_fmt = None


# ═══════════════════════════════════════════════════════════════════════════
#  TurnPageWidget
#  One full-panel page per agent turn.
# ═══════════════════════════════════════════════════════════════════════════

class TurnPageWidget(QFrame):
    """Displays a single agent turn: sidebar metadata + locked body text."""

    source_clicked = pyqtSignal(str, str)   # (doc_id, text)

    def __init__(
        self,
        msg_idx:       int,
        speaker:       str,
        turn_num:      int,
        total_turns:   int,
        talking_point: str,
        body_text:     str,
        quality:       dict,
        citations:     list,
        model_name:    str,
        agent_color:   str,
        bg_color:      str,
        parent:        QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.msg_idx   = msg_idx
        self.speaker   = speaker
        self._color    = agent_color
        self._bg       = bg_color

        # ── Import TTSPlaybackWorker.clean_text at runtime to avoid circular ──
        try:
            from tts.speech_engine import TTSPlaybackWorker
            self._cleaned_text: str = TTSPlaybackWorker.clean_text(body_text)
        except Exception:
            self._cleaned_text = body_text

        self._setup_ui(speaker, turn_num, total_turns, talking_point,
                       quality, citations, model_name)

    # ── Build layout ───────────────────────────────────────────────────────
    def _setup_ui(
        self,
        speaker:       str,
        turn_num:      int,
        total_turns:   int,
        talking_point: str,
        quality:       dict,
        citations:     list,
        model_name:    str,
    ) -> None:
        color = self._color
        bg    = self._bg

        self.setStyleSheet(f"QFrame {{ background: {bg}; border: none; }}")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ──────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(46)
        header.setStyleSheet(
            f"background: #050b16; border-bottom: 2px solid {color}44;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 0, 18, 0)

        icon     = "\u25c6" if "astra" in speaker.lower() else "\u25cf"
        spk_lbl  = QLabel(
            f"<span style='color:{color}; font-size:13pt; font-weight:900; "
            f"letter-spacing:2px;'>{icon} {escape(speaker).upper()}</span>"
        )
        spk_lbl.setTextFormat(Qt.TextFormat.RichText)
        hl.addWidget(spk_lbl)
        hl.addStretch()

        if total_turns > 0:
            turn_lbl = QLabel(
                f"<span style='color:#37506a; font-size:8.5pt;'>TURN &nbsp;</span>"
                f"<span style='color:{color}; font-size:15pt; "
                f"font-weight:900;'>{turn_num}</span>"
                f"<span style='color:#2a3a55; font-size:9pt;'> / {total_turns}</span>"
            )
            turn_lbl.setTextFormat(Qt.TextFormat.RichText)
            hl.addWidget(turn_lbl)

        root_layout.addWidget(header)

        # ── Sidebar + body ──────────────────────────────────────────────────
        content_row = QWidget()
        content_row.setStyleSheet(f"background: {bg};")
        cl = QHBoxLayout(content_row)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        root_layout.addWidget(content_row, stretch=1)

        # LEFT SIDEBAR
        sb_scroll = QScrollArea()
        sb_scroll.setWidgetResizable(True)
        sb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sb_scroll.setFixedWidth(210)
        sb_scroll.setStyleSheet(
            f"QScrollArea {{ background: {bg}; border: none; "
            f"border-right: 2px solid {color}18; }}"
            "QScrollBar:vertical { width: 4px; background: transparent; }"
            f"QScrollBar::handle:vertical {{ background: {color}30; "
            "border-radius: 2px; }}"
        )

        sidebar = QWidget()
        sidebar.setStyleSheet(f"background: {bg};")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(14, 16, 14, 16)
        sl.setSpacing(12)

        self._populate_sidebar(sl, color, talking_point, quality,
                               citations, model_name)
        sl.addStretch()
        sb_scroll.setWidget(sidebar)
        cl.addWidget(sb_scroll)

        # RIGHT BODY
        body_wrap = QWidget()
        body_wrap.setStyleSheet(f"background: {bg};")
        bwl = QVBoxLayout(body_wrap)
        bwl.setContentsMargins(16, 14, 16, 10)
        bwl.setSpacing(6)

        # Outlined document frame
        doc_frame = QFrame()
        doc_frame.setFrameShape(QFrame.Shape.StyledPanel)
        doc_frame.setStyleSheet(
            f"QFrame {{ background: #040c18; "
            f"border: 1px solid {color}40; border-radius: 6px; }}"
        )
        dfl = QVBoxLayout(doc_frame)
        dfl.setContentsMargins(0, 0, 0, 0)
        dfl.setSpacing(0)

        # LockedTextBrowser setup
        self._body = LockedTextBrowser(doc_frame)
        self._body.setReadOnly(True)
        self._body.setOpenLinks(False)
        self._body.setStyleSheet(
            "QTextBrowser {"
            "  background: transparent;"
            "  border: none;"
            "  color: #c8d8e8;"
            f" font-size: 10.5pt;"
            "  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;"
            "  padding: 14px 18px 14px 18px;"
            "  line-height: 175%;"
            "}"
            "QScrollBar:vertical { width: 7px; background: transparent; }"
            f"QScrollBar::handle:vertical {{ background: {color}35; "
            "border-radius: 3px; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "height:0; }"
        )
        self._body.setPlainText(self._cleaned_text)
        self._body.on_scroll_changed = self._on_scroll_changed
        dfl.addWidget(self._body, stretch=1)

        # ▼ more indicator bar (hidden when no overflow)
        self._arrow_bar = self._make_arrow_bar(color)
        dfl.addWidget(self._arrow_bar)
        self._arrow_bar.setVisible(False)

        bwl.addWidget(doc_frame, stretch=1)

        # Page counter label (e.g. "pg 1 / 2")
        self._page_lbl = QLabel("")
        self._page_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._page_lbl.setStyleSheet(
            f"color: {color}50; font-size: 7.5pt; "
            "background: transparent; border: none;"
        )
        bwl.addWidget(self._page_lbl)
        self._update_page_label()

        cl.addWidget(body_wrap, stretch=1)

        # Connect scroll for arrow visibility
        vb = self._body.verticalScrollBar()
        vb.rangeChanged.connect(lambda *_: self._on_scroll_changed(vb.value(), vb.maximum()))
        vb.valueChanged.connect(lambda v: self._on_scroll_changed(v, vb.maximum()))

    # ── Sidebar population ─────────────────────────────────────────────────
    def _populate_sidebar(
        self,
        sl:            QVBoxLayout,
        color:         str,
        talking_point: str,
        quality:       dict,
        citations:     list,
        model_name:    str,
    ) -> None:

        def _sec(txt: str) -> QLabel:
            lb = QLabel(
                f"<span style='color:#2e4a62; font-size:7pt; "
                f"font-weight:800; letter-spacing:2.5px;'>{txt}</span>"
            )
            lb.setTextFormat(Qt.TextFormat.RichText)
            lb.setStyleSheet("background: transparent;")
            return lb

        def _hsep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.HLine)
            f.setFixedHeight(1)
            f.setStyleSheet(f"background: {color}10; border: none;")
            return f

        bg = self._bg

        if talking_point:
            sl.addWidget(_sec("TALKING POINT"))
            tp = QLabel(
                f"<span style='color:#5d8a84; font-size:8.5pt; "
                f"font-style:italic; line-height:160%;'>"
                f"{escape(talking_point)}</span>"
            )
            tp.setTextFormat(Qt.TextFormat.RichText)
            tp.setWordWrap(True)
            tp.setStyleSheet(f"background: {bg};")
            sl.addWidget(tp)
            sl.addWidget(_hsep())

        if quality:
            rv = quality.get("relevance", quality.get("r", 0))
            nv = quality.get("novelty",   quality.get("n", 0))
            ev = quality.get("evidence",  quality.get("e", 0))
            qc = round((rv + nv + ev) / 3, 2)
            bc  = "#69f0ae" if qc >= 0.65 else ("#ffd740" if qc >= 0.45 else "#ef5350")
            dot = "\u25b2"  if qc >= 0.65 else ("\u25cf"  if qc >= 0.45 else "\u25bc")
            sl.addWidget(_sec("QUALITY"))
            ql = QLabel(
                f"<span style='color:{bc}; font-size:18pt; "
                f"font-weight:900;'>{dot} {qc:.2f}</span>"
                f"<div style='margin-top:5px; color:#3d5a6e; font-size:7.5pt; "
                f"line-height:175%;'>"
                f"Relevance &nbsp;<b style='color:#57788e'>{rv:.2f}</b><br>"
                f"Novelty &nbsp;&nbsp;&nbsp;<b style='color:#57788e'>{nv:.2f}</b><br>"
                f"Evidence &nbsp;<b style='color:#57788e'>{ev:.2f}</b>"
                f"</div>"
            )
            ql.setTextFormat(Qt.TextFormat.RichText)
            ql.setWordWrap(True)
            ql.setStyleSheet(f"background: {bg};")
            sl.addWidget(ql)
            sl.addWidget(_hsep())

        if model_name:
            sl.addWidget(_sec("MODEL"))
            ml = QLabel(
                f"<span style='color:#354f63; font-size:8pt; "
                f"font-style:italic;'>{escape(model_name)}</span>"
            )
            ml.setTextFormat(Qt.TextFormat.RichText)
            ml.setWordWrap(True)
            ml.setStyleSheet(f"background: {bg};")
            sl.addWidget(ml)
            sl.addWidget(_hsep())

        if citations:
            sl.addWidget(_sec("SOURCES"))
            for i, cit in enumerate(citations, 1):
                title  = cit.get("title",  cit.get("t", ""))
                source = cit.get("source", cit.get("s", ""))
                score  = cit.get("score",  cit.get("sc", 0.0))
                cl2 = QLabel(
                    f"<div style='margin:3px 0;'>"
                    f"<span style='color:#3d5a6e; font-size:7.5pt; "
                    f"font-weight:600; line-height:155%;'>"
                    f"[{i}] {escape(str(title))}</span><br>"
                    f"<span style='color:#2e4455; font-size:7pt;'>"
                    f"{escape(str(source))}  "
                    f"<b style='color:#4a6a7e'>{float(score):.2f}</b></span>"
                    f"</div>"
                )
                cl2.setTextFormat(Qt.TextFormat.RichText)
                cl2.setWordWrap(True)
                cl2.setStyleSheet(f"background: {bg};")
                sl.addWidget(cl2)

    # ── Arrow bar ──────────────────────────────────────────────────────────
    @staticmethod
    def _make_arrow_bar(color: str) -> QLabel:
        bar = QLabel(
            "<div style='text-align:center; padding:4px 0 2px 0;'>"
            f"<span style='color:{color}; font-size:11pt;'>&#9660;</span>"
            f"<span style='color:{color}70; font-size:7.5pt; "
            "margin-left:6px;'>more</span>"
            "</div>"
        )
        bar.setTextFormat(Qt.TextFormat.RichText)
        bar.setFixedHeight(26)
        bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.setStyleSheet(
            "border: none; border-top: 1px solid "
            + color + "20; background: transparent;"
        )
        return bar

    # ── Scroll callbacks ───────────────────────────────────────────────────
    def _on_scroll_changed(self, value: int, maximum: int) -> None:
        has_more = maximum > 0 and value < maximum
        self._arrow_bar.setVisible(has_more)
        self._update_page_label()

    def _update_page_label(self) -> None:
        vb     = self._body.verticalScrollBar()
        maxv   = vb.maximum()
        vh     = self._body.viewport().height()
        if maxv <= 0 or vh <= 0:
            self._page_lbl.setText("")
            return
        page_h   = max(vh, 1)
        total_pg = max(1, -(-int(vb.maximum() + vh) // page_h))   # ceil div
        cur_pg   = min(total_pg, (vb.value() // page_h) + 1)
        self._page_lbl.setText(f"pg {cur_pg} / {total_pg}")

    # ── Public highlight API ───────────────────────────────────────────────
    def highlight_word(self, char_offset: int, word_len: int) -> None:
        self._body.highlight_word(char_offset, word_len)

    def clear_highlight(self) -> None:
        self._body.clear_highlight()


# ═══════════════════════════════════════════════════════════════════════════
#  CenterDebatePanel — public-facing widget used by main_window.py
# ═══════════════════════════════════════════════════════════════════════════

_AGENT_BG_COLORS = [
    "#07101e",   # default blue-dark
    "#110905",   # warm dark (second agent)
    "#070f10",   # teal dark
    "#12070f",   # purple dark
]

_AGENT_ACCENT_COLORS = [
    "#00e5ff",
    "#ff6e40",
    "#69f0ae",
    "#ea80fc",
]


class CenterDebatePanel(QWidget):
    """Page-per-turn live debate stream.

    Public API (unchanged from previous version):
        append_message(...)      — add a turn page
        highlight_message(idx)   — flip to a turn page
        highlight_word(idx, offset, length) — TTS word highlight
        clear_highlight()
        clear_messages()
        get_messages()
        set_follow_mode(bool)    — no-op; always follows
        update_turn_indicator(current, total)
        set_source_click_handler(callback)
        follow_mode_changed      — signal (kept for backward compat)
        public_view              — alias for self
    """

    follow_mode_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.public_view = self

        self._pages:       list[TurnPageWidget] = []
        self._agent_map:   dict[str, int]       = {}   # name → color index
        self._total_turns: int                  = 0
        self._source_cb:   Optional[Callable]   = None
        self._view_locked: bool                  = False

        self._setup_ui()

    # ── Build chrome ───────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        self.setStyleSheet(f"background: {_PANEL_BG}; border: none;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Turn indicator strip
        self._turn_bar = QWidget()
        self._turn_bar.setFixedHeight(24)
        self._turn_bar.setStyleSheet(
            "background: #030810; border-bottom: 1px solid #0d1e30;"
        )
        til = QHBoxLayout(self._turn_bar)
        til.setContentsMargins(12, 0, 12, 0)
        self._turn_lbl = QLabel("")
        self._turn_lbl.setStyleSheet(
            "color: #1e3a52; font-size: 7.5pt; font-weight: 700; "
            "letter-spacing: 1.5px; background: transparent;"
        )
        til.addStretch()
        til.addWidget(self._turn_lbl)
        outer.addWidget(self._turn_bar)

        # Stacked pages
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {_PANEL_BG}; border: none;")
        outer.addWidget(self._stack, stretch=1)

        # Empty placeholder
        placeholder = QWidget()
        placeholder.setStyleSheet(f"background: {_PANEL_BG};")
        pl_lbl = QLabel(
            "<span style='color:#1a2d40; font-size:14pt; "
            "font-weight:300; letter-spacing:3px;'>DEBATE STREAM</span>"
        )
        pl_lbl.setTextFormat(Qt.TextFormat.RichText)
        pl_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pll = QVBoxLayout(placeholder)
        pll.addStretch()
        pll.addWidget(pl_lbl)
        pll.addStretch()
        self._stack.addWidget(placeholder)

    # ── Colour assignment ──────────────────────────────────────────────────
    def _color_for_agent(self, name: str) -> tuple[str, str]:
        """Return (accent_color, bg_color) for an agent name."""
        if name not in self._agent_map:
            idx = len(self._agent_map) % len(_AGENT_ACCENT_COLORS)
            self._agent_map[name] = idx
        idx = self._agent_map[name]
        return _AGENT_ACCENT_COLORS[idx], _AGENT_BG_COLORS[idx]

    # ── Public API ─────────────────────────────────────────────────────────
    def append_message(
        self,
        *args,
        talking_point: str  = "",
        quality:       dict | None = None,
        citations:     list | None = None,
        model_name:    str  = "",
        turn_num:      int  = 0,
        **_legacy_kwargs,
    ) -> int:
        """Create a new turn page and return its msg_idx.

        Supports both call orders:
            append_message(text, agent_name, ...)
            append_message(agent_name, text, ...)

        Extra legacy kwargs are accepted and ignored for compatibility.
        """
        if len(args) < 2:
            raise TypeError("append_message requires two positional args")

        first = str(args[0])
        second = str(args[1])
        if first.strip().lower() in {"astra", "nova"}:
            agent_name, text = first, second
        else:
            text, agent_name = first, second

        msg_idx = len(self._pages)
        accent, bg = self._color_for_agent(agent_name)
        page = TurnPageWidget(
            msg_idx       = msg_idx,
            speaker       = agent_name,
            turn_num      = turn_num or (msg_idx + 1),
            total_turns   = self._total_turns,
            talking_point = talking_point,
            body_text     = text,
            quality       = quality or {},
            citations     = citations or [],
            model_name    = model_name,
            agent_color   = accent,
            bg_color      = bg,
            parent        = None,
        )
        if self._source_cb:
            page.source_clicked.connect(self._source_cb)

        self._pages.append(page)
        self._stack.addWidget(page)
        if not self._view_locked:
            self._stack.setCurrentWidget(page)   # auto-advance to newest
        return msg_idx

    def highlight_message(self, msg_idx: int) -> None:
        """Flip the stack to the page for msg_idx (called by now_speaking)."""
        if self._view_locked:
            return
        if 0 <= msg_idx < len(self._pages):
            self._stack.setCurrentWidget(self._pages[msg_idx])

    def highlight_word(self, msg_idx: int, char_offset: int, word_len: int) -> None:
        """Highlight a word during TTS playback (called by word_at signal).

        char_offset is a byte offset into clean_text() — same string used by
        setPlainText(), so it maps directly to document position.
        """
        if 0 <= msg_idx < len(self._pages):
            self._pages[msg_idx].highlight_word(char_offset, word_len)

    def clear_highlight(self) -> None:
        """Clear any active TTS word highlight."""
        page = self._current_page()
        if page:
            page.clear_highlight()

    def clear_messages(self) -> None:
        """Remove all turn pages."""
        for page in self._pages:
            self._stack.removeWidget(page)
            page.deleteLater()
        self._pages.clear()
        self._agent_map.clear()
        self._stack.setCurrentIndex(0)   # back to placeholder

    def get_messages(self) -> list[tuple[str, str]]:
        """Return messages as (agent_name, text) tuples in debate order.

        This is the canonical format consumed by TTSPlaybackWorker and
        capture utilities.
        """
        return [(p.speaker, p._cleaned_text) for p in self._pages]

    def set_follow_mode(self, enabled: bool) -> None:
        """No-op — the panel always follows TTS (kept for API compatibility)."""
        # Emit so any connected slots still receive the signal
        self.follow_mode_changed.emit(True)

    def set_view_locked(self, locked: bool) -> None:
        """Lock/unlock page switching in the center panel.

        When locked, both auto-advance on new messages and explicit page flips
        from highlight_message() are ignored, preserving the current view.
        """
        self._view_locked = bool(locked)

    def update_turn_indicator(self, *args) -> None:
        """Update top-bar turn counter.

        Supports both legacy and current call patterns:
            update_turn_indicator(current, total)
            update_turn_indicator(agent_name, current, total)
        """
        current = 0
        total = 0
        if len(args) == 2:
            current, total = args
        elif len(args) == 3:
            _agent_name, current, total = args
        else:
            return

        try:
            current = int(current)
            total = int(total)
        except Exception:
            return

        self._total_turns = total
        if total > 0:
            self._turn_lbl.setText(f"TURN  {current} / {total}")
        else:
            self._turn_lbl.setText("")

    def set_source_click_handler(self, callback: Callable) -> None:
        """Register a callback(doc_id, text) for citation clicks."""
        self._source_cb = callback
        for page in self._pages:
            try:
                page.source_clicked.disconnect()
            except Exception:
                pass
            page.source_clicked.connect(callback)

    # ── Internal helpers ───────────────────────────────────────────────────
    def _current_page(self) -> Optional[TurnPageWidget]:
        w = self._stack.currentWidget()
        return w if isinstance(w, TurnPageWidget) else None
