from __future__ import annotations

import re
from html import escape

from PyQt6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget


class LeftAgentPanel(QWidget):
    """Astra private reasoning panel - Cyan theme."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("leftAgentPanel")

        self._header = QLabel("ASTRA - Private Reasoning")
        self._header.setObjectName("astraHeader")
        self._header.setStyleSheet(
            "font-size: 11pt; font-weight: 700; color: #00e5ff; "
            "padding: 4px 0; letter-spacing: 1px;"
        )

        self._memory_label = QLabel("Memory: 0 facts")
        self._memory_label.setObjectName("astraMemoryLabel")
        self._memory_label.setStyleSheet("color: #4dd0e1; font-size: 8pt;")

        self.private_view = QTextBrowser()
        self.private_view.setObjectName("astraPrivateView")
        self.private_view.setReadOnly(True)
        self.private_view.setPlaceholderText("Astra inner monologue will appear here...")
        self.private_view.setStyleSheet(
            "QTextBrowser { border: 1px solid #006064; border-radius: 8px; "
            "background-color: #0a1a20; color: #b2ebf2; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._header)
        layout.addWidget(self._memory_label)
        layout.addWidget(self.private_view)

    def append_thought(self, text: str) -> None:
        m = re.match(r'^\[T(\d+)\]\s*(.*)', text, re.DOTALL)
        turn_label = f"T{m.group(1)}" if m else "▸"
        body = escape(m.group(2)) if m else escape(text)
        html = (
            "<div style='"
            "margin:6px 3px 3px 3px;"
            "border:1px solid #00838f;"
            "border-left:3px solid #00e5ff;"
            "border-radius:12px;"
            "background:#0c2238;"
            "'>"
            # header strip
            "<div style='"
            "background:#0f3254;"
            "color:#4dd0e1;"
            "font-size:7.5pt;"
            "font-weight:700;"
            "letter-spacing:0.5px;"
            "padding:3px 10px;"
            "border-radius:10px 10px 0 0;"
            "font-family:'Segoe UI',system-ui,sans-serif;"
            f"'>&#9670; {turn_label}</div>"
            # body
            "<div style='"
            "color:#b2ebf2;"
            "font-size:8.5pt;"
            "font-family:'Segoe UI',system-ui,sans-serif;"
            "padding:6px 10px 8px 12px;"
            "line-height:1.55;"
            f"'>{body}</div>"
            "</div>"
        )
        self.private_view.append(html)

    def update_memory_count(self, count: int) -> None:
        self._memory_label.setText(f"Memory: {count} facts")
