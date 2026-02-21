from __future__ import annotations

import re
from html import escape

from PyQt6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget


class RightAgentPanel(QWidget):
    """Nova private reasoning panel - Orange theme."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("rightAgentPanel")

        self._header = QLabel("NOVA - Private Reasoning")
        self._header.setObjectName("novaHeader")
        self._header.setStyleSheet(
            "font-size: 11pt; font-weight: 700; color: #ff6e40; "
            "padding: 4px 0; letter-spacing: 1px;"
        )

        self._memory_label = QLabel("Memory: 0 facts")
        self._memory_label.setObjectName("novaMemoryLabel")
        self._memory_label.setStyleSheet("color: #ff8a65; font-size: 8pt;")

        self.private_view = QTextBrowser()
        self.private_view.setObjectName("novaPrivateView")
        self.private_view.setReadOnly(True)
        self.private_view.setPlaceholderText("Nova inner monologue will appear here...")
        self.private_view.setStyleSheet(
            "QTextBrowser { border: 1px solid #bf360c; border-radius: 8px; "
            "background-color: #1a0f0a; color: #ffccbc; }"
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
            "border:1px solid #7a3120;"
            "border-left:3px solid #ff6e40;"
            "border-radius:12px;"
            "background:#281508;"
            "'>"
            # header strip
            "<div style='"
            "background:#3d2008;"
            "color:#ff8a65;"
            "font-size:7.5pt;"
            "font-weight:700;"
            "letter-spacing:0.5px;"
            "padding:3px 10px;"
            "border-radius:10px 10px 0 0;"
            "font-family:'Segoe UI',system-ui,sans-serif;"
            f"'>&#9670; {turn_label}</div>"
            # body
            "<div style='"
            "color:#ffccbc;"
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
