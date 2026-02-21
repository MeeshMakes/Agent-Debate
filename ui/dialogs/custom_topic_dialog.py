"""Custom Topic Dialog with file ingestion support.

Provides:
  - Rich multi-paragraph topic name + description editor
  - Drag-and-drop file area
  - File browser button
  - Ingestion queue display
  - Confirm / Cancel

On confirmation, emits topic_configured(title, description, file_paths).
Actual ingestion runs after the session is created (on Start).
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _DropZone(QWidget):
    """Drag-drop target for files and folders."""

    files_dropped = pyqtSignal(list)   # list[str]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setStyleSheet(
            "background: #0d1520; border: 2px dashed #2a3a55; border-radius: 8px;"
            "color: #546e7a;"
        )
        layout = QVBoxLayout(self)
        lbl = QLabel("⬇  Drag files or folders here  ⬇")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #546e7a; font-size: 13px; border: none;")
        layout.addWidget(lbl)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                "background: #0d2030; border: 2px dashed #00acc1; border-radius: 8px;"
            )

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(
            "background: #0d1520; border: 2px dashed #2a3a55; border-radius: 8px;"
        )

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.setStyleSheet(
            "background: #0d1520; border: 2px dashed #2a3a55; border-radius: 8px;"
        )
        if paths:
            self.files_dropped.emit(paths)


class CustomTopicDialog(QDialog):
    """Full-featured topic configuration with optional file ingestion."""

    topic_configured = pyqtSignal(str, str, list)   # title, description, [file_paths]

    def __init__(self, initial_title: str = "", initial_description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚙  Configure Debate Topic")
        self.resize(780, 620)
        self.setModal(True)
        self._queued_files: list[Path] = []
        self._apply_style()
        self._build_ui(initial_title, initial_description)

    def _build_ui(self, title: str, desc: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # ---- Topic title
        title_box = QGroupBox("Topic Title")
        title_box.setObjectName("groupBox")
        tbl = QVBoxLayout(title_box)
        self._title_edit = QLineEdit(title)
        self._title_edit.setPlaceholderText("Enter a clear, specific debate topic…")
        self._title_edit.setStyleSheet(
            "QLineEdit { background: #0d1520; color: #e8eaf6; border: 1px solid #2a3a55;"
            " border-radius: 6px; padding: 7px 10px; font-size: 14px; }"
            "QLineEdit:focus { border-color: #00acc1; }"
        )
        tbl.addWidget(self._title_edit)
        root.addWidget(title_box)

        # ---- Description / prompt
        desc_box = QGroupBox("Topic Description & Debate Prompt  (multi-paragraph OK)")
        desc_box.setObjectName("groupBox")
        dbl = QVBoxLayout(desc_box)
        self._desc_edit = QTextEdit()
        self._desc_edit.setPlaceholderText(
            "Describe the debate in detail.\n\n"
            "You may include:\n"
            "  • Background context\n"
            "  • Specific questions to answer\n"
            "  • Constraints or focus areas\n"
            "  • Desired depth (historical, philosophical, technical…)"
        )
        self._desc_edit.setPlainText(desc)
        self._desc_edit.setStyleSheet(
            "QTextEdit { background: #0a1018; color: #cfd8dc; border: 1px solid #1e2d42;"
            " border-radius: 6px; padding: 8px; font-size: 13px; }"
            "QTextEdit:focus { border-color: #00acc1; }"
        )
        self._desc_edit.setMinimumHeight(160)
        dbl.addWidget(self._desc_edit)
        root.addWidget(desc_box, stretch=1)

        # ---- File ingestion
        ingest_box = QGroupBox("Knowledge Ingestion  (optional — add files/folders for this debate)")
        ingest_box.setObjectName("groupBox")
        ingest_box.setCheckable(True)
        ingest_box.setChecked(False)
        self._ingest_box = ingest_box
        ibl = QVBoxLayout(ingest_box)

        self._drop_zone = _DropZone()
        self._drop_zone.files_dropped.connect(self._add_files)
        ibl.addWidget(self._drop_zone)

        browse_btn = QPushButton("📁  Browse Files / Folders")
        browse_btn.setStyleSheet(
            "QPushButton { background: #1a2840; color: #90a4ae; border-radius: 6px; padding: 5px 12px; }"
            "QPushButton:hover { background: #1e3a5f; color: #fff; }"
        )
        browse_btn.clicked.connect(self._browse_files)
        ibl.addWidget(browse_btn)

        file_list_label = QLabel("Queued files:")
        file_list_label.setStyleSheet("color: #546e7a; font-size: 11px;")
        ibl.addWidget(file_list_label)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(100)
        self._file_list.setStyleSheet(
            "QListWidget { background: #080d14; color: #80cbc4; border: 1px solid #1e2d42;"
            " border-radius: 4px; font-size: 11px; }"
            "QListWidget::item { padding: 3px 6px; }"
        )
        ibl.addWidget(self._file_list)

        remove_btn = QPushButton("✕  Remove Selected")
        remove_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #546e7a; font-size: 11px; }"
            "QPushButton:hover { color: #ef5350; }"
        )
        remove_btn.clicked.connect(self._remove_selected)
        ibl.addWidget(remove_btn)

        root.addWidget(ingest_box)

        # ---- Buttons
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: #263238; color: #90a4ae; border-radius: 6px; padding: 7px 20px; }"
            "QPushButton:hover { background: #37474f; color: #fff; }"
        )
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("✓  Set Topic & Continue")
        ok_btn.setStyleSheet(
            "QPushButton { background: #00695c; color: #fff; font-weight: 700;"
            " border-radius: 6px; padding: 7px 20px; border: 1px solid #00897b; }"
            "QPushButton:hover { background: #00796b; }"
        )
        ok_btn.clicked.connect(self._on_confirm)

        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # File handling

    def _add_files(self, paths: list[str]) -> None:
        for p_str in paths:
            p = Path(p_str)
            if p not in self._queued_files:
                self._queued_files.append(p)
                icon = "📁" if p.is_dir() else "📄"
                self._file_list.addItem(f"{icon}  {p.name}  ({p})")

    def _browse_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files to ingest",
            str(Path.home()),
            "All supported (*.txt *.md *.py *.json *.yaml *.yml *.csv *.html *.pdf *.rst *.log *.*)",
        )
        if paths:
            self._add_files(paths)
        # Also let user pick a folder
        folder = QFileDialog.getExistingDirectory(self, "Or select a folder to ingest")
        if folder:
            self._add_files([folder])

    def _remove_selected(self) -> None:
        row = self._file_list.currentRow()
        if row >= 0:
            self._file_list.takeItem(row)
            if row < len(self._queued_files):
                self._queued_files.pop(row)

    # ------------------------------------------------------------------
    # Confirm

    def _on_confirm(self) -> None:
        title = self._title_edit.text().strip()
        if not title:
            self._title_edit.setStyleSheet(
                "QLineEdit { border: 1px solid #ef5350; background: #0d1520; color: #e8eaf6;"
                " border-radius: 6px; padding: 7px 10px; }"
            )
            return
        desc = self._desc_edit.toPlainText().strip()
        files = list(self._queued_files) if self._ingest_box.isChecked() else []
        self.topic_configured.emit(title, desc, [str(f) for f in files])
        self.accept()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog   { background: #080d14; }
            QGroupBox#groupBox { color: #4dd0e1; font-weight: 600; border: 1px solid #1e2d42;
                                  border-radius: 8px; margin-top: 8px; padding: 10px 10px 6px 10px; }
            QGroupBox#groupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)
