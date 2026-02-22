from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QWidget

from core.session_manager import get_session_manager
from ui.dialogs.analytics_dialog import AnalyticsDialog
from ui.dialogs.session_browser_dialog import SessionBrowserDialog
from ui.dialogs.topic_picker_dialog import TopicPickerDialog
from ui.main_window import MainWindow


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images" / "smoke"


def _save_widget(widget: QWidget, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    pix = widget.grab()
    pix.save(str(path), "PNG")
    print(f"saved: {path.relative_to(ROOT)}")


def _process(app: QApplication) -> None:
    app.processEvents()
    app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)

    theme_path = ROOT / "ui" / "themes" / "dark_theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))

    main_win = MainWindow(project_root=ROOT)
    main_win.resize(1680, 1000)
    main_win.show()
    main_win.raise_()
    main_win.activateWindow()
    _process(app)

    _save_widget(main_win, "01-main-window-overview.png")
    _save_widget(main_win.center_panel, "02-center-live-debate-panel.png")
    _save_widget(main_win.left_panel, "03-left-astra-inner-monologue-panel.png")
    _save_widget(main_win.right_panel, "04-right-nova-inner-monologue-panel.png")
    _save_widget(main_win.arbiter_panel, "05-arbiter-panel.png")
    _save_widget(main_win.graph_panel, "06-debate-graph-panel.png")
    _save_widget(main_win.scoring_panel, "07-scoring-verdict-panel.png")

    topic_dialog = TopicPickerDialog(current_title=main_win._current_topic_title, parent=main_win)
    topic_dialog.resize(1620, 860)
    topic_dialog.show()
    topic_dialog.raise_()
    topic_dialog.activateWindow()
    _process(app)
    _save_widget(topic_dialog, "08-debate-studio-topic-picker.png")
    topic_dialog.close()
    _process(app)

    session_dialog = SessionBrowserDialog(get_session_manager(), parent=main_win)
    session_dialog.resize(1100, 680)
    session_dialog.show()
    session_dialog.raise_()
    session_dialog.activateWindow()
    _process(app)
    _save_widget(session_dialog, "09-session-browser-dialog.png")
    session_dialog.close()
    _process(app)

    analytics_dialog = AnalyticsDialog(parent=main_win)
    analytics_dialog.resize(1140, 720)
    analytics_dialog.show()
    analytics_dialog.raise_()
    analytics_dialog.activateWindow()
    _process(app)
    _save_widget(analytics_dialog, "10-analytics-dialog.png")
    analytics_dialog.close()
    _process(app)

    main_win.close()

    QTimer.singleShot(0, app.quit)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
