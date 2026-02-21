from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    app = QApplication(sys.argv)

    theme_path = project_root / "ui" / "themes" / "dark_theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))

    window = MainWindow(project_root=project_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
