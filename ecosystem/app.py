"""Application entry point.

Run with ``python -m ecosystem.app`` or via the ``project-ecosystem``
console script declared in ``setup.py``.
"""

from __future__ import annotations

import os
import sys
import signal


def main() -> int:
    # Make Ctrl-C kill the app cleanly (Qt otherwise swallows SIGINT).
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Some environments (headless CI, sandboxed dev boxes) lack a display
    # or EGL. Allow forcing a platform with QT_QPA_PLATFORM.
    if os.environ.get("ECOSYSTEM_FORCE_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = os.environ["ECOSYSTEM_FORCE_PLATFORM"]

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from .gui.main_window import MainWindow

    # High-DPI awareness (default on Qt6 but explicit doesn't hurt).
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Project Ecosystem")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ProjectEcosystem")
    # On some platforms the native menu bar needs this.
    app.setDesktopFileName("project-ecosystem")

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
