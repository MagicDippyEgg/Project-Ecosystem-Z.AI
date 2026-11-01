"""Headless screenshot helper.

Renders the main window offscreen, lets the simulation tick for a few
seconds, and saves PNGs of the window + each tab to ``screenshots/``.

Used during development to visually verify the GUI without a display.
Not part of the application itself.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force offscreen platform before importing Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer, Qt, QEventLoop
from PySide6.QtGui import QPixmap, QGuiApplication
from PySide6.QtWidgets import QApplication

# Make sure the project root is importable.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ecosystem.gui.main_window import MainWindow


def grab(window: QMainWindow, path: Path) -> None:
    """Grab the window's pixels into a PNG."""
    window.grab().save(str(path))
    print(f"saved {path} ({path.stat().st_size} bytes)")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    # Run at high speed for a bit so panels/charts have data.
    win.sim.set_speed(8.0)
    # Pump events for ~3 seconds of wall-clock; the sim ticks in between.
    end = time.monotonic() + 3.0
    while time.monotonic() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        time.sleep(0.02)
    out_dir = PROJECT_ROOT / "screenshots"
    out_dir.mkdir(exist_ok=True)
    grab(win, out_dir / "01_main_default.png")
    # Switch to each tab and capture.
    tabs = win.tabs
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        app.processEvents(QEventLoop.AllEvents, 100)
        time.sleep(0.1)
        name = tabs.tabText(i).lower().replace(" ", "_")
        grab(win, out_dir / f"02_tab_{name}.png")
    # Try clicking an organism - simulate selection.
    orgs = win.sim.living_organisms()
    if orgs:
        org = orgs[0]
        win._select_organism(org.id)
        tabs.setCurrentIndex(0)  # Inspector
        app.processEvents(QEventLoop.AllEvents, 100)
        time.sleep(0.1)
        grab(win, out_dir / "03_inspector_with_selection.png")
    # Capture at higher zoom.
    win.canvas.set_zoom(12.0)
    win.canvas.refresh_terrain()
    app.processEvents(QEventLoop.AllEvents, 100)
    time.sleep(0.1)
    grab(win, out_dir / "04_zoomed_world.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
