"""Smoke tests for the GUI (constructed under offscreen QPA platform).

These tests don't simulate user interaction; they verify that the main
window and its panels can be constructed and refreshed without exceptions,
and that the world canvas paints without crashing.
"""

from __future__ import annotations

import os
import sys

import pytest

# The offscreen platform is forced in conftest.py before Qt is imported.

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    """One QApplication per module (Qt insists)."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_main_window_constructs(qapp):
    from ecosystem.gui.main_window import MainWindow
    win = MainWindow()
    assert win.windowTitle().startswith("Project Ecosystem")
    assert win.sim is not None
    assert win.sim.population() > 0
    win.deleteLater()
    qapp.processEvents()


def test_main_window_panels_have_data(qapp):
    from ecosystem.gui.main_window import MainWindow
    win = MainWindow()
    # Tick a few times so charts have data.
    for _ in range(30):
        win.sim.tick()
    win.stats_panel.refresh()
    win.species_panel.refresh()
    win.env_panel.refresh()
    win.journal_panel.refresh()
    # Stats summary should now show real numbers.
    assert win.stats_panel.lbl_pop[1].text() != "—"
    assert win.stats_panel.lbl_tick[1].text() != "—"
    # Species table should have rows.
    assert win.species_panel.table.rowCount() > 0
    win.deleteLater()
    qapp.processEvents()


def test_world_canvas_paints_without_error(qapp):
    from ecosystem.gui.main_window import MainWindow
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    qapp.processEvents()
    # Force a paint event by grabbing the canvas.
    pix = win.canvas.grab()
    assert not pix.isNull()
    win.deleteLater()
    qapp.processEvents()


def test_inspector_panel_shows_selected_organism(qapp):
    from ecosystem.gui.main_window import MainWindow
    win = MainWindow()
    orgs = win.sim.living_organisms()
    assert len(orgs) > 0
    win._select_organism(orgs[0].id)
    win.inspector.refresh()
    assert win.inspector.lbl_title.text() == f"Organism #{orgs[0].id}"
    # Stats should be populated.
    assert win.inspector.stat_labels["ID"].text() == str(orgs[0].id)
    assert win.inspector.stat_labels["Species"].text() != "—"
    win.deleteLater()
    qapp.processEvents()


def test_control_panel_signals(qapp):
    from ecosystem.gui.main_window import MainWindow
    from ecosystem.core.weather import WeatherType
    win = MainWindow()
    # Trigger pause via the control panel.
    win.controls.btn_play.setChecked(True)
    win.controls._on_play_clicked()
    assert win.sim.paused
    # Speed change.
    win.controls.speed_slider.setValue(300)
    assert win.sim.speed == pytest.approx(3.0)
    # Weather trigger.
    win.controls.weather_combo.setCurrentIndex(WeatherType.RAIN)
    win.controls._on_weather_triggered()
    assert win.sim.weather.active_event is not None
    win.deleteLater()
    qapp.processEvents()


def test_save_and_load_from_manager(qapp, tmp_path):
    from ecosystem.gui.main_window import MainWindow
    from ecosystem.persistence import SaveManager
    win = MainWindow()
    for _ in range(10):
        win.sim.tick()
    mgr = SaveManager(base_dir=tmp_path)
    mgr.save(win.sim, "GUI Test")
    slots = mgr.list_saves()
    assert len(slots) == 1
    assert slots[0].name == "GUI Test"
    win.deleteLater()
    qapp.processEvents()
