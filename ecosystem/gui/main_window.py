"""Main application window.

Composes the :class:`WorldCanvas`, :class:`ControlPanel`, and the various
info panels into a single native window, drives the tick timer, and wires
up menu actions (new world, save, load, quit, about).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QIcon, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QDialog, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox, QLabel, QFormLayout, QLineEdit,
    QApplication, QStatusBar,
)

from ..core.simulation import Simulation, SimulationConfig
from ..core.weather import WeatherType, WEATHER_LABELS
from ..persistence import SaveManager, save_simulation, load_simulation
from .theme import THEME
from .world_view import WorldCanvas
from .panels import (
    ControlPanel, StatsPanel, SpeciesPanel, InspectorPanel,
    JournalPanel, EnvironmentPanel,
)


__all__ = ["MainWindow"]


class SaveLoadDialog(QDialog):
    """Modal dialog for picking a saved world to load."""

    def __init__(self, save_manager: SaveManager, mode: str = "load", parent=None) -> None:
        super().__init__(parent)
        self.save_manager = save_manager
        self.mode = mode
        self.selected_name: Optional[str] = None
        self.setWindowTitle("Load World" if mode == "load" else "Manage Saves")
        self.setMinimumWidth(520)
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Saved worlds:"))
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)
        # Buttons
        btn_box = QDialogButtonBox()
        if mode == "load":
            btn_box.addButton("Load", QDialogButtonBox.AcceptRole)
            self.del_btn = btn_box.addButton("Delete", QDialogButtonBox.ActionRole)
            self.del_btn.clicked.connect(self._delete_selected)
        else:
            btn_box.addButton(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        btn_box.accepted.connect(self._accept)
        layout.addWidget(btn_box)
        self._populate()

    def _populate(self) -> None:
        self.list_widget.clear()
        slots = self.save_manager.list_saves()
        if not slots:
            item = QListWidgetItem("(no saved worlds yet)")
            item.setData(Qt.UserRole, None)
            self.list_widget.addItem(item)
            return
        for s in slots:
            text = (f"{s.name}   —   tick {s.tick}, pop {s.population}, "
                    f"'{s.world_name}'   ({s.size_bytes // 1024} KB)")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, s.name)
            self.list_widget.addItem(item)

    def _accept(self) -> None:
        item = self.list_widget.currentItem()
        if item is not None:
            self.selected_name = item.data(Qt.UserRole)
        self.accept()

    def _delete_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if name is None:
            return
        if QMessageBox.question(self, "Delete save", f"Delete '{name}'?") == QMessageBox.Yes:
            self.save_manager.delete(name)
            self._populate()


class MainWindow(QMainWindow):
    """The application's main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Project Ecosystem — Artificial Life Simulation")
        self.resize(1280, 800)
        self.setMinimumSize(960, 640)
        self.save_manager = SaveManager()
        self._last_frame_time: float = time.monotonic()

        # Build a default simulation right away so the window isn't blank.
        self.sim = Simulation(SimulationConfig())
        # Wire simulation callbacks.
        self.sim.on_stats = self._on_stats_updated

        # Central widget: splitter with canvas + side tab panel.
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Control panel across the top.
        self.controls = ControlPanel()
        self.controls.set_simulation(self.sim)
        outer.addWidget(self.controls)

        splitter = QSplitter(Qt.Horizontal)
        # Left: world canvas
        self.canvas = WorldCanvas()
        self.canvas.set_simulation(self.sim)
        splitter.addWidget(self.canvas)
        # Right: tabbed info panels
        self.tabs = QTabWidget()
        self.inspector = InspectorPanel()
        self.inspector.set_simulation(self.sim)
        self.species_panel = SpeciesPanel()
        self.species_panel.set_simulation(self.sim)
        self.stats_panel = StatsPanel()
        self.stats_panel.set_simulation(self.sim)
        self.env_panel = EnvironmentPanel()
        self.env_panel.set_simulation(self.sim)
        self.journal_panel = JournalPanel()
        self.journal_panel.set_simulation(self.sim)
        self.tabs.addTab(self.inspector, "Inspector")
        self.tabs.addTab(self.species_panel, "Species")
        self.tabs.addTab(self.stats_panel, "Stats")
        self.tabs.addTab(self.env_panel, "Environment")
        self.tabs.addTab(self.journal_panel, "Journal")
        self.tabs.setMinimumWidth(360)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        self.setCentralWidget(central)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_label = QLabel("Ready")
        self.status.addWidget(self.status_label, 1)

        self._build_menus()

        # Connect signals
        self.controls.pause_toggled.connect(self._on_pause_toggled)
        self.controls.speed_changed.connect(self._on_speed_changed)
        self.controls.regenerate_requested.connect(self._on_regenerate)
        self.controls.weather_triggered.connect(self._on_weather_triggered)
        self.controls.weather_cleared.connect(self._on_weather_cleared)
        self.canvas.organism_clicked.connect(self._on_organism_clicked)
        self.species_panel.species_selected.connect(self._on_species_selected)

        # Tick timer - runs at ~30Hz, advances the sim by wall-clock time.
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        # Apply dark styling.
        self._apply_style()

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        mb = self.menuBar()
        # File menu
        file_menu = mb.addMenu("&File")
        self.act_new = QAction("&New World…", self)
        self.act_new.setShortcut(QKeySequence("Ctrl+N"))
        self.act_new.triggered.connect(self._on_new_world_dialog)
        file_menu.addAction(self.act_new)

        self.act_save = QAction("&Save World…", self)
        self.act_save.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save.triggered.connect(self._on_save)
        file_menu.addAction(self.act_save)

        self.act_load = QAction("&Load World…", self)
        self.act_load.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load.triggered.connect(self._on_load)
        file_menu.addAction(self.act_load)

        file_menu.addSeparator()
        self.act_save_as = QAction("Save As File…", self)
        self.act_save_as.triggered.connect(self._on_save_as_file)
        file_menu.addAction(self.act_save_as)

        self.act_load_file = QAction("Load From File…", self)
        self.act_load_file.triggered.connect(self._on_load_from_file)
        file_menu.addAction(self.act_load_file)

        file_menu.addSeparator()
        self.act_quit = QAction("&Quit", self)
        self.act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self.act_quit.triggered.connect(self.close)
        file_menu.addAction(self.act_quit)

        # Simulation menu
        sim_menu = mb.addMenu("&Simulation")
        self.act_pause = QAction("&Pause / Play", self)
        self.act_pause.setShortcut(QKeySequence("Space"))
        self.act_pause.triggered.connect(self._toggle_pause)
        sim_menu.addAction(self.act_pause)

        self.act_step = QAction("&Step One Tick", self)
        self.act_step.setShortcut(QKeySequence("."))
        self.act_step.triggered.connect(lambda: self.sim.tick())
        sim_menu.addAction(self.act_step)

        sim_menu.addSeparator()
        self.act_reset_view = QAction("Reset &View", self)
        self.act_reset_view.setShortcut(QKeySequence("R"))
        self.act_reset_view.triggered.connect(self.canvas.reset_view)
        sim_menu.addAction(self.act_reset_view)

        sim_menu.addSeparator()
        self.act_clear_sel = QAction("&Clear Selection", self)
        self.act_clear_sel.setShortcut(QKeySequence("Escape"))
        self.act_clear_sel.triggered.connect(lambda: self._select_organism(None))
        sim_menu.addAction(self.act_clear_sel)

        # Weather menu
        weather_menu = mb.addMenu("&Weather")
        for wt in WeatherType:
            act = QAction(WEATHER_LABELS[wt], self)
            act.triggered.connect(lambda checked, w=wt: self._on_weather_triggered(w, 80))
            weather_menu.addAction(act)
        weather_menu.addSeparator()
        act_clear_w = QAction("Clear Weather Event", self)
        act_clear_w.triggered.connect(self._on_weather_cleared)
        weather_menu.addAction(act_clear_w)

        # Help menu
        help_menu = mb.addMenu("&Help")
        self.act_about = QAction("&About", self)
        self.act_about.triggered.connect(self._on_about)
        help_menu.addAction(self.act_about)
        self.act_help = QAction("&Quick Help", self)
        self.act_help.setShortcut(QKeySequence("F1"))
        self.act_help.triggered.connect(self._on_help)
        help_menu.addAction(self.act_help)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _apply_style(self) -> None:
        qss = f"""
        QMainWindow, QDialog {{ background: {THEME.bg.name()}; }}
        QWidget {{ color: {THEME.text.name()}; font-family: 'DejaVu Sans', sans-serif; }}
        QMenuBar {{ background: {THEME.panel_bg.name()}; color: {THEME.text.name()}; }}
        QMenuBar::item:selected {{ background: {THEME.accent_dim.name()}; }}
        QMenu {{ background: {THEME.panel_bg.name()}; color: {THEME.text.name()};
                 border: 1px solid {THEME.border.name()}; }}
        QMenu::item:selected {{ background: {THEME.accent_dim.name()}; }}
        QTabWidget::pane {{ border: 1px solid {THEME.border.name()}; top: -1px; }}
        QTabBar::tab {{ background: {THEME.panel_bg_alt.name()}; color: {THEME.text_dim.name()};
                        padding: 6px 14px; border: 1px solid {THEME.border.name()};
                        border-bottom: none; }}
        QTabBar::tab:selected {{ background: {THEME.panel_bg.name()}; color: {THEME.text_bright.name()};
                                 border-bottom: 2px solid {THEME.accent.name()}; }}
        QTabBar::btn {{ background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};
                        border: 1px solid {THEME.border.name()}; }}
        QTabBar::btn:hover {{ background: {THEME.accent_dim.name()}; }}
        QTabWidget::right-corner {{ background: {THEME.panel_bg.name()}; }}
        QPushButton {{ background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};
                       border: 1px solid {THEME.border.name()}; padding: 5px 12px; border-radius: 3px; }}
        QPushButton:hover {{ background: {THEME.accent_dim.name()}; color: {THEME.text_bright.name()}; }}
        QPushButton:pressed {{ background: {THEME.border.name()}; }}
        QPushButton:checked {{ background: {THEME.accent_dim.name()}; color: {THEME.text_bright.name()}; }}
        QSlider::groove:horizontal {{ height: 4px; background: {THEME.border.name()}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {THEME.accent.name()}; width: 14px;
                                      margin: -6px 0; border-radius: 7px; }}
        QGroupBox {{ color: {THEME.text_dim.name()}; border: 1px solid {THEME.border.name()};
                     border-radius: 4px; margin-top: 8px; padding-top: 10px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QTableWidget {{ background: {THEME.panel_bg_alt.name()}; alternate-background-color: {THEME.panel_bg.name()};
                        gridline-color: {THEME.border.name()}; selection-background-color: {THEME.accent_dim.name()};
                        border: 1px solid {THEME.border.name()}; }}
        QHeaderView::section {{ background: {THEME.panel_bg.name()}; color: {THEME.text_dim.name()};
                                padding: 4px; border: none; }}
        QListWidget {{ background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};
                       border: 1px solid {THEME.border.name()}; }}
        QListWidget::item:selected {{ background: {THEME.accent_dim.name()}; }}
        QPlainTextEdit {{ border: 1px solid {THEME.border.name()}; }}
        QSpinBox, QDoubleSpinBox, QComboBox {{ background: {THEME.panel_bg_alt.name()};
                                                color: {THEME.text.name()}; border: 1px solid {THEME.border.name()};
                                                padding: 2px 4px; border-radius: 2px; }}
        QComboBox QAbstractItemView {{ background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};
                                        selection-background-color: {THEME.accent_dim.name()}; }}
        QCheckBox {{ color: {THEME.text.name()}; }}
        QCheckBox::indicator {{ width: 14px; height: 14px; }}
        QStatusBar {{ background: {THEME.panel_bg.name()}; color: {THEME.text_dim.name()}; }}
        QSplitter::handle {{ background: {THEME.border.name()}; }}
        QSplitter::handle:horizontal {{ width: 3px; }}
        QSplitter::handle:vertical {{ height: 3px; }}
        QLabel {{ background: transparent; }}
        """
        self.setStyleSheet(qss)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last_frame_time, 0.1)
        self._last_frame_time = now
        ticks_done = self.sim.step_frames(dt)
        # Refresh UI panels roughly every frame; this is cheap because the
        # panels cache their state.
        self.canvas.update()
        # Don't refresh tables every frame (expensive) - only when sim ticks.
        if ticks_done > 0:
            self.stats_panel.refresh()
            self.species_panel.refresh()
            self.env_panel.refresh()
            self.journal_panel.refresh()
            self.inspector.refresh()
            # Status bar summary.
            self.status_label.setText(
                f"Tick {self.sim.tick_count}  •  Pop {self.sim.population()}  •  "
                f"{self.sim.weather.current.name.title()}  •  "
                f"{'⏸ paused' if self.sim.paused else f'{self.sim.speed:.2f}×'}")

    def _on_stats_updated(self) -> None:
        # Hook for future async stats updates.
        pass

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_pause_toggled(self, paused: bool) -> None:
        self.sim.set_paused(paused)
        self.status_label.setText(
            f"Tick {self.sim.tick_count}  •  Pop {self.sim.population()}  •  "
            f"{'⏸ paused' if paused else f'{self.sim.speed:.2f}×'}")

    def _on_speed_changed(self, speed: float) -> None:
        self.sim.set_speed(speed)

    def _on_regenerate(self, width: int, height: int, seed: int) -> None:
        self.sim.config.width = width
        self.sim.config.height = height
        self.sim.config.seed = seed
        self.sim.rng = random.Random(seed)
        self.sim.generate_world(seed, name=f"World #{seed & 0xFFFF:X}")
        self._wire_sim()
        self.status_label.setText(f"New world generated (seed {seed}, {width}×{height}).")

    def _on_weather_triggered(self, wt: WeatherType, duration: int) -> None:
        self.sim.trigger_weather(wt, duration)

    def _on_weather_cleared(self) -> None:
        self.sim.clear_weather()

    def _on_organism_clicked(self, oid: int) -> None:
        if oid < 0:
            self._select_organism(None)
        else:
            self._select_organism(oid)
            self.tabs.setCurrentWidget(self.inspector)

    def _on_species_selected(self, sid: int) -> None:
        # Select the first living organism of this species (if any).
        for org in self.sim.living_organisms():
            if org.species_id == sid:
                self._select_organism(org.id)
                self.tabs.setCurrentWidget(self.inspector)
                return

    def _select_organism(self, oid: Optional[int]) -> None:
        self.inspector.set_organism(oid)
        self.canvas.select_organism(oid)

    def _toggle_pause(self) -> None:
        self.sim.set_paused(not self.sim.paused)
        self.controls.btn_play.setChecked(self.sim.paused)
        self.controls.btn_play.setText("▶ Play" if self.sim.paused else "⏸ Pause")

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------
    def _on_new_world_dialog(self) -> None:
        from PySide6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("New World")
        form = QFormLayout(dlg)
        w = QSpinBox(); w.setRange(24, 256); w.setValue(96)
        h = QSpinBox(); h.setRange(24, 192); h.setValue(64)
        s = QSpinBox(); s.setRange(0, 0x7FFFFFFF); s.setValue(random.randint(0, 0x7FFFFFFF))
        pop = QSpinBox(); pop.setRange(4, 200); pop.setValue(60)
        spc = QSpinBox(); spc.setRange(1, 6); spc.setValue(3)
        form.addRow("Width:", w)
        form.addRow("Height:", h)
        form.addRow("Seed:", s)
        form.addRow("Initial population:", pop)
        form.addRow("Initial species:", spc)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() == QDialog.Accepted:
            self.sim.config = SimulationConfig(
                width=w.value(), height=h.value(), seed=s.value(),
                initial_population=pop.value(), initial_species_count=spc.value())
            self.sim.rng = random.Random(s.value())
            self.sim.generate_world(s.value(), name=f"World #{s.value() & 0xFFFF:X}")
            self._wire_sim()
            self.status_label.setText(f"New world generated (seed {s.value()}).")

    def _on_save(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save World", "Save name:",
                                        text=f"world_{self.sim.tick_count}")
        if ok and name:
            try:
                path = self.save_manager.save(self.sim, name)
                self.status_label.setText(f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))

    def _on_load(self) -> None:
        dlg = SaveLoadDialog(self.save_manager, mode="load", parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.selected_name:
            try:
                self.sim = self.save_manager.load(dlg.selected_name)
                self._wire_sim()
                self.status_label.setText(f"Loaded '{dlg.selected_name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Load failed", str(e))

    def _on_save_as_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "ecosystem_save.json", "JSON (*.json)")
        if path:
            try:
                save_simulation(self.sim, path, name=Path(path).stem)
                self.status_label.setText(f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))

    def _on_load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load File", "", "JSON (*.json)")
        if path:
            try:
                self.sim = load_simulation(path)
                self._wire_sim()
                self.status_label.setText(f"Loaded {path}")
            except Exception as e:
                QMessageBox.critical(self, "Load failed", str(e))

    def _on_about(self) -> None:
        QMessageBox.about(self, "About Project Ecosystem",
            "<h2>Project Ecosystem</h2>"
            "<p>A native Linux desktop artificial life simulation.</p>"
            "<p>Organisms live, forage, hunt, drink, mate, age and die in a "
            "procedurally generated world with day/night cycles, weather "
            "events and evolving biomes.</p>"
            "<p>Each organism has a heritable genome with mutation and "
            "recombination; new species arise naturally through genetic drift.</p>"
            "<p><i>Built with PySide6 and pure-Python simulation logic.</i></p>")

    def _on_help(self) -> None:
        QMessageBox.information(self, "Quick Help",
            "<b>Controls</b><ul>"
            "<li><b>Left-click</b> an organism to inspect it.</li>"
            "<li><b>Left-drag</b> on empty space to pan the world.</li>"
            "<li><b>Mouse wheel</b> to zoom; <b>R</b> to reset view.</li>"
            "<li><b>Space</b> pause/play; <b>.</b> step one tick.</li>"
            "<li><b>Ctrl+N</b> new world; <b>Ctrl+S</b> save; <b>Ctrl+O</b> load.</li>"
            "</ul>"
            "<b>Tabs</b><ul>"
            "<li><b>Inspector</b>: selected organism details and life events.</li>"
            "<li><b>Species</b>: living and extinct species, populations.</li>"
            "<li><b>Stats</b>: population/species/energy/temperature charts.</li>"
            "<li><b>Environment</b>: biome distribution, weather, day/night.</li>"
            "<li><b>Journal</b>: chronological log of notable events.</li>"
            "</ul>")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _wire_sim(self) -> None:
        """Re-bind the new simulation to all panels/widgets."""
        self.sim.on_stats = self._on_stats_updated
        self.controls.set_simulation(self.sim)
        self.canvas.set_simulation(self.sim)
        self.inspector.set_simulation(self.sim)
        self.inspector.set_organism(None)
        self.species_panel.set_simulation(self.sim)
        self.stats_panel.set_simulation(self.sim)
        self.env_panel.set_simulation(self.sim)
        self.journal_panel.set_simulation(self.sim)
