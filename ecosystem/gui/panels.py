"""Side panels: controls, inspector, species list, stats charts, journal.

Each panel is a self-contained QWidget that the main window docks into a
QSplitter / QTabWidget. They talk to the :class:`Simulation` through a
shared reference set by :meth:`set_simulation` and refresh themselves on
:meth:`refresh` (called by the main window's tick timer).
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPalette, QAction, QShortcut, QKeySequence,
    QPainterPath,
)
from PySide6.QtWidgets import (
    QWidget, QPushButton, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QTabWidget, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFormLayout, QFrame, QSizePolicy, QScrollArea, QToolButton,
)

from ..core.simulation import Simulation, SimulationConfig
from ..core.organism import Organism, OrganismState, STATE_LABELS
from ..core.world import Biome
from ..core.weather import WeatherType, WEATHER_LABELS
from ..core.genome import TRAIT_SPECS
from ..persistence import SaveManager
from .theme import THEME, hsl_to_qcolor


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------
class ControlPanel(QWidget):
    """Top toolbar-style controls: play/pause, speed, regenerate, weather."""

    pause_toggled = Signal(bool)
    speed_changed = Signal(float)
    regenerate_requested = Signal(int, int, int)   # width, height, seed
    weather_triggered = Signal(WeatherType, int)
    weather_cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Play/Pause
        self.btn_play = QPushButton("⏸ Pause")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._on_play_clicked)
        layout.addWidget(self.btn_play)

        # Speed slider
        layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(25)
        self.speed_slider.setMaximum(800)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(160)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0×")
        self.speed_label.setMinimumWidth(38)
        layout.addWidget(self.speed_label)

        # Step button (advance 1 tick while paused)
        self.btn_step = QPushButton("⏭ Step")
        self.btn_step.clicked.connect(self._on_step_clicked)
        layout.addWidget(self.btn_step)

        layout.addStretch(1)

        # World generation
        layout.addWidget(QLabel("W:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(24, 256)
        self.width_spin.setValue(96)
        layout.addWidget(self.width_spin)
        layout.addWidget(QLabel("H:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(24, 192)
        self.height_spin.setValue(64)
        layout.addWidget(self.height_spin)
        layout.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 0x7FFFFFFF)
        self.seed_spin.setValue(0xC0FFEE)
        layout.addWidget(self.seed_spin)
        self.btn_regen = QPushButton("↻ New World")
        self.btn_regen.clicked.connect(self._on_regen_clicked)
        layout.addWidget(self.btn_regen)

        layout.addStretch(1)

        # Weather trigger
        layout.addWidget(QLabel("Weather:"))
        self.weather_combo = QComboBox()
        for wt in WeatherType:
            self.weather_combo.addItem(WEATHER_LABELS[wt], wt)
        layout.addWidget(self.weather_combo)
        self.btn_weather = QPushButton("⚡ Trigger")
        self.btn_weather.clicked.connect(self._on_weather_triggered)
        layout.addWidget(self.btn_weather)
        self.btn_clear_weather = QPushButton("Clear")
        self.btn_clear_weather.clicked.connect(self._on_weather_cleared)
        layout.addWidget(self.btn_clear_weather)

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self.btn_play.setChecked(sim.paused)
        self.btn_play.setText("▶ Play" if sim.paused else "⏸ Pause")
        v = int(sim.speed * 100)
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(v)
        self.speed_slider.blockSignals(False)
        self.speed_label.setText(f"{sim.speed:.2f}×")
        self.width_spin.setValue(sim.config.width)
        self.height_spin.setValue(sim.config.height)
        self.seed_spin.setValue(sim.config.seed)

    def _on_play_clicked(self) -> None:
        if self._sim is None:
            return
        paused = self.btn_play.isChecked()
        self.btn_play.setText("▶ Play" if paused else "⏸ Pause")
        self.pause_toggled.emit(paused)

    def _on_speed_changed(self, value: int) -> None:
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.2f}×")
        self.speed_changed.emit(speed)

    def _on_step_clicked(self) -> None:
        if self._sim is not None:
            self._sim.tick()

    def _on_regen_clicked(self) -> None:
        self.regenerate_requested.emit(
            self.width_spin.value(), self.height_spin.value(), self.seed_spin.value())

    def _on_weather_triggered(self) -> None:
        wt = self.weather_combo.currentData()
        self.weather_triggered.emit(wt, 80)

    def _on_weather_cleared(self) -> None:
        self.weather_cleared.emit()


# ---------------------------------------------------------------------------
# Stats panel with simple charts
# ---------------------------------------------------------------------------
class StatsPanel(QWidget):
    """Population/species/energy charts + current numeric stats."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Numeric summary grid.
        self.summary = QGridLayout()
        self.summary.setSpacing(6)
        self.lbl_pop = self._mk_stat("Population")
        self.lbl_species = self._mk_stat("Living species")
        self.lbl_births = self._mk_stat("Recent births")
        self.lbl_deaths = self._mk_stat("Recent deaths")
        self.lbl_tick = self._mk_stat("Tick")
        self.lbl_time = self._mk_stat("Time of day")
        self.lbl_weather = self._mk_stat("Weather")
        self.lbl_temp = self._mk_stat("Temperature")
        self.summary.addWidget(self.lbl_pop[0], 0, 0); self.summary.addWidget(self.lbl_pop[1], 0, 1)
        self.summary.addWidget(self.lbl_species[0], 0, 2); self.summary.addWidget(self.lbl_species[1], 0, 3)
        self.summary.addWidget(self.lbl_births[0], 1, 0); self.summary.addWidget(self.lbl_births[1], 1, 1)
        self.summary.addWidget(self.lbl_deaths[0], 1, 2); self.summary.addWidget(self.lbl_deaths[1], 1, 3)
        self.summary.addWidget(self.lbl_tick[0], 2, 0); self.summary.addWidget(self.lbl_tick[1], 2, 1)
        self.summary.addWidget(self.lbl_time[0], 2, 2); self.summary.addWidget(self.lbl_time[1], 2, 3)
        self.summary.addWidget(self.lbl_weather[0], 3, 0); self.summary.addWidget(self.lbl_weather[1], 3, 1)
        self.summary.addWidget(self.lbl_temp[0], 3, 2); self.summary.addWidget(self.lbl_temp[1], 3, 3)
        layout.addLayout(self.summary)

        # Population chart.
        self.pop_chart = ChartWidget("Population over time", THEME.chart_line_pop)
        layout.addWidget(self.pop_chart, 2)
        # Species count chart.
        self.species_chart = ChartWidget("Species count", THEME.chart_line_species)
        layout.addWidget(self.species_chart, 1)
        # Mean energy chart.
        self.energy_chart = ChartWidget("Mean energy", THEME.chart_line_energy)
        layout.addWidget(self.energy_chart, 1)
        # Temperature chart.
        self.temp_chart = ChartWidget("Effective temperature", THEME.chart_line_temp)
        layout.addWidget(self.temp_chart, 1)

    def _mk_stat(self, label: str):
        a = QLabel(label)
        a.setStyleSheet(f"color: {THEME.text_dim.name()}; font-size: 11px;")
        b = QLabel("—")
        b.setStyleSheet(f"color: {THEME.text_bright.name()}; font-size: 13px; font-weight: 600;")
        return a, b

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self.refresh()

    def refresh(self) -> None:
        if self._sim is None:
            return
        sim = self._sim
        stats = sim.stats
        self.lbl_pop[1].setText(str(sim.population()))
        self.lbl_species[1].setText(str(len(sim.species.living_species())))
        self.lbl_births[1].setText(str(sim._recent_births))
        self.lbl_deaths[1].setText(str(sim._recent_deaths))
        self.lbl_tick[1].setText(str(sim.tick_count))
        tod = sim.world.time_of_day if sim.world else 0.0
        hour = int((tod * 24.0) % 24.0)
        minute = int(((tod * 24.0) % 1.0) * 60.0)
        self.lbl_time[1].setText(f"{hour:02d}:{minute:02d}")
        self.lbl_weather[1].setText(WEATHER_LABELS[sim.weather.current])
        self.lbl_temp[1].setText(f"{sim.weather.effective_temp:.2f}")
        # Charts.
        self.pop_chart.set_data(stats.ticks, stats.population)
        self.species_chart.set_data(stats.ticks, stats.species_count)
        self.energy_chart.set_data(stats.ticks, stats.mean_energy)
        self.temp_chart.set_data(stats.ticks, stats.temperature)


class ChartWidget(QWidget):
    """A minimal line chart - draws a list of (x, y) samples."""

    def __init__(self, title: str, line_color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self.line_color = line_color
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, xs: List[float], ys: List[float]) -> None:
        self.xs = list(xs)
        self.ys = list(ys)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), THEME.chart_bg)
        # Title
        painter.setPen(THEME.text_dim)
        f = QFont("DejaVu Sans", 9)
        painter.setFont(f)
        painter.drawText(6, 14, self.title)
        if len(self.xs) < 2:
            painter.setPen(THEME.text_dim)
            painter.drawText(self.rect(), Qt.AlignCenter, "No data yet")
            return
        # Plot area
        pad_top = 20
        pad_bottom = 6
        pad_left = 6
        pad_right = 6
        plot = QRectF(pad_left, pad_top,
                      self.width() - pad_left - pad_right,
                      self.height() - pad_top - pad_bottom)
        # Grid
        painter.setPen(QPen(THEME.chart_grid, 1))
        for i in range(1, 4):
            y = plot.top() + plot.height() * i / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        # Scale
        xmin, xmax = self.xs[0], self.xs[-1]
        ymin = min(self.ys)
        ymax = max(self.ys)
        if ymax - ymin < 1e-6:
            ymax = ymin + 1.0
        def sx(x): return plot.left() + (x - xmin) / (xmax - xmin) * plot.width()
        def sy(y): return plot.bottom() - (y - ymin) / (ymax - ymin) * plot.height()
        # Line
        path = QPainterPath()
        path.moveTo(QPointF(sx(self.xs[0]), sy(self.ys[0])))
        for x, y in zip(self.xs[1:], self.ys[1:]):
            path.lineTo(QPointF(sx(x), sy(y)))
        painter.setPen(QPen(self.line_color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        # Min/max labels
        painter.setPen(THEME.text_dim)
        painter.drawText(QPointF(plot.right() - 60, plot.top() + 10),
                         f"max {ymax:.1f}")
        painter.drawText(QPointF(plot.right() - 60, plot.bottom() - 2),
                         f"min {ymin:.1f}")
        painter.end()


# ---------------------------------------------------------------------------
# Species panel
# ---------------------------------------------------------------------------
class SpeciesPanel(QWidget):
    """List of species with population and ancestry info."""

    species_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Species"))
        hdr.addStretch(1)
        self.chk_extinct = QCheckBox("Show extinct")
        self.chk_extinct.setChecked(True)
        hdr.addWidget(self.chk_extinct)
        self.chk_extinct.toggled.connect(self.refresh)
        layout.addLayout(hdr)
        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Name", "Pop", "Peak", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table)

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self.refresh()

    def refresh(self) -> None:
        if self._sim is None:
            return
        summaries = self._sim.species_summary()
        show_extinct = self.chk_extinct.isChecked()
        self.table.setRowCount(0)
        for s in summaries:
            if not show_extinct and s["extinct"]:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            # Colour swatch
            swatch = QTableWidgetItem()
            color = hsl_to_qcolor(s["color_hue"], s["color_sat"], s["color_light"])
            swatch.setBackground(color)
            swatch.setText("")
            self.table.setItem(row, 0, swatch)
            name_item = QTableWidgetItem(s["name"])
            if s["extinct"]:
                name_item.setForeground(THEME.text_dim)
                f = name_item.font()
                f.setStrikeOut(True)
                name_item.setFont(f)
            self.table.setItem(row, 1, name_item)
            self.table.setItem(row, 2, QTableWidgetItem(str(s["population"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(s["peak"])))
            status = "Extinct" if s["extinct"] else "Living"
            self.table.setItem(row, 4, QTableWidgetItem(status))

    def _on_selection(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        row = items[0].row()
        name_item = self.table.item(row, 1)
        if name_item is None or self._sim is None:
            return
        # Find species by name.
        for sp in self._sim.species.all_species():
            if sp.name == name_item.text():
                self.species_selected.emit(sp.id)
                return


# ---------------------------------------------------------------------------
# Inspector panel - selected organism details
# ---------------------------------------------------------------------------
class InspectorPanel(QWidget):
    """Detailed view of a single organism (or 'nothing selected')."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        self._organism_id: Optional[int] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.lbl_title = QLabel("No organism selected")
        self.lbl_title.setStyleSheet(f"color: {THEME.text_bright.name()}; font-size: 14px; font-weight: 700;")
        layout.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel("Click an organism in the world to inspect it.")
        self.lbl_subtitle.setStyleSheet(f"color: {THEME.text_dim.name()}; font-size: 11px;")
        self.lbl_subtitle.setWordWrap(True)
        layout.addWidget(self.lbl_subtitle)

        # Stats grid
        self.stats_grid = QGridLayout()
        self.stats_grid.setSpacing(4)
        self.stat_labels: dict[str, QLabel] = {}
        for i, key in enumerate([
            "ID", "Species", "State", "Age", "Max age",
            "Energy", "Hydration", "Health", "Position",
            "Offspring", "Kills", "Diet",
        ]):
            r, c = i // 2, (i % 2) * 2
            lbl = QLabel(key)
            lbl.setStyleSheet(f"color: {THEME.text_dim.name()}; font-size: 11px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {THEME.text.name()}; font-size: 12px;")
            self.stats_grid.addWidget(lbl, r, c)
            self.stats_grid.addWidget(val, r, c + 1)
            self.stat_labels[key] = val
        layout.addLayout(self.stats_grid)

        # Traits section
        traits_box = QGroupBox("Genome traits")
        traits_layout = QGridLayout()
        traits_layout.setSpacing(3)
        self.trait_labels: dict[str, QLabel] = {}
        for i, (name, spec) in enumerate(TRAIT_SPECS.items()):
            r, c = i // 3, (i % 3) * 2
            lbl = QLabel(name.replace("_", " ").title())
            lbl.setStyleSheet(f"color: {THEME.text_dim.name()}; font-size: 10px;")
            val = QLabel("—")
            val.setStyleSheet(f"color: {THEME.text.name()}; font-size: 11px;")
            traits_layout.addWidget(lbl, r, c)
            traits_layout.addWidget(val, r, c + 1)
            self.trait_labels[name] = val
        traits_box.setLayout(traits_layout)
        layout.addWidget(traits_box)

        # Life events
        events_box = QGroupBox("Life events")
        ev_layout = QVBoxLayout()
        self.events_view = QPlainTextEdit()
        self.events_view.setReadOnly(True)
        self.events_view.setMaximumHeight(140)
        self.events_view.setStyleSheet(
            f"background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};"
            f" font-family: 'DejaVu Sans Mono', monospace; font-size: 10px;")
        ev_layout.addWidget(self.events_view)
        events_box.setLayout(ev_layout)
        layout.addWidget(events_box)

        layout.addStretch(1)

        self.lbl_action = QLabel("Current behaviour: —")
        self.lbl_action.setStyleSheet(
            f"color: {THEME.accent.name()}; font-size: 11px; font-style: italic;")
        self.lbl_action.setWordWrap(True)
        layout.addWidget(self.lbl_action)

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim

    def set_organism(self, oid: Optional[int]) -> None:
        self._organism_id = oid
        self.refresh()

    def refresh(self) -> None:
        if self._sim is None or self._organism_id is None:
            self.lbl_title.setText("No organism selected")
            self.lbl_subtitle.setText("Click an organism in the world to inspect it.")
            for lbl in self.stat_labels.values():
                lbl.setText("—")
            for lbl in self.trait_labels.values():
                lbl.setText("—")
            self.events_view.clear()
            self.lbl_action.setText("Current behaviour: —")
            return
        org = self._sim.get_organism(self._organism_id)
        if org is None:
            self.lbl_title.setText(f"Organism #{self._organism_id} (gone)")
            return
        sp = self._sim.species.get(org.species_id)
        sp_name = sp.name if sp else "?"
        self.lbl_title.setText(f"Organism #{org.id}")
        self.lbl_subtitle.setText(
            f"Species: {sp_name}  •  Age {org.age:.0f}/{org.genome['max_age']:.0f}  •  "
            f"{'Alive' if org.alive else 'Dead ('+org.death_cause+')'}")
        self.stat_labels["ID"].setText(str(org.id))
        self.stat_labels["Species"].setText(sp_name)
        self.stat_labels["State"].setText(STATE_LABELS[org.state])
        self.stat_labels["Age"].setText(f"{org.age:.0f}")
        self.stat_labels["Max age"].setText(f"{org.genome['max_age']:.0f}")
        self.stat_labels["Energy"].setText(f"{org.energy:.0f} / {org.max_energy:.0f}")
        self.stat_labels["Hydration"].setText(f"{org.hydration:.0f}")
        self.stat_labels["Health"].setText(f"{org.health:.0f}")
        self.stat_labels["Position"].setText(f"({org.x:.1f}, {org.y:.1f})")
        self.stat_labels["Offspring"].setText(str(org.offspring_count))
        self.stat_labels["Kills"].setText(str(org.kills))
        diet = "Herbivore" if org.is_herbivore else ("Carnivore" if org.is_predator else "Omnivore")
        self.stat_labels["Diet"].setText(diet)
        # Traits
        for name, lbl in self.trait_labels.items():
            v = org.genome[name]
            if name == "color_hue":
                lbl.setText(f"{v:.0f}°")
            else:
                lbl.setText(f"{v:.2f}")
        # Events
        lines = []
        for e in org.events[-20:]:
            lines.append(f"[tick {e.tick}] {e.kind}: {e.detail}")
        self.events_view.setPlainText("\n".join(lines))
        self.lbl_action.setText(f"Current behaviour: {org.last_action}")


# ---------------------------------------------------------------------------
# Journal panel
# ---------------------------------------------------------------------------
class JournalPanel(QWidget):
    """Chronological log of world events."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Observation journal"))
        hdr.addStretch(1)
        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        hdr.addWidget(self.chk_autoscroll)
        layout.addLayout(hdr)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setStyleSheet(
            f"background: {THEME.panel_bg_alt.name()}; color: {THEME.text.name()};"
            f" font-family: 'DejaVu Sans Mono', monospace; font-size: 11px;")
        layout.addWidget(self.view)

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self.refresh()

    def refresh(self) -> None:
        if self._sim is None:
            return
        lines = []
        for e in self._sim.journal.recent(120):
            cat = e.category
            tag = {
                "species": "🧬",
                "weather": "🌦",
                "milestone": "★",
                "population": "pop",
                "event": "!",
            }.get(cat, cat)
            lines.append(f"[{e.tick:>5}] {tag:<5} {e.text}")
        text = "\n".join(lines)
        self.view.setPlainText(text)
        if self.chk_autoscroll.isChecked():
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Environment panel - shows world details
# ---------------------------------------------------------------------------
class EnvironmentPanel(QWidget):
    """World/environment info: biome distribution, daylight, weather."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sim: Optional[Simulation] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.lbl_world = QLabel("World: —")
        self.lbl_world.setStyleSheet(f"color: {THEME.text_bright.name()}; font-size: 13px; font-weight: 600;")
        layout.addWidget(self.lbl_world)

        self.lbl_size = QLabel("Size: —")
        layout.addWidget(self.lbl_size)
        self.lbl_day = QLabel("Day/Night: —")
        layout.addWidget(self.lbl_day)
        self.lbl_weather = QLabel("Weather: —")
        layout.addWidget(self.lbl_weather)
        self.lbl_temp = QLabel("Temperature: —")
        layout.addWidget(self.lbl_temp)

        # Biome distribution table
        layout.addWidget(QLabel("Biome distribution"))
        self.biome_table = QTableWidget(0, 3)
        self.biome_table.setHorizontalHeaderLabels(["Biome", "Tiles", "%"])
        self.biome_table.verticalHeader().setVisible(False)
        self.biome_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.biome_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.biome_table)

        layout.addStretch(1)

    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self.refresh()

    def refresh(self) -> None:
        if self._sim is None or self._sim.world is None:
            return
        w = self._sim.world
        self.lbl_world.setText(f"World: {w.name}")
        self.lbl_size.setText(f"Size: {w.width} × {w.height} tiles  •  seed {w.seed}")
        tod = w.time_of_day
        hour = int((tod * 24.0) % 24.0)
        minute = int(((tod * 24.0) % 1.0) * 60.0)
        phase = "Day" if w.is_day() else "Night"
        self.lbl_day.setText(f"Day/Night: {hour:02d}:{minute:02d} ({phase})  •  day {w.tick // w.day_length + 1}")
        self.lbl_weather.setText(f"Weather: {WEATHER_LABELS[self._sim.weather.current]}")
        ev = self._sim.weather.active_event
        if ev is not None:
            self.lbl_weather.setText(
                self.lbl_weather.text() + f"  •  event: {WEATHER_LABELS[ev.kind]} ({ev.duration}t left)")
        self.lbl_temp.setText(f"Temperature: {self._sim.weather.effective_temp:.2f}  •  base {self._sim.weather.base_temp:.2f}")
        # Biome counts
        counts = w.biome_counts()
        total = sum(counts.values()) or 1
        self.biome_table.setRowCount(0)
        for biome in sorted(counts.keys(), key=lambda b: -counts[b]):
            row = self.biome_table.rowCount()
            self.biome_table.insertRow(row)
            self.biome_table.setItem(row, 0, QTableWidgetItem(biome.name.replace("_", " ").title()))
            self.biome_table.setItem(row, 1, QTableWidgetItem(str(counts[biome])))
            self.biome_table.setItem(row, 2, QTableWidgetItem(f"{100.0 * counts[biome] / total:.1f}%"))
