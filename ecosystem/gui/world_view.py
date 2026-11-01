"""The interactive world canvas.

Renders the simulated world as a tile grid with organisms drawn on top,
supports pan & zoom, click-to-select an organism, and exposes a small
API the rest of the GUI uses (``set_simulation``, ``select_organism``,
``refresh_terrain``).

Rendering strategy:

* Terrain is rendered into a cached :class:`QPixmap` whenever the world
  is regenerated or its overall lighting changes substantially. Drawing
  96x64 = 6144 tiles every frame would be wasteful.
* Organisms are drawn every frame on top of the cached terrain pixmap -
  they're the dynamic part of the scene.
* A translucent night overlay is drawn on top, its opacity derived from
  the world's :meth:`~ecosystem.core.world.World.daylight`.
* Weather effects are drawn as a subtle full-canvas tint.

The widget is intentionally framework-light: it only depends on PySide6
and the simulation backend.
"""

from __future__ import annotations

import math
from typing import List, Optional

from PySide6.QtCore import Qt, QPoint, QPointF, QRectF, QSize, Signal, QTimer
from PySide6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath, QPolygonF,
    QFont, QRadialGradient, QLinearGradient, QPalette,
)
from PySide6.QtWidgets import QWidget

from ..core.simulation import Simulation
from ..core.organism import Organism, OrganismState
from ..core.world import World, Biome
from ..core.weather import WeatherType, WEATHER_LABELS
from .theme import THEME, biome_color, organism_color, hsl_to_qcolor

__all__ = ["WorldCanvas"]


class WorldCanvas(QWidget):
    """Custom-painted view of the simulation world."""

    organism_clicked = Signal(int)   # emits organism id, or -1 if empty space

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        # State
        self._sim: Optional[Simulation] = None
        self._terrain_pixmap: Optional[QPixmap] = None
        self._terrain_dirty: bool = True
        # View transform
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._dragging: bool = False
        self._drag_start: Optional[QPoint] = None
        self._drag_pan_start = (0.0, 0.0)
        # Selection
        self._selected_id: Optional[int] = None
        self._hover_id: Optional[int] = None
        # Tooltip-ish status text drawn at bottom-left.
        self._status_text: str = ""
        # Background
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, THEME.bg)
        self.setPalette(pal)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_simulation(self, sim: Simulation) -> None:
        self._sim = sim
        self._terrain_dirty = True
        self._selected_id = None
        self._hover_id = None
        # Fit world to viewport.
        self._fit_to_view()
        self.update()

    def refresh_terrain(self) -> None:
        self._terrain_dirty = True
        self.update()

    def select_organism(self, oid: Optional[int]) -> None:
        self._selected_id = oid
        self.update()

    def selected_organism_id(self) -> Optional[int]:
        return self._selected_id

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.4, min(8.0, zoom))
        self.update()

    def reset_view(self) -> None:
        self._fit_to_view()
        self.update()

    # ------------------------------------------------------------------
    # View transforms
    # ------------------------------------------------------------------
    def _fit_to_view(self) -> None:
        if self._sim is None or self._sim.world is None:
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            return
        w = self.width() or 640
        h = self.height() or 480
        world = self._sim.world
        # Tile size in pixels.
        ts = min(w / world.width, h / world.height) * 0.95
        self._zoom = ts  # zoom == pixels per tile
        # Centre.
        self._pan_x = (w - world.width * ts) / 2.0
        self._pan_y = (h - world.height * ts) / 2.0

    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (self._pan_x + x * self._zoom, self._pan_y + y * self._zoom)

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self._pan_x) / self._zoom, (sy - self._pan_y) / self._zoom)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:
        if self._sim is not None and self._terrain_pixmap is None:
            self._fit_to_view()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), THEME.bg)

        if self._sim is None or self._sim.world is None:
            painter.setPen(THEME.text_dim)
            painter.drawText(self.rect(), Qt.AlignCenter, "No world loaded")
            return

        # 1) Render or reuse the terrain pixmap.
        if self._terrain_dirty or self._terrain_pixmap is None:
            self._render_terrain_pixmap()
        if self._terrain_pixmap is not None:
            painter.drawPixmap(QPointF(self._pan_x, self._pan_y), self._terrain_pixmap)

        # 2) Draw subtle tile grid for orientation at high zoom.
        if self._zoom >= 6.0:
            painter.setPen(QPen(THEME.grid_line, 1))
            world = self._sim.world
            for i in range(0, world.width + 1):
                x = self._pan_x + i * self._zoom
                painter.drawLine(QPointF(x, self._pan_y),
                                 QPointF(x, self._pan_y + world.height * self._zoom))
            for j in range(0, world.height + 1):
                y = self._pan_y + j * self._zoom
                painter.drawLine(QPointF(self._pan_x, y),
                                 QPointF(self._pan_x + world.width * self._zoom, y))

        # 3) Draw water resource nodes (springs).
        self._draw_resources(painter)

        # 4) Draw organisms.
        self._draw_organisms(painter)

        # 5) Night overlay.
        daylight = self._sim.world.daylight()
        if daylight < 0.85:
            alpha = int(120 * (1.0 - daylight))
            painter.fillRect(self.rect(), QColor(8, 12, 30, alpha))

        # 6) Weather tint.
        self._draw_weather_tint(painter)

        # 7) Selection ring & hover tooltip.
        self._draw_selection(painter)

        # 8) Status text bottom-left.
        self._draw_status(painter)

        painter.end()

    def _render_terrain_pixmap(self) -> None:
        """Render the world's terrain tiles into ``self._terrain_pixmap``."""
        if self._sim is None or self._sim.world is None:
            return
        world = self._sim.world
        ts = max(1.0, self._zoom)
        pm = QPixmap(int(world.width * ts), int(world.height * ts))
        pm.fill(THEME.bg)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, False)
        # Draw tiles as solid rects. Faster than path fills.
        for tile in world.tiles:
            color = biome_color(tile.biome)
            # Slight shading by elevation for a bit of depth.
            shade = 0.85 + 0.3 * tile.elevation
            r = min(255, int(color.red() * shade))
            g = min(255, int(color.green() * shade))
            b = min(255, int(color.blue() * shade))
            painter.fillRect(
                QRectF(tile.x * ts, tile.y * ts, ts + 0.5, ts + 0.5),
                QColor(r, g, b),
            )
            # Show food density as a faint green tint on fertile tiles.
            if tile.food > 0.5 and tile.params["max_food"] > 0:
                food_frac = min(1.0, tile.food / tile.params["max_food"])
                overlay = QColor(120, 200, 80, int(60 * food_frac))
                painter.fillRect(
                    QRectF(tile.x * ts, tile.y * ts, ts + 0.5, ts + 0.5),
                    overlay,
                )
        painter.end()
        self._terrain_pixmap = pm
        self._terrain_dirty = False

    def _draw_resources(self, painter: QPainter) -> None:
        if self._sim is None or self._sim.world is None:
            return
        for res in self._sim.world.resources:
            sx, sy = self._world_to_screen(res.x + 0.5, res.y + 0.5)
            if res.kind.name == "WATER":
                # A small blue circle with a lighter centre.
                r = max(2.0, self._zoom * 0.35)
                painter.setBrush(QBrush(QColor(120, 180, 230, 200)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(sx, sy), r, r)
                painter.setBrush(QBrush(QColor(200, 230, 255, 220)))
                painter.drawEllipse(QPointF(sx, sy), r * 0.4, r * 0.4)

    def _draw_organisms(self, painter: QPainter) -> None:
        if self._sim is None:
            return
        # Determine marker radius from zoom.
        base = max(2.5, self._zoom * 0.42)
        sel_id = self._selected_id
        hover_id = self._hover_id
        for org in self._sim.living_organisms():
            sx, sy = self._world_to_screen(org.x, org.y)
            # Skip drawing if completely outside view.
            if sx < -20 or sy < -20 or sx > self.width() + 20 or sy > self.height() + 20:
                continue
            r = base * (0.7 + 0.5 * org.genome["size"])
            color = organism_color(org.genome["color_hue"], org.genome["color_sat"], org.genome["color_light"])
            # Body.
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(0, 0, 0, 140), 1))
            painter.drawEllipse(QPointF(sx, sy), r, r)
            # Diet indicator: small outline ring colour.
            if org.is_predator:
                ring = QColor(220, 80, 80, 220)
            elif org.is_herbivore:
                ring = QColor(120, 200, 80, 200)
            else:
                ring = QColor(220, 200, 80, 200)
            painter.setPen(QPen(ring, max(0.8, r * 0.18)))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(sx, sy), r * 1.1, r * 1.1)
            # State indicator for selected/hovered.
            if org.id == sel_id or org.id == hover_id:
                painter.setPen(QPen(THEME.text_bright, 1))
                f = QFont("DejaVu Sans", max(6, int(self._zoom * 0.3)))
                painter.setFont(f)
                painter.drawText(QPointF(sx + r + 2, sy - r - 1), f"#{org.id}")

    def _draw_selection(self, painter: QPainter) -> None:
        if self._sim is None or self._selected_id is None:
            return
        org = self._sim.get_organism(self._selected_id)
        if org is None or not org.alive:
            return
        sx, sy = self._world_to_screen(org.x, org.y)
        r = max(4.0, self._zoom * 0.55) * (0.7 + 0.5 * org.genome["size"])
        # Pulsing ring.
        t = self._sim.tick_count
        pulse = 0.5 + 0.5 * math.sin(t * 0.2)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(THEME.selection, 2))
        painter.drawEllipse(QPointF(sx, sy), r * 1.6 + pulse * 2, r * 1.6 + pulse * 2)
        # Vision range circle (faint).
        vision = org.genome["vision_range"] * self._sim.world.daylight()
        if vision > 0.5:
            painter.setPen(QPen(QColor(255, 220, 80, 70), 1, Qt.DashLine))
            painter.drawEllipse(QPointF(sx, sy), vision * self._zoom, vision * self._zoom)

    def _draw_weather_tint(self, painter: QPainter) -> None:
        if self._sim is None:
            return
        w = self._sim.weather
        kind = w.current
        if kind == WeatherType.RAIN:
            painter.fillRect(self.rect(), QColor(80, 110, 160, 35))
        elif kind == WeatherType.STORM:
            painter.fillRect(self.rect(), QColor(60, 70, 110, 60))
            # Random lightning flashes.
            if (self._sim.tick_count % 37) == 0:
                painter.fillRect(self.rect(), QColor(255, 255, 255, 90))
        elif kind == WeatherType.DROUGHT:
            painter.fillRect(self.rect(), QColor(220, 160, 80, 30))
        elif kind == WeatherType.HEATWAVE:
            painter.fillRect(self.rect(), QColor(220, 90, 60, 35))
        elif kind == WeatherType.COLD_SNAP:
            painter.fillRect(self.rect(), QColor(180, 220, 255, 40))
        elif kind == WeatherType.FOG:
            painter.fillRect(self.rect(), QColor(220, 220, 220, 55))

    def _draw_status(self, painter: QPainter) -> None:
        if self._sim is None:
            return
        painter.setPen(THEME.text_bright)
        f = QFont("DejaVu Sans", 9)
        painter.setFont(f)
        # Status: time, weather, population.
        tod = self._sim.world.time_of_day
        hour = int((tod * 24.0) % 24)
        minute = int(((tod * 24.0) % 1.0) * 60.0)
        weather_label = WEATHER_LABELS[self._sim.weather.current]
        day_phase = "Day" if self._sim.world.is_day() else "Night"
        text = (f"Tick {self._sim.tick_count}   "
                f"Time {hour:02d}:{minute:02d} ({day_phase})   "
                f"Pop {self._sim.population()}   "
                f"Weather: {weather_label}")
        painter.drawText(10, self.height() - 10, text)

    # ------------------------------------------------------------------
    # Mouse / keyboard
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            # Click selects an organism if any near; else begins pan.
            wx, wy = self._screen_to_world(event.position().x(), event.position().y())
            org = self._sim.organism_at_point(wx, wy, tolerance=1.5) if self._sim else None
            if org is not None:
                self._selected_id = org.id
                self.organism_clicked.emit(org.id)
                self.update()
                return
            # Else start panning.
            self._selected_id = None
            self.organism_clicked.emit(-1)
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_pan_start = (self._pan_x, self._pan_y)
            self.update()
        elif event.button() == Qt.RightButton:
            # Right-drag also pans.
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._drag_pan_start = (self._pan_x, self._pan_y)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._drag_start is not None:
            dx = event.position().x() - self._drag_start.x()
            dy = event.position().y() - self._drag_start.y()
            self._pan_x = self._drag_pan_start[0] + dx
            self._pan_y = self._drag_pan_start[1] + dy
            self.update()
        else:
            # Hover: highlight nearest organism.
            if self._sim is not None:
                wx, wy = self._screen_to_world(event.position().x(), event.position().y())
                org = self._sim.organism_at_point(wx, wy, tolerance=1.0)
                new_hover = org.id if org is not None else None
                if new_hover != self._hover_id:
                    self._hover_id = new_hover
                    self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        self._drag_start = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 ** delta
        # Zoom around cursor.
        mx = event.position().x()
        my = event.position().y()
        wx_before, wy_before = self._screen_to_world(mx, my)
        self._zoom = max(2.0, min(40.0, self._zoom * factor))
        # Adjust pan so cursor stays over same world point.
        wx_after = (mx - self._pan_x) / self._zoom
        wy_after = (my - self._pan_y) / self._zoom
        self._pan_x += (wx_after - wx_before) * self._zoom
        self._pan_y += (wy_after - wy_before) * self._zoom
        # Terrain pixmap needs to be re-rendered at higher resolution.
        self._terrain_dirty = True
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key_Escape:
            self._selected_id = None
            self.organism_clicked.emit(-1)
            self.update()
        else:
            super().keyPressEvent(event)
