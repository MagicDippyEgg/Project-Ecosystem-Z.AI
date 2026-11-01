"""Visual theme: colours, fonts, and palette helpers.

Kept in one place so the whole GUI uses a consistent dark-with-vivid-accents
aesthetic that makes organisms pop against the terrain.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Tuple

from PySide6.QtGui import QColor, QPalette, QFont, QFontDatabase
from PySide6.QtCore import Qt

__all__ = [
    "Theme",
    "THEME",
    "biome_color",
    "hsl_to_qcolor",
    "organism_color",
]


@dataclass(frozen=True)
class Theme:
    # Window chrome
    bg: QColor
    panel_bg: QColor
    panel_bg_alt: QColor
    border: QColor
    text: QColor
    text_dim: QColor
    text_bright: QColor
    accent: QColor
    accent_dim: QColor
    danger: QColor
    success: QColor
    warning: QColor
    # World overlay
    grid_line: QColor
    selection: QColor
    night_overlay: QColor
    # Charts
    chart_bg: QColor
    chart_grid: QColor
    chart_line_pop: QColor
    chart_line_species: QColor
    chart_line_energy: QColor
    chart_line_temp: QColor


def _build_theme() -> Theme:
    p = QPalette()
    return Theme(
        bg=QColor("#0E1116"),
        panel_bg=QColor("#161A22"),
        panel_bg_alt=QColor("#1E232D"),
        border=QColor("#2A2F3A"),
        text=QColor("#D7DCE5"),
        text_dim=QColor("#7E8593"),
        text_bright=QColor("#F4F6FA"),
        accent=QColor("#56B4E9"),
        accent_dim=QColor("#2C6F95"),
        danger=QColor("#E06C75"),
        success=QColor("#7FB069"),
        warning=QColor("#E5C07B"),
        grid_line=QColor(255, 255, 255, 18),
        selection=QColor(255, 220, 80, 220),
        night_overlay=QColor(10, 18, 40, 90),
        chart_bg=QColor("#11151C"),
        chart_grid=QColor(255, 255, 255, 18),
        chart_line_pop=QColor("#56B4E9"),
        chart_line_species=QColor("#E5C07B"),
        chart_line_energy=QColor("#7FB069"),
        chart_line_temp=QColor("#E06C75"),
    )


THEME = _build_theme()


# ---------------------------------------------------------------------------
# Biome colours
# ---------------------------------------------------------------------------
from ..core.world import Biome

_BIOME_COLORS = {
    Biome.DEEP_WATER:    QColor("#1A3A5C"),
    Biome.SHALLOW_WATER: QColor("#2E6F8E"),
    Biome.BEACH:         QColor("#D9C28A"),
    Biome.GRASSLAND:     QColor("#5C8A3A"),
    Biome.FOREST:        QColor("#2F5F2A"),
    Biome.JUNGLE:        QColor("#1F4A1F"),
    Biome.SAVANNA:       QColor("#A39744"),
    Biome.DESERT:        QColor("#D4A95E"),
    Biome.ROCK:          QColor("#6E6E72"),
    Biome.SNOW:          QColor("#E4ECF2"),
    Biome.TUNDRA:        QColor("#8B98A6"),
}


def biome_color(b: Biome) -> QColor:
    return _BIOME_COLORS[b]


# ---------------------------------------------------------------------------
# HSL helpers
# ---------------------------------------------------------------------------
def hsl_to_qcolor(h: float, s: float, l: float, alpha: int = 255) -> QColor:
    """h in degrees, s & l in 0..1."""
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, l, s)
    c = QColor(int(r * 255), int(g * 255), int(b * 255))
    c.setAlpha(alpha)
    return c


def organism_color(hue: float, sat: float, light: float, alpha: int = 255) -> QColor:
    return hsl_to_qcolor(hue, sat, light, alpha)
