"""PySide6 native desktop GUI for Project Ecosystem."""

from .main_window import MainWindow
from .world_view import WorldCanvas
from .panels import (
    ControlPanel, StatsPanel, SpeciesPanel, InspectorPanel,
    JournalPanel, EnvironmentPanel,
)
from .theme import THEME

__all__ = [
    "MainWindow",
    "WorldCanvas",
    "ControlPanel",
    "StatsPanel",
    "SpeciesPanel",
    "InspectorPanel",
    "JournalPanel",
    "EnvironmentPanel",
    "THEME",
]
