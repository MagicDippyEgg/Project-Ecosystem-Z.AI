"""
Project Ecosystem - Artificial Life Simulation.

A self-sustaining simulated ecosystem where organisms live, interact,
reproduce, compete for resources and evolve over time.

This package is split into three layers:

* ``ecosystem.core``      - pure-Python simulation backend (no GUI deps).
* ``ecosystem.persistence`` - JSON save/load.
* ``ecosystem.gui``       - PySide6 native desktop front-end.

Importing the top-level package is cheap and side-effect free.
"""

__version__ = "1.0.0"
__all__ = ["core", "persistence", "gui", "__version__"]
