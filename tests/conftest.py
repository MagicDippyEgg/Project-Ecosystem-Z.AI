"""Pytest configuration & shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `import ecosystem` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Headless GUI testing: force the offscreen QPA platform before any Qt
# import happens, *unless* the caller has explicitly chosen a platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def rng_seed() -> int:
    return 12345


@pytest.fixture
def small_sim():
    """A tiny, fast-running simulation useful for many tests."""
    from ecosystem.core import Simulation, SimulationConfig
    cfg = SimulationConfig(
        width=32, height=24, seed=12345,
        initial_population=10, initial_species_count=2,
        max_population=120,
    )
    return Simulation(cfg)
