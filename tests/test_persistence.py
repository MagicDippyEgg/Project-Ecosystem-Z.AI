"""Tests for the persistence layer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from ecosystem.core.simulation import Simulation, SimulationConfig
from ecosystem.persistence import (
    save_simulation, load_simulation, SaveManager, SAVE_FORMAT_VERSION,
)


def test_save_and_load_roundtrip(tmp_path):
    sim = Simulation(SimulationConfig(width=24, height=16, seed=42, initial_population=8))
    for _ in range(15):
        sim.tick()
    path = tmp_path / "test.json"
    save_simulation(sim, path, name="TestRun")
    assert path.exists()
    assert path.stat().st_size > 0
    sim2 = load_simulation(path)
    assert sim2.tick_count == sim.tick_count
    assert sim2.population() == sim.population()
    assert sim2.world.width == sim.world.width
    assert len(sim2.organisms) == len(sim.organisms)


def test_save_file_is_valid_json(tmp_path):
    sim = Simulation(SimulationConfig(width=16, height=12, seed=1, initial_population=4))
    path = tmp_path / "valid.json"
    save_simulation(sim, path)
    # Should be parseable as JSON.
    with path.open() as f:
        payload = json.load(f)
    assert payload["format_version"] == SAVE_FORMAT_VERSION
    assert "simulation" in payload
    assert "saved_at" in payload


def test_save_load_preserves_organisms(tmp_path):
    sim = Simulation(SimulationConfig(width=24, height=16, seed=2, initial_population=10))
    for _ in range(20):
        sim.tick()
    path = tmp_path / "orgs.json"
    save_simulation(sim, path)
    sim2 = load_simulation(path)
    # Each living organism in sim should be present in sim2 with same id/genome.
    for oid, org in sim.organisms.items():
        org2 = sim2.organisms.get(oid)
        assert org2 is not None
        assert org2.genome == org.genome
        assert org2.x == pytest.approx(org.x)
        assert org2.y == pytest.approx(org.y)
        assert org2.alive == org.alive


def test_save_manager_list_and_delete(tmp_path):
    sim = Simulation(SimulationConfig(width=16, height=12, seed=1, initial_population=4))
    mgr = SaveManager(base_dir=tmp_path)
    mgr.save(sim, "First World")
    mgr.save(sim, "Second World")
    slots = mgr.list_saves()
    assert len(slots) == 2
    names = {s.name for s in slots}
    assert names == {"First World", "Second World"}
    # Delete one.
    assert mgr.delete("First World")
    slots = mgr.list_saves()
    assert len(slots) == 1
    assert slots[0].name == "Second World"


def test_save_manager_load_by_name(tmp_path):
    sim = Simulation(SimulationConfig(width=16, height=12, seed=1, initial_population=4))
    sim.tick(); sim.tick()
    mgr = SaveManager(base_dir=tmp_path)
    mgr.save(sim, "MyWorld")
    sim2 = mgr.load("MyWorld")
    assert sim2.tick_count == sim.tick_count


def test_save_manager_safe_name():
    assert SaveManager._safe_name("Hello World!") == "Hello_World_"
    assert SaveManager._safe_name("../../etc/passwd") == "______etc_passwd"
    assert SaveManager._safe_name("") == "save"


def test_load_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_simulation(tmp_path / "does_not_exist.json")


def test_save_atomic_no_partial_files_on_error(tmp_path, monkeypatch):
    """If save_simulation fails mid-write, no half-written file should remain."""
    sim = Simulation(SimulationConfig(width=16, height=12, seed=1, initial_population=4))
    path = tmp_path / "atomic.json"
    # Patch os.replace to raise.
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        save_simulation(sim, path)
    # The target file should not exist; the temp file should be cleaned up.
    assert not path.exists()
    # No leftover .tmp files.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert len(leftovers) == 0


def test_save_load_preserves_species(tmp_path):
    sim = Simulation(SimulationConfig(width=24, height=16, seed=3,
                                      initial_population=12, initial_species_count=3))
    for _ in range(40):
        sim.tick()
    path = tmp_path / "species.json"
    save_simulation(sim, path)
    sim2 = load_simulation(path)
    assert len(sim2.species.all_species()) == len(sim.species.all_species())
    for sp_id in sim.species._species:
        sp1 = sim.species.get(sp_id)
        sp2 = sim2.species.get(sp_id)
        assert sp2 is not None
        assert sp2.name == sp1.name
        assert sp2.population == sp1.population
        assert sp2.peak_population == sp1.peak_population


def test_save_load_preserves_journal(tmp_path):
    sim = Simulation(SimulationConfig(width=24, height=16, seed=5, initial_population=8))
    sim.trigger_weather(__import__("ecosystem.core.weather", fromlist=["WeatherType"]).WeatherType.STORM, 30)
    for _ in range(40):
        sim.tick()
    path = tmp_path / "journal.json"
    save_simulation(sim, path)
    sim2 = load_simulation(path)
    assert len(sim2.journal) == len(sim.journal)
    for e1, e2 in zip(sim.journal.entries, sim2.journal.entries):
        assert e1.tick == e2.tick
        assert e1.text == e2.text
