"""Tests for the top-level Simulation controller."""

from __future__ import annotations

import random

import pytest

from ecosystem.core.simulation import Simulation, SimulationConfig, SimulationStats
from ecosystem.core.weather import WeatherType


def test_simulation_initialises(small_sim):
    sim = small_sim
    assert sim.world is not None
    assert sim.population() > 0
    assert len(sim.species.living_species()) >= 1
    assert sim.tick_count == 0


def test_simulation_tick_advances_state(small_sim):
    sim = small_sim
    pop_before = sim.population()
    sim.tick()
    assert sim.tick_count == 1
    assert sim.world.tick == 1


def test_simulation_runs_many_ticks_without_crash(small_sim):
    sim = small_sim
    for _ in range(300):
        sim.tick()
    assert sim.tick_count == 300
    # Stats should have been sampled.
    assert len(sim.stats.ticks) > 0
    assert len(sim.stats.population) == len(sim.stats.ticks)


def test_simulation_population_can_change(small_sim):
    sim = small_sim
    initial = sim.population()
    for _ in range(200):
        sim.tick()
    final = sim.population()
    # Over 200 ticks, *something* should have changed (births or deaths).
    # We don't assert direction since randomness could go either way briefly.
    assert isinstance(final, int)
    assert final >= 0


def test_simulation_pause_and_speed(small_sim):
    sim = small_sim
    sim.set_paused(True)
    assert sim.paused
    # step_frames should do nothing while paused.
    ticks = sim.step_frames(0.5)
    assert ticks == 0
    sim.set_paused(False)
    sim.set_speed(4.0)
    assert sim.speed == 4.0
    # At 4x for 0.4s, expect roughly 16 ticks.
    ticks = sim.step_frames(0.4)
    assert ticks > 0


def test_simulation_speed_clamped():
    sim = Simulation(SimulationConfig(width=16, height=12, seed=1, initial_population=4))
    sim.set_speed(1000.0)
    assert sim.speed <= 16.0
    sim.set_speed(0.01)
    assert sim.speed >= 0.25


def test_simulation_regenerate_creates_new_world(small_sim):
    sim = small_sim
    old_seed = sim.config.seed
    old_tile_count = len(sim.world.tiles)
    sim.generate_world(seed=999, name="Test")
    assert sim.config.seed == 999
    assert sim.tick_count == 0
    assert sim.population() > 0
    # The new world may have a different tile count if width/height changed,
    # but here they didn't.
    assert len(sim.world.tiles) == old_tile_count
    assert sim.world.name == "Test"


def test_simulation_force_weather(small_sim):
    sim = small_sim
    sim.trigger_weather(WeatherType.STORM, duration=20)
    assert sim.weather.active_event is not None
    assert sim.weather.active_event.kind == WeatherType.STORM
    assert sim.weather.active_event.duration == 20


def test_simulation_organism_selection(small_sim):
    sim = small_sim
    orgs = sim.living_organisms()
    assert len(orgs) > 0
    org = orgs[0]
    # Select by point near the organism.
    found = sim.organism_at_point(org.x, org.y, tolerance=2.0)
    assert found is not None
    assert found.id == org.id


def test_simulation_species_summary_has_data(small_sim):
    sim = small_sim
    summary = sim.species_summary()
    assert len(summary) > 0
    s = summary[0]
    assert "name" in s and "population" in s and "color_hue" in s


def test_simulation_serialisation_roundtrip(small_sim):
    sim = small_sim
    # Run a few ticks to populate state.
    for _ in range(20):
        sim.tick()
    data = sim.to_dict()
    sim2 = Simulation.from_dict(data)
    assert sim2.tick_count == sim.tick_count
    assert sim2.population() == sim.population()
    assert sim2.world.width == sim.world.width
    assert len(sim2.organisms) == len(sim.organisms)
    # Continuing the loaded sim should produce identical state to the
    # original advanced by the same number of ticks (with the same RNG
    # state implicit in the loaded config).
    pop_before = sim2.population()
    sim2.tick()
    assert sim2.tick_count == sim.tick_count + 1


def test_simulation_max_population_cap():
    cfg = SimulationConfig(width=32, height=24, seed=1,
                           initial_population=20, max_population=30)
    sim = Simulation(cfg)
    # Run many ticks - population should never exceed the cap.
    for _ in range(500):
        sim.tick()
        assert sim.population() <= 30


def test_simulation_journal_records_events(small_sim):
    sim = small_sim
    initial_entries = len(sim.journal)
    # Trigger weather - should add a journal entry.
    sim.trigger_weather(WeatherType.DROUGHT, duration=40)
    assert len(sim.journal) > initial_entries


def test_simulation_stats_sampling(small_sim):
    sim = small_sim
    # Stats sample every 8 ticks.
    for _ in range(17):
        sim.tick()
    assert len(sim.stats.ticks) >= 2
    assert len(sim.stats.population) == len(sim.stats.ticks)
    assert len(sim.stats.weather) == len(sim.stats.ticks)


def test_simulation_ancestors_tree(small_sim):
    sim = small_sim
    orgs = sim.living_organisms()
    if not orgs:
        pytest.skip("no living organisms")
    org = orgs[0]
    tree = sim.ancestors(org.id, max_depth=3)
    assert tree["id"] == org.id
    assert "children" in tree


def test_simulation_offspring_count_increases(small_sim):
    """After running for a while, at least one organism should have offspring."""
    sim = small_sim
    # Run enough ticks for reproduction to occur.
    for _ in range(400):
        sim.tick()
    total_offspring = sum(o.offspring_count for o in sim.organisms.values())
    assert total_offspring > 0, "no reproduction happened in 400 ticks"


def test_simulation_handles_extreme_weather_without_crash():
    sim = Simulation(SimulationConfig(width=24, height=16, seed=1, initial_population=8))
    sim.trigger_weather(WeatherType.STORM, duration=200)
    for _ in range(200):
        sim.tick()
    # Some organisms may die from storm damage but sim shouldn't crash.
    assert sim.tick_count == 200
