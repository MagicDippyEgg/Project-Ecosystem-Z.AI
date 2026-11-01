"""Tests for organisms: behaviour, metabolism, reproduction, death."""

from __future__ import annotations

import math
import random

import pytest

from ecosystem.core.genome import Genome, random_genome, recombine
from ecosystem.core.organism import (
    Organism, OrganismState, BirthRequest, STATE_LABELS,
)
from ecosystem.core.world import World
from ecosystem.core.weather import WeatherSystem
from ecosystem.core.species import SpeciesRegistry


def make_test_world(width=24, height=16, seed=42):
    rng = random.Random(seed)
    return World.generate(width=width, height=height, seed=seed, rng=rng), rng


def make_test_organism(oid=1, x=8.0, y=8.0, overrides=None):
    rng = random.Random(oid * 7)
    g = random_genome(rng)
    if overrides:
        data = g.as_dict()
        data.update(overrides)
        g = Genome(data)
    return Organism(
        id=oid, genome=g, x=x, y=y, species_id=1,
        birth_tick=0, energy=80.0, hydration=80.0,
    )


def test_organism_initial_state():
    org = make_test_organism()
    assert org.alive
    assert org.age == 0.0
    assert org.energy > 0.0
    assert org.state == OrganismState.IDLE


def test_organism_derived_stats_consistent():
    org = make_test_organism(overrides={"size": 1.5})
    # max_energy = 60 + 50 * size = 60 + 75 = 135
    assert org.max_energy == pytest.approx(60.0 + 50.0 * 1.5)
    # radius = 0.18 + 0.22 * size
    assert org.radius == pytest.approx(0.18 + 0.22 * 1.5)


def test_organism_diet_classification():
    herb = make_test_organism(overrides={"diet": 0.1})
    carn = make_test_organism(oid=2, overrides={"diet": 0.9})
    omni = make_test_organism(oid=3, overrides={"diet": 0.5})
    assert herb.is_herbivore and not herb.is_predator
    assert carn.is_predator and not carn.is_herbivore
    assert omni.is_omnivore


def test_organism_matures_with_age():
    org = make_test_organism(overrides={"maturity_age": 30})
    assert not org.is_mature
    org.age = 30
    assert org.is_mature
    org.age = 60
    assert org.is_mature


def test_organism_starves_to_death():
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    org = make_test_organism()
    org.energy = 0.0  # exactly zero -> metabolism pushes negative -> clamped to 0 -> death
    # Clear all nearby food so it can't graze to survive.
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            t = world.tile_at(int(org.x) + dx, int(org.y) + dy)
            if t is not None:
                t.food = 0.0
    org.step(world, weather, rng, 1, [])
    assert not org.alive
    assert org.death_cause == "starvation"


def test_organism_dies_of_dehydration():
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    org = make_test_organism()
    org.hydration = 0.1
    org.energy = 90.0
    org.step(world, weather, rng, 1, [])
    assert not org.alive
    assert org.death_cause == "dehydration"


def test_organism_dies_of_old_age():
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    # Use a max_age within the trait's valid range (200-1000).
    org = make_test_organism(overrides={"max_age": 200.0})
    org.age = 200.0
    org.energy = 90.0
    org.hydration = 90.0
    org.step(world, weather, rng, 1, [])
    assert not org.alive
    assert org.death_cause == "old age"


def test_organism_forages_when_hungry():
    """A hungry herbivore on a food tile should eat."""
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    org = make_test_organism(overrides={"diet": 0.0, "vision_range": 5.0})
    # Place on a grassland tile with food.
    grass = next(t for t in world.tiles if t.biome.name == "GRASSLAND")
    grass.food = 20.0
    org.x = grass.x + 0.5
    org.y = grass.y + 0.5
    org.energy = 30.0  # hungry
    energy_before = org.energy
    org.step(world, weather, rng, 1, [])
    assert org.energy >= energy_before
    assert org.state == OrganismState.FORAGING


def test_organism_reproduces_with_mate():
    """Two mature, well-fed same-species organisms should produce offspring."""
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    g = random_genome(rng)
    parent_a = Organism(id=1, genome=g, x=8.0, y=8.0, species_id=1,
                        birth_tick=0, energy=90.0, hydration=80.0)
    parent_b = Organism(id=2, genome=g, x=8.5, y=8.5, species_id=1,
                        birth_tick=0, energy=90.0, hydration=80.0)
    # Make them mature.
    parent_a.age = parent_a.genome["maturity_age"] + 1
    parent_b.age = parent_b.genome["maturity_age"] + 1
    req = parent_a.step(world, weather, rng, 1, [parent_b])
    # Should produce a birth request (or at least move toward the mate).
    # Since they're adjacent, reproduction should fire.
    assert req is not None or parent_a.state == OrganismState.MATING
    if req is not None:
        assert isinstance(req, BirthRequest)
        assert req.parent_ids == (1, 2)
        # Energy was deducted from both parents.
        assert parent_a.energy < 90.0
        assert parent_b.energy < 90.0
        # Cooldowns set.
        assert parent_a.cooldown_repro > 0
        assert parent_b.cooldown_repro > 0


def test_organism_predator_attacks_prey():
    """A predator adjacent to smaller prey should attack."""
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    predator = make_test_organism(
        oid=1, overrides={"diet": 1.0, "size": 1.5, "aggression": 0.9})
    prey = make_test_organism(
        oid=2, x=8.5, y=8.5, overrides={"diet": 0.0, "size": 0.8})
    prey.species_id = 99  # different species
    predator.species_id = 1
    predator.energy = 20.0  # hungry
    prey_health_before = prey.health
    predator.step(world, weather, rng, 1, [prey])
    # Predator should be hunting.
    assert predator.state == OrganismState.HUNTING
    # Prey should have lost health.
    assert prey.health < prey_health_before


def test_organism_flees_predator():
    """A prey near a predator should flee."""
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    predator = make_test_organism(
        oid=1, x=10.0, y=10.0,
        overrides={"diet": 1.0, "size": 1.5, "aggression": 0.9})
    predator.species_id = 1
    prey = make_test_organism(
        oid=2, x=9.0, y=9.0,
        overrides={"diet": 0.0, "size": 0.8, "vision_range": 12.0})
    prey.species_id = 99
    prey.energy = 50.0
    pos_before = (prey.x, prey.y)
    prey.step(world, weather, rng, 1, [predator])
    # Prey should be fleeing and have moved away.
    assert prey.state == OrganismState.FLEEING
    moved = math.hypot(prey.x - pos_before[0], prey.y - pos_before[1])
    assert moved > 0.0


def test_organism_to_dict_roundtrip():
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    org = make_test_organism()
    # Run a few ticks to populate events.
    for t in range(5):
        org.step(world, weather, rng, t, [])
    data = org.to_dict()
    org2 = Organism.from_dict(data)
    assert org2.id == org.id
    assert org2.x == pytest.approx(org.x)
    assert org2.y == pytest.approx(org.y)
    assert org2.energy == pytest.approx(org.energy)
    assert org2.genome == org.genome
    assert org2.alive == org.alive
    assert len(org2.events) == len(org.events)


def test_organism_does_not_attack_same_species():
    """Predators shouldn't cannibalise their own species."""
    world, rng = make_test_world()
    weather = WeatherSystem(rng=rng)
    g = random_genome(rng)
    a = Organism(id=1, genome=Genome({**g.as_dict(), "diet": 1.0, "size": 1.5}),
                 x=8.0, y=8.0, species_id=7, birth_tick=0, energy=20.0)
    b = Organism(id=2, genome=Genome({**g.as_dict(), "diet": 0.0, "size": 0.8}),
                 x=8.5, y=8.5, species_id=7, birth_tick=0, energy=80.0)
    # Same species - a should NOT consider b edible.
    assert not a._is_edible_to(b)
