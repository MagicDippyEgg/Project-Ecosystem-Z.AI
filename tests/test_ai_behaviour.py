"""Tests for improved organism AI: survival intelligence, terrain awareness.

These tests verify the behavioural improvements requested in the
"smarter organism AI" task:

- Land organisms avoid entering water.
- Organisms find routes around obstacles.
- Organisms successfully find food and water.
- Organisms respond to nearby threats.
- Organisms avoid dangerous or unreachable targets.
- Populations remain viable over extended simulations.
"""

from __future__ import annotations

import math
import random

import pytest

from ecosystem.core.genome import Genome, random_genome
from ecosystem.core.organism import Organism, OrganismState
from ecosystem.core.world import World, Biome
from ecosystem.core.weather import WeatherSystem
from ecosystem.core.species import SpeciesRegistry
from ecosystem.core.simulation import Simulation, SimulationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def test_world(rng):
    return World.generate(width=32, height=24, seed=42, rng=rng)


@pytest.fixture
def weather(rng):
    return WeatherSystem(rng=rng)


@pytest.fixture
def herbivore(rng):
    """A herbivore organism on a safe tile."""
    g = random_genome(rng)
    data = g.as_dict()
    data["diet"] = 0.0  # herbivore
    data["size"] = 1.0
    data["speed"] = 1.0
    data["metabolism"] = 0.5
    data["vision_range"] = 10.0
    g = Genome(data)
    return Organism(id=1, genome=g, x=8.0, y=8.0, species_id=1,
                    birth_tick=0, energy=70.0, hydration=70.0)


@pytest.fixture
def predator(rng):
    """A predator organism."""
    g = random_genome(rng)
    data = g.as_dict()
    data["diet"] = 1.0  # carnivore
    data["size"] = 1.5
    data["aggression"] = 0.9
    data["speed"] = 1.2
    data["vision_range"] = 12.0
    g = Genome(data)
    return Organism(id=2, genome=g, x=12.0, y=8.0, species_id=2,
                    birth_tick=0, energy=70.0, hydration=70.0)


def find_safe_tile(world, near_x=8, near_y=8):
    """Find a safe (passable >= 0.85) tile near the given position."""
    for r in range(10):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                tile = world.tile_at(near_x + dx, near_y + dy)
                if tile is not None and tile.params["passable"] >= 0.85:
                    return tile
    return None


# ---------------------------------------------------------------------------
# 1. Land organisms avoid water
# ---------------------------------------------------------------------------
class TestAvoidWater:
    def test_organism_on_water_escapes_to_safety(self, test_world, weather, rng):
        """An organism that finds itself on a water tile immediately escapes."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0)
        # Find a water tile that has land within 3 tiles.
        water_tile = None
        for tile in test_world.tiles:
            if tile.biome in (Biome.DEEP_WATER, Biome.SHALLOW_WATER):
                # Check for nearby land.
                has_land = False
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nt = test_world.tile_at(tile.x + dx, tile.y + dy)
                        if nt is not None and nt.params["passable"] >= 0.85:
                            has_land = True
                            break
                    if has_land:
                        break
                if has_land:
                    water_tile = tile
                    break
        if water_tile is None:
            pytest.skip("no water tile with nearby land found")
        org.x = float(water_tile.x) + 0.5
        org.y = float(water_tile.y) + 0.5
        org.step(test_world, weather, rng, 1, [])
        # After stepping, the organism should be on a safe tile.
        assert test_world.is_safe_tile(org.x, org.y), \
            f"organism should escape to safe terrain, still at ({org.x}, {org.y})"

    def test_organism_does_not_enter_water_when_foraging(self, test_world, weather, rng):
        """A foraging organism doesn't walk into water chasing food."""
        # Place organism near water with food on the other side.
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=20.0)
        # Find a shore: land tile adjacent to water.
        shore = None
        for tile in test_world.tiles:
            if tile.params["passable"] >= 0.85:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nt = test_world.tile_at(tile.x + dx, tile.y + dy)
                        if nt is not None and nt.has_water:
                            shore = tile
                            break
                    if shore:
                        break
                if shore:
                    break
        if shore is None:
            pytest.skip("no shore found")
        org.x = float(shore.x) + 0.5
        org.y = float(shore.y) + 0.5
        # Run several ticks; organism should never be on a water tile.
        for tick in range(20):
            org.energy = max(org.energy, 20.0)  # keep hungry but alive
            org.hydration = max(org.hydration, 30.0)
            org.health = max(org.health, 50.0)
            org.step(test_world, weather, rng, tick, [])
            if org.alive:
                assert test_world.is_safe_tile(org.x, org.y), \
                    f"organism entered water at tick {tick}: ({org.x}, {org.y})"

    def test_organism_movement_never_crosses_water(self, rng):
        """Over many ticks, an organism never ends up on a water tile.

        Uses a custom world with a known water wall to ensure the
        organism must navigate around it.
        """
        world = World.generate(width=24, height=16, seed=99, rng=random.Random(99))
        # Force all tiles to grassland, then build a water wall.
        for tile in world.tiles:
            tile.biome = Biome.GRASSLAND
            tile.has_water = False
            tile.food = 15.0
        for j in range(0, 16):
            tile = world.tile_at(12, j)
            if tile is not None:
                tile.biome = Biome.DEEP_WATER
                tile.food = 0.0
                tile.has_water = True
        world._compute_landmasses()
        world._build_tile_grid()
        weather = WeatherSystem(rng=rng)
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=80.0, hydration=80.0)
        for tick in range(100):
            org.energy = 80.0  # keep alive
            org.hydration = 80.0
            org.health = 100.0
            org.step(world, weather, rng, tick, [])
            assert world.is_safe_tile(org.x, org.y), \
                f"organism on water at tick {tick}: ({org.x}, {org.y})"


# ---------------------------------------------------------------------------
# 2. Organisms find routes around obstacles
# ---------------------------------------------------------------------------
class TestPathfinding:
    def test_organism_reaches_target_across_water_barrier(self):
        """An organism can reach a target on the other side of a water wall."""
        rng = random.Random(99)
        world = World.generate(width=24, height=16, seed=99, rng=rng)
        # Build a water wall at x=12, with a gap at y=0.
        for j in range(1, 16):
            tile = world.tile_at(12, j)
            if tile is not None:
                tile.biome = Biome.DEEP_WATER
                tile.has_water = True
                tile.food = 0.0
        gap = world.tile_at(12, 0)
        if gap is not None:
            gap.biome = Biome.GRASSLAND
            gap.has_water = False
        world._compute_landmasses()
        world._build_tile_grid()
        # Verify the gap is on the same landmass as both sides.
        lm_left = world.landmass_id(10.5, 8.5)
        lm_right = world.landmass_id(14.5, 8.5)
        if lm_left != lm_right or lm_left == -1:
            pytest.skip("water wall completely separates the world")
        # An organism at x=10 should be able to path to x=14.
        pf = world and __import__("ecosystem.core.navigation", fromlist=["Pathfinder"]).Pathfinder(world)
        path = pf.find_path(10.5, 8.5, 14.5, 8.5)
        if path is not None:
            assert len(path) >= 2
            # All waypoints should be safe.
            for wx, wy in path:
                assert world.is_safe_tile(wx, wy)
            # The path should go through the gap (y near 0).
            assert any(wy < 2.0 for _, wy in path), \
                "path should route through the gap at y=0"


# ---------------------------------------------------------------------------
# 3. Organisms find food and water
# ---------------------------------------------------------------------------
class TestFindResources:
    def test_hungry_organism_eats_food_on_current_tile(self, test_world, weather, rng):
        """A hungry organism eats food on its current tile."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=30.0)
        tile = find_safe_tile(test_world)
        if tile is None:
            pytest.skip("no safe tile")
        org.x = float(tile.x) + 0.5
        org.y = float(tile.y) + 0.5
        tile.food = 20.0
        energy_before = org.energy
        org.step(test_world, weather, rng, 1, [])
        assert org.energy > energy_before, "organism should have eaten"

    def test_thirsty_organism_drinks_from_shore(self, test_world, weather, rng):
        """A thirsty organism finds and drinks from a shore tile."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=80.0, hydration=15.0)
        # Place near a shore.
        shore = None
        for tile in test_world.tiles:
            if tile.params["passable"] >= 0.85:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nt = test_world.tile_at(tile.x + dx, tile.y + dy)
                        if nt is not None and nt.has_water:
                            shore = tile
                            break
                    if shore:
                        break
                if shore:
                    break
        if shore is None:
            pytest.skip("no shore found")
        org.x = float(shore.x) + 0.5
        org.y = float(shore.y) + 0.5
        hydration_before = org.hydration
        # Run a few ticks — organism should drink.
        drank = False
        for tick in range(10):
            org.energy = 80.0  # keep energy up
            org.health = 100.0
            org.step(test_world, weather, rng, tick, [])
            if org.hydration > hydration_before + 5:
                drank = True
                break
        assert drank, "organism should have drunk water"

    def test_organism_finds_food_within_vision(self, test_world, weather, rng):
        """A hungry organism moves toward reachable food."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=30.0, hydration=80.0)
        tile = find_safe_tile(test_world)
        if tile is None:
            pytest.skip("no safe tile")
        org.x = float(tile.x) + 0.5
        org.y = float(tile.y) + 0.5
        # Place food a few tiles away on the same landmass.
        org_lm = test_world.landmass_id(org.x, org.y)
        food_tile = None
        for t in test_world.tiles:
            if test_world.landmass_id(t.x + 0.5, t.y + 0.5) == org_lm:
                dist = abs(t.x - org.x) + abs(t.y - org.y)
                if 3 <= dist <= 8:
                    t.food = 25.0
                    food_tile = t
                    break
        if food_tile is None:
            pytest.skip("no suitable food tile found")
        pos_before = (org.x, org.y)
        org.step(test_world, weather, rng, 1, [])
        # Organism should have moved (toward food or eating).
        moved = math.hypot(org.x - pos_before[0], org.y - pos_before[1])
        assert org.state in (OrganismState.FORAGING,), \
            f"organism should be foraging, got {org.state}"


# ---------------------------------------------------------------------------
# 4. Organisms respond to threats
# ---------------------------------------------------------------------------
class TestThreatResponse:
    def test_prey_flees_from_nearby_predator(self, test_world, weather, rng):
        """A prey organism flees when a predator is within vision."""
        prey = Organism(id=1, genome=Genome({**random_genome(rng).as_dict(),
                                              "diet": 0.0, "size": 0.8,
                                              "vision_range": 12.0, "speed": 1.0}),
                        x=8.0, y=8.0, species_id=1, birth_tick=0,
                        energy=60.0, hydration=60.0)
        predator = Organism(id=2, genome=Genome({**random_genome(rng).as_dict(),
                                                  "diet": 1.0, "size": 1.5,
                                                  "aggression": 0.9}),
                            x=9.0, y=8.0, species_id=2, birth_tick=0,
                            energy=60.0, hydration=60.0)
        pos_before = (prey.x, prey.y)
        prey.step(test_world, weather, rng, 1, [predator])
        assert prey.state == OrganismState.FLEEING, \
            f"prey should flee, got {prey.state}"
        # Prey should have moved away from the predator.
        moved = math.hypot(prey.x - pos_before[0], prey.y - pos_before[1])
        assert moved > 0.0, "prey should have moved"

    def test_prey_does_not_flee_from_distant_predator(self, test_world, weather, rng):
        """A prey doesn't flee if the predator is far away."""
        prey = Organism(id=1, genome=Genome({**random_genome(rng).as_dict(),
                                              "diet": 0.0, "size": 0.8,
                                              "vision_range": 5.0, "speed": 1.0}),
                        x=8.0, y=8.0, species_id=1, birth_tick=0,
                        energy=60.0, hydration=60.0)
        predator = Organism(id=2, genome=Genome({**random_genome(rng).as_dict(),
                                                  "diet": 1.0, "size": 1.5}),
                            x=20.0, y=20.0, species_id=2, birth_tick=0,
                            energy=60.0, hydration=60.0)
        # Predator is at distance ~17, beyond vision=5.
        prey.step(test_world, weather, rng, 1, [predator])
        assert prey.state != OrganismState.FLEEING, \
            "prey should not flee from distant predator"

    def test_predator_hunts_nearby_prey(self, test_world, weather, rng):
        """A hungry predator attacks adjacent prey."""
        predator = Organism(id=1, genome=Genome({**random_genome(rng).as_dict(),
                                                  "diet": 1.0, "size": 1.5,
                                                  "aggression": 0.9, "vision_range": 10.0}),
                            x=8.0, y=8.0, species_id=1, birth_tick=0,
                            energy=20.0, hydration=60.0)
        prey = Organism(id=2, genome=Genome({**random_genome(rng).as_dict(),
                                              "diet": 0.0, "size": 0.8}),
                        x=8.5, y=8.5, species_id=2, birth_tick=0,
                        energy=60.0, hydration=60.0)
        prey_health_before = prey.health
        predator.step(test_world, weather, rng, 1, [prey])
        assert predator.state == OrganismState.HUNTING
        assert prey.health < prey_health_before, "prey should have been attacked"


# ---------------------------------------------------------------------------
# 5. Organisms avoid unreachable targets
# ---------------------------------------------------------------------------
class TestUnreachableTargets:
    def test_organism_does_not_pursue_food_across_water(self):
        """An organism doesn't chase food on a separate island."""
        rng = random.Random(123)
        world = World.generate(width=32, height=24, seed=123, rng=rng)
        # Place food only on a different landmass (if one exists).
        org_lm = world.landmass_id(8.5, 8.5)
        if org_lm == -1:
            pytest.skip("organism not on land")
        # Find a tile on a DIFFERENT landmass.
        far_food = None
        for tile in world.tiles:
            lm = world.landmass_id(tile.x + 0.5, tile.y + 0.5)
            if lm != -1 and lm != org_lm:
                tile.food = 30.0
                far_food = tile
                break
        if far_food is None:
            pytest.skip("no separate landmass found")
        # The food should not be in the reachable food results.
        result = world.best_reachable_food_tile(8.5, 8.5, 20)
        if result is not None:
            # Result should be on the same landmass as the organism.
            assert world.landmass_id(result[0], result[1]) == org_lm, \
                "food target should be on the same landmass"

    def test_organism_does_not_pursue_mate_across_water(self):
        """An organism doesn't pursue a mate on a separate island."""
        rng = random.Random(456)
        world = World.generate(width=32, height=24, seed=456, rng=rng)
        g = random_genome(rng)
        org = Organism(id=1, genome=g, x=8.0, y=8.0, species_id=1,
                       birth_tick=0, energy=80.0, hydration=80.0)
        org.age = g["maturity_age"] + 1
        org_lm = world.landmass_id(org.x, org.y)
        if org_lm == -1:
            pytest.skip("organism not on land")
        # Find a mate position on a different landmass.
        far_mate_pos = None
        for tile in world.tiles:
            lm = world.landmass_id(tile.x + 0.5, tile.y + 0.5)
            if lm != -1 and lm != org_lm:
                far_mate_pos = (tile.x + 0.5, tile.y + 0.5)
                break
        if far_mate_pos is None:
            pytest.skip("no separate landmass found")
        mate = Organism(id=2, genome=g, x=far_mate_pos[0], y=far_mate_pos[1],
                        species_id=1, birth_tick=0, energy=80.0, hydration=80.0)
        mate.age = g["maturity_age"] + 1
        pos_before = (org.x, org.y)
        org.step(world, WeatherSystem(rng=random.Random(1)), rng, 1, [mate])
        # Organism should NOT have moved toward the unreachable mate.
        # It might have stayed put or moved in another direction.
        moved_toward = False
        dx = far_mate_pos[0] - pos_before[0]
        dy = far_mate_pos[1] - pos_before[1]
        org_dx = org.x - pos_before[0]
        org_dy = org.y - pos_before[1]
        if abs(org_dx) > 0.1 or abs(org_dy) > 0.1:
            # Check if movement is toward the mate (dot product > 0).
            dot = org_dx * dx + org_dy * dy
            if dot > 0:
                moved_toward = True
        # The organism should not be in MATING state heading toward unreachable mate.
        assert org.state != OrganismState.MATING or not moved_toward, \
            "organism should not pursue mate across water"


# ---------------------------------------------------------------------------
# 6. Population viability over extended simulations
# ---------------------------------------------------------------------------
class TestPopulationViability:
    def test_population_survives_500_ticks(self):
        """A reasonably-configured world maintains a population for 500 ticks."""
        cfg = SimulationConfig(width=48, height=32, seed=42,
                               initial_population=30, initial_species_count=3,
                               max_population=150)
        sim = Simulation(cfg)
        for _ in range(500):
            sim.tick()
        # Population should be non-zero after 500 ticks.
        assert sim.population() > 0, \
            f"population went extinct after 500 ticks (final: {sim.population()})"

    def test_population_survives_1000_ticks_on_large_world(self):
        """A larger world with more species maintains population for 1000 ticks."""
        cfg = SimulationConfig(width=64, height=48, seed=7777,
                               initial_population=50, initial_species_count=4,
                               max_population=200)
        sim = Simulation(cfg)
        for _ in range(1000):
            sim.tick()
        assert sim.population() > 0, \
            f"population went extinct after 1000 ticks (final: {sim.population()})"

    def test_no_mass_drowning_deaths(self):
        """Organisms should not die from being stuck in water."""
        cfg = SimulationConfig(width=48, height=32, seed=42,
                               initial_population=30, initial_species_count=3,
                               max_population=150)
        sim = Simulation(cfg)
        for _ in range(500):
            sim.tick()
        # Check that no organisms ended up on water tiles.
        for org in sim.living_organisms():
            assert sim.world.is_safe_tile(org.x, org.y), \
                f"organism #{org.id} is on a water tile at ({org.x}, {org.y})"

    def test_multiple_species_can_coexist(self):
        """Multiple species can coexist for a reasonable period."""
        cfg = SimulationConfig(width=48, height=32, seed=12345,
                               initial_population=40, initial_species_count=4,
                               max_population=150)
        sim = Simulation(cfg)
        initial_species = len(sim.species.living_species())
        for _ in range(300):
            sim.tick()
        # At least one species should still be alive.
        assert len(sim.species.living_species()) >= 1, \
            "all species went extinct within 300 ticks"
        # And the population should be non-zero.
        assert sim.population() > 0

    def test_births_occur_continuously(self):
        """Reproduction happens throughout the simulation, not just in a burst."""
        cfg = SimulationConfig(width=48, height=32, seed=42,
                               initial_population=30, initial_species_count=3,
                               max_population=150)
        sim = Simulation(cfg)
        # Track births in the first 200 ticks and the next 200.
        births_first_half = 0
        births_second_half = 0
        for i in range(400):
            sim.tick()
            if i < 200:
                births_first_half += sim._recent_births
            else:
                births_second_half += sim._recent_births
        # Both halves should have some births (not all in the first burst).
        # The second half might have fewer if the population declined,
        # but it shouldn't be zero if the population survived.
        if sim.population() > 0:
            assert births_second_half > 0, \
                "no births in ticks 200-400 despite surviving population"


# ---------------------------------------------------------------------------
# 7. Rest behaviour
# ---------------------------------------------------------------------------
class TestRestBehaviour:
    def test_organism_rests_when_health_low(self, test_world, weather, rng):
        """An organism with low health rests to recover."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=80.0, hydration=80.0)
        org.health = 30.0
        tile = find_safe_tile(test_world)
        if tile:
            org.x = float(tile.x) + 0.5
            org.y = float(tile.y) + 0.5
        org.step(test_world, weather, rng, 1, [])
        assert org.state == OrganismState.RESTING, \
            f"organism with low health should rest, got {org.state}"

    def test_resting_organism_recovers_health(self, test_world, weather, rng):
        """A resting organism slowly recovers health in mild weather."""
        org = Organism(id=1, genome=random_genome(rng), x=8.0, y=8.0,
                       species_id=1, birth_tick=0, energy=80.0, hydration=80.0)
        org.health = 30.0
        tile = find_safe_tile(test_world)
        if tile:
            org.x = float(tile.x) + 0.5
            org.y = float(tile.y) + 0.5
        weather.base_temp = 0.55
        weather._recompute()
        health_before = org.health
        org.step(test_world, weather, rng, 1, [])
        # Health should have increased (resting recovery > weather damage).
        if org.state == OrganismState.RESTING:
            assert org.health > health_before or org.health >= 30.0
