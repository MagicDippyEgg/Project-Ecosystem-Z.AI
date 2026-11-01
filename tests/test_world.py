"""Tests for the world: terrain generation, biomes, resources."""

from __future__ import annotations

import random

import pytest

from ecosystem.core.world import (
    World, Biome, BIOME_PARAMS, value_noise, fractal_noise,
)
from ecosystem.core.resource import ResourceNode, ResourceType


def test_world_generation_basic():
    rng = random.Random(2024)
    w = World.generate(width=32, height=24, seed=42, rng=rng)
    assert w.width == 32 and w.height == 24
    assert len(w.tiles) == 32 * 24
    # Every tile has a valid biome and is indexed.
    for tile in w.tiles:
        assert isinstance(tile.biome, Biome)
        assert BIOME_PARAMS[tile.biome]["max_food"] >= 0
    # Index lookup works.
    t = w.tile_at(0, 0)
    assert t is not None and t.x == 0 and t.y == 0
    # Out of bounds returns None.
    assert w.tile_at(-1, 0) is None
    assert w.tile_at(0, 9999) is None


def test_world_generation_is_deterministic():
    w1 = World.generate(width=32, height=24, seed=42)
    w2 = World.generate(width=32, height=24, seed=42)
    for t1, t2 in zip(w1.tiles, w2.tiles):
        assert t1.biome == t2.biome
        assert t1.elevation == pytest.approx(t2.elevation)
        assert t1.moisture == pytest.approx(t2.moisture)


def test_world_has_water_tiles():
    w = World.generate(width=48, height=32, seed=123)
    has_water = any(t.biome in (Biome.DEEP_WATER, Biome.SHALLOW_WATER) for t in w.tiles)
    assert has_water, "every world should have some water"


def test_world_has_land_tiles():
    w = World.generate(width=48, height=32, seed=123)
    has_land = any(t.biome not in (Biome.DEEP_WATER, Biome.SHALLOW_WATER) for t in w.tiles)
    assert has_land


def test_day_night_cycle():
    w = World.generate(width=24, height=16, seed=1)
    w.time_of_day = 0.5  # noon
    assert w.daylight() > 0.9
    assert w.is_day()
    w.time_of_day = 0.0  # midnight
    assert w.daylight() < 0.05
    assert not w.is_day()


def test_food_regeneration():
    w = World.generate(width=24, height=16, seed=7)
    # Find a grassland tile.
    tile = next(t for t in w.tiles if t.biome == Biome.GRASSLAND)
    tile.food = 0.0
    cap = tile.params["max_food"]
    for _ in range(200):
        tile.grow_food(1.0, moisture_bonus=0.0)
    assert tile.food > 0.0
    assert tile.food <= cap


def test_food_consumption_depletes():
    w = World.generate(width=24, height=16, seed=7)
    tile = next(t for t in w.tiles if t.biome == Biome.GRASSLAND)
    tile.food = 10.0
    taken = tile.consume_food(4.0)
    assert taken == 4.0
    assert tile.food == 6.0
    taken = tile.consume_food(100.0)
    assert taken == 6.0
    assert tile.food == 0.0


def test_value_noise_in_unit_range():
    table = list(range(256))
    random.Random(0).shuffle(table)
    for _ in range(50):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        v = value_noise(x, y, table)
        assert 0.0 <= v <= 1.0


def test_fractal_noise_in_unit_range():
    table = list(range(256))
    random.Random(0).shuffle(table)
    for _ in range(50):
        x = random.uniform(0, 5)
        y = random.uniform(0, 5)
        v = fractal_noise(x, y, table)
        assert 0.0 <= v <= 1.0


def test_resource_node_consume_and_regen():
    node = ResourceNode(x=1, y=2, kind=ResourceType.WATER,
                       amount=50.0, max_amount=80.0, regen_rate=2.0)
    taken = node.consume(10.0)
    assert taken == 10.0
    assert node.amount == 40.0
    node.regen(5.0)
    assert node.amount == 50.0
    # Regen caps at max_amount.
    for _ in range(100):
        node.regen(5.0)
    assert node.amount == 80.0


def test_world_to_dict_roundtrip():
    w = World.generate(width=24, height=16, seed=11)
    w.advance_time(37)
    w.tick = 37
    data = w.to_dict()
    w2 = World.from_dict(data)
    assert w2.width == w.width
    assert w2.height == w.height
    assert w2.tick == w.tick
    assert w2.time_of_day == pytest.approx(w.time_of_day)
    assert len(w2.tiles) == len(w.tiles)
    for t1, t2 in zip(w.tiles, w2.tiles):
        assert t1.biome == t2.biome
        assert t1.food == pytest.approx(t2.food)
        assert t1.elevation == pytest.approx(t2.elevation)


def test_world_drink_water_infinite_from_lake():
    w = World.generate(width=24, height=16, seed=3)
    # Find a water tile.
    water_tile = next((t for t in w.tiles if t.has_water), None)
    assert water_tile is not None
    drank = w.drink_water(water_tile.x, water_tile.y, 25.0)
    assert drank == 25.0
    # Drinking again should still work (lakes are infinite).
    drank2 = w.drink_water(water_tile.x, water_tile.y, 25.0)
    assert drank2 == 25.0


def test_world_passable_checks():
    w = World.generate(width=24, height=16, seed=9)
    for tile in w.tiles:
        if tile.biome == Biome.DEEP_WATER:
            assert not tile.passable
        elif tile.biome == Biome.GRASSLAND:
            assert tile.passable


def test_world_resource_nodes_in_radius():
    w = World.generate(width=24, height=16, seed=5)
    # Add a water node.
    node = ResourceNode(x=12, y=8, kind=ResourceType.WATER,
                       amount=30.0, max_amount=50.0, regen_rate=1.0)
    w.resources.append(node)
    found = w.resource_nodes_in_radius(12, 8, 3.0, kind=ResourceType.WATER)
    assert len(found) == 1
    assert found[0].amount == 30.0
    # Outside radius.
    found2 = w.resource_nodes_in_radius(20, 20, 3.0, kind=ResourceType.WATER)
    assert len(found2) == 0
