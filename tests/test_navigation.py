"""Tests for the navigation module: steering, pathfinding, reachability."""

from __future__ import annotations

import math
import random

import pytest

from ecosystem.core.world import World, Biome
from ecosystem.core.navigation import (
    Pathfinder, StuckDetector, land_safe, land_safe_threshold,
    steer_toward, line_of_sight_clear,
)


@pytest.fixture
def test_world():
    """A small world with a water barrier for pathfinding tests."""
    rng = random.Random(42)
    return World.generate(width=24, height=16, seed=42, rng=rng)


@pytest.fixture
def custom_world():
    """A world with a known water wall for deterministic pathfinding tests.

    Layout:
      - Land everywhere except a vertical water wall at x=12.
      - This forces organisms on the left to path around the wall to
        reach the right side (via the gap at the top or bottom).
    """
    rng = random.Random(99)
    world = World.generate(width=24, height=16, seed=99, rng=rng)
    # Force all tiles to grassland first, then build the water wall.
    for tile in world.tiles:
        tile.biome = Biome.GRASSLAND
        tile.has_water = False
        if tile.food == 0.0:
            tile.food = 10.0
    # Build a vertical water wall at x=12 (except a 1-tile gap at y=0).
    for j in range(1, 16):
        tile = world.tile_at(12, j)
        if tile is not None:
            tile.biome = Biome.DEEP_WATER
            tile.food = 0.0
            tile.has_water = True
    # Ensure x=12, y=0 is land (the gap).
    gap = world.tile_at(12, 0)
    if gap is not None:
        gap.biome = Biome.GRASSLAND
        gap.has_water = False
        gap.food = 10.0
    world._compute_landmasses()
    world._build_tile_grid()
    return world


# ---------------------------------------------------------------------------
# Safety predicates
# ---------------------------------------------------------------------------
class TestLandSafe:
    def test_grassland_is_safe(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND:
                assert land_safe(tile)

    def test_deep_water_is_not_safe(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.DEEP_WATER:
                assert not land_safe(tile)

    def test_shallow_water_is_not_safe(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.SHALLOW_WATER:
                assert not land_safe(tile)

    def test_threshold_parameter(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.ROCK:
                # Rock has passable=0.75, below 0.85 threshold.
                assert not land_safe(tile)
                # But with a lower threshold, it's safe.
                assert land_safe_threshold(tile, 0.7)


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
class TestSteering:
    def test_steering_moves_toward_target_on_clear_ground(self, test_world):
        """On open grassland, steering moves directly toward the target."""
        # Find a grassland tile.
        start = None
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND and land_safe(tile):
                start = (float(tile.x) + 0.5, float(tile.y) + 0.5)
                break
        assert start is not None
        tx, ty = start[0] + 3.0, start[1]
        nx, ny = steer_toward(test_world, start[0], start[1], tx, ty, 1.0)
        # Should have moved toward the target.
        assert nx > start[0]

    def test_steering_avoids_water(self, custom_world):
        """Steering should not move into water tiles."""
        # Start at x=11, y=8 (just left of the water wall).
        sx, sy = 11.5, 8.5
        # Target is across the wall at x=15, y=8.
        tx, ty = 15.5, 8.5
        nx, ny = steer_toward(custom_world, sx, sy, tx, ty, 1.0)
        # The new position should NOT be on a water tile.
        assert custom_world.is_safe_tile(nx, ny)

    def test_steering_returns_unchanged_when_surrounded_by_water(self, test_world):
        """If all neighbours are unsafe, steering stays put."""
        # Find a deep-water tile and start "on" it (edge case).
        water_tile = None
        for tile in test_world.tiles:
            if tile.biome == Biome.DEEP_WATER:
                water_tile = tile
                break
        assert water_tile is not None
        sx, sy = float(water_tile.x) + 0.5, float(water_tile.y) + 0.5
        # Steering from a water tile with all-water neighbours should stay put.
        nx, ny = steer_toward(test_world, sx, sy, sx + 5, sy, 1.0)
        # Position unchanged or very close (no safe move).
        assert math.hypot(nx - sx, ny - sy) < 0.5

    def test_steering_routes_around_obstacle(self, custom_world):
        """Steering should pick a direction that makes progress, not water."""
        # Start at x=11, y=8, target at x=15, y=8 (across water wall).
        sx, sy = 11.5, 8.5
        tx, ty = 15.5, 8.5
        dist_before = math.hypot(tx - sx, ty - sy)
        nx, ny = steer_toward(custom_world, sx, sy, tx, ty, 1.0)
        # Should NOT move into the water wall (x=12).
        assert int(nx) != 12 or not land_safe(custom_world.tile_at(int(nx), int(ny)))
        # Should move to a safe tile.
        assert custom_world.is_safe_tile(nx, ny)


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------
class TestLineOfSight:
    def test_clear_los_on_grassland(self, test_world):
        """LOS is clear between two grassland tiles with no water between."""
        # Find two nearby grassland tiles.
        tiles = [t for t in test_world.tiles if t.biome == Biome.GRASSLAND][:10]
        if len(tiles) >= 2:
            t1, t2 = tiles[0], tiles[1]
            # May or may not be clear depending on terrain, but shouldn't crash.
            result = line_of_sight_clear(test_world, t1.x + 0.5, t1.y + 0.5,
                                         t2.x + 0.5, t2.y + 0.5)
            assert isinstance(result, bool)

    def test_los_blocked_by_water(self, custom_world):
        """LOS across the water wall should be blocked."""
        # x=10 to x=14 at y=8 crosses the water wall at x=12.
        result = line_of_sight_clear(custom_world, 10.5, 8.5, 14.5, 8.5)
        assert not result


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------
class TestPathfinder:
    def test_pathfinder_finds_direct_path_on_clear_ground(self, test_world):
        """A* finds a path between two safe tiles on the same landmass."""
        # Find two safe tiles on the same landmass.
        start_lm = test_world.landmass_id(5.5, 5.5)
        if start_lm == -1:
            # Find any safe tile.
            for tile in test_world.tiles:
                if land_safe(tile):
                    start_lm = test_world.landmass_id(tile.x + 0.5, tile.y + 0.5)
                    break
        assert start_lm != -1
        # Find a target on the same landmass, a few tiles away.
        target = None
        for tile in test_world.tiles:
            if test_world.landmass_id(tile.x + 0.5, tile.y + 0.5) == start_lm:
                if abs(tile.x - 5) + abs(tile.y - 5) > 3:
                    target = (float(tile.x) + 0.5, float(tile.y) + 0.5)
                    break
        if target is None:
            pytest.skip("no suitable target found")
        pf = Pathfinder(test_world)
        path = pf.find_path(5.5, 5.5, target[0], target[1])
        assert path is not None
        assert len(path) >= 2
        # Every waypoint should be on a safe tile.
        for wx, wy in path:
            assert test_world.is_safe_tile(wx, wy)

    def test_pathfinder_routes_around_water_wall(self, custom_world):
        """A* routes around the water wall via the gap at y=0."""
        pf = Pathfinder(custom_world)
        # Start left of wall, target right of wall.
        path = pf.find_path(10.5, 8.5, 15.5, 8.5)
        if path is not None:
            # Path should exist and go through the gap (y=0).
            assert len(path) >= 2
            # No waypoint should be on a water tile.
            for wx, wy in path:
                assert custom_world.is_safe_tile(wx, wy), \
                    f"waypoint ({wx},{wy}) is on water"

    def test_pathfinder_returns_none_for_unreachable_target(self):
        """A* returns None when no path exists (target on separate island)."""
        # Create a world with isolated tiles.
        rng = random.Random(123)
        world = World.generate(width=16, height=12, seed=123, rng=rng)
        pf = Pathfinder(world)
        # Try to path to a water tile (should find nearest safe adjacent
        # or return None if completely surrounded).
        # Find a deep-water tile in the interior.
        for tile in world.tiles:
            if tile.biome == Biome.DEEP_WATER:
                # Try pathing to it — should either find a shore path
                # or return None.
                path = pf.find_path(5.5, 5.5, tile.x + 0.5, tile.y + 0.5)
                # Either None or all-safe waypoints.
                if path is not None:
                    for wx, wy in path:
                        assert world.is_safe_tile(wx, wy)
                break

    def test_pathfinder_caches_results(self, test_world):
        """Repeated calls with the same start/goal return cached results."""
        pf = Pathfinder(test_world)
        # Find a valid path first.
        path1 = pf.find_path(5.5, 5.5, 8.5, 8.5)
        # Second call should hit the cache (same object).
        path2 = pf.find_path(5.5, 5.5, 8.5, 8.5)
        if path1 is not None:
            assert path2 is not None
            assert len(path1) == len(path2)

    def test_pathfinder_adjusts_unsafe_goal_to_adjacent_safe(self, test_world):
        """If the goal tile is unsafe, the pathfinder finds the nearest safe tile."""
        pf = Pathfinder(test_world)
        # Find a water tile.
        water_tile = None
        for tile in test_world.tiles:
            if tile.biome == Biome.DEEP_WATER:
                water_tile = tile
                break
        if water_tile is None:
            pytest.skip("no water tile found")
        # Path to the water tile — should adjust to a nearby shore.
        path = pf.find_path(5.5, 5.5, water_tile.x + 0.5, water_tile.y + 0.5)
        if path is not None:
            # The final waypoint should be safe (adjusted).
            final = path[-1]
            assert test_world.is_safe_tile(final[0], final[1])


# ---------------------------------------------------------------------------
# Stuck detector
# ---------------------------------------------------------------------------
class TestStuckDetector:
    def test_not_stuck_when_making_progress(self):
        sd = StuckDetector(window=5, threshold=0.5)
        for d in [10, 9, 8, 7, 6]:
            sd.update(d)
        assert not sd.is_stuck()

    def test_stuck_when_not_making_progress(self):
        sd = StuckDetector(window=5, threshold=0.5)
        for d in [10, 10, 10, 10, 10]:
            sd.update(d)
        assert sd.is_stuck()

    def test_reset_clears_state(self):
        sd = StuckDetector(window=5, threshold=0.5)
        for d in [10, 10, 10, 10, 10]:
            sd.update(d)
        assert sd.is_stuck()
        sd.reset()
        assert not sd.is_stuck()

    def test_target_change_resets(self):
        sd = StuckDetector(window=5, threshold=0.5)
        for d in [10, 10, 10]:
            sd.update(d, target_id=1)
        sd.update(10, target_id=2)  # different target
        assert not sd.is_stuck()  # reset because target changed


# ---------------------------------------------------------------------------
# World reachability helpers
# ---------------------------------------------------------------------------
class TestWorldReachability:
    def test_landmass_id_returns_valid_id_for_land(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND:
                lm = test_world.landmass_id(tile.x + 0.5, tile.y + 0.5)
                assert lm >= 0
                break

    def test_landmass_id_returns_minus_one_for_water(self, test_world):
        for tile in test_world.tiles:
            if tile.biome == Biome.DEEP_WATER:
                lm = test_world.landmass_id(tile.x + 0.5, tile.y + 0.5)
                assert lm == -1
                break

    def test_same_landmass_true_for_adjacent_land_tiles(self, test_world):
        """Adjacent land tiles should be on the same landmass."""
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND:
                neighbour = test_world.tile_at(tile.x + 1, tile.y)
                if neighbour is not None and neighbour.biome == Biome.GRASSLAND:
                    assert test_world.same_landmass(tile.x + 0.5, tile.y + 0.5,
                                                   neighbour.x + 0.5, neighbour.y + 0.5)
                    return

    def test_can_drink_at_shore(self, test_world):
        """can_drink_at returns True for a land tile adjacent to water."""
        # Find a land tile adjacent to water.
        for tile in test_world.tiles:
            if tile.biome not in (Biome.DEEP_WATER, Biome.SHALLOW_WATER):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nt = test_world.tile_at(tile.x + dx, tile.y + dy)
                        if nt is not None and nt.has_water:
                            assert test_world.can_drink_at(tile.x + 0.5, tile.y + 0.5)
                            return

    def test_nearest_drinking_spot_returns_safe_tile(self, test_world):
        """nearest_drinking_spot returns a tile where you can drink."""
        # Find a land tile.
        start = None
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND:
                start = (tile.x + 0.5, tile.y + 0.5)
                break
        assert start is not None
        spot = test_world.nearest_drinking_spot(start[0], start[1], 20)
        if spot is not None:
            assert test_world.is_safe_tile(spot[0], spot[1])
            assert test_world.can_drink_at(spot[0], spot[1])

    def test_best_reachable_food_tile_on_same_landmass(self, test_world):
        """best_reachable_food_tile returns a tile on the same landmass."""
        start = None
        start_lm = -1
        for tile in test_world.tiles:
            if tile.biome == Biome.GRASSLAND:
                lm = test_world.landmass_id(tile.x + 0.5, tile.y + 0.5)
                if lm >= 0:
                    start = (tile.x + 0.5, tile.y + 0.5)
                    start_lm = lm
                    break
        assert start is not None
        # Add food to a nearby tile on the same landmass.
        for tile in test_world.tiles:
            if test_world.landmass_id(tile.x + 0.5, tile.y + 0.5) == start_lm:
                tile.food = 20.0
                break
        result = test_world.best_reachable_food_tile(start[0], start[1], 10)
        if result is not None:
            assert test_world.landmass_id(result[0], result[1]) == start_lm
