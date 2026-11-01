"""Navigation: terrain-aware movement and pathfinding.

This module provides the movement intelligence that organisms use to get
around the world safely. It has three layers, from cheapest to most
expensive:

1. **Steering** (:func:`steer_toward`) — every-tick local obstacle
   avoidance. Picks the best adjacent tile to move toward based on
   progress toward the target, terrain safety and movement cost.
   Automatically routes around water and impassable terrain without
   needing a full path.

2. **Line-of-sight** (:func:`line_of_sight_clear`) — Bresenham-line
   check that returns ``True`` if the direct path between two points
   crosses only safe tiles. Used to decide whether steering is enough
   or whether full pathfinding is needed.

3. **A-star** (:class:`Pathfinder`) — grid pathfinding with caching and
   a node-expansion cap. Only invoked when the direct path is blocked
   and the organism is "stuck" (hasn't made progress for several
   ticks). Paths are cached per (start, goal) tile pair so they're
   reused across ticks and across organisms heading the same way.

All three layers respect a **safety predicate**: a function
``(tile) -> bool`` that decides whether a tile is enterable for a
given organism. Land organisms use :func:`land_safe` (passable >= 0.85),
which excludes water, deep rock and snow.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World, Tile

__all__ = [
    "land_safe",
    "land_safe_threshold",
    "steer_toward",
    "line_of_sight_clear",
    "Pathfinder",
    "StuckDetector",
]

# ---------------------------------------------------------------------------
# Safety predicates
# ---------------------------------------------------------------------------
#: Tiles with ``passable >= 0.85`` are solid ground safe for land organisms.
#: This excludes deep/shallow water (0.0 / 0.4), rock (0.75) and snow (0.6).
LAND_SAFE_THRESHOLD: float = 0.85


def land_safe_threshold(tile: "Tile", threshold: float = LAND_SAFE_THRESHOLD) -> bool:
    """True if ``tile`` is solid ground (passable ≥ ``threshold``)."""
    return tile is not None and tile.params["passable"] >= threshold


def land_safe(tile: "Tile") -> bool:
    """Default land safety predicate (threshold = 0.85)."""
    return land_safe_threshold(tile, LAND_SAFE_THRESHOLD)


SafetyFn = Callable[["Tile"], bool]


# ---------------------------------------------------------------------------
# Steering — local obstacle avoidance
# ---------------------------------------------------------------------------
#: 8 directional offsets (N, NE, E, SE, S, SW, W, NW).
_DIRECTIONS_8: Tuple[Tuple[int, int], ...] = (
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
)


def steer_toward(
    world: "World",
    x: float,
    y: float,
    tx: float,
    ty: float,
    speed: float,
    safe: SafetyFn = land_safe,
) -> Tuple[float, float]:
    """Move ``(x, y)`` toward ``(tx, ty)`` by ``speed``, avoiding unsafe tiles.

    Returns the new ``(x, y)``. If every candidate move is unsafe the
    position is unchanged (the caller should treat this as "stuck" and
    consider pathfinding or picking a new target).

    Algorithm: sample the 8 neighbouring tiles plus the current tile,
    score each by how much closer it puts us to the target minus a
    penalty proportional to movement cost, and move toward the best
    one. Water and other unsafe tiles are excluded entirely (score
    = -infinity) so the organism naturally veers around them.
    """
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return x, y
    # Normalised desired direction.
    ux, uy = dx / dist, dy / dist
    # The ideal landing spot if there were no obstacles.
    ideal_x = x + ux * speed
    ideal_y = y + uy * speed

    # If the ideal spot is safe AND the target is safe AND the path to
    # the ideal spot doesn't cross unsafe tiles, take it (fast path).
    # The path check prevents "jumping" over narrow water channels when
    # speed > 1.0 tile/tick.
    ideal_tile = world.tile_at(int(ideal_x), int(ideal_y))
    target_tile = world.tile_at(int(tx), int(ty))
    if ideal_tile is not None and safe(ideal_tile) and \
       (target_tile is None or safe(target_tile)):
        # Check intermediate tiles if the step is longer than 1 tile.
        step_dist = math.hypot(ideal_x - x, ideal_y - y)
        if step_dist > 1.0:
            if not line_of_sight_clear(world, x, y, ideal_x, ideal_y, safe):
                pass  # fall through to neighbour evaluation
            else:
                if speed >= dist:
                    return tx, ty
                return ideal_x, ideal_y
        else:
            if speed >= dist:
                return tx, ty
            return ideal_x, ideal_y

    # Otherwise, evaluate the 8 neighbouring tiles and pick the one
    # that maximises progress toward the target while being safe.
    cx, cy = int(x), int(y)
    best_x, best_y = x, y
    best_score = -float("inf")
    for dx_t, dy_t in _DIRECTIONS_8:
        nx, ny = cx + dx_t, cy + dy_t
        if nx < 0 or ny < 0 or nx >= world.width or ny >= world.height:
            continue
        tile = world.tile_at(nx, ny)
        if tile is None or not safe(tile):
            continue
        # Progress = reduction in distance to target.
        new_dist = math.hypot(tx - (nx + 0.5), ty - (ny + 0.5))
        progress = dist - new_dist
        # Penalise high-cost terrain (jungle, snow, etc.).
        cost_penalty = 0.3 * (tile.movement_cost - 1.0)
        score = progress - cost_penalty
        if score > best_score:
            best_score = score
            # Move to the centre of the chosen tile.
            best_x = float(nx) + 0.5
            best_y = float(ny) + 0.5

    if best_score <= 0:
        # No neighbour makes progress — stay put (will trigger stuck logic).
        return x, y

    # Move toward the best neighbour at the organism's speed, but don't
    # overshoot it.
    ndx = best_x - x
    ndy = best_y - y
    nd = math.hypot(ndx, ndy)
    if nd < 1e-6:
        return x, y
    step = min(speed, nd)
    return x + ndx / nd * step, y + ndy / nd * step


# ---------------------------------------------------------------------------
# Line of sight
# ---------------------------------------------------------------------------
def line_of_sight_clear(
    world: "World",
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    safe: SafetyFn = land_safe,
) -> bool:
    """Bresenham-line check: True if every tile on the line is safe."""
    ix0, iy0 = int(x0), int(y0)
    ix1, iy1 = int(x1), int(y1)
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy
    cx, cy = ix0, iy0
    while True:
        tile = world.tile_at(cx, cy)
        if tile is None or not safe(tile):
            return False
        if cx == ix1 and cy == iy1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


# ---------------------------------------------------------------------------
# A* pathfinding with caching
# ---------------------------------------------------------------------------
class Pathfinder:
    """A* pathfinder on the tile grid with LRU-style caching.

    Paths are cached by (start_tile, goal_tile) integer pairs. A cache
    entry is reused until it's been looked up ``cache_ttl`` times (to
    bound memory) or the path is exhausted.

    The search is bounded by ``max_nodes`` to keep latency predictable;
    if no path is found within the budget, :meth:`find_path` returns
    ``None`` and the caller should fall back to steering or give up the
    target.
    """

    def __init__(
        self,
        world: "World",
        safe: SafetyFn = land_safe,
        max_nodes: int = 400,
        cache_size: int = 256,
    ) -> None:
        self.world = world
        self.safe = safe
        self.max_nodes = max_nodes
        self._cache_size = cache_size
        self._cache: Dict[Tuple[int, int, int, int], Optional[List[Tuple[int, int]]]] = {}
        self._cache_hits: Dict[Tuple[int, int, int, int], int] = {}

    def find_path(
        self,
        sx: float,
        sy: float,
        gx: float,
        gy: float,
    ) -> Optional[List[Tuple[float, float]]]:
        """Return a list of waypoints from ``(sx, sy)`` to ``(gx, gy)``.

        Each waypoint is the centre of a tile. Returns ``None`` if no
        safe path exists within the node budget. If the goal tile itself
        is unsafe, the nearest safe tile adjacent to it is used instead
        (so organisms can still path to "drink from the shore" spots).
        """
        # Snap to tile centres.
        start = (int(sx), int(sy))
        goal = (int(gx), int(gy))
        # Validate / adjust goal: if the goal tile is unsafe, find an
        # adjacent safe tile closest to the start.
        goal_tile = self.world.tile_at(goal[0], goal[1])
        if goal_tile is None or not self.safe(goal_tile):
            adjusted = self._nearest_safe_adjacent(goal[0], goal[1], start)
            if adjusted is None:
                return None
            goal = adjusted
        # Validate start: if the organism is somehow on an unsafe tile,
        # find the nearest safe tile.
        start_tile = self.world.tile_at(start[0], start[1])
        if start_tile is None or not self.safe(start_tile):
            adjusted = self._nearest_safe_adjacent(start[0], start[1], goal)
            if adjusted is None:
                return None
            start = adjusted

        cache_key = (start[0], start[1], goal[0], goal[1])
        if cache_key in self._cache:
            self._cache_hits[cache_key] = self._cache_hits.get(cache_key, 0) + 1
            return self._cache[cache_key]

        path = self._astar(start, goal)
        # Store in cache (as float waypoints).
        if path is not None:
            waypoints = [(float(x) + 0.5, float(y) + 0.5) for (x, y) in path]
        else:
            waypoints = None
        self._cache[cache_key] = waypoints
        self._cache_hits[cache_key] = 1
        # Evict if cache is too big.
        if len(self._cache) > self._cache_size:
            # Drop the least-recently-used entry.
            oldest = min(self._cache_hits, key=self._cache_hits.get)
            del self._cache[oldest]
            del self._cache_hits[oldest]
        return waypoints

    def _nearest_safe_adjacent(
        self, tx: int, ty: int, reference: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """BFS outward from ``(tx, ty)`` to find the nearest safe tile."""
        if not (0 <= tx < self.world.width and 0 <= ty < self.world.height):
            return None
        visited = {(tx, ty)}
        # Search rings 1..6.
        for ring in range(1, 7):
            best = None
            best_d2 = float("inf")
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    nx, ny = tx + dx, ty + dy
                    if (nx, ny) in visited:
                        continue
                    visited.add((nx, ny))
                    if not (0 <= nx < self.world.width and 0 <= ny < self.world.height):
                        continue
                    tile = self.world.tile_at(nx, ny)
                    if tile is not None and self.safe(tile):
                        d2 = (nx - reference[0]) ** 2 + (ny - reference[1]) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (nx, ny)
            if best is not None:
                return best
        return None

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* on the tile grid. Returns tile-coordinate path or None."""
        if start == goal:
            return [start]
        w, h = self.world.width, self.world.height

        def heuristic(x: int, y: int) -> float:
            return abs(x - goal[0]) + abs(y - goal[1])

        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(*start), counter, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        nodes_expanded = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                # Reconstruct path.
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            nodes_expanded += 1
            if nodes_expanded > self.max_nodes:
                return None
            cx, cy = current
            for dx, dy in _DIRECTIONS_8:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                tile = self.world.tile_at(nx, ny)
                if tile is None or not self.safe(tile):
                    continue
                # Diagonal moves can't cut through corners of unsafe tiles.
                if dx != 0 and dy != 0:
                    t1 = self.world.tile_at(cx + dx, cy)
                    t2 = self.world.tile_at(cx, cy + dy)
                    if (t1 is None or not self.safe(t1)) and (t2 is None or not self.safe(t2)):
                        continue
                tentative = g_score[current] + tile.movement_cost * (1.414 if dx != 0 and dy != 0 else 1.0)
                neighbour = (nx, ny)
                if tentative < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    counter += 1
                    heapq.heappush(open_heap, (tentative + heuristic(nx, ny), counter, neighbour))
        return None


# ---------------------------------------------------------------------------
# Stuck detection
# ---------------------------------------------------------------------------
class StuckDetector:
    """Tracks whether an organism has made progress toward its target.

    Call :meth:`update` every tick with the organism's current distance
    to its target. If the distance hasn't decreased by at least
    ``threshold`` tiles over ``window`` ticks, :meth:`is_stuck` returns
    ``True`` — signalling that steering is failing and full pathfinding
    (or a new target) is needed.
    """

    def __init__(self, window: int = 12, threshold: float = 0.5) -> None:
        self.window = window
        self.threshold = threshold
        self._distances: List[float] = []
        self._target_id: Optional[int] = None

    def reset(self, target_id: Optional[int] = None) -> None:
        self._distances.clear()
        self._target_id = target_id

    def update(self, distance: float, target_id: Optional[int] = None) -> None:
        if target_id != self._target_id:
            self.reset(target_id)
        self._distances.append(distance)
        if len(self._distances) > self.window:
            self._distances.pop(0)

    def is_stuck(self) -> bool:
        if len(self._distances) < self.window:
            return False
        return (self._distances[0] - self._distances[-1]) < self.threshold
