"""Procedural world: terrain, biomes and resources.

The world is a 2D grid of :class:`Tile` objects. Each tile has a
:class:`Biome`, a moisture value, an elevation, and optionally a
:class:`ResourceNode` (food plant or water source).

Terrain is generated with value noise (a simple, dependency-free
approximation of Perlin noise) seeded by the simulation's RNG. The same
seed always yields the same terrain, which makes saved worlds
reproducible.

Day/night cycle and weather live in :mod:`ecosystem.core.weather` and
:mod:`ecosystem.core.simulation`, but the *effect* of weather on tiles
(soil moisture, plant growth) is computed here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .resource import ResourceNode, ResourceType

__all__ = [
    "Biome",
    "Tile",
    "World",
    "value_noise",
    "fractal_noise",
]


class Biome(IntEnum):
    """Biome categories used for rendering and ecology rules."""

    DEEP_WATER = 0
    SHALLOW_WATER = 1
    BEACH = 2
    GRASSLAND = 3
    FOREST = 4
    JUNGLE = 5
    SAVANNA = 6
    DESERT = 7
    ROCK = 8
    SNOW = 9
    TUNDRA = 10


# Per-biome ecology parameters.
#   growth_rate : plant-food units regenerated per tick (multiplied by moisture)
#   max_food    : cap on plant food per tile
#   passable    : whether organisms can enter the tile
#   water_tile  : whether drinking from this tile hydrates organisms
#   food_value  : nutrition per unit of plant food eaten
BIOME_PARAMS: Dict[Biome, Dict[str, float]] = {
    Biome.DEEP_WATER:   {"growth_rate": 0.0, "max_food": 0.0, "passable": 0.0, "water_tile": 1.0, "food_value": 0.0},
    Biome.SHALLOW_WATER:{"growth_rate": 0.0, "max_food": 0.0, "passable": 0.4, "water_tile": 1.0, "food_value": 0.0},
    Biome.BEACH:        {"growth_rate": 0.05,"max_food": 8.0, "passable": 1.0, "water_tile": 0.0, "food_value": 1.0},
    Biome.GRASSLAND:    {"growth_rate": 0.28,"max_food": 28.0,"passable": 1.0, "water_tile": 0.0, "food_value": 1.0},
    Biome.FOREST:       {"growth_rate": 0.36,"max_food": 38.0,"passable": 0.9, "water_tile": 0.0, "food_value": 1.2},
    Biome.JUNGLE:       {"growth_rate": 0.42,"max_food": 45.0,"passable": 0.7, "water_tile": 0.0, "food_value": 1.4},
    Biome.SAVANNA:      {"growth_rate": 0.17,"max_food": 18.0,"passable": 1.0, "water_tile": 0.0, "food_value": 1.0},
    Biome.DESERT:       {"growth_rate": 0.02,"max_food": 4.0, "passable": 0.85,"water_tile": 0.0, "food_value": 0.6},
    Biome.ROCK:         {"growth_rate": 0.0, "max_food": 0.0, "passable": 0.75,"water_tile": 0.0, "food_value": 0.0},
    Biome.SNOW:         {"growth_rate": 0.01,"max_food": 3.0, "passable": 0.6, "water_tile": 0.0, "food_value": 0.4},
    Biome.TUNDRA:       {"growth_rate": 0.05,"max_food": 10.0,"passable": 0.85,"water_tile": 0.0, "food_value": 0.8},
}


@dataclass
class Tile:
    """A single cell of the world grid."""

    x: int
    y: int
    biome: Biome = Biome.GRASSLAND
    elevation: float = 0.0       # 0..1
    moisture: float = 0.5        # 0..1
    temperature: float = 0.5     # 0..1 (0 cold, 1 hot) - base, before weather
    food: float = 0.0            # current plant food on the tile
    # Resource nodes are stored separately on the World for spatial queries;
    # tiles just remember if one is present for rendering.
    has_water: bool = False

    @property
    def params(self) -> Dict[str, float]:
        return BIOME_PARAMS[self.biome]

    @property
    def passable(self) -> bool:
        return self.params["passable"] > 0.0

    @property
    def movement_cost(self) -> float:
        """Multiplier on movement effort; higher = harder to traverse."""
        p = self.params["passable"]
        if p <= 0:
            return 1e9
        return 1.0 / max(p, 0.05)

    def grow_food(self, dt: float, moisture_bonus: float = 0.0) -> None:
        """Regenerate plant food using logistic growth.

        Growth is fastest when the tile is nearly empty and slows as it
        approaches the carrying capacity. This prevents the boom-bust
        dynamic where herbivores strip all food and then starve: after a
        tile is grazed bare it rebounds quickly.
        """
        rate = self.params["growth_rate"]
        cap = self.params["max_food"]
        if rate <= 0 or cap <= 0:
            return
        # Logistic growth factor: 1 at food=0, 0 at food=cap.
        space_factor = max(0.0, 1.0 - self.food / cap)
        eff = rate * (0.5 + self.moisture + moisture_bonus) * (0.3 + 0.7 * space_factor)
        self.food = min(cap, self.food + eff * dt)

    def consume_food(self, amount: float) -> float:
        """Eat up to ``amount`` food; return how much was actually consumed."""
        taken = min(self.food, amount)
        self.food -= taken
        return taken


# ---------------------------------------------------------------------------
# Value noise
# ---------------------------------------------------------------------------
def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise(
    x: float,
    y: float,
    seed_table: List[int],
) -> float:
    """Cheap value noise in ``[0, 1]`` using a permutation table.

    Not as smooth as Perlin but good enough for biome generation and
    entirely deterministic given the seed table.
    """
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    xf = x - xi
    yf = y - yi
    n = len(seed_table)
    def grad(ix: int, iy: int) -> float:
        idx = (seed_table[(ix % n + n) % n] + (iy % n + n) % n) % n
        return seed_table[idx] / (n - 1)
    v00 = grad(xi, yi)
    v10 = grad(xi + 1, yi)
    v01 = grad(xi, yi + 1)
    v11 = grad(xi + 1, yi + 1)
    sx = _smoothstep(xf)
    sy = _smoothstep(yf)
    a = v00 + sx * (v10 - v00)
    b = v01 + sx * (v11 - v01)
    return a + sy * (b - a)


def fractal_noise(
    x: float,
    y: float,
    seed_table: List[int],
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    """Sum several octaves of :func:`value_noise`."""
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * value_noise(x * freq, y * freq, seed_table)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
@dataclass
class World:
    """A 2D grid world with biomes, food, water and time-of-day."""

    width: int
    height: int
    seed: int
    tiles: List[Tile] = field(default_factory=list)
    resources: List[ResourceNode] = field(default_factory=list)
    # Grid index from (x, y) -> tile. Stored as a flat list for O(1)
    # access without dict/tuple overhead (tile_at is called millions of
    # times per second during simulation).
    _index: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    _tile_grid: List[List[Optional["Tile"]]] = field(default_factory=list, repr=False)
    time_of_day: float = 0.25  # 0..1, 0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk
    day_length: int = 240      # ticks per in-world day
    tick: int = 0
    mean_temperature: float = 0.55
    name: str = "Primary World"
    # Landmass labels: _landmass[x][y] = connected-component id of safe
    # (passable >= 0.85) tiles. Two tiles are mutually reachable on foot
    # iff they share the same landmass id. Computed once at generation.
    _landmass: List[List[int]] = field(default_factory=list, repr=False)
    _landmass_count: int = 0

    # ----- construction --------------------------------------------------
    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        seed: int,
        *,
        scale: float = 0.10,
        sea_level: float = 0.42,
        rng: Optional[random.Random] = None,
        name: str = "Primary World",
    ) -> "World":
        """Generate a new world from ``seed``."""
        if rng is None:
            rng = random.Random(seed)
        # Permutation table for value noise.
        table = list(range(512))
        rng.shuffle(table)

        world = cls(width=width, height=height, seed=seed, name=name)
        cx, cy = width / 2.0, height / 2.0
        max_r = math.hypot(cx, cy)

        tiles: List[Tile] = []
        for j in range(height):
            for i in range(width):
                nx = i * scale
                ny = j * scale
                elev = fractal_noise(nx, ny, table, octaves=5, persistence=0.55)
                moist = fractal_noise(nx + 100.0, ny + 100.0, table, octaves=4, persistence=0.5)
                # Island falloff so worlds feel like continents not endless plains.
                r = math.hypot(i - cx, j - cy) / max_r
                falloff = max(0.0, 1.0 - r * r * 1.1)
                elev *= falloff
                # Temperature: warmer near equator (centre y), colder at poles.
                lat_factor = 1.0 - abs(j - cy) / cy
                temp = 0.15 + 0.7 * lat_factor + 0.15 * elev
                biome = _classify_biome(elev, moist, temp, sea_level)
                tile = Tile(
                    x=i, y=j, biome=biome,
                    elevation=elev, moisture=moist, temperature=temp,
                    food=0.0, has_water=(biome in (Biome.DEEP_WATER, Biome.SHALLOW_WATER)),
                )
                # Initial plant food so herbivores don't starve on tick 1.
                # 35% of cap — enough to start but not so much that the
                # population overshoots carrying capacity instantly.
                cap = tile.params["max_food"]
                tile.food = rng.uniform(0.0, cap * 0.35) if cap > 0 else 0.0
                idx = len(tiles)
                tiles.append(tile)
                world._index[(i, j)] = idx
        world.tiles = tiles

        # Sprinkle water springs in dry-but-passable biomes so creatures
        # can drink away from lakes/rivers. More springs = better
        # hydration coverage for large continents.
        for _ in range(max(4, (width * height) // 350)):
            for _attempt in range(20):
                i = rng.randrange(width)
                j = rng.randrange(height)
                tile = world.tile_at(i, j)
                if tile is None or tile.biome in (Biome.DEEP_WATER, Biome.SHALLOW_WATER, Biome.ROCK, Biome.SNOW):
                    continue
                if tile.moisture > 0.45:
                    node = ResourceNode(
                        x=float(i), y=float(j),
                        kind=ResourceType.WATER,
                        amount=60.0, max_amount=80.0,
                        regen_rate=0.4,
                    )
                    world.resources.append(node)
                    break
        world._compute_landmasses()
        world._build_tile_grid()
        return world

    # ----- landmass / reachability --------------------------------------
    def _compute_landmasses(self) -> None:
        """Flood-fill connected components of solid-ground tiles.

        Two tiles are on the same landmass (mutually reachable on foot
        without entering water) iff they share the same id in
        ``self._landmass``. Water and low-passability tiles get id -1.
        """
        w, h = self.width, self.height
        # Use a flat list indexed by x * h + y for speed.
        labels = [-1] * (w * h)
        current_id = 0
        from collections import deque
        for sx in range(w):
            for sy in range(h):
                idx = sx * h + sy
                if labels[idx] != -1:
                    continue
                tile = self.tile_at(sx, sy)
                if tile is None or tile.params["passable"] < 0.85:
                    continue
                # BFS from this seed.
                queue = deque()
                queue.append((sx, sy))
                labels[idx] = current_id
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nx, ny = cx + dx, cy + dy
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue
                        nidx = nx * h + ny
                        if labels[nidx] != -1:
                            continue
                        nt = self.tile_at(nx, ny)
                        if nt is None or nt.params["passable"] < 0.85:
                            continue
                        labels[nidx] = current_id
                        queue.append((nx, ny))
                current_id += 1
        # Convert flat list to 2D for O(1) lookup.
        self._landmass = [[labels[x * h + y] for y in range(h)] for x in range(w)]
        self._landmass_count = current_id

    def landmass_id(self, x: float, y: float) -> int:
        """Return the landmass id at ``(x, y)``, or -1 if water/unsafe."""
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return -1
        if not self._landmass:
            return -1
        return self._landmass[ix][iy]

    def same_landmass(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """True if both points are on the same connected landmass."""
        a = self.landmass_id(x1, y1)
        b = self.landmass_id(x2, y2)
        return a != -1 and a == b

    def is_safe_tile(self, x: float, y: float, threshold: float = 0.85) -> bool:
        """True if the tile at ``(x, y)`` is solid ground for a land organism."""
        tile = self.tile_at(int(x), int(y))
        return tile is not None and tile.params["passable"] >= threshold

    def can_drink_at(self, x: float, y: float) -> bool:
        """True if an organism at ``(x, y)`` can drink without moving.

        Checks for adjacent (8-neighbourhood) water tiles or a water
        resource node on the same tile. The organism stays on solid
        ground and drinks from the shore.
        """
        ix, iy = int(x), int(y)
        # Water resource node on this tile?
        for res in self.resources:
            if res.kind.name == "WATER" and abs(res.x - x) <= 1.5 and abs(res.y - y) <= 1.5:
                return True
        # Adjacent water tile?
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tile = self.tile_at(ix + dx, iy + dy)
                if tile is not None and tile.has_water:
                    return True
        return False

    def nearest_drinking_spot(self, x: float, y: float, radius: int) -> Optional[Tuple[float, float]]:
        """Find the nearest safe tile from which the organism can drink.

        Scans tiles within ``radius`` of ``(x, y)`` and returns the
        closest one that is safe (solid ground) AND has water adjacent.
        Only returns tiles on the same landmass as the organism.
        """
        cx, cy = int(x), int(y)
        my_landmass = self.landmass_id(x, y)
        best: Optional[Tuple[float, float]] = None
        best_d2 = float("inf")
        r = int(radius)
        for j in range(max(0, cy - r), min(self.height, cy + r + 1)):
            for i in range(max(0, cx - r), min(self.width, cx + r + 1)):
                if my_landmass != -1 and self._landmass[i][j] != my_landmass:
                    continue
                tile = self.tile_at(i, j)
                if tile is None or tile.params["passable"] < 0.85:
                    continue
                if not self.can_drink_at(i + 0.5, j + 0.5):
                    continue
                d2 = (i + 0.5 - x) ** 2 + (j + 0.5 - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = (float(i) + 0.5, float(j) + 0.5)
        return best

    def best_reachable_food_tile(
        self, x: float, y: float, radius: int, min_food: float = 2.0,
    ) -> Optional[Tuple[float, float]]:
        """Find the tile with the most food within ``radius`` on the same landmass."""
        cx, cy = int(x), int(y)
        my_lm = self.landmass_id(x, y)
        if my_lm == -1 or not self._landmass:
            return None
        lm = self._landmass
        best: Optional[Tuple[float, float]] = None
        best_food = min_food
        r = int(radius)
        xlo = max(0, cx - r)
        xhi = min(self.width, cx + r + 1)
        ylo = max(0, cy - r)
        yhi = min(self.height, cy + r + 1)
        for i in range(xlo, xhi):
            row = lm[i]
            trow = self._tile_grid[i]
            for j in range(ylo, yhi):
                if row[j] != my_lm:
                    continue
                tile = trow[j]
                if tile is None:
                    continue
                if tile.food > best_food:
                    best_food = tile.food
                    best = (float(i) + 0.5, float(j) + 0.5)
        return best

    # ----- access --------------------------------------------------------
    def tile_at(self, x: int, y: int) -> Optional[Tile]:
        """O(1) tile lookup via 2D grid (falls back to dict if needed)."""
        if self._tile_grid:
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return None
            return self._tile_grid[x][y]
        # Fallback for worlds loaded from dict before grid is built.
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        idx = self._index.get((x, y))
        if idx is None:
            return None
        return self.tiles[idx]

    def tile_index(self, x: int, y: int) -> int:
        return self._index[(x, y)]

    def _build_tile_grid(self) -> None:
        """Build the 2D tile grid for O(1) ``tile_at`` lookups."""
        grid: List[List[Optional[Tile]]] = [[None] * self.height for _ in range(self.width)]
        for tile in self.tiles:
            grid[tile.x][tile.y] = tile
        self._tile_grid = grid

    def in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    def biome_at(self, x: float, y: float) -> Optional[Biome]:
        t = self.tile_at(int(x), int(y))
        return t.biome if t is not None else None

    def passable_at(self, x: float, y: float) -> bool:
        t = self.tile_at(int(x), int(y))
        return t is not None and t.passable

    # ----- simulation hooks ---------------------------------------------
    def advance_time(self, ticks: int = 1) -> None:
        self.tick += ticks
        self.time_of_day = (self.time_of_day + ticks / self.day_length) % 1.0

    def daylight(self) -> float:
        """Brightness in ``[0, 1]``; 0 = pitch black (midnight), 1 = full noon."""
        # Cosine centred on noon (time_of_day == 0.5). At t=0.5 phase=0,
        # cos(0)=1, daylight=1; at t=0 or 1 phase=±pi, cos=-1, daylight=0.
        phase = (self.time_of_day - 0.5) * 2.0 * math.pi
        return max(0.0, 0.5 + 0.5 * math.cos(phase))

    def is_day(self) -> bool:
        return 0.25 <= self.time_of_day <= 0.75

    def grow_resources(self, dt: float, moisture_bonus: float = 0.0) -> None:
        """Regenerate plant food on every tile and refill water springs."""
        for tile in self.tiles:
            tile.grow_food(dt, moisture_bonus)
        for res in self.resources:
            res.regen(dt)

    def consume_plant_food(self, x: int, y: int, amount: float) -> float:
        tile = self.tile_at(x, y)
        if tile is None:
            return 0.0
        return tile.consume_food(amount)

    def drink_water(self, x: int, y: int, amount: float) -> float:
        """Drink from a water tile or nearby spring; return amount drunk."""
        tile = self.tile_at(x, y)
        if tile is not None and tile.has_water:
            return amount  # Lakes/oceans are effectively infinite.
        # Else look for a nearby spring resource node.
        for res in self.resources:
            if res.kind != ResourceType.WATER:
                continue
            if abs(res.x - x) <= 1.5 and abs(res.y - y) <= 1.5:
                return res.consume(amount)
        return 0.0

    def resource_nodes_in_radius(
        self, x: float, y: float, radius: float, kind: Optional[ResourceType] = None,
    ) -> List[ResourceNode]:
        out: List[ResourceNode] = []
        r2 = radius * radius
        for res in self.resources:
            if kind is not None and res.kind != kind:
                continue
            dx = res.x - x
            dy = res.y - y
            if dx * dx + dy * dy <= r2 and res.amount > 0.1:
                out.append(res)
        return out

    # ----- serialisation -------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "time_of_day": self.time_of_day,
            "day_length": self.day_length,
            "tick": self.tick,
            "mean_temperature": self.mean_temperature,
            "name": self.name,
            "tiles": [
                {
                    "x": t.x, "y": t.y, "biome": int(t.biome),
                    "elevation": t.elevation, "moisture": t.moisture,
                    "temperature": t.temperature, "food": t.food,
                    "has_water": t.has_water,
                }
                for t in self.tiles
            ],
            "resources": [r.to_dict() for r in self.resources],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "World":
        world = cls(
            width=data["width"], height=data["height"], seed=data["seed"],
            time_of_day=data["time_of_day"], day_length=data["day_length"],
            tick=data["tick"], mean_temperature=data.get("mean_temperature", 0.55),
            name=data.get("name", "Primary World"),
        )
        tiles: List[Tile] = []
        for td in data["tiles"]:
            tile = Tile(
                x=td["x"], y=td["y"], biome=Biome(td["biome"]),
                elevation=td["elevation"], moisture=td["moisture"],
                temperature=td["temperature"], food=td["food"],
                has_water=td["has_water"],
            )
            world._index[(tile.x, tile.y)] = len(tiles)
            tiles.append(tile)
        world.tiles = tiles
        world.resources = [ResourceNode.from_dict(r) for r in data["resources"]]
        # Recompute landmass labels (cheaper than serialising the grid).
        world._compute_landmasses()
        world._build_tile_grid()
        return world

    # ----- queries used by the GUI --------------------------------------
    def biome_counts(self) -> Dict[Biome, int]:
        counts: Dict[Biome, int] = {}
        for tile in self.tiles:
            counts[tile.biome] = counts.get(tile.biome, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Biome classification
# ---------------------------------------------------------------------------
def _classify_biome(elev: float, moist: float, temp: float, sea_level: float) -> Biome:
    """Pick a biome from elevation/moisture/temperature."""
    if elev < sea_level * 0.55:
        return Biome.DEEP_WATER
    if elev < sea_level:
        return Biome.SHALLOW_WATER
    if elev < sea_level + 0.04:
        return Biome.BEACH
    if elev > 0.82:
        return Biome.SNOW if temp < 0.45 else Biome.ROCK
    if temp < 0.25:
        return Biome.SNOW if elev > 0.7 else Biome.TUNDRA
    if temp > 0.78 and moist < 0.35:
        return Biome.DESERT
    if moist < 0.30:
        return Biome.SAVANNA if temp > 0.55 else Biome.TUNDRA
    if moist > 0.70 and temp > 0.65:
        return Biome.JUNGLE
    if moist > 0.55:
        return Biome.FOREST
    return Biome.GRASSLAND
