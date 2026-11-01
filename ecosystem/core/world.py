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
    Biome.GRASSLAND:    {"growth_rate": 0.20,"max_food": 22.0,"passable": 1.0, "water_tile": 0.0, "food_value": 1.0},
    Biome.FOREST:       {"growth_rate": 0.26,"max_food": 32.0,"passable": 0.9, "water_tile": 0.0, "food_value": 1.2},
    Biome.JUNGLE:       {"growth_rate": 0.30,"max_food": 40.0,"passable": 0.7, "water_tile": 0.0, "food_value": 1.4},
    Biome.SAVANNA:      {"growth_rate": 0.12,"max_food": 14.0,"passable": 1.0, "water_tile": 0.0, "food_value": 1.0},
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
    # Grid index from (x, y) -> tile index.
    _index: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    time_of_day: float = 0.25  # 0..1, 0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk
    day_length: int = 240      # ticks per in-world day
    tick: int = 0
    mean_temperature: float = 0.55
    name: str = "Primary World"

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
                cap = tile.params["max_food"]
                tile.food = rng.uniform(0.0, cap * 0.35) if cap > 0 else 0.0
                idx = len(tiles)
                tiles.append(tile)
                world._index[(i, j)] = idx
        world.tiles = tiles

        # Sprinkle a few water resource nodes (springs) in dry-but-passable
        # biomes so creatures can drink away from lakes.
        for _ in range(max(2, (width * height) // 600)):
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
        return world

    # ----- access --------------------------------------------------------
    def tile_at(self, x: int, y: int) -> Optional[Tile]:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        idx = self._index.get((x, y))
        if idx is None:
            return None
        return self.tiles[idx]

    def tile_index(self, x: int, y: int) -> int:
        return self._index[(x, y)]

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
