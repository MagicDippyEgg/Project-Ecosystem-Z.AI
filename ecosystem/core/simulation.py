"""Simulation controller.

The :class:`Simulation` owns the :class:`~ecosystem.core.world.World`,
:class:`~ecosystem.core.weather.WeatherSystem`,
:class:`~ecosystem.core.species.SpeciesRegistry`, the population of
:class:`~ecosystem.core.organism.Organism` instances, and the
:class:`~ecosystem.core.journal.Journal`.

Each call to :meth:`Simulation.tick` advances the world by one tick:
updates weather, regrows resources, asks every living organism to step
(resolving birth/death requests), updates species populations and
records statistics.

The class is GUI-agnostic: the GUI calls :meth:`tick` (possibly several
times per frame for speed > 1x), reads state for rendering, and listens
to the ``stats_updated`` signal-like callback for charts.

Speed/pause are controlled by :meth:`set_speed` and :meth:`set_paused`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .genome import Genome, random_genome, mutate
from .journal import Journal
from .navigation import Pathfinder
from .organism import Organism, OrganismState, BirthRequest
from .species import SpeciesRegistry, Species
from .weather import WeatherSystem, WeatherType, WEATHER_LABELS
from .world import World, Biome

__all__ = ["SimulationConfig", "SimulationStats", "Simulation"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Knobs that control a fresh simulation."""

    width: int = 112
    height: int = 72
    seed: int = 0xC0FFEE
    initial_population: int = 70
    initial_species_count: int = 4
    # Spatial hash cell size (in tiles) used for neighbour queries.
    cell_size: int = 12
    # Cap on the number of organisms (performance guardrail).
    max_population: int = 600
    # Whether to seed multiple distinct founding species.
    diverse_founders: bool = True

    def to_dict(self) -> Dict:
        return {
            "width": self.width, "height": self.height, "seed": self.seed,
            "initial_population": self.initial_population,
            "initial_species_count": self.initial_species_count,
            "cell_size": self.cell_size,
            "max_population": self.max_population,
            "diverse_founders": self.diverse_founders,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationConfig":
        return cls(
            width=int(data.get("width", 96)),
            height=int(data.get("height", 64)),
            seed=int(data.get("seed", 0xC0FFEE)),
            initial_population=int(data.get("initial_population", 60)),
            initial_species_count=int(data.get("initial_species_count", 3)),
            cell_size=int(data.get("cell_size", 8)),
            max_population=int(data.get("max_population", 800)),
            diverse_founders=bool(data.get("diverse_founders", True)),
        )


# ---------------------------------------------------------------------------
# Stats - sampled periodically for the charts
# ---------------------------------------------------------------------------
@dataclass
class SimulationStats:
    """Time-series of ecosystem metrics for the charts/inspector."""

    ticks: List[int] = field(default_factory=list)
    population: List[int] = field(default_factory=list)
    births: List[int] = field(default_factory=list)
    deaths: List[int] = field(default_factory=list)
    species_count: List[int] = field(default_factory=list)
    mean_energy: List[float] = field(default_factory=list)
    mean_age: List[float] = field(default_factory=list)
    mean_hydration: List[float] = field(default_factory=list)
    weather: List[int] = field(default_factory=list)
    temperature: List[float] = field(default_factory=list)
    # Per-species population history, keyed by species id.
    per_species_pop: Dict[int, List[int]] = field(default_factory=dict)

    def sample(self, tick: int, sim: "Simulation") -> None:
        living = sim.living_organisms()
        self.ticks.append(tick)
        self.population.append(len(living))
        self.births.append(sim._recent_births)
        self.deaths.append(sim._recent_deaths)
        self.species_count.append(len(sim.species.living_species()))
        if living:
            self.mean_energy.append(sum(o.energy for o in living) / len(living))
            self.mean_age.append(sum(o.age for o in living) / len(living))
            self.mean_hydration.append(sum(o.hydration for o in living) / len(living))
        else:
            self.mean_energy.append(0.0)
            self.mean_age.append(0.0)
            self.mean_hydration.append(0.0)
        self.weather.append(int(sim.weather.current))
        self.temperature.append(sim.weather.effective_temp)
        # Per-species populations.
        for sp in sim.species.all_species():
            self.per_species_pop.setdefault(sp.id, []).append(sp.population)
        # Trailing cap to keep memory bounded.
        MAX_SAMPLES = 600
        if len(self.ticks) > MAX_SAMPLES:
            for fld in ("ticks", "population", "births", "deaths",
                        "species_count", "mean_energy", "mean_age",
                        "mean_hydration", "weather", "temperature"):
                lst = getattr(self, fld)
                del lst[:-MAX_SAMPLES]
            for sid in list(self.per_species_pop.keys()):
                lst = self.per_species_pop[sid]
                del lst[:-MAX_SAMPLES]
                if not lst:
                    del self.per_species_pop[sid]

    def to_dict(self) -> Dict:
        return {
            "ticks": list(self.ticks),
            "population": list(self.population),
            "births": list(self.births),
            "deaths": list(self.deaths),
            "species_count": list(self.species_count),
            "mean_energy": list(self.mean_energy),
            "mean_age": list(self.mean_age),
            "mean_hydration": list(self.mean_hydration),
            "weather": list(self.weather),
            "temperature": list(self.temperature),
            "per_species_pop": {str(k): list(v) for k, v in self.per_species_pop.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationStats":
        return cls(
            ticks=list(data.get("ticks", [])),
            population=list(data.get("population", [])),
            births=list(data.get("births", [])),
            deaths=list(data.get("deaths", [])),
            species_count=list(data.get("species_count", [])),
            mean_energy=list(data.get("mean_energy", [])),
            mean_age=list(data.get("mean_age", [])),
            mean_hydration=list(data.get("mean_hydration", [])),
            weather=list(data.get("weather", [])),
            temperature=list(data.get("temperature", [])),
            per_species_pop={int(k): list(v) for k, v in data.get("per_species_pop", {}).items()},
        )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class Simulation:
    """Top-level simulation controller."""

    def __init__(self, config: SimulationConfig | None = None,
                 rng: random.Random | None = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = rng or random.Random(self.config.seed)
        # Callbacks the GUI can subscribe to.
        self.on_stats: Optional[Callable[[], None]] = None
        self.on_event: Optional[Callable[[str, str], None]] = None
        # State.
        self.world: Optional[World] = None
        self.weather = WeatherSystem(rng=self.rng, base_temp=0.55)
        self.species = SpeciesRegistry(rng=self.rng)
        self.organisms: Dict[int, Organism] = {}
        self.journal = Journal()
        self.stats = SimulationStats()
        self.tick_count: int = 0
        self._next_organism_id: int = 1
        self._recent_births: int = 0
        self._recent_deaths: int = 0
        self._stats_every: int = 8        # sample stats every N ticks
        self._neighbour_cell_size: int = self.config.cell_size
        self._spatial: Dict[Tuple[int, int], List[int]] = {}
        self.paused: bool = False
        self.speed: float = 1.0           # ticks per frame; can be fractional
        self._tick_accumulator: float = 0.0
        # Track deceased organisms briefly for the inspector (last N dead).
        self._recent_dead: List[Organism] = []
        # Shared pathfinder (created lazily after the world exists).
        self._pathfinder: Optional[Pathfinder] = None
        # Generate the world on construction.
        self.generate_world(self.config.seed, name=self.config_default_world_name())

    def config_default_world_name(self) -> str:
        return f"World #{self.config.seed & 0xFFFF:X}"

    @property
    def pathfinder(self) -> Pathfinder:
        """Lazily-created shared :class:`Pathfinder` for the current world."""
        if self._pathfinder is None and self.world is not None:
            self._pathfinder = Pathfinder(self.world, max_nodes=350, cache_size=512)
        if self._pathfinder is None:
            # Should not happen, but be defensive.
            raise RuntimeError("pathfinder requested before world exists")
        return self._pathfinder

    # ------------------------------------------------------------------
    # World (re)generation
    # ------------------------------------------------------------------
    def generate_world(self, seed: int, name: str = "Primary World",
                       width: Optional[int] = None, height: Optional[int] = None) -> World:
        """Discard the current world/population and generate a new one."""
        self.config.seed = int(seed)
        if width is not None:
            self.config.width = int(width)
        if height is not None:
            self.config.height = int(height)
        self.rng = random.Random(self.config.seed)
        self.world = World.generate(
            width=self.config.width, height=self.config.height,
            seed=self.config.seed, name=name, rng=self.rng,
        )
        self.weather = WeatherSystem(rng=self.rng, base_temp=0.55)
        self.species = SpeciesRegistry(rng=self.rng)
        self.organisms = {}
        self.journal = Journal()
        self.stats = SimulationStats()
        self.tick_count = 0
        self._next_organism_id = 1
        self._recent_births = 0
        self._recent_deaths = []
        self._pathfinder = None  # will be recreated on first use
        self.journal.add(0, "milestone", f"World '{name}' generated "
                                          f"({self.config.width}x{self.config.height}).")
        self._seed_initial_population()
        self.journal.add(0, "population", f"Seeded {len(self.organisms)} founders "
                                          f"across {len(self.species)} species.")
        if self.on_stats:
            self.on_stats()
        return self.world

    def _seed_initial_population(self) -> None:
        """Spawn the initial founders, optionally across multiple species."""
        if self.world is None:
            return
        # Build N founder genomes - either truly random (diverse) or
        # small variations of a single ancestor.
        founders: List[Genome] = []
        if self.config.diverse_founders and self.config.initial_species_count > 1:
            for _ in range(self.config.initial_species_count):
                founders.append(random_genome(self.rng))
        else:
            base = random_genome(self.rng)
            founders.append(base)
        # Distribute the population across founders.
        per = max(1, self.config.initial_population // len(founders))
        for fg in founders:
            for _ in range(per):
                # Slight mutation per individual so they're not identical.
                g = mutate(fg, self.rng, rate=0.3)
                # Place on a random dry-land tile (passable == 1.0).
                placed = False
                for _attempt in range(80):
                    i = self.rng.randrange(self.world.width)
                    j = self.rng.randrange(self.world.height)
                    tile = self.world.tile_at(i, j)
                    if tile is not None and tile.passable and tile.params["passable"] >= 0.85:
                        placed = True
                        break
                if not placed:
                    continue
                org = Organism(
                    id=self._next_organism_id,
                    genome=g,
                    x=float(i) + 0.5, y=float(j) + 0.5,
                    species_id=0,
                    birth_tick=0,
                    parent_ids=(),
                    # Varied starting energy and age staggers reproduction
                    # so the population doesn't boom in a single burst.
                    energy=self.rng.uniform(35.0, 55.0),
                    hydration=75.0,
                    age=self.rng.uniform(0.0, float(g["maturity_age"]) * 0.8),
                )
                sp = self.species.register_birth(org.id, g, 0, parent_species_id=None)
                org.species_id = sp.id
                self.organisms[org.id] = org
                self._next_organism_id += 1
        # Rebuild spatial hash.
        self._rebuild_spatial()

    # ------------------------------------------------------------------
    # Pause / speed
    # ------------------------------------------------------------------
    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.25, min(16.0, float(speed)))

    def step_frames(self, dt_seconds: float) -> int:
        """Advance the simulation based on wall-clock time.

        Returns the number of ticks actually executed. The GUI calls this
        from its timer; for tests you can call :meth:`tick` directly.
        """
        if self.paused:
            return 0
        # 10 ticks per simulated second at 1x speed.
        self._tick_accumulator += dt_seconds * 10.0 * self.speed
        ticks = 0
        while self._tick_accumulator >= 1.0:
            self.tick()
            self._tick_accumulator -= 1.0
            ticks += 1
            # Safety guard against spiral-of-death on slow frames.
            if ticks >= 64:
                self._tick_accumulator = 0.0
                break
        return ticks

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def tick(self) -> None:
        if self.world is None:
            return
        self.tick_count += 1
        self._recent_births = 0
        self._recent_deaths = 0
        # 1. Advance time.
        self.world.advance_time(1)
        # 2. Weather.
        prev_event_kind = self.weather.active_event.kind if self.weather.active_event else None
        self.weather.update(1.0)
        # 3. Grow resources, influenced by weather.
        moisture_bonus = self.weather.effective_moisture_bonus
        self.world.grow_resources(1.0, moisture_bonus=moisture_bonus * self.weather.growth_multiplier)
        # 4. Rebuild spatial hash for neighbour queries.
        self._rebuild_spatial()
        # 5. Step every living organism.
        living_ids = [oid for oid, o in self.organisms.items() if o.alive]
        # Shuffle to avoid positional bias in hunt/mate resolution.
        self.rng.shuffle(living_ids)
        birth_requests: List[BirthRequest] = []
        for oid in living_ids:
            org = self.organisms[oid]
            neighbours = self._neighbours_of(org)
            req = org.step(self.world, self.weather, self.rng, self.tick_count, neighbours, sim=self)
            if req is not None:
                birth_requests.append(req)
        # 6. Process births.
        for req in birth_requests:
            self._process_birth(req)
        # 7. Handle deaths from this tick (already flagged alive=False).
        newly_dead = [o for o in self.organisms.values() if not o.alive and o.death_tick == self.tick_count]
        self._recent_deaths = len(newly_dead)
        for org in newly_dead:
            sp = self.species.get(org.species_id)
            if sp is not None:
                self.species.register_death(sp, self.tick_count)
        # 8. Periodic species-extinction sweep.
        for sp in self.species.tick_check(self.tick_count):
            self.journal.add(self.tick_count, "species",
                             f"Species '{sp.name}' went extinct "
                             f"(peaked at {sp.peak_population}).")
            if self.on_event:
                self.on_event("extinct", sp.name)
        # 9. Weather event journal.
        if prev_event_kind is None and self.weather.active_event is not None:
            ev = self.weather.active_event
            label = WEATHER_LABELS[ev.kind]
            self.journal.add(self.tick_count, "weather",
                             f"A {label.lower()} began "
                             f"(intensity {ev.intensity:.2f}).")
            if self.on_event:
                self.on_event("weather_start", label)
        elif prev_event_kind is not None and self.weather.active_event is None:
            label = WEATHER_LABELS[prev_event_kind]
            self.journal.add(self.tick_count, "weather", f"The {label.lower()} ended.")
            if self.on_event:
                self.on_event("weather_end", label)
        # 10. Periodic stats sampling.
        if self.tick_count % self._stats_every == 0:
            self.stats.sample(self.tick_count, self)
            if self.on_stats:
                self.on_stats()
        # 11. Periodic population milestones.
        if self.tick_count % 120 == 0:
            pop = len(self.living_organisms())
            self.journal.add(self.tick_count, "milestone",
                             f"Tick {self.tick_count}: population {pop}, "
                             f"{len(self.species.living_species())} species.")

    # ------------------------------------------------------------------
    # Birth processing
    # ------------------------------------------------------------------
    def _process_birth(self, req: BirthRequest) -> None:
        if self.world is None:
            return
        if len(self.organisms) >= self.config.max_population:
            # Population cap reached - skip this birth.
            return
        for _ in range(req.litter_size):
            if len(self.organisms) >= self.config.max_population:
                break
            # Place offspring at a valid (safe, land) location.
            x, y = req.x, req.y
            if not self.world.is_safe_tile(x, y):
                # Nudge to nearest safe tile.
                found = False
                for r in range(1, 6):
                    for dj in range(-r, r + 1):
                        for di in range(-r, r + 1):
                            nx, ny = int(x) + di, int(y) + dj
                            if self.world.is_safe_tile(nx + 0.5, ny + 0.5):
                                x, y = nx + 0.5, ny + 0.5
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    return
            child = Organism(
                id=self._next_organism_id,
                genome=req.genome,
                x=x, y=y,
                species_id=0,
                birth_tick=req.tick,
                parent_ids=req.parent_ids,
                energy=35.0,
                hydration=65.0,
            )
            sp = self.species.register_birth(
                child.id, req.genome, req.tick,
                parent_species_id=req.parent_species_id,
            )
            child.species_id = sp.id
            # Detect speciation event for the journal.
            if sp.founding_tick == req.tick and sp.births == 1:
                parent_name = "unknown"
                if req.parent_species_id is not None:
                    psp = self.species.get(req.parent_species_id)
                    if psp is not None:
                        parent_name = psp.name
                self.journal.add(req.tick, "species",
                                 f"New species '{sp.name}' split from "
                                 f"'{parent_name}'.")
                if self.on_event:
                    self.on_event("new_species", sp.name)
            self.organisms[child.id] = child
            self._next_organism_id += 1
            self._recent_births += 1

    # ------------------------------------------------------------------
    # Spatial hashing for neighbour queries
    # ------------------------------------------------------------------
    def _rebuild_spatial(self) -> None:
        self._spatial.clear()
        cs = self._neighbour_cell_size
        for oid, org in self.organisms.items():
            if not org.alive:
                continue
            cx = int(org.x) // cs
            cy = int(org.y) // cs
            self._spatial.setdefault((cx, cy), []).append(oid)

    def _neighbours_of(self, org: Organism) -> List[Organism]:
        # 3x3 cell window for general neighbour queries (foraging, threats).
        cs = self._neighbour_cell_size
        cx = int(org.x) // cs
        cy = int(org.y) // cs
        out: List[Organism] = []
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                ids = self._spatial.get((cx + di, cy + dj))
                if not ids:
                    continue
                for oid in ids:
                    if oid == org.id:
                        continue
                    other = self.organisms.get(oid)
                    if other is not None and other.alive:
                        out.append(other)
        return out

    def find_nearest_potential_mate(self, org: "Organism", max_cells: int = 4) -> Optional["Organism"]:
        """Wider search for a same-species mature mate.

        Used by :meth:`Organism._seek_mate` when no mate is in the local
        neighbour window. Walks outward in rings of spatial-hash cells up
        to ``max_cells`` away and returns the closest candidate.
        """
        cs = self._neighbour_cell_size
        cx = int(org.x) // cs
        cy = int(org.y) // cs
        best: Optional["Organism"] = None
        best_d2 = float("inf")
        for ring in range(1, max_cells + 1):
            for dj in range(-ring, ring + 1):
                for di in range(-ring, ring + 1):
                    if max(abs(di), abs(dj)) != ring:
                        continue  # only walk the ring perimeter
                    ids = self._spatial.get((cx + di, cy + dj))
                    if not ids:
                        continue
                    for oid in ids:
                        if oid == org.id:
                            continue
                        other = self.organisms.get(oid)
                        if other is None or not other.alive:
                            continue
                        if other.species_id != org.species_id:
                            continue
                        if not other.is_mature or other.cooldown_repro > 0:
                            continue
                        if other.energy < other.genome["repro_threshold"]:
                            continue
                        d2 = (other.x - org.x) ** 2 + (other.y - org.y) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = other
            if best is not None:
                return best  # closest in the nearest ring is good enough
        return best

    # ------------------------------------------------------------------
    # Queries used by the GUI
    # ------------------------------------------------------------------
    def living_organisms(self) -> List[Organism]:
        return [o for o in self.organisms.values() if o.alive]

    def population(self) -> int:
        return sum(1 for o in self.organisms.values() if o.alive)

    def get_organism(self, oid: int) -> Optional[Organism]:
        return self.organisms.get(oid)

    def organisms_near(self, x: float, y: float, radius: float) -> List[Organism]:
        r2 = radius * radius
        return [o for o in self.organisms.values()
                if o.alive and (o.x - x) ** 2 + (o.y - y) ** 2 <= r2]

    def organism_at_point(self, x: float, y: float, tolerance: float = 1.5) -> Optional[Organism]:
        best: Optional[Organism] = None
        best_d = tolerance * tolerance
        for o in self.organisms.values():
            if not o.alive:
                continue
            d2 = (o.x - x) ** 2 + (o.y - y) ** 2
            if d2 <= best_d:
                best_d = d2
                best = o
        return best

    def species_summary(self) -> List[Dict]:
        out: List[Dict] = []
        for sp in sorted(self.species.all_species(), key=lambda s: (-s.population, s.id)):
            out.append({
                "id": sp.id, "name": sp.name,
                "population": sp.population,
                "births": sp.births, "deaths": sp.deaths,
                "peak": sp.peak_population,
                "founding_tick": sp.founding_tick,
                "extinct": sp.extinct,
                "extinction_tick": sp.extinction_tick,
                "parent_species_id": sp.parent_species_id,
                "color_hue": sp.color_hue,
                "color_sat": sp.color_sat,
                "color_light": sp.color_light,
                "centroid": dict(sp.centroid),
            })
        return out

    def ancestors(self, oid: int, max_depth: int = 5) -> Dict:
        """Return a small ancestor tree for an organism (for the family tree view)."""
        out: Dict = {"id": oid, "children": []}
        org = self.organisms.get(oid)
        if org is None or max_depth <= 0:
            return out
        for pid in org.parent_ids:
            child = self.ancestors(pid, max_depth - 1)
            if self.organisms.get(pid) is not None:
                child["name"] = f"#{pid}"
                child["species"] = self.organisms[pid].species_id
            out["children"].append(child)
        return out

    def descendants_count(self, oid: int) -> int:
        """Count living descendants of ``oid`` by walking parent_ids."""
        # Build a child index once.
        if not hasattr(self, "_child_index_cache") or self._child_index_dirty:
            self._child_index_cache: Dict[int, List[int]] = {}
            for o in self.organisms.values():
                for pid in o.parent_ids:
                    self._child_index_cache.setdefault(pid, []).append(o.id)
            self._child_index_dirty = False
        count = 0
        stack = list(self._child_index_cache.get(oid, []))
        visited = set(stack)
        while stack:
            cur = stack.pop()
            org = self.organisms.get(cur)
            if org is not None and org.alive:
                count += 1
            for cid in self._child_index_cache.get(cur, []):
                if cid not in visited:
                    visited.add(cid)
                    stack.append(cid)
        return count

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "tick_count": self.tick_count,
            "next_organism_id": self._next_organism_id,
            "world": self.world.to_dict() if self.world else None,
            "weather": self.weather.to_dict(),
            "species": self.species.to_dict(),
            "organisms": [o.to_dict() for o in self.organisms.values()],
            "journal": self.journal.to_dict(),
            "stats": self.stats.to_dict(),
            "paused": self.paused,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Simulation":
        config = SimulationConfig.from_dict(data["config"])
        sim = cls.__new__(cls)
        sim.config = config
        sim.rng = random.Random(config.seed)
        sim.on_stats = None
        sim.on_event = None
        sim.world = World.from_dict(data["world"]) if data.get("world") else None
        sim.weather = WeatherSystem.from_dict(data["weather"], sim.rng)
        sim.species = SpeciesRegistry.from_dict(data["species"], sim.rng)
        sim.organisms = {}
        for od in data.get("organisms", []):
            org = Organism.from_dict(od)
            sim.organisms[org.id] = org
        sim.journal = Journal.from_dict(data.get("journal", {}))
        sim.stats = SimulationStats.from_dict(data.get("stats", {}))
        sim.tick_count = int(data["tick_count"])
        sim._next_organism_id = int(data["next_organism_id"])
        sim._recent_births = 0
        sim._recent_deaths = 0
        sim._stats_every = 8
        sim._neighbour_cell_size = config.cell_size
        sim._spatial = {}
        sim._rebuild_spatial()
        sim.paused = bool(data.get("paused", False))
        sim.speed = float(data.get("speed", 1.0))
        sim._tick_accumulator = 0.0
        sim._recent_dead = []
        sim._pathfinder = None
        sim._child_index_dirty = True
        return sim

    # ------------------------------------------------------------------
    # Force-trigger weather (GUI helper)
    # ------------------------------------------------------------------
    def trigger_weather(self, kind: WeatherType, duration: int = 60) -> None:
        ev = self.weather.force_event(kind, duration)
        self.journal.add(self.tick_count, "weather",
                         f"Forced {WEATHER_LABELS[kind].lower()} for {duration} ticks.")
        if self.on_event:
            self.on_event("weather_start", WEATHER_LABELS[kind])

    def clear_weather(self) -> None:
        self.weather.clear_event()
