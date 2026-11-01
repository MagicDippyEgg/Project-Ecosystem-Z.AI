"""Organism: an individual living agent with survival intelligence.

Each organism is an autonomous agent that, every tick:

1. Ages and consumes energy/water proportional to metabolism, size and
   local crowding (competition stress).
2. Assesses its situation: threats, vital needs (health, hydration,
   energy), and available targets — all filtered by **reachability**
   (same landmass, not across water).
3. Acts on the highest-priority need using terrain-aware movement that
   steers around water and obstacles. If steering gets it stuck, it
   falls back to A* pathfinding (cached) to route around large
   barriers.
4. May reproduce, rest (to conserve energy), or die.

The decision logic is a **needs-urgency hierarchy** rather than a flat
priority list, so an organism that's both hungry and thirsty handles
the more urgent one first. Targets are only pursued if reachable on
the same landmass — organisms no longer walk into water chasing
impossible goals.

The class is plain Python (no Qt) and serialises itself to a dict, so
the whole simulation can be saved/loaded as JSON.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .genome import Genome, recombine, mutate, random_genome
from .navigation import (
    Pathfinder, StuckDetector, land_safe, line_of_sight_clear, steer_toward,
)

if TYPE_CHECKING:  # avoid runtime circular import
    from .world import World
    from .weather import WeatherSystem
    from .species import Species
    from .simulation import Simulation

__all__ = ["OrganismState", "Organism", "OrganismEvent", "BirthRequest"]


class OrganismState(IntEnum):
    """High-level behavioural state, for rendering and the inspector."""

    IDLE = 0
    FORAGING = 1     # seeking plant food / prey
    HUNTING = 2      # actively chasing prey
    DRINKING = 3
    FLEEING = 4
    MATING = 5
    WANDERING = 6
    RESTING = 7      # low-energy / low-health conservation


STATE_LABELS = {
    OrganismState.IDLE: "Idle",
    OrganismState.FORAGING: "Foraging",
    OrganismState.HUNTING: "Hunting",
    OrganismState.DRINKING: "Drinking",
    OrganismState.FLEEING: "Fleeing",
    OrganismState.MATING: "Seeking mate",
    OrganismState.WANDERING: "Wandering",
    OrganismState.RESTING: "Resting",
}


@dataclass
class OrganismEvent:
    """A single notable event in this organism's life - for the journal/inspector."""

    tick: int
    kind: str
    detail: str = ""

    def to_dict(self) -> Dict:
        return {"tick": self.tick, "kind": self.kind, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Dict) -> "OrganismEvent":
        return cls(tick=int(data["tick"]), kind=str(data["kind"]), detail=str(data.get("detail", "")))


# ---------------------------------------------------------------------------
# Need-urgency thresholds. Tuned so that an organism with comfortable
# stats (energy > 60, hydration > 60, health > 70) has no urgent needs
# and can reproduce or explore, while a struggling organism prioritises
# survival.
# ---------------------------------------------------------------------------
HEALTH_CRITICAL = 35.0
HEALTH_LOW = 55.0
WATER_CRITICAL = 20.0
WATER_LOW = 45.0
ENERGY_CRITICAL_FRAC = 0.15   # fraction of max_energy
ENERGY_LOW_FRAC = 0.50
REST_HEALTH_RECOVER = 0.15    # health regained per tick while resting


@dataclass
class Organism:
    """A single living (or dead) organism."""

    id: int
    genome: Genome
    x: float
    y: float
    species_id: int
    birth_tick: int
    parent_ids: Tuple[int, ...] = ()
    # Vital stats
    age: float = 0.0
    energy: float = 60.0
    hydration: float = 70.0
    health: float = 100.0
    # Behaviour
    state: OrganismState = OrganismState.IDLE
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_id: Optional[int] = None
    cooldown_repro: int = 0
    # Memory
    events: List[OrganismEvent] = field(default_factory=list)
    last_action: str = "Just born"
    offspring_count: int = 0
    kills: int = 0
    alive: bool = True
    death_tick: Optional[int] = None
    death_cause: str = ""
    # --- Navigation state (not serialised; recomputed on load) ---
    _path: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    _path_goal: Optional[Tuple[float, float]] = field(default=None, repr=False)
    _stuck: StuckDetector = field(default_factory=StuckDetector, repr=False)
    _resting_ticks: int = 0
    # Cached needs assessment — only re-scan for targets every N ticks to
    # avoid O(vision^2) tile scans every tick for every organism.
    _needs_cache: List = field(default_factory=list, repr=False)
    _needs_cache_tick: int = -999
    _NEEDS_RESCAN_EVERY: int = 3
    # Cached derived stats
    _max_energy: float = 100.0
    _size_radius: float = 0.35

    def __post_init__(self) -> None:
        self._recompute_derived()

    def _recompute_derived(self) -> None:
        self._max_energy = 60.0 + 50.0 * self.genome["size"]
        self._size_radius = 0.18 + 0.22 * self.genome["size"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def max_energy(self) -> float:
        return self._max_energy

    @property
    def radius(self) -> float:
        return self._size_radius

    @property
    def is_predator(self) -> bool:
        return self.genome["diet"] > 0.6

    @property
    def is_herbivore(self) -> bool:
        return self.genome["diet"] < 0.4

    @property
    def is_omnivore(self) -> bool:
        return 0.4 <= self.genome["diet"] <= 0.6

    @property
    def maturity_progress(self) -> float:
        return min(1.0, self.age / self.genome["maturity_age"])

    @property
    def is_mature(self) -> bool:
        return self.age >= self.genome["maturity_age"]

    @property
    def age_progress(self) -> float:
        return min(1.0, self.age / self.genome["max_age"])

    @property
    def color(self) -> Tuple[float, float, float]:
        return (self.genome["color_hue"], self.genome["color_sat"], self.genome["color_light"])

    @property
    def energy_critical(self) -> bool:
        return self.energy < self._max_energy * ENERGY_CRITICAL_FRAC

    @property
    def energy_low(self) -> bool:
        return self.energy < self._max_energy * ENERGY_LOW_FRAC

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------
    def step(self, world: "World", weather: "WeatherSystem", rng: random.Random, tick: int,
             neighbours: List["Organism"], sim: Optional["Simulation"] = None) -> Optional["BirthRequest"]:
        if not self.alive:
            return None
        # Determine if resting (reduces metabolism).
        resting = self.state == OrganismState.RESTING
        crowd = len(neighbours)
        self._metabolise(world, weather, crowd, resting)
        self.age += 1.0
        if self.cooldown_repro > 0:
            self.cooldown_repro -= 1
        # Resting recovers a little health if not under extreme weather.
        if resting and self.health < 100.0:
            if 0.2 < weather.effective_temp < 0.8:
                self.health = min(100.0, self.health + REST_HEALTH_RECOVER)
        if self._check_death(tick, world):
            return None
        return self._decide_and_act(world, weather, rng, tick, neighbours, sim)

    def _metabolise(self, world: "World", weather: "WeatherSystem",
                    crowd: int = 0, resting: bool = False) -> None:
        size_cost = self.genome["size"] ** 2 * 0.14
        meta_cost = self.genome["metabolism"] * 0.22
        # Night or resting reduces metabolism.
        factor = 0.65 if (resting or not world.is_day()) else 1.0
        # Mild crowding stress.
        stress = 1.0 + 0.04 * min(crowd, 8)
        env_damage = weather.damage_per_tick * (1.0 - 0.6 * self.genome["hardiness"])
        self.energy -= (size_cost + meta_cost) * factor * stress
        self.hydration -= self.genome["water_need"] * 0.15 * factor * stress
        if weather.effective_temp > 0.85 or weather.effective_temp < 0.15:
            self.health -= env_damage
        else:
            self.health -= env_damage * 0.35
        self.energy = max(0.0, self.energy)
        self.hydration = max(0.0, self.hydration)
        self.health = max(0.0, min(100.0, self.health))

    def _check_death(self, tick: int, world: "World") -> bool:
        if self.health <= 0.0:
            self._die(tick, "injury")
            return True
        if self.energy <= 0.0:
            self._die(tick, "starvation")
            return True
        if self.hydration <= 0.0:
            self._die(tick, "dehydration")
            return True
        if self.age >= self.genome["max_age"]:
            self._die(tick, "old age")
            return True
        return False

    def _die(self, tick: int, cause: str) -> None:
        self.alive = False
        self.death_tick = tick
        self.death_cause = cause
        self.events.append(OrganismEvent(tick=tick, kind="died", detail=cause))
        self.last_action = f"Died ({cause})"

    # ------------------------------------------------------------------
    # Decision making — needs-urgency hierarchy
    # ------------------------------------------------------------------
    def _decide_and_act(
        self,
        world: "World",
        weather: "WeatherSystem",
        rng: random.Random,
        tick: int,
        neighbours: List["Organism"],
        sim: Optional["Simulation"] = None,
    ) -> Optional["BirthRequest"]:
        vision = self.genome["vision_range"] * world.daylight() * weather.vision_multiplier
        vision = max(2.5, vision)

        # 1. Safety check: if somehow on an unsafe tile, escape immediately.
        if not world.is_safe_tile(self.x, self.y):
            return self._escape_to_safety(world, weather, rng, tick)

        # 2. Detect and respond to threats.
        threats = self._detect_threats(neighbours, vision)
        if threats and self._should_flee(threats):
            return self._flee_from(world, weather, rng, tick, threats)

        # 3. Assess needs in urgency order (cached to avoid per-tick scans).
        needs = self._get_needs(world, vision, tick)
        if needs:
            top = needs[0]
            result = self._handle_need(top, world, weather, rng, tick, neighbours, vision, sim)
            if result is not None:
                return result
            # If we acted on the need (state changed), don't also reproduce.
            if self.state in (OrganismState.DRINKING, OrganismState.FORAGING,
                              OrganismState.HUNTING, OrganismState.RESTING):
                return None

        # 4. Reproduction (if healthy and mature).
        if self._can_reproduce_safely():
            result = self._seek_mate(world, weather, rng, tick, neighbours, vision, sim)
            if result is not None:
                return result
            if self.state == OrganismState.MATING:
                return None

        # 5. Opportunistic hunting for hungry predators.
        if self.is_predator and self.energy < self.genome["repro_threshold"]:
            result = self._hunt(world, weather, rng, tick, neighbours, vision)
            if result is not None or self.state == OrganismState.HUNTING:
                return None

        # 6. Top up energy if not full.
        if self.energy < self._max_energy * 0.80:
            return self._forage(world, weather, rng, tick, vision=vision, sim=sim)

        # 7. Default: explore.
        return self._explore(world, weather, rng, tick)

    # ----- needs assessment --------------------------------------------
    def _get_needs(self, world: "World", vision: float, tick: int):
        """Cached wrapper around :meth:`_assess_needs`.

        Needs are only re-scanned every ``_NEEDS_RESCAN_EVERY`` ticks (or
        on the first call). Between scans, the cached target list is
        returned. This avoids O(radius²) tile scans every tick for every
        organism, which was the main performance bottleneck.
        """
        if tick - self._needs_cache_tick >= self._NEEDS_RESCAN_EVERY:
            self._needs_cache = self._assess_needs(world, vision)
            self._needs_cache_tick = tick
        return self._needs_cache

    def _assess_needs(self, world: "World", vision: float) -> List[Tuple[str, float, Optional[Tuple[float, float]]]]:
        """Return needs sorted by urgency (highest first).

        Each entry is ``(need_type, urgency_0_to_1, target_xy_or_None)``.
        Targets are pre-filtered to be reachable (same landmass).
        """
        needs: List[Tuple[str, float, Optional[Tuple[float, float]]]] = []
        max_e = self._max_energy
        # Health
        if self.health < HEALTH_LOW:
            urg = (HEALTH_LOW - self.health) / HEALTH_LOW
            needs.append(("health", urg, None))
        # Hydration — search a wider radius than vision so interior
        # dwellers can still find shore/springs.
        if self.hydration < WATER_LOW:
            urg = max(0.0, (WATER_LOW - self.hydration) / WATER_LOW)
            search_r = max(int(vision) + 1, 15)
            target = world.nearest_drinking_spot(self.x, self.y, search_r)
            needs.append(("water", urg, target))
        # Energy
        if self.energy < max_e * ENERGY_LOW_FRAC:
            urg = max(0.0, (max_e * ENERGY_LOW_FRAC - self.energy) / (max_e * ENERGY_LOW_FRAC))
            target = world.best_reachable_food_tile(self.x, self.y, int(vision) + 1)
            needs.append(("food", urg, target))
        needs.sort(key=lambda n: -n[1])
        return needs

    def _handle_need(
        self,
        need: Tuple[str, float, Optional[Tuple[float, float]]],
        world: "World",
        weather: "WeatherSystem",
        rng: random.Random,
        tick: int,
        neighbours: List["Organism"],
        vision: float,
        sim: Optional["Simulation"],
    ) -> Optional["BirthRequest"]:
        need_type, urgency, target = need
        if need_type == "water":
            return self._seek_water(world, weather, rng, tick, target, vision, sim)
        if need_type == "food":
            # Predators below critical energy try hunting first.
            if self.is_predator and self.energy < self._max_energy * ENERGY_CRITICAL_FRAC:
                result = self._hunt(world, weather, rng, tick, neighbours, vision)
                if result is not None or self.state == OrganismState.HUNTING:
                    return None
            return self._forage(world, weather, rng, tick, target, vision, sim)
        if need_type == "health":
            return self._rest(world, weather, rng, tick, reason="recovering")
        return None

    # ----- threat detection --------------------------------------------
    def _detect_threats(self, neighbours: List["Organism"], vision: float) -> List["Organism"]:
        """Return predators within vision that can eat us."""
        return [o for o in neighbours if o.alive and o.id != self.id and
                self._dist_to(o) <= vision and o.is_predator and
                self._is_edible_to(o)]

    def _should_flee(self, threats: List["Organism"]) -> bool:
        """Decide if fleeing is worth it.

        Flee if the closest predator is within ~60% of our vision and
        we're not so critically starved that we'd die trying to run.
        """
        if not threats:
            return False
        closest = min(threats, key=lambda o: self._dist_to(o))
        d = self._dist_to(closest)
        # Flee if predator is close (within 8 tiles) or actively hunting us.
        if d < 8.0:
            return self.energy > self._max_energy * ENERGY_CRITICAL_FRAC * 0.5
        return False

    # ----- actions ------------------------------------------------------
    def _flee_from(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                   tick: int, threats: List["Organism"]) -> None:
        threat = min(threats, key=lambda o: self._dist_to(o))
        # Move directly away from the threat, using safe steering.
        dx = self.x - threat.x
        dy = self.y - threat.y
        d = math.hypot(dx, dy) or 1.0
        flee_dist = 4.0
        tx = self.x + dx / d * flee_dist
        ty = self.y + dy / d * flee_dist
        # Clamp to world bounds.
        tx = max(1.0, min(world.width - 1.0, tx))
        ty = max(1.0, min(world.height - 1.0, ty))
        self._move_toward(world, tx, ty)
        self.state = OrganismState.FLEEING
        self.last_action = f"Fleeing from predator #{threat.id}"
        self.events.append(OrganismEvent(tick=tick, kind="fled", detail=f"#{threat.id}"))
        # Clear any pending path.
        self._path.clear()
        self._path_goal = None

    def _seek_water(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                    tick: int, target: Optional[Tuple[float, float]],
                    vision: float, sim: Optional["Simulation"]) -> None:
        # If we can drink right here, do so.
        if world.can_drink_at(self.x, self.y):
            drank = self._drink_from(world)
            if drank > 0:
                self.state = OrganismState.DRINKING
                self.last_action = f"Drinking (+{drank:.0f} hydration)"
                self.events.append(OrganismEvent(tick=tick, kind="drank", detail=f"+{drank:.0f}"))
                return
        if target is None:
            # No water in vision — wander to search.
            self._explore(world, weather, rng, tick, prefer_near_water=True)
            self.last_action = "Thirsty; searching for water"
            return
        # Move toward the drinking spot safely.
        d = math.hypot(target[0] - self.x, target[1] - self.y)
        if d < 1.5:
            drank = self._drink_from(world)
            if drank > 0:
                self.state = OrganismState.DRINKING
                self.last_action = f"Drinking (+{drank:.0f} hydration)"
                self.events.append(OrganismEvent(tick=tick, kind="drank", detail=f"+{drank:.0f}"))
                return
        self._move_toward(world, target[0], target[1], sim=sim)
        self.state = OrganismState.DRINKING
        self.last_action = "Heading to water"

    def _drink_from(self, world: "World") -> float:
        """Drink from adjacent water without entering it."""
        # Try water resource nodes first.
        for res in world.resources:
            if res.kind.name == "WATER" and abs(res.x - self.x) <= 1.5 and abs(res.y - self.y) <= 1.5:
                return res.consume(25.0)
        # Else drink from adjacent water tile (infinite).
        ix, iy = int(self.x), int(self.y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tile = world.tile_at(ix + dx, iy + dy)
                if tile is not None and tile.has_water:
                    drank = 25.0
                    self.hydration = min(100.0, self.hydration + drank)
                    return drank
        return 0.0

    def _forage(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                tick: int, target: Optional[Tuple[float, float]] = None,
                vision: float = 0.0, sim: Optional["Simulation"] = None) -> None:
        # Eat plant food at current tile if there's a meaningful amount.
        # This keeps organisms dispersed instead of all rushing to the
        # single best food tile (which causes local overgrazing).
        tile = world.tile_at(int(self.x), int(self.y))
        if tile is not None and tile.food > 0.5:
            want = min(self._max_energy - self.energy, 10.0)
            eaten = world.consume_plant_food(tile.x, tile.y, want)
            if eaten > 0:
                gained = eaten * tile.params["food_value"]
                self.energy = min(self._max_energy, self.energy + gained)
                self.state = OrganismState.FORAGING
                self.last_action = f"Grazing (+{gained:.0f} energy)"
                self.events.append(OrganismEvent(tick=tick, kind="ate", detail=f"plant +{gained:.0f}"))
                return
        # Current tile is bare — move toward the target or search.
        if target is not None:
            self._move_toward(world, target[0], target[1], sim=sim)
            self.state = OrganismState.FORAGING
            self.last_action = "Foraging toward food"
            return
        if vision > 0:
            reachable = world.best_reachable_food_tile(self.x, self.y, int(vision) + 1)
            if reachable is not None:
                self._move_toward(world, reachable[0], reachable[1], sim=sim)
                self.state = OrganismState.FORAGING
                self.last_action = "Foraging toward food"
                return
        # No food found — explore toward food-rich areas.
        self._explore(world, weather, rng, tick)
        self.last_action = "Foraging; no food in sight"

    def _hunt(self, world: "World", weather: "WeatherSystem", rng: random.Random,
              tick: int, neighbours: List["Organism"], vision: float) -> Optional["BirthRequest"]:
        # Only hunt prey on the same landmass.
        prey = [o for o in neighbours if o.alive and o.id != self.id and
                self._can_eat(o) and self._dist_to(o) <= vision and
                world.same_landmass(self.x, self.y, o.x, o.y)]
        if not prey:
            return None
        # Prefer the closest, weakest prey.
        target = min(prey, key=lambda o: self._dist_to(o) + o.health * 0.01)
        d = self._dist_to(target)
        if d < 0.8:
            damage = 30.0 + 40.0 * self.genome["aggression"] + 20.0 * self.genome["size"]
            target.health -= damage
            self.state = OrganismState.HUNTING
            self.last_action = f"Attacking #{target.id}"
            self.events.append(OrganismEvent(tick=tick, kind="attacked", detail=f"#{target.id}"))
            target.events.append(OrganismEvent(tick=tick, kind="attacked", detail=f"by #{self.id}"))
            if target.health <= 0:
                gained = min(self._max_energy - self.energy, target.energy * 0.7 + 30.0)
                self.energy = min(self._max_energy, self.energy + gained)
                self.kills += 1
                target._die(tick, f"hunted by #{self.id}")
                self.last_action = f"Killed #{target.id} (+{gained:.0f})"
                self.events.append(OrganismEvent(tick=tick, kind="ate", detail=f"prey #{target.id} +{gained:.0f}"))
            return None
        self._move_toward(world, target.x, target.y)
        self.target_id = target.id
        self.state = OrganismState.HUNTING
        self.last_action = f"Chasing #{target.id}"
        return None

    def _seek_mate(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                   tick: int, neighbours: List["Organism"], vision: float,
                   sim: Optional["Simulation"] = None,
                   ) -> Optional["BirthRequest"]:
        # Eligible mates on the same landmass.
        mates = [o for o in neighbours if o.alive and o.id != self.id and
                 o.species_id == self.species_id and o.is_mature and
                 o.energy >= o.genome["repro_threshold"] and o.cooldown_repro <= 0 and
                 self._dist_to(o) <= vision and
                 world.same_landmass(self.x, self.y, o.x, o.y)]
        if not mates and sim is not None:
            # Long-range search through the spatial hash, filtered by landmass.
            far_mate = sim.find_nearest_potential_mate(self, max_cells=4)
            if far_mate is not None and world.same_landmass(self.x, self.y, far_mate.x, far_mate.y):
                self._move_toward(world, far_mate.x, far_mate.y, sim=sim)
                if far_mate.state not in (OrganismState.FLEEING, OrganismState.HUNTING):
                    far_mate._move_toward(world, self.x, self.y)
                    far_mate.state = OrganismState.MATING
                    far_mate.last_action = f"Heading toward distant mate #{self.id}"
                self.state = OrganismState.MATING
                self.last_action = f"Heading toward distant mate #{far_mate.id}"
                return None
        if not mates:
            same_species = [o for o in neighbours if o.alive and o.id != self.id and
                            o.species_id == self.species_id and self._dist_to(o) <= vision and
                            world.same_landmass(self.x, self.y, o.x, o.y)]
            if same_species:
                target = min(same_species, key=lambda o: self._dist_to(o))
                self._move_toward(world, target.x, target.y, sim=sim)
                self.state = OrganismState.MATING
                self.last_action = f"Approaching mate #{target.id}"
                return None
            return None  # Don't wander — let the caller decide.
        mate = min(mates, key=lambda o: self._dist_to(o))
        d = self._dist_to(mate)
        repro_dist = 2.0 + 0.8 * (self.genome["size"] + mate.genome["size"]) * 0.5
        if d < repro_dist:
            return self._reproduce_with(mate, rng, tick)
        self._move_toward(world, mate.x, mate.y, sim=sim)
        if mate.state not in (OrganismState.FLEEING, OrganismState.HUNTING):
            mate._move_toward(world, self.x, self.y)
        self.state = OrganismState.MATING
        self.last_action = f"Moving toward mate #{mate.id}"
        return None

    def _can_reproduce_safely(self) -> bool:
        """True if reproduction won't immediately endanger the organism."""
        if not self.is_mature or self.cooldown_repro > 0:
            return False
        if self.energy < self.genome["repro_threshold"]:
            return False
        if self.hydration < WATER_LOW:
            return False
        if self.health < HEALTH_LOW:
            return False
        # Don't reproduce if it would drop energy below critical — but
        # allow it if the organism is well above the threshold.
        if self.energy - self.genome["offspring_energy"] < self._max_energy * ENERGY_CRITICAL_FRAC:
            return False
        return True

    def _reproduce_with(self, mate: "Organism", rng: random.Random, tick: int) -> "BirthRequest":
        child_genome = recombine(self.genome, mate.genome, rng)
        cost_a = self.genome["offspring_energy"]
        cost_b = mate.genome["offspring_energy"]
        self.energy -= cost_a
        mate.energy -= cost_b
        self.cooldown_repro = 70
        mate.cooldown_repro = 70
        self.offspring_count += 1
        mate.offspring_count += 1
        litter = max(1, min(int(child_genome["litter_size"]), 3))
        self.events.append(OrganismEvent(tick=tick, kind="gave_birth", detail=f"#{mate.id} x{litter_count_label(litter)}"))
        mate.events.append(OrganismEvent(tick=tick, kind="gave_birth", detail=f"#{self.id} x{litter_count_label(litter)}"))
        self.last_action = f"Mated with #{mate.id}"
        mate.last_action = f"Mated with #{self.id}"
        cx = 0.5 * (self.x + mate.x) + rng.uniform(-1.0, 1.0)
        cy = 0.5 * (self.y + mate.y) + rng.uniform(-1.0, 1.0)
        return BirthRequest(
            genome=child_genome,
            x=cx, y=cy,
            parent_ids=(self.id, mate.id),
            parent_species_id=self.species_id,
            litter_size=litter,
            tick=tick,
        )

    def _rest(self, world: "World", weather: "WeatherSystem", rng: random.Random,
              tick: int, reason: str = "") -> None:
        self.state = OrganismState.RESTING
        self._resting_ticks += 1
        self.last_action = f"Resting ({reason})" if reason else "Resting"
        # Clear movement targets.
        self.target_x = None
        self.target_y = None
        self._path.clear()
        self._path_goal = None

    def _explore(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                 tick: int, prefer_near_water: bool = False) -> None:
        """Wander toward a safe, food-rich area. Stays on safe terrain."""
        # Pick a wander target that's on safe ground.
        if self.target_x is None or self.target_y is None or rng.random() < 0.05:
            # Try to find a nearby food-rich tile to wander toward.
            best = None
            best_food = 0.5
            cx, cy = int(self.x), int(self.y)
            my_lm = world.landmass_id(self.x, self.y)
            r = 8
            for j in range(max(0, cy - r), min(world.height, cy + r + 1)):
                for i in range(max(0, cx - r), min(world.width, cx + r + 1)):
                    if my_lm != -1 and world._landmass[i][j] != my_lm:
                        continue
                    tile = world.tile_at(i, j)
                    if tile is None or tile.params["passable"] < 0.85:
                        continue
                    score = tile.food * 0.5
                    if prefer_near_water and world.can_drink_at(i + 0.5, j + 0.5):
                        score += 10.0
                    if score > best_food:
                        best_food = score
                        best = (float(i) + 0.5, float(j) + 0.5)
            if best is not None:
                self.target_x, self.target_y = best
            else:
                # Random safe direction.
                for _attempt in range(10):
                    angle = rng.uniform(0, 2 * math.pi)
                    dist = rng.uniform(4, 10)
                    tx = self.x + math.cos(angle) * dist
                    ty = self.y + math.sin(angle) * dist
                    if world.is_safe_tile(tx, ty):
                        self.target_x = tx
                        self.target_y = ty
                        break
                else:
                    self.target_x = self.x
                    self.target_y = self.y
        if self.target_x is not None and self.target_y is not None:
            self._move_toward(world, self.target_x, self.target_y)
            if math.hypot(self.target_x - self.x, self.target_y - self.y) < 1.5:
                self.target_x = None
                self.target_y = None
        if not world.is_day() and self.energy > self._max_energy * 0.65:
            self.state = OrganismState.RESTING
            self.last_action = "Resting (night)"
        else:
            self.state = OrganismState.WANDERING
            self.last_action = "Exploring"

    def _escape_to_safety(self, world: "World", weather: "WeatherSystem",
                          rng: random.Random, tick: int) -> None:
        """Move to the nearest safe tile (used if somehow on water/unsafe)."""
        best = None
        best_d2 = float("inf")
        cx, cy = int(self.x), int(self.y)
        for ring in range(1, 6):
            found = False
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if max(abs(dx), abs(dy)) != ring:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < world.width and 0 <= ny < world.height):
                        continue
                    tile = world.tile_at(nx, ny)
                    if tile is not None and tile.params["passable"] >= 0.85:
                        d2 = (nx - self.x) ** 2 + (ny - self.y) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (float(nx) + 0.5, float(ny) + 0.5)
                            found = True
            if found:
                break
        if best is not None:
            speed = self.genome["speed"] * 0.7  # move faster when escaping
            self.x = best[0]
            self.y = best[1]
            self.x = max(0.1, min(world.width - 0.1, self.x))
            self.y = max(0.1, min(world.height - 0.1, self.y))
        self.state = OrganismState.FLEEING
        self.last_action = "Escaping unsafe terrain"
        self.events.append(OrganismEvent(tick=tick, kind="fled", detail="unsafe terrain"))

    # ------------------------------------------------------------------
    # Movement — terrain-aware steering with A* fallback
    # ------------------------------------------------------------------
    def _move_toward(self, world: "World", tx: float, ty: float,
                     sim: Optional["Simulation"] = None) -> None:
        """Move toward ``(tx, ty)`` using safe steering, with A* fallback.

        The organism first tries :func:`steer_toward` (cheap local
        avoidance). If it gets stuck (no progress for several ticks),
        it requests an A* path from the simulation's shared
        :class:`Pathfinder` and follows the waypoints until the goal is
        reached or the path becomes stale.
        """
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        speed = self.genome["speed"] * 0.5
        # Update stuck detector.
        self._stuck.update(dist, target_id=int(tx * 10000 + ty))

        # If we have a cached path toward this goal, follow it.
        goal_key = (int(tx), int(ty))
        if self._path and self._path_goal == goal_key:
            waypoint = self._path[0]
            wdist = math.hypot(waypoint[0] - self.x, waypoint[1] - self.y)
            if wdist < 1.0:
                self._path.pop(0)
                if not self._path:
                    self._path_goal = None
                else:
                    waypoint = self._path[0]
            nx, ny = steer_toward(world, self.x, self.y, waypoint[0], waypoint[1], speed)
            self.x = nx
            self.y = ny
            self._clamp_to_bounds(world)
            self.target_x = tx
            self.target_y = ty
            return

        # Try direct steering first.
        nx, ny = steer_toward(world, self.x, self.y, tx, ty, speed)
        # Final safety check: never move to an unsafe tile. If steering
        # somehow returned an unsafe position (e.g. the organism was
        # nudged by another organism's mate-seeking), stay put instead.
        if not world.is_safe_tile(nx, ny):
            # Try to find a safe adjacent tile to move to.
            cx, cy = int(self.x), int(self.y)
            escaped = False
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    test_tile = world.tile_at(cx + ddx, cy + ddy)
                    if test_tile is not None and test_tile.params["passable"] >= 0.85:
                        nx = float(cx + ddx) + 0.5
                        ny = float(cy + ddy) + 0.5
                        escaped = True
                        break
                if escaped:
                    break
            if not escaped:
                nx, ny = self.x, self.y  # stay put
        moved = math.hypot(nx - self.x, ny - self.y) > 0.01
        self.x = nx
        self.y = ny
        self._clamp_to_bounds(world)
        self.target_x = tx
        self.target_y = ty

        # If steering made no progress, try A* pathfinding.
        if not moved and sim is not None and not self._stuck.is_stuck():
            # Mark as stuck to avoid re-triggering every tick.
            self._stuck.update(dist + 0.001, target_id=int(tx * 10000 + ty))
        if sim is not None and self._stuck.is_stuck() and not self._path:
            pathfinder = sim.pathfinder
            path = pathfinder.find_path(self.x, self.y, tx, ty)
            if path is not None and len(path) > 1:
                self._path = path[1:]  # skip the start tile
                self._path_goal = goal_key
            else:
                # No path found — give up on this target.
                self.target_x = None
                self.target_y = None
                self._stuck.reset()
            # Reset stuck detector either way.
            self._stuck.reset()

    def _clamp_to_bounds(self, world: "World") -> None:
        self.x = max(0.1, min(world.width - 0.1, self.x))
        self.y = max(0.1, min(world.height - 0.1, self.y))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dist_to(self, other: "Organism") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def _is_edible_to(self, predator: "Organism") -> bool:
        """Return True if ``predator`` can eat *us*."""
        if predator.species_id == self.species_id:
            return False
        if predator.genome["diet"] < 0.4:
            return False
        return predator.genome["size"] >= self.genome["size"] * 0.85

    def _can_eat(self, other: "Organism") -> bool:
        """Return True if *we* can eat ``other``."""
        if self.species_id == other.species_id:
            return False
        if self.genome["diet"] < 0.4:
            return False
        return self.genome["size"] >= other.genome["size"] * 0.85

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genome": self.genome.to_json(),
            "x": self.x, "y": self.y,
            "species_id": self.species_id,
            "birth_tick": self.birth_tick,
            "parent_ids": list(self.parent_ids),
            "age": self.age,
            "energy": self.energy,
            "hydration": self.hydration,
            "health": self.health,
            "state": int(self.state),
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_id": self.target_id,
            "cooldown_repro": self.cooldown_repro,
            "events": [e.to_dict() for e in self.events[-64:]],
            "last_action": self.last_action,
            "offspring_count": self.offspring_count,
            "kills": self.kills,
            "alive": self.alive,
            "death_tick": self.death_tick,
            "death_cause": self.death_cause,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Organism":
        return cls(
            id=int(data["id"]),
            genome=Genome.from_json(data["genome"]),
            x=float(data["x"]), y=float(data["y"]),
            species_id=int(data["species_id"]),
            birth_tick=int(data["birth_tick"]),
            parent_ids=tuple(data.get("parent_ids", [])),
            age=float(data["age"]),
            energy=float(data["energy"]),
            hydration=float(data["hydration"]),
            health=float(data["health"]),
            state=OrganismState(int(data["state"])),
            target_x=data.get("target_x"),
            target_y=data.get("target_y"),
            target_id=data.get("target_id"),
            cooldown_repro=int(data["cooldown_repro"]),
            events=[OrganismEvent.from_dict(e) for e in data.get("events", [])],
            last_action=str(data.get("last_action", "")),
            offspring_count=int(data.get("offspring_count", 0)),
            kills=int(data.get("kills", 0)),
            alive=bool(data["alive"]),
            death_tick=data.get("death_tick"),
            death_cause=str(data.get("death_cause", "")),
        )


# ---------------------------------------------------------------------------
# BirthRequest
# ---------------------------------------------------------------------------
@dataclass
class BirthRequest:
    genome: Genome
    x: float
    y: float
    parent_ids: Tuple[int, ...]
    parent_species_id: int
    litter_size: int
    tick: int


def litter_count_label(n: int) -> str:
    return str(n)
