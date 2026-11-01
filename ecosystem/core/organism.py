"""Organism: an individual living agent.

Each organism is an autonomous agent that, every tick:

1. Ages (and consumes energy proportional to its metabolism & size).
2. Evaluates its needs (hunger, thirst, reproduction, safety).
3. Picks a target (food, water, mate, or flee-from-predator) based on the
   most pressing need and what it can *see* within its vision range.
4. Moves toward that target, possibly interacting when it arrives.
5. May reproduce, die of old age / starvation / dehydration / injury.

The decision logic is deliberately simple (greedy over a need-weighted
score) so that emergent behaviour arises from the interaction of many
agents rather than from clever individual AI.

The class is plain Python (no Qt) and serialises itself to a dict, so the
whole simulation can be saved/loaded as JSON.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .genome import Genome, recombine, mutate, random_genome

if TYPE_CHECKING:  # avoid runtime circular import
    from .world import World
    from .weather import WeatherSystem
    from .species import Species
    from .simulation import Simulation

__all__ = ["OrganismState", "Organism", "OrganismEvent"]


class OrganismState(IntEnum):
    """High-level behavioural state, for rendering and the inspector."""

    IDLE = 0
    FORAGING = 1     # seeking plant food / prey
    HUNTING = 2      # actively chasing prey
    DRINKING = 3
    FLEEING = 4
    MATING = 5
    WANDERING = 6
    RESTING = 7      # night-time low-energy state


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
    kind: str          # 'birth','ate','drank','mated','gave_birth','fled','attacked','mutated','died','aged'
    detail: str = ""

    def to_dict(self) -> Dict:
        return {"tick": self.tick, "kind": self.kind, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Dict) -> "OrganismEvent":
        return cls(tick=int(data["tick"]), kind=str(data["kind"]), detail=str(data.get("detail", "")))


@dataclass
class Organism:
    """A single living (or dead) organism.

    The fields are kept as plain attributes so PySide6's model/view code
    can introspect them cheaply. ``genome`` is a :class:`Genome` instance.
    """

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
    target_id: Optional[int] = None   # id of organism being chased/fled
    cooldown_repro: int = 0           # ticks until can reproduce again
    # Memory
    events: List[OrganismEvent] = field(default_factory=list)
    # Cached last action description, for the inspector's "current behaviour".
    last_action: str = "Just born"
    # Reproduction counter
    offspring_count: int = 0
    # Kills counter (for predators)
    kills: int = 0
    alive: bool = True
    death_tick: Optional[int] = None
    death_cause: str = ""
    # Cached derived stats (recomputed when needed)
    _max_energy: float = 100.0
    _size_radius: float = 0.35

    # ------------------------------------------------------------------
    # Construction & derived stats
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._recompute_derived()

    def _recompute_derived(self) -> None:
        # Max energy scales with size: bigger = more storage but more upkeep.
        self._max_energy = 60.0 + 50.0 * self.genome["size"]
        self._size_radius = 0.18 + 0.22 * self.genome["size"]

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
        """HSL colour derived from the genome."""
        return (self.genome["color_hue"], self.genome["color_sat"], self.genome["color_light"])

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------
    def step(self, world: "World", weather: "WeatherSystem", rng: random.Random, tick: int,
             neighbours: List["Organism"], sim: Optional["Simulation"] = None) -> Optional["BirthRequest"]:
        """Advance this organism by one tick.

        Returns a :class:`BirthRequest` if it reproduced this tick (so the
        simulation can spawn the child without this module needing to know
        about the wider population), otherwise ``None``.

        ``sim`` is optional and used only for the long-range mate search
        when the local neighbour window has no eligible mates.
        """
        if not self.alive:
            return None
        # 1. Metabolism: lose energy & water per tick. Crowding nearby
        #    organisms increases energy cost (competition stress).
        crowd = len(neighbours)
        self._metabolise(world, weather, crowd)
        # 2. Ageing.
        self.age += 1.0
        # 3. Cooldowns.
        if self.cooldown_repro > 0:
            self.cooldown_repro -= 1
        # 4. Death checks.
        if self._check_death(tick, world):
            return None
        # 5. Decision making + action.
        return self._decide_and_act(world, weather, rng, tick, neighbours, sim)

    def _metabolise(self, world: "World", weather: "WeatherSystem", crowd: int = 0) -> None:
        # Energy cost: base + size^2 + speed (if moving) + metabolism.
        size_cost = self.genome["size"] ** 2 * 0.18
        meta_cost = self.genome["metabolism"] * 0.28
        # Reduced metabolism at night (resting).
        night_factor = 0.7 if not world.is_day() else 1.0
        # Crowding stress: up to ~1.4x energy cost when 8+ neighbours.
        # Mild enough to disperse populations but not prevent reproduction.
        stress = 1.0 + 0.05 * min(crowd, 8)
        # Hardiness reduces environmental damage.
        env_damage = weather.damage_per_tick * (1.0 - 0.6 * self.genome["hardiness"])
        self.energy -= (size_cost + meta_cost) * night_factor * stress
        self.hydration -= self.genome["water_need"] * 0.18 * night_factor * stress
        # Temperature stress: if effective temp far from comfort zone, lose health.
        if weather.effective_temp > 0.85 or weather.effective_temp < 0.15:
            self.health -= env_damage
        else:
            self.health -= env_damage * 0.4
        # Clamp.
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
    # Decision making
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
        # Vision range affected by daylight and weather.
        vision = self.genome["vision_range"] * world.daylight() * weather.vision_multiplier
        vision = max(1.5, vision)

        # Threat detection: any predator within vision that targets us.
        threats = [o for o in neighbours if o.alive and o.id != self.id and
                   self._dist_to(o) <= vision and o.is_predator and
                   o.genome["size"] >= self.genome["size"] * 0.8 and
                   self._is_edible_to(o)]

        # 1) Flee if a predator is close and we're not starving.
        if threats and self.energy > 12.0:
            threat = min(threats, key=lambda o: self._dist_to(o))
            return self._flee_from(world, weather, rng, tick, threat)

        # 2) Critical needs: water or food.
        critical_water = self.hydration < 25.0
        critical_food = self.energy < 25.0

        if critical_water:
            return self._seek_water(world, weather, rng, tick)

        if critical_food:
            # Predators prefer hunting; herbivores prefer foraging.
            if self.is_predator:
                req = self._hunt(world, weather, rng, tick, neighbours, vision)
                if req is not None:
                    return req
                # If we just attacked (state flipped to HUNTING), don't
                # immediately forage on top of the prey.
                if self.state == OrganismState.HUNTING:
                    return None
                return self._forage(world, weather, rng, tick, vision)
            return self._forage(world, weather, rng, tick, vision)

        # 3) Reproduce if mature and well-fed.
        if (self.is_mature and self.energy >= self.genome["repro_threshold"]
                and self.cooldown_repro <= 0 and self.hydration > 35.0):
            return self._seek_mate(world, weather, rng, tick, neighbours, vision, sim)

        # 4) Opportunistic hunting for predators.
        if self.is_predator and self.energy < self.genome["repro_threshold"] * 0.8:
            req = self._hunt(world, weather, rng, tick, neighbours, vision)
            if req is not None:
                return req

        # 5) Default: forage or wander toward denser food.
        if self.energy < self._max_energy * 0.85:
            return self._forage(world, weather, rng, tick, vision)
        return self._wander(world, weather, rng, tick)

    # ----- actions -------------------------------------------------------
    def _flee_from(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                   tick: int, threat: "Organism") -> None:
        # Move directly away from the threat.
        dx = self.x - threat.x
        dy = self.y - threat.y
        d = math.hypot(dx, dy) or 1.0
        self._move_toward(world, self.x + dx / d * 3.0, self.y + dy / d * 3.0)
        self.state = OrganismState.FLEEING
        self.last_action = f"Fleeing from predator #{threat.id}"
        self.events.append(OrganismEvent(tick=tick, kind="fled", detail=f"#{threat.id}"))

    def _seek_water(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                    tick: int) -> None:
        # Look for water tiles or water resource nodes in vision.
        tx, ty = self._nearest_water(world, vision=self.genome["vision_range"])
        if tx is None:
            self._wander(world, weather, rng, tick)
            self.last_action = "Thirsty; searching for water"
            return
        if math.hypot(tx - self.x, ty - self.y) < 1.5:
            drank = world.drink_water(int(self.x), int(self.y), 25.0)
            self.hydration = min(100.0, self.hydration + drank)
            if drank > 0:
                self.state = OrganismState.DRINKING
                self.last_action = f"Drinking (+{drank:.0f} hydration)"
                self.events.append(OrganismEvent(tick=tick, kind="drank", detail=f"+{drank:.0f}"))
        else:
            self._move_toward(world, tx, ty)
            self.state = OrganismState.DRINKING
            self.last_action = "Heading to water"

    def _forage(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                tick: int, vision: float) -> None:
        # Eat plant food at current tile if any.
        tile = world.tile_at(int(self.x), int(self.y))
        if tile is not None and tile.food > 1.0:
            want = min(self._max_energy - self.energy, 9.0)
            eaten = world.consume_plant_food(tile.x, tile.y, want)
            if eaten > 0:
                gained = eaten * tile.params["food_value"]
                self.energy = min(self._max_energy, self.energy + gained)
                self.state = OrganismState.FORAGING
                self.last_action = f"Grazing (+{gained:.0f} energy)"
                self.events.append(OrganismEvent(tick=tick, kind="ate", detail=f"plant +{gained:.0f}"))
                return
        # Else move toward the tile with the most food within vision.
        best = self._best_food_tile(world, vision)
        if best is not None:
            self._move_toward(world, best[0], best[1])
            self.state = OrganismState.FORAGING
            self.last_action = "Foraging toward food"
            return
        self._wander(world, weather, rng, tick)
        self.last_action = "Foraging; no food in sight"

    def _hunt(self, world: "World", weather: "WeatherSystem", rng: random.Random,
              tick: int, neighbours: List["Organism"], vision: float) -> Optional["BirthRequest"]:
        prey = [o for o in neighbours if o.alive and o.id != self.id and
                self._can_eat(o) and self._dist_to(o) <= vision]
        if not prey:
            return None
        target = min(prey, key=lambda o: self._dist_to(o))
        d = self._dist_to(target)
        if d < 0.8:
            # Attack!
            damage = 30.0 + 40.0 * self.genome["aggression"] + 20.0 * self.genome["size"]
            target.health -= damage
            self.state = OrganismState.HUNTING
            self.last_action = f"Attacking #{target.id}"
            self.events.append(OrganismEvent(tick=tick, kind="attacked", detail=f"#{target.id}"))
            target.events.append(OrganismEvent(tick=tick, kind="attacked", detail=f"by #{self.id}"))
            if target.health <= 0:
                # Consume the prey's energy.
                gained = min(self._max_energy - self.energy, target.energy * 0.7 + 30.0)
                self.energy = min(self._max_energy, self.energy + gained)
                self.kills += 1
                target._die(tick, f"hunted by #{self.id}")
                self.last_action = f"Killed #{target.id} (+{gained:.0f})"
                self.events.append(OrganismEvent(tick=tick, kind="ate", detail=f"prey #{target.id} +{gained:.0f}"))
            return None
        # Move toward prey.
        self._move_toward(world, target.x, target.y)
        self.target_id = target.id
        self.state = OrganismState.HUNTING
        self.last_action = f"Chasing #{target.id}"
        return None

    def _seek_mate(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                   tick: int, neighbours: List["Organism"], vision: float,
                   sim: Optional["Simulation"] = None,
                   ) -> Optional["BirthRequest"]:
        # Eligible mates: same species, mature, high energy, off cooldown.
        mates = [o for o in neighbours if o.alive and o.id != self.id and
                 o.species_id == self.species_id and o.is_mature and
                 o.energy >= o.genome["repro_threshold"] and o.cooldown_repro <= 0 and
                 self._dist_to(o) <= vision]
        if not mates:
            # Try a longer-range search through the simulation's spatial hash.
            if sim is not None:
                far_mate = sim.find_nearest_potential_mate(self, max_cells=4)
                if far_mate is not None:
                    self._move_toward(world, far_mate.x, far_mate.y)
                    # Nudge the mate to head toward us too, so they actually meet.
                    if far_mate.state not in (OrganismState.FLEEING, OrganismState.HUNTING):
                        far_mate._move_toward(world, self.x, self.y)
                        far_mate.state = OrganismState.MATING
                        far_mate.last_action = f"Heading toward distant mate #{self.id}"
                    self.state = OrganismState.MATING
                    self.last_action = f"Heading toward distant mate #{far_mate.id}"
                    return None
            # Move toward nearest same-species mature organism in vision.
            same_species = [o for o in neighbours if o.alive and o.id != self.id and
                            o.species_id == self.species_id and self._dist_to(o) <= vision]
            if same_species:
                target = min(same_species, key=lambda o: self._dist_to(o))
                self._move_toward(world, target.x, target.y)
                self.state = OrganismState.MATING
                self.last_action = f"Approaching mate #{target.id}"
                return None
            self._wander(world, weather, rng, tick)
            self.last_action = "Searching for a mate"
            return None
        mate = min(mates, key=lambda o: self._dist_to(o))
        d = self._dist_to(mate)
        # Reproduction distance scales with organism size so bigger creatures
        # don't need to be pixel-perfect.
        repro_dist = 2.0 + 0.8 * (self.genome["size"] + mate.genome["size"]) * 0.5
        if d < repro_dist:
            return self._reproduce_with(mate, rng, tick)
        # Both move toward each other for faster convergence.
        self._move_toward(world, mate.x, mate.y)
        if mate.state not in (OrganismState.FLEEING, OrganismState.HUNTING):
            mate._move_toward(world, self.x, self.y)
        self.state = OrganismState.MATING
        self.last_action = f"Moving toward mate #{mate.id}"
        return None

    def _reproduce_with(self, mate: "Organism", rng: random.Random, tick: int) -> "BirthRequest":
        child_genome = recombine(self.genome, mate.genome, rng)
        # Split offspring energy cost between parents.
        cost_a = self.genome["offspring_energy"]
        cost_b = mate.genome["offspring_energy"]
        self.energy -= cost_a
        mate.energy -= cost_b
        self.cooldown_repro = 60
        mate.cooldown_repro = 60
        self.offspring_count += 1
        mate.offspring_count += 1
        litter = int(child_genome["litter_size"])
        litter = max(1, min(litter, 3))
        self.events.append(OrganismEvent(tick=tick, kind="gave_birth", detail=f"#{mate.id} x{litter_count_label(litter)}"))
        mate.events.append(OrganismEvent(tick=tick, kind="gave_birth", detail=f"#{self.id} x{litter_count_label(litter)}"))
        self.last_action = f"Mated with #{mate.id}"
        mate.last_action = f"Mated with #{self.id}"
        # Position offspring near the parents.
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

    def _wander(self, world: "World", weather: "WeatherSystem", rng: random.Random,
                tick: int) -> None:
        # Drift in the current heading, occasionally turning.
        if self.target_x is None or self.target_y is None or rng.random() < 0.08:
            self.target_x = self.x + rng.uniform(-6.0, 6.0)
            self.target_y = self.y + rng.uniform(-6.0, 6.0)
        self._move_toward(world, self.target_x, self.target_y)
        # If reached target, clear it.
        if math.hypot(self.target_x - self.x, self.target_y - self.y) < 1.0:
            self.target_x = None
            self.target_y = None
        # Resting at night if well-fed.
        if not world.is_day() and self.energy > self._max_energy * 0.7:
            self.state = OrganismState.RESTING
            self.last_action = "Resting (night)"
        else:
            self.state = OrganismState.WANDERING
            self.last_action = "Wandering"

    # ----- helpers -------------------------------------------------------
    def _move_toward(self, world: "World", tx: float, ty: float) -> None:
        dx = tx - self.x
        dy = ty - self.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        speed = self.genome["speed"] * 0.5
        nx = self.x + dx / d * speed
        ny = self.y + dy / d * speed
        # Try to move; if blocked by impassable terrain, slide along axis.
        if world.passable_at(nx, ny):
            self.x = nx
            self.y = ny
        elif world.passable_at(nx, self.y):
            self.x = nx
        elif world.passable_at(self.x, ny):
            self.y = ny
        # Clamp to world bounds.
        self.x = max(0.1, min(world.width - 0.1, self.x))
        self.y = max(0.1, min(world.height - 0.1, self.y))
        self.target_x = tx
        self.target_y = ty

    def _dist_to(self, other: "Organism") -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def _is_edible_to(self, predator: "Organism") -> bool:
        """Return True if ``predator`` can eat *us* (``self``)."""
        if predator.species_id == self.species_id:
            return False
        if predator.genome["diet"] < 0.4:
            return False
        # Size advantage needed.
        return predator.genome["size"] >= self.genome["size"] * 0.85

    def _can_eat(self, other: "Organism") -> bool:
        """Return True if *we* (``self``) can eat ``other``."""
        if self.species_id == other.species_id:
            return False
        if self.genome["diet"] < 0.4:
            return False
        return self.genome["size"] >= other.genome["size"] * 0.85

    def _nearest_water(self, world: "World", vision: float) -> Tuple[Optional[float], Optional[float]]:
        # Check water resource nodes first.
        nodes = world.resource_nodes_in_radius(self.x, self.y, vision, kind=None)
        water_nodes = [r for r in nodes if r.kind.name == "WATER"]
        if water_nodes:
            best = min(water_nodes, key=lambda r: (r.x - self.x) ** 2 + (r.y - self.y) ** 2)
            return best.x, best.y
        # Scan a window of tiles for water.
        cx, cy = int(self.x), int(self.y)
        r = int(vision) + 1
        best: Optional[Tuple[float, float]] = None
        best_d = float("inf")
        for j in range(max(0, cy - r), min(world.height, cy + r + 1)):
            for i in range(max(0, cx - r), min(world.width, cx + r + 1)):
                tile = world.tile_at(i, j)
                if tile is not None and tile.has_water:
                    d = (i - self.x) ** 2 + (j - self.y) ** 2
                    if d < best_d:
                        best_d = d
                        best = (float(i), float(j))
        if best is None:
            return None, None
        return best

    def _best_food_tile(self, world: "World", vision: float) -> Optional[Tuple[float, float]]:
        cx, cy = int(self.x), int(self.y)
        r = int(vision) + 1
        best: Optional[Tuple[float, float]] = None
        best_food = 1.0  # only consider tiles with food > 1
        for j in range(max(0, cy - r), min(world.height, cy + r + 1)):
            for i in range(max(0, cx - r), min(world.width, cx + r + 1)):
                tile = world.tile_at(i, j)
                if tile is None or not tile.passable:
                    continue
                if tile.food > best_food:
                    best_food = tile.food
                    best = (float(i), float(j))
        return best

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
            "events": [e.to_dict() for e in self.events[-64:]],  # cap memory
            "last_action": self.last_action,
            "offspring_count": self.offspring_count,
            "kills": self.kills,
            "alive": self.alive,
            "death_tick": self.death_tick,
            "death_cause": self.death_cause,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Organism":
        org = cls(
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
        return org


# ---------------------------------------------------------------------------
# BirthRequest - a value object returned by step() when reproduction happens
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
    """Tiny helper so the mating-event label says 'x2' etc."""
    return str(n)
