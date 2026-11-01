"""Species tracking and speciation.

A *Species* is a cluster of genetically-similar organisms. The registry
uses online clustering: when a new organism is born, its genome is compared
to every existing species' centroid; if the closest match is within
``species_threshold`` genetic distance, the organism joins that species
(and the centroid is updated incrementally). Otherwise a new species is
created.

This is intentionally simple - we don't try to maintain phylogenetic
trees - but it does produce the emergent behaviour the project asks for:
over many generations, mutation drift causes populations to split into
new species that compete with their ancestors.

Each :class:`Species` keeps a few useful statistics (population, founding
tick, average traits, colour) for the GUI inspector and the family-tree
view.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .genome import Genome, genetic_distance

__all__ = ["Species", "SpeciesRegistry"]


@dataclass
class Species:
    """A cluster of genetically similar organisms."""

    id: int
    name: str
    founding_tick: int
    parent_species_id: Optional[int]
    centroid: Dict[str, float]            # mean trait values
    population: int = 0
    births: int = 0
    deaths: int = 0
    peak_population: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    # Rolling mean colour for the GUI badge.
    color_hue: float = 180.0
    color_sat: float = 0.7
    color_light: float = 0.55
    # Sample member ids used to render the family tree (most recent first).
    representative_ids: List[int] = field(default_factory=list)
    # A short log of notable events (e.g. founding, peak, extinction).
    log: List[Tuple[int, str]] = field(default_factory=list)

    def update_centroid(self, genome: Genome, weight: float = 0.05) -> None:
        """Move the centroid toward ``genome`` by ``weight`` (0..1)."""
        for name, value in genome.as_dict().items():
            cur = self.centroid.get(name, value)
            self.centroid[name] = cur * (1.0 - weight) + value * weight

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "founding_tick": self.founding_tick,
            "parent_species_id": self.parent_species_id,
            "centroid": dict(self.centroid),
            "population": self.population,
            "births": self.births,
            "deaths": self.deaths,
            "peak_population": self.peak_population,
            "extinct": self.extinct,
            "extinction_tick": self.extinction_tick,
            "color_hue": self.color_hue,
            "color_sat": self.color_sat,
            "color_light": self.color_light,
            "representative_ids": list(self.representative_ids),
            "log": [list(entry) for entry in self.log],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Species":
        return cls(
            id=int(data["id"]),
            name=data["name"],
            founding_tick=int(data["founding_tick"]),
            parent_species_id=data["parent_species_id"],
            centroid=dict(data["centroid"]),
            population=int(data["population"]),
            births=int(data["births"]),
            deaths=int(data["deaths"]),
            peak_population=int(data["peak_population"]),
            extinct=bool(data["extinct"]),
            extinction_tick=data["extinction_tick"],
            color_hue=float(data["color_hue"]),
            color_sat=float(data["color_sat"]),
            color_light=float(data["color_light"]),
            representative_ids=list(data["representative_ids"]),
            log=[(int(t), str(s)) for t, s in data["log"]],
        )


class SpeciesRegistry:
    """Tracks species and handles speciation."""

    # Genetic distance above which a new species is created.
    SPECIES_THRESHOLD: float = 0.18

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._species: Dict[int, Species] = {}
        self._next_id: int = 1
        self.rng = rng or random.Random()

    # ----- queries -------------------------------------------------------
    def all_species(self) -> List[Species]:
        return list(self._species.values())

    def living_species(self) -> List[Species]:
        return [s for s in self._species.values() if not s.extinct]

    def get(self, species_id: int) -> Optional[Species]:
        return self._species.get(species_id)

    def __len__(self) -> int:
        return len(self._species)

    # ----- registration --------------------------------------------------
    def register_birth(
        self,
        organism_id: int,
        genome: Genome,
        tick: int,
        parent_species_id: Optional[int],
    ) -> Species:
        """Classify a newborn into a species (or found a new one)."""
        best: Optional[Species] = None
        best_dist = float("inf")
        for sp in self._species.values():
            if sp.extinct:
                continue
            centroid_genome = Genome(sp.centroid)
            d = genetic_distance(genome, centroid_genome)
            if d < best_dist:
                best_dist = d
                best = sp
        if best is not None and best_dist <= self.SPECIES_THRESHOLD:
            best.population += 1
            best.births += 1
            best.peak_population = max(best.peak_population, best.population)
            best.update_centroid(genome)
            # Update rolling colour from the genome.
            best.color_hue = best.color_hue * 0.92 + genome["color_hue"] * 0.08
            best.color_sat = best.color_sat * 0.92 + genome["color_sat"] * 0.08
            best.color_light = best.color_light * 0.92 + genome["color_light"] * 0.08
            best.representative_ids.insert(0, organism_id)
            if len(best.representative_ids) > 12:
                best.representative_ids.pop()
            return best
        # Found a new species.
        new_sp = Species(
            id=self._next_id,
            name=self._generate_name(parent_species_id),
            founding_tick=tick,
            parent_species_id=parent_species_id,
            centroid=genome.as_dict(),
            population=1,
            births=1,
            peak_population=1,
            color_hue=genome["color_hue"],
            color_sat=genome["color_sat"],
            color_light=genome["color_light"],
            representative_ids=[organism_id],
            log=[(tick, "founded")],
        )
        self._species[self._next_id] = new_sp
        self._next_id += 1
        return new_sp

    def register_death(self, species: Species, tick: int) -> None:
        species.population = max(0, species.population - 1)
        species.deaths += 1
        if species.population == 0 and not species.extinct:
            species.extinct = True
            species.extinction_tick = tick
            species.log.append((tick, "extinct"))

    def tick_check(self, tick: int) -> List[Species]:
        """Return species that just went extinct this tick (if any)."""
        newly_extinct: List[Species] = []
        for sp in self._species.values():
            if not sp.extinct and sp.population == 0:
                sp.extinct = True
                sp.extinction_tick = tick
                sp.log.append((tick, "extinct"))
                newly_extinct.append(sp)
        return newly_extinct

    # ----- naming --------------------------------------------------------
    def _generate_name(self, parent_id: Optional[int]) -> str:
        """Generate a quasi-scientific species name.

        Names are pseudo-Latin: a genus prefix + a species suffix. Daughter
        species share the genus of their parent, so families are readable
        in the species list / family tree.
        """
        genus_prefixes = [
            "Albio", "Brevi", "Crypta", "Dento", "Eo", "Fera", "Gravi",
            "Helio", "Icto", "Juxta", "Kino", "Lepido", "Macro", "Necto",
            "Oculi", "Ptero", "Quasi", "Rudi", "Steno", "Tachy", "Umbra",
            "Viri", "Xeno", "Ypsi", "Zono",
        ]
        species_suffixes = [
            "ferox", "gracilis", "robustus", "minutus", "auratus", "niger",
            "silvus", "aquaticus", "nocturna", "solaris", "vorus", "gregis",
            "saltans", "fugax", "tenax", "lurida", "ignis", "glacies",
        ]
        if parent_id is not None and parent_id in self._species:
            parent = self._species[parent_id]
            # Take parent's genus (text before the first space).
            parent_name = parent.name
            genus = parent_name.split(" ")[0]
        else:
            genus = self.rng.choice(genus_prefixes)
        suffix = self.rng.choice(species_suffixes)
        return f"{genus} {suffix}"

    # ----- serialisation -------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "next_id": self._next_id,
            "species": [s.to_dict() for s in self._species.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict, rng: random.Random) -> "SpeciesRegistry":
        reg = cls(rng=rng)
        reg._next_id = int(data["next_id"])
        for sd in data["species"]:
            sp = Species.from_dict(sd)
            reg._species[sp.id] = sp
        return reg
