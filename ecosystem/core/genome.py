"""Genome and genetics.

A *Genome* is a dictionary of named float traits, each constrained to a
``[min, max]`` range defined in :data:`TRAIT_SPECS`.

The genetics layer supports three operations:

* :func:`mutate`      - perturb traits by a small random amount.
* :func:`recombine`   - blend two parent genomes to produce a child.
* :func:`genetic_distance` - measure how different two genomes are.

Speciation (see :mod:`ecosystem.core.species`) is driven by
:func:`genetic_distance`: when a child is sufficiently far from any
existing species' centroid, a new species is created.

All functions are deterministic given the same :class:`random.Random`
instance, which is threaded through the whole simulation so that
save/load can reproduce runs (when desired) and tests are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Tuple

__all__ = [
    "TraitSpec",
    "TRAIT_SPECS",
    "Genome",
    "random_genome",
    "mutate",
    "recombine",
    "genetic_distance",
    "clamp_trait",
]


@dataclass(frozen=True)
class TraitSpec:
    """Specification of a single heritable trait.

    Attributes
    ----------
    name:
        Trait key in the genome dict.
    minimum, maximum:
        Inclusive bounds. Values are clamped after mutation/recombination.
    default:
        Value used when a genome is generated without explicit value.
    wrap:
        If ``True`` the value is interpreted as an angle in degrees and
        wraps modulo 360 (used for ``color_hue``). Distance for wrapped
        traits is the shortest angular distance.
    integer:
        If ``True`` the value is rounded to the nearest int. Reserved for
        future discrete traits; currently unused.
    """

    name: str
    minimum: float
    maximum: float
    default: float
    wrap: bool = False
    integer: bool = False


# ---------------------------------------------------------------------------
# Trait catalogue
# ---------------------------------------------------------------------------
# Each entry controls a real gameplay/statistical property of an organism.
# Ranges were chosen so that a randomly initialised genome produces a
# viable (if marginal) organism and so that evolution has meaningful room
# to drift in either direction.
TRAIT_SPECS: Dict[str, TraitSpec] = {
    # Body plan
    "size": TraitSpec("size", 0.4, 2.0, 1.0),
    "speed": TraitSpec("speed", 0.4, 3.0, 1.0),
    "metabolism": TraitSpec("metabolism", 0.3, 1.8, 1.0),
    "vision_range": TraitSpec("vision_range", 2.0, 22.0, 10.0),

    # Life history
    "max_age": TraitSpec("max_age", 200.0, 1000.0, 600.0),
    "maturity_age": TraitSpec("maturity_age", 18.0, 90.0, 40.0),
    "repro_threshold": TraitSpec("repro_threshold", 45.0, 92.0, 55.0),
    "offspring_energy": TraitSpec("offspring_energy", 18.0, 55.0, 32.0),
    "litter_size": TraitSpec("litter_size", 1.0, 3.0, 1.0, integer=True),

    # Ecology
    "diet": TraitSpec("diet", 0.0, 1.0, 0.3),  # 0=herbivore, 1=carnivore
    "aggression": TraitSpec("aggression", 0.0, 1.0, 0.25),
    "hardiness": TraitSpec("hardiness", 0.0, 1.0, 0.4),
    "water_need": TraitSpec("water_need", 0.3, 1.5, 0.8),

    # Genetics
    "mutation_rate": TraitSpec("mutation_rate", 0.02, 0.18, 0.08),

    # Visual / species identity
    "color_hue": TraitSpec("color_hue", 0.0, 360.0, 180.0, wrap=True),
    "color_sat": TraitSpec("color_sat", 0.3, 1.0, 0.7),
    "color_light": TraitSpec("color_light", 0.35, 0.75, 0.55),
}

# Traits that participate in genetic-distance computation. ``color_*`` are
# excluded from speciation distance so that visually similar organisms are
# not automatically lumped together; species identity is driven by
# morphology and life history.
_DISTANCE_TRAITS: Tuple[str, ...] = (
    "size", "speed", "metabolism", "vision_range", "max_age",
    "maturity_age", "repro_threshold", "offspring_energy", "diet",
    "aggression", "hardiness", "water_need",
)

# Per-trait weights used when computing genetic distance. Traits that have
# a bigger ecological impact count more.
_DISTANCE_WEIGHTS: Dict[str, float] = {
    "size": 1.6, "speed": 1.4, "metabolism": 1.2,
    "vision_range": 0.8, "max_age": 0.6, "maturity_age": 0.6,
    "repro_threshold": 0.5, "offspring_energy": 0.6,
    "diet": 2.2, "aggression": 1.0, "hardiness": 0.9, "water_need": 0.8,
}


class Genome(Mapping[str, float]):
    """An immutable-ish bag of trait values.

    Implemented as a plain dict subclass for fast attribute access during
    simulation. Values are validated (clamped to spec) on construction.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, float] | None = None) -> None:
        if data is None:
            data = {name: spec.default for name, spec in TRAIT_SPECS.items()}
        else:
            data = dict(data)
            # Fill any missing traits with defaults.
            for name, spec in TRAIT_SPECS.items():
                if name not in data:
                    data[name] = spec.default
        # Clamp / wrap every value to its valid range.
        clamped = {name: clamp_trait(name, v) for name, v in data.items()}
        object.__setattr__(self, "_data", clamped)

    # Mapping protocol ----------------------------------------------------
    def __getitem__(self, key: str) -> float:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:  # noqa: D401
        return key in self._data

    # Convenience ---------------------------------------------------------
    def get(self, key: str, default: float = 0.0) -> float:  # type: ignore[override]
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, float]:
        """Return a shallow copy as a plain dict (for serialisation)."""
        return dict(self._data)

    def to_json(self) -> Dict[str, float]:
        return self.as_dict()

    @classmethod
    def from_json(cls, data: Mapping[str, float]) -> "Genome":
        return cls(data)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Genome({self._data})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Genome):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        # Hash on a tuple of sorted items so equal genomes hash equally.
        return hash(tuple(sorted(self._data.items())))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp_trait(name: str, value: float) -> float:
    """Clamp (or wrap) ``value`` to the valid range of trait ``name``.

    Unknown traits are returned unchanged so custom traits can be carried
    without crashing the simulation.
    """
    spec = TRAIT_SPECS.get(name)
    if spec is None:
        return float(value)
    v = float(value)
    if spec.wrap:
        return v % spec.maximum
    if spec.integer:
        v = round(v)
    return max(spec.minimum, min(spec.maximum, v))


def random_genome(rng: random.Random) -> Genome:
    """Generate a fully random viable genome."""
    data: Dict[str, float] = {}
    for name, spec in TRAIT_SPECS.items():
        if spec.integer:
            data[name] = float(rng.randint(int(spec.minimum), int(spec.maximum)))
        else:
            data[name] = rng.uniform(spec.minimum, spec.maximum)
    return Genome(data)


def mutate(genome: Genome, rng: random.Random, rate: float | None = None) -> Genome:
    """Return a mutated copy of ``genome``.

    ``rate`` overrides the genome's own ``mutation_rate`` if given; this is
    used by tests and by external stressors (e.g. radiation events).
    Each trait has an independent chance of being perturbed; the perturbation
    magnitude is proportional to the trait's range so all traits mutate on a
    comparable scale.
    """
    own_rate = rate if rate is not None else genome["mutation_rate"]
    out = dict(genome.as_dict())
    for name, spec in TRAIT_SPECS.items():
        if rng.random() < own_rate:
            span = spec.maximum - spec.minimum
            # Gaussian perturbation centred on 0 with stddev = 8% of span.
            delta = rng.gauss(0.0, 0.08 * span)
            out[name] = out[name] + delta
    return Genome(out)


def recombine(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
) -> Genome:
    """Blend two parent genomes into a child genome.

    For each trait the child inherits the value from one parent chosen
    uniformly at random, then a small interpolated jitter is added so that
    children are not exact clones. The result is then passed through
    :func:`mutate` using the average parental mutation rate.
    """
    child: Dict[str, float] = {}
    avg_rate = 0.5 * (parent_a["mutation_rate"] + parent_b["mutation_rate"])
    for name, spec in TRAIT_SPECS.items():
        if rng.random() < 0.5:
            base = parent_a[name]
        else:
            base = parent_b[name]
        # Small blend toward the other parent to avoid discrete jumps.
        blend = rng.uniform(0.0, 0.2)
        other = parent_b[name] if base is parent_a[name] else parent_a[name]
        if spec.wrap:
            # Angular interpolation - take the shorter arc.
            diff = ((other - base + 180.0) % 360.0) - 180.0
            child[name] = base + blend * diff
        else:
            child[name] = base + blend * (other - base)
    genome = Genome(child)
    return mutate(genome, rng, rate=avg_rate)


def genetic_distance(a: Genome, b: Genome) -> float:
    """Weighted normalised distance between two genomes.

    Returns a float in ``[0, 1]``-ish range (can exceed 1 for maximally
    divergent genomes due to weights summing above 1). Used by the
    :class:`~ecosystem.core.species.SpeciesRegistry` to decide speciation.
    """
    total = 0.0
    weight_sum = 0.0
    for name in _DISTANCE_TRAITS:
        spec = TRAIT_SPECS[name]
        span = spec.maximum - spec.minimum
        if span <= 0:
            continue
        va = a[name]
        vb = b[name]
        if spec.wrap:
            diff = abs(((va - vb + 180.0) % 360.0) - 180.0) / span
        else:
            diff = abs(va - vb) / span
        w = _DISTANCE_WEIGHTS.get(name, 1.0)
        total += diff * w
        weight_sum += w
    if weight_sum == 0:
        return 0.0
    return total / weight_sum
