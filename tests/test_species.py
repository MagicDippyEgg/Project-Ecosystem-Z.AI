"""Tests for species tracking and speciation."""

from __future__ import annotations

import random

import pytest

from ecosystem.core.genome import Genome, random_genome
from ecosystem.core.species import SpeciesRegistry, Species


def test_registry_starts_empty():
    reg = SpeciesRegistry(rng=random.Random(1))
    assert len(reg) == 0
    assert reg.living_species() == []


def test_first_organism_founds_new_species():
    rng = random.Random(2)
    reg = SpeciesRegistry(rng=rng)
    g = random_genome(rng)
    sp = reg.register_birth(organism_id=1, genome=g, tick=0, parent_species_id=None)
    assert sp.id == 1
    assert sp.population == 1
    assert sp.parent_species_id is None
    assert sp.founding_tick == 0
    assert len(reg) == 1


def test_similar_organisms_join_same_species():
    rng = random.Random(3)
    reg = SpeciesRegistry(rng=rng)
    base = random_genome(rng)
    sp1 = reg.register_birth(1, base, 0, None)
    # Slightly mutated copies should be within the species threshold.
    for i in range(5):
        slightly_mutated = Genome(base.as_dict())  # identical -> distance 0
        sp = reg.register_birth(i + 2, slightly_mutated, 0, sp1.id)
        assert sp.id == sp1.id
    assert sp1.population == 6


def test_divergent_organism_founds_new_species():
    rng = random.Random(4)
    reg = SpeciesRegistry(rng=rng)
    # First species.
    g1 = Genome({name: spec.minimum for name, spec in __import__("ecosystem.core.genome", fromlist=["TRAIT_SPECS"]).TRAIT_SPECS.items()})
    sp1 = reg.register_birth(1, g1, 0, None)
    # Maximally divergent genome.
    g2 = Genome({name: spec.maximum for name, spec in __import__("ecosystem.core.genome", fromlist=["TRAIT_SPECS"]).TRAIT_SPECS.items()})
    sp2 = reg.register_birth(2, g2, 1, sp1.id)
    assert sp2.id != sp1.id
    assert sp2.parent_species_id == sp1.id


def test_species_extinction_when_population_drops_to_zero():
    rng = random.Random(5)
    reg = SpeciesRegistry(rng=rng)
    g = random_genome(rng)
    sp = reg.register_birth(1, g, 0, None)
    reg.register_death(sp, 10)
    assert sp.population == 0
    assert sp.extinct
    assert sp.extinction_tick == 10


def test_species_centroid_drifts_with_new_members():
    rng = random.Random(6)
    reg = SpeciesRegistry(rng=rng)
    base = random_genome(rng)
    sp = reg.register_birth(1, base, 0, None)
    centroid_before = dict(sp.centroid)
    # Register many similar organisms; centroid should stay close to base.
    for i in range(20):
        g = Genome(base.as_dict())  # identical
        reg.register_birth(i + 2, g, i, sp.id)
    for k, v in sp.centroid.items():
        assert abs(v - centroid_before[k]) < 0.5


def test_species_naming_uses_parent_genus():
    rng = random.Random(7)
    reg = SpeciesRegistry(rng=rng)
    g = random_genome(rng)
    parent = reg.register_birth(1, g, 0, None)
    # Force a divergent child to found a daughter species.
    far_genome = Genome({name: spec.maximum for name, spec in __import__("ecosystem.core.genome", fromlist=["TRAIT_SPECS"]).TRAIT_SPECS.items()})
    child = reg.register_birth(2, far_genome, 1, parent.id)
    # Daughter genus should match parent's genus (text before first space).
    assert child.name.split(" ")[0] == parent.name.split(" ")[0]
    # But species suffix differs.
    assert child.name != parent.name


def test_species_serialisation_roundtrip():
    rng = random.Random(8)
    reg = SpeciesRegistry(rng=rng)
    g1 = random_genome(rng)
    g2 = random_genome(rng)
    # Make them divergent.
    g2_data = g2.as_dict()
    g2_data["diet"] = 1.0 - g2_data["diet"]
    g2_data["size"] = 0.4 if g2_data["size"] > 1.0 else 2.0
    g2 = Genome(g2_data)
    reg.register_birth(1, g1, 0, None)
    reg.register_birth(2, g2, 1, 1)
    data = reg.to_dict()
    reg2 = SpeciesRegistry.from_dict(data, rng)
    assert len(reg2) == len(reg)
    for sp_id in reg._species:
        sp_a = reg.get(sp_id)
        sp_b = reg2.get(sp_id)
        assert sp_a is not None and sp_b is not None
        assert sp_a.name == sp_b.name
        assert sp_a.population == sp_b.population


def test_species_peak_population_tracks_max():
    rng = random.Random(9)
    reg = SpeciesRegistry(rng=rng)
    g = random_genome(rng)
    sp = reg.register_birth(1, g, 0, None)
    # Add 10 more.
    for i in range(10):
        reg.register_birth(i + 2, Genome(g.as_dict()), i, sp.id)
    assert sp.peak_population == 11
    # Remove some.
    for _ in range(5):
        reg.register_death(sp, 100)
    assert sp.peak_population == 11  # peak unchanged
    assert sp.population == 6
