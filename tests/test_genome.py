"""Tests for the genome / genetics layer."""

from __future__ import annotations

import random

import pytest

from ecosystem.core.genome import (
    Genome, TRAIT_SPECS, random_genome, mutate, recombine,
    genetic_distance, clamp_trait,
)


def test_genome_defaults_are_valid():
    g = Genome()
    for name, spec in TRAIT_SPECS.items():
        v = g[name]
        if spec.wrap:
            assert 0.0 <= v < spec.maximum
        else:
            assert spec.minimum <= v <= spec.maximum


def test_genome_clamps_out_of_range():
    g = Genome({"size": 999.0, "speed": -5.0})
    assert g["size"] == TRAIT_SPECS["size"].maximum
    assert g["speed"] == TRAIT_SPECS["speed"].minimum


def test_genome_wraps_color_hue():
    g = Genome({"color_hue": 400.0})
    assert g["color_hue"] == pytest.approx(40.0)


def test_random_genome_in_range():
    rng = random.Random(42)
    for _ in range(100):
        g = random_genome(rng)
        for name, spec in TRAIT_SPECS.items():
            v = g[name]
            if spec.wrap:
                assert 0.0 <= v < spec.maximum
            else:
                assert spec.minimum <= v <= spec.maximum


def test_mutate_preserves_ranges():
    rng = random.Random(7)
    g = random_genome(rng)
    for _ in range(50):
        g = mutate(g, rng, rate=0.5)
        for name, spec in TRAIT_SPECS.items():
            v = g[name]
            if spec.wrap:
                assert 0.0 <= v < spec.maximum
            else:
                assert spec.minimum <= v <= spec.maximum


def test_mutate_with_zero_rate_is_identity():
    rng = random.Random(1)
    g = random_genome(rng)
    g2 = mutate(g, rng, rate=0.0)
    assert g2.as_dict() == g.as_dict()


def test_recombine_produces_child_in_range():
    rng = random.Random(123)
    a = random_genome(rng)
    b = random_genome(rng)
    child = recombine(a, b, rng)
    for name, spec in TRAIT_SPECS.items():
        v = child[name]
        if spec.wrap:
            assert 0.0 <= v < spec.maximum
        else:
            assert spec.minimum <= v <= spec.maximum


def test_recombine_inherits_average_mutation_rate():
    rng = random.Random(99)
    a = Genome({"mutation_rate": 0.05})
    b = Genome({"mutation_rate": 0.15})
    # Average is 0.1 - recombine should pass this through mutate as the rate.
    # We can't check the exact child value but the operation should not crash
    # and the child's mutation_rate must remain in range.
    child = recombine(a, b, rng)
    assert 0.02 <= child["mutation_rate"] <= 0.18


def test_genetic_distance_self_is_zero():
    rng = random.Random(5)
    g = random_genome(rng)
    assert genetic_distance(g, g) == pytest.approx(0.0)


def test_genetic_distance_max_for_opposite_genomes():
    # Build two genomes with every trait at the opposite extreme.
    a_data = {name: spec.minimum for name, spec in TRAIT_SPECS.items()}
    b_data = {name: spec.maximum for name, spec in TRAIT_SPECS.items()}
    a = Genome(a_data)
    b = Genome(b_data)
    d = genetic_distance(a, b)
    # Distance is normalised; maximally divergent should be close to 1.
    assert d > 0.9


def test_genetic_distance_is_symmetric():
    rng = random.Random(8)
    a = random_genome(rng)
    b = random_genome(rng)
    assert genetic_distance(a, b) == pytest.approx(genetic_distance(b, a))


def test_genome_immutable_mapping():
    g = Genome({"size": 1.0})
    with pytest.raises(TypeError):
        g["size"] = 2.0  # type: ignore[index]


def test_clamp_trait_unknown_returns_value():
    assert clamp_trait("not_a_real_trait", 3.14) == 3.14


def test_genome_equality_and_hash():
    g1 = Genome({"size": 1.0, "speed": 1.0})
    g2 = Genome({"size": 1.0, "speed": 1.0})
    assert g1 == g2
    assert hash(g1) == hash(g2)


def test_genome_to_json_roundtrip():
    g = Genome({"size": 1.5, "speed": 0.8})
    data = g.to_json()
    g2 = Genome.from_json(data)
    assert g == g2
