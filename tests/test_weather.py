"""Tests for the WeatherSystem."""

from __future__ import annotations

import random

import pytest

from ecosystem.core.weather import (
    WeatherSystem, WeatherType, WeatherEvent, WEATHER_EFFECTS, WEATHER_LABELS,
)


def test_weather_starts_clear_or_baseline():
    ws = WeatherSystem(rng=random.Random(1))
    assert ws.current in (WeatherType.CLEAR, WeatherType.CLOUDY, WeatherType.RAIN, WeatherType.FOG)


def test_force_event_sets_active_event():
    ws = WeatherSystem(rng=random.Random(2))
    ev = ws.force_event(WeatherType.STORM, duration=30)
    assert ws.active_event is not None
    assert ws.active_event.kind == WeatherType.STORM
    assert ws.active_event.duration == 30
    assert ws.current == WeatherType.STORM
    # Effects should now reflect storm.
    assert ws.damage_per_tick > 0.0


def test_event_expires_after_duration():
    ws = WeatherSystem(rng=random.Random(3))
    ws.force_event(WeatherType.RAIN, duration=5)
    for _ in range(5):
        ws.update(1.0)
    assert ws.active_event is None


def test_clear_event_resets_to_baseline():
    ws = WeatherSystem(rng=random.Random(4))
    ws.force_event(WeatherType.HEATWAVE, duration=100)
    ws.clear_event()
    assert ws.active_event is None


def test_effects_are_consistent_with_table():
    ws = WeatherSystem(rng=random.Random(5))
    for wt in WeatherType:
        ws.force_event(wt, duration=10)
        eff = WEATHER_EFFECTS[wt]
        assert ws.damage_per_tick == pytest.approx(eff["damage"])
        assert ws.vision_multiplier == pytest.approx(eff["vision_mul"])
        assert ws.growth_multiplier == pytest.approx(eff["growth_mul"])


def test_all_weathers_have_labels():
    for wt in WeatherType:
        assert wt in WEATHER_LABELS
        assert isinstance(WEATHER_LABELS[wt], str)
        assert len(WEATHER_LABELS[wt]) > 0


def test_weather_serialisation_roundtrip():
    ws = WeatherSystem(rng=random.Random(6))
    ws.force_event(WeatherType.DROUGHT, duration=50)
    # Tick it a bit.
    ws.update(1.0)
    data = ws.to_dict()
    ws2 = WeatherSystem.from_dict(data, random.Random(7))
    assert ws2.current == ws.current
    assert ws2.effective_temp == pytest.approx(ws.effective_temp)
    if ws.active_event:
        assert ws2.active_event is not None
        assert ws2.active_event.kind == ws.active_event.kind
        assert ws2.active_event.duration == ws.active_event.duration


def test_random_event_spawning_does_not_crash_over_long_run():
    """Run many updates and ensure no exceptions and stats stay sane."""
    ws = WeatherSystem(rng=random.Random(7))
    ws.base_temp = 0.55
    for _ in range(2000):
        ws.update(1.0)
        assert 0.0 <= ws.effective_temp <= 1.5
        assert ws.damage_per_tick >= 0.0
