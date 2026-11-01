"""Weather and environmental events.

Weather is driven by a small state machine: at any given moment the world
has a *current* weather condition (clear, cloudy, rain, storm, drought,
heatwave, cold snap) plus an active :class:`WeatherEvent` if an unusual
event is in progress. Events are spawned probabilistically each tick and
have a finite duration; while active they modify the world's effective
moisture, temperature, plant growth and organism stress.

The system is deliberately simple but produces real gameplay effects:
droughts cause food shortages (and population crashes), storms injure
organisms, heatwaves punish cold-adapted species, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

__all__ = ["WeatherType", "WeatherEvent", "WeatherSystem"]


class WeatherType(IntEnum):
    CLEAR = 0
    CLOUDY = 1
    RAIN = 2
    STORM = 3
    DROUGHT = 4
    HEATWAVE = 5
    COLD_SNAP = 6
    FOG = 7


# Per-weather effect multipliers. ``moisture_delta`` is added to the world's
# moisture bonus each tick; ``temp_delta`` shifts temperature; ``growth_mul``
# multiplies plant food growth; ``damage`` is per-tick HP damage applied to
# exposed organisms; ``vision_mul`` scales effective vision range.
WEATHER_EFFECTS: Dict[WeatherType, Dict[str, float]] = {
    WeatherType.CLEAR:    {"moisture_delta": 0.0,  "temp_delta": 0.0,  "growth_mul": 1.0, "damage": 0.0,  "vision_mul": 1.0},
    WeatherType.CLOUDY:   {"moisture_delta": 0.01, "temp_delta": -0.02,"growth_mul": 0.95,"damage": 0.0,  "vision_mul": 0.9},
    WeatherType.RAIN:     {"moisture_delta": 0.10, "temp_delta": -0.05,"growth_mul": 1.6, "damage": 0.0,  "vision_mul": 0.75},
    WeatherType.STORM:    {"moisture_delta": 0.12, "temp_delta": -0.08,"growth_mul": 1.1, "damage": 0.6,  "vision_mul": 0.55},
    WeatherType.DROUGHT:  {"moisture_delta": -0.04,"temp_delta": 0.08, "growth_mul": 0.45,"damage": 0.04, "vision_mul": 1.1},
    WeatherType.HEATWAVE: {"moisture_delta": -0.03,"temp_delta": 0.20, "growth_mul": 0.55,"damage": 0.25, "vision_mul": 1.0},
    WeatherType.COLD_SNAP:{"moisture_delta": 0.0,  "temp_delta": -0.20,"growth_mul": 0.55,"damage": 0.25, "vision_mul": 0.9},
    WeatherType.FOG:      {"moisture_delta": 0.02, "temp_delta": -0.02,"growth_mul": 1.0, "damage": 0.0,  "vision_mul": 0.45},
}


# Display-friendly labels (kept short for the GUI status bar).
WEATHER_LABELS: Dict[WeatherType, str] = {
    WeatherType.CLEAR: "Clear",
    WeatherType.CLOUDY: "Cloudy",
    WeatherType.RAIN: "Rain",
    WeatherType.STORM: "Storm",
    WeatherType.DROUGHT: "Drought",
    WeatherType.HEATWAVE: "Heatwave",
    WeatherType.COLD_SNAP: "Cold Snap",
    WeatherType.FOG: "Fog",
}


@dataclass
class WeatherEvent:
    """An active weather event with a remaining duration."""

    kind: WeatherType
    duration: int          # ticks remaining
    intensity: float = 1.0 # 0..1, scales effects

    def tick_down(self) -> None:
        self.duration -= 1

    @property
    def expired(self) -> bool:
        return self.duration <= 0

    def to_dict(self) -> Dict:
        return {"kind": int(self.kind), "duration": self.duration, "intensity": self.intensity}

    @classmethod
    def from_dict(cls, data: Dict) -> "WeatherEvent":
        return cls(
            kind=WeatherType(int(data["kind"])),
            duration=int(data["duration"]),
            intensity=float(data.get("intensity", 1.0)),
        )


@dataclass
class WeatherSystem:
    """Tracks current weather and rolls for events."""

    rng: random.Random = field(default_factory=random.Random)
    current: WeatherType = WeatherType.CLEAR
    base_temp: float = 0.55
    effective_temp: float = 0.55
    effective_moisture_bonus: float = 0.0
    growth_multiplier: float = 1.0
    damage_per_tick: float = 0.0
    vision_multiplier: float = 1.0
    active_event: Optional[WeatherEvent] = None
    # Rolling history of recent events for the GUI.
    history: List[WeatherEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._recompute()

    # ----- public API ----------------------------------------------------
    def update(self, dt: float = 1.0) -> None:
        """Advance weather by ``dt`` ticks; spawn/expire events as needed."""
        if self.active_event is not None:
            for _ in range(int(dt)):
                self.active_event.tick_down()
                if self.active_event.expired:
                    self.history.append(self.active_event)
                    if len(self.history) > 20:
                        self.history.pop(0)
                    self.active_event = None
                    self.current = self._pick_baseline()
                    break
        # Maybe spawn a new event.
        if self.active_event is None:
            self._maybe_spawn_event()
        self._recompute()

    def force_event(self, kind: WeatherType, duration: int) -> WeatherEvent:
        """Force a specific event (used by the GUI's "trigger weather" menu)."""
        self.active_event = WeatherEvent(kind=kind, duration=duration, intensity=1.0)
        self.current = kind
        self._recompute()
        return self.active_event

    def clear_event(self) -> None:
        self.active_event = None
        self.current = self._pick_baseline()
        self._recompute()

    # ----- internals -----------------------------------------------------
    def _pick_baseline(self) -> WeatherType:
        # Mostly clear/cloudy, occasional rain or fog.
        r = self.rng.random()
        if r < 0.45:
            return WeatherType.CLEAR
        if r < 0.75:
            return WeatherType.CLOUDY
        if r < 0.92:
            return WeatherType.RAIN
        return WeatherType.FOG

    def _maybe_spawn_event(self) -> None:
        # ~0.6% chance per tick of a notable event => roughly one every ~3 in-world days.
        if self.rng.random() < 0.006:
            choices = [
                (WeatherType.STORM, 40, 90),
                (WeatherType.DROUGHT, 80, 200),
                (WeatherType.HEATWAVE, 50, 120),
                (WeatherType.COLD_SNAP, 50, 120),
                (WeatherType.RAIN, 40, 80),
            ]
            kind, lo, hi = self.rng.choice(choices)
            self.active_event = WeatherEvent(
                kind=kind,
                duration=self.rng.randint(lo, hi),
                intensity=self.rng.uniform(0.6, 1.0),
            )
            self.current = kind

    def _recompute(self) -> None:
        eff = WEATHER_EFFECTS[self.current]
        intensity = self.active_event.intensity if self.active_event else 1.0
        self.effective_temp = self.base_temp + eff["temp_delta"] * intensity
        self.effective_moisture_bonus = eff["moisture_delta"] * intensity
        self.growth_multiplier = eff["growth_mul"]
        self.damage_per_tick = eff["damage"] * intensity
        self.vision_multiplier = eff["vision_mul"]

    # ----- serialisation -------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "current": int(self.current),
            "base_temp": self.base_temp,
            "effective_temp": self.effective_temp,
            "effective_moisture_bonus": self.effective_moisture_bonus,
            "growth_multiplier": self.growth_multiplier,
            "damage_per_tick": self.damage_per_tick,
            "vision_multiplier": self.vision_multiplier,
            "active_event": self.active_event.to_dict() if self.active_event else None,
            "history": [e.to_dict() for e in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict, rng: random.Random) -> "WeatherSystem":
        ws = cls(rng=rng)
        ws.current = WeatherType(int(data["current"]))
        ws.base_temp = float(data.get("base_temp", 0.55))
        ws.effective_temp = float(data.get("effective_temp", ws.base_temp))
        ws.effective_moisture_bonus = float(data.get("effective_moisture_bonus", 0.0))
        ws.growth_multiplier = float(data.get("growth_multiplier", 1.0))
        ws.damage_per_tick = float(data.get("damage_per_tick", 0.0))
        ws.vision_multiplier = float(data.get("vision_multiplier", 1.0))
        ws.active_event = WeatherEvent.from_dict(data["active_event"]) if data.get("active_event") else None
        ws.history = [WeatherEvent.from_dict(e) for e in data.get("history", [])]
        return ws
