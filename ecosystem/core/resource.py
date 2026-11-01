"""Resource nodes (food plants, water springs, etc.).

A :class:`ResourceNode` is a discrete, positionable resource that organisms
can seek out and consume - distinct from the per-tile plant food managed by
:class:`~ecosystem.core.world.World`. Currently the main use is water
springs placed in dry biomes so creatures can drink away from lakes.

The class is intentionally tiny and self-serialising so the world module
can store and reload it without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

__all__ = ["ResourceType", "ResourceNode"]


class ResourceType(IntEnum):
    """Discrete resource kinds placed on the map."""

    WATER = 0
    FRUIT = 1     # high-value plant food
    CARCASS = 2   # left behind when an organism dies; food for scavengers


@dataclass
class ResourceNode:
    """A discrete resource on the map with regenerating amount."""

    x: float
    y: float
    kind: ResourceType
    amount: float
    max_amount: float
    regen_rate: float = 0.0  # units per tick (only meaningful for WATER/FRUIT)

    def consume(self, amount: float) -> float:
        taken = min(self.amount, amount)
        self.amount -= taken
        return taken

    def regen(self, dt: float) -> None:
        if self.regen_rate <= 0 or self.amount >= self.max_amount:
            return
        self.amount = min(self.max_amount, self.amount + self.regen_rate * dt)

    def to_dict(self) -> Dict:
        return {
            "x": self.x, "y": self.y,
            "kind": int(self.kind),
            "amount": self.amount, "max_amount": self.max_amount,
            "regen_rate": self.regen_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ResourceNode":
        return cls(
            x=float(data["x"]), y=float(data["y"]),
            kind=ResourceType(int(data["kind"])),
            amount=float(data["amount"]),
            max_amount=float(data["max_amount"]),
            regen_rate=float(data.get("regen_rate", 0.0)),
        )
