"""Observation journal.

The journal is a rolling log of noteworthy world events: births of new
species, extinctions, weather events, population milestones. The GUI
shows it as a chronological list so the user can read the "story" of
their world.

Entries are intentionally short text snippets; the simulation also
records quantitative data into :class:`~ecosystem.core.simulation.SimulationStats`
for graphing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["JournalEntry", "Journal"]


@dataclass
class JournalEntry:
    """A single line in the observation journal."""

    tick: int
    category: str   # 'species','population','weather','event','milestone'
    text: str

    def to_dict(self) -> Dict:
        return {"tick": self.tick, "category": self.category, "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict) -> "JournalEntry":
        return cls(tick=int(data["tick"]), category=str(data["category"]), text=str(data["text"]))


@dataclass
class Journal:
    """Append-only log with a configurable max length."""

    entries: List[JournalEntry] = field(default_factory=list)
    max_entries: int = 500

    def add(self, tick: int, category: str, text: str) -> JournalEntry:
        entry = JournalEntry(tick=tick, category=category, text=text)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            # Drop oldest, but never the first few founding entries.
            self.entries = self.entries[-self.max_entries:]
        return entry

    def recent(self, n: int = 50) -> List[JournalEntry]:
        return self.entries[-n:]

    def by_category(self, category: str) -> List[JournalEntry]:
        return [e for e in self.entries if e.category == category]

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> Dict:
        return {"entries": [e.to_dict() for e in self.entries], "max_entries": self.max_entries}

    @classmethod
    def from_dict(cls, data: Dict) -> "Journal":
        return cls(
            entries=[JournalEntry.from_dict(e) for e in data.get("entries", [])],
            max_entries=int(data.get("max_entries", 500)),
        )
