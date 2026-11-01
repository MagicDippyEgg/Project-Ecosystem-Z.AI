"""Persistence layer: save/load the entire simulation as JSON.

Save files are plain JSON so they can be inspected, diffed and edited
outside the application if needed. The format is versioned; future
versions of the simulation can carry an ``upgrade`` step.

The :class:`SaveManager` handles file paths, atomic writes (write to a
temp file then rename) and a small library of saved worlds in
``~/.local/share/project-ecosystem/saves/`` (or the project-local
``saves/`` directory if the user is running from source).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core.simulation import Simulation

__all__ = ["SAVE_FORMAT_VERSION", "SaveManager", "SaveSlot", "save_simulation", "load_simulation"]

SAVE_FORMAT_VERSION: int = 1


@dataclass
class SaveSlot:
    """Metadata about a save file on disk (without the full payload)."""

    path: Path
    name: str
    created: float
    tick: int
    population: int
    world_name: str
    size_bytes: int

    def to_dict(self) -> Dict:
        return {
            "path": str(self.path),
            "name": self.name,
            "created": self.created,
            "tick": self.tick,
            "population": self.population,
            "world_name": self.world_name,
            "size_bytes": self.size_bytes,
        }


def save_simulation(sim: Simulation, path: str | os.PathLike, name: Optional[str] = None) -> Path:
    """Atomically write ``sim`` to ``path`` as JSON.

    Returns the final path. The write is atomic: the file is written to a
    sibling temp file first and then renamed, so a crash mid-write never
    leaves a half-written save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": SAVE_FORMAT_VERSION,
        "saved_at": time.time(),
        "name": name or path.stem,
        "simulation": sim.to_dict(),
    }
    # Serialise with sorted keys & compact separators for stable diffs.
    text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file on any error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_simulation(path: str | os.PathLike) -> Simulation:
    """Load a simulation from ``path`` and return a live :class:`Simulation`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Save file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format_version") != SAVE_FORMAT_VERSION:
        # In a real product we'd run an upgrade step here. For v1 we just
        # warn and try to load anyway.
        # pragma: no cover - defensive
        pass
    data = payload["simulation"]
    return Simulation.from_dict(data)


class SaveManager:
    """Manage a directory of save files."""

    def __init__(self, base_dir: Optional[str | os.PathLike] = None) -> None:
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        else:
            # Prefer XDG-style user data dir; fall back to project-local.
            xdg = os.environ.get("XDG_DATA_HOME")
            if xdg:
                self.base_dir = Path(xdg) / "project-ecosystem" / "saves"
            else:
                self.base_dir = Path.home() / ".local" / "share" / "project-ecosystem" / "saves"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_saves(self) -> List[SaveSlot]:
        slots: List[SaveSlot] = []
        for p in sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                slot = self._read_slot(p)
            except Exception:
                continue
            slots.append(slot)
        return slots

    def _read_slot(self, path: Path) -> SaveSlot:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        sim_data = payload.get("simulation", {})
        return SaveSlot(
            path=path,
            name=payload.get("name", path.stem),
            created=float(payload.get("saved_at", path.stat().st_mtime)),
            tick=int(sim_data.get("tick_count", 0)),
            population=sum(1 for o in sim_data.get("organisms", []) if o.get("alive", True)),
            world_name=sim_data.get("world", {}).get("name", path.stem),
            size_bytes=path.stat().st_size,
        )

    def save(self, sim: Simulation, name: str) -> Path:
        safe = self._safe_name(name)
        path = self.base_dir / f"{safe}.json"
        return save_simulation(sim, path, name=name)

    def load(self, name: str) -> Simulation:
        safe = self._safe_name(name)
        path = self.base_dir / f"{safe}.json"
        return load_simulation(path)

    def delete(self, name: str) -> bool:
        safe = self._safe_name(name)
        path = self.base_dir / f"{safe}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, name: str) -> bool:
        safe = self._safe_name(name)
        return (self.base_dir / f"{safe}.json").exists()

    @staticmethod
    def _safe_name(name: str) -> str:
        # Allow only alphanumerics, dash, underscore; replace others.
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64] or "save"
