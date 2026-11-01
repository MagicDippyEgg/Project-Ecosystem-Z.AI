"""Simulation backend.

All modules in this package are pure Python and have no dependency on Qt,
so the simulation can be imported, scripted and tested headlessly.
"""

from .genome import Genome, TRAIT_SPECS, mutate, recombine, genetic_distance, random_genome
from .resource import ResourceType, ResourceNode
from .organism import Organism, OrganismState, BirthRequest
from .species import SpeciesRegistry, Species
from .world import World, Tile, Biome, BIOME_PARAMS
from .weather import WeatherSystem, WeatherEvent, WeatherType, WEATHER_LABELS, WEATHER_EFFECTS
from .simulation import Simulation, SimulationConfig, SimulationStats
from .journal import Journal, JournalEntry

__all__ = [
    "Genome",
    "TRAIT_SPECS",
    "mutate",
    "recombine",
    "genetic_distance",
    "random_genome",
    "ResourceType",
    "ResourceNode",
    "Organism",
    "OrganismState",
    "BirthRequest",
    "SpeciesRegistry",
    "Species",
    "World",
    "Tile",
    "Biome",
    "BIOME_PARAMS",
    "WeatherSystem",
    "WeatherEvent",
    "WeatherType",
    "WEATHER_LABELS",
    "WEATHER_EFFECTS",
    "Simulation",
    "SimulationConfig",
    "SimulationStats",
    "Journal",
    "JournalEntry",
]
