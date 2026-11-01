# Project Ecosystem

**A native Linux desktop artificial life simulation.**

Organisms live, forage, hunt, drink, mate, age and die in a procedurally
generated world with day/night cycles, weather events and evolving
biomes. Each organism carries a heritable genome with mutation and
recombination; new species arise naturally through genetic drift.

This is not a scripted animation. Every organism makes its own
decisions based on its needs and surroundings, and the ecosystem evolves
without user intervention.

---

## Features

### World
- **Procedural terrain** generated from seed using fractal value noise.
- **11 biomes**: deep/shallow water, beach, grassland, forest, jungle,
  savanna, desert, rock, snow, tundra - each with distinct fertility,
  passability and food value.
- **Resources**: per-tile plant food that regrows logistically, plus
  discrete water springs in dry biomes.
- **Day/night cycle**: brightness follows a cosine curve centred on
  noon; vision range scales with daylight.
- **Weather system**: clear, cloudy, rain, storm, drought, heatwave,
  cold snap, fog - each with real gameplay effects (growth multipliers,
  damage, vision modifiers). Events spawn randomly and can be
  force-triggered from the menu.
- **Island falloff**: worlds are continents surrounded by ocean.

### Organisms
- **16 heritable traits**: size, speed, metabolism, vision range,
  max age, maturity age, reproduction threshold, offspring energy,
  litter size, diet (herbivore→carnivore), aggression, hardiness,
  water need, mutation rate, and three colour genes.
- **Survival needs**: energy, hydration, health, age.
- **Autonomous decision-making**: each tick an organism evaluates
  threats, critical needs, reproduction readiness and opportunistic
  foraging/hunting, then acts.
- **Behaviours**: foraging (plant food), hunting (prey), drinking,
  fleeing predators, seeking mates, wandering, resting at night.
- **Birth and death**: reproduction requires two mature, well-fed
  same-species parents; deaths from starvation, dehydration, injury,
  old age, or being hunted.
- **Crowding stress**: metabolism scales with local population density,
  creating natural carrying capacity.

### Genetics & Evolution
- **Genome** with clamped/wrapped traits and a heritable mutation rate.
- **Recombination**: children blend parental traits with small jitter,
  then mutate.
- **Speciation**: online clustering by genetic distance - when a child
  diverges far enough from all existing species centroids, a new
  species is founded with a pseudo-Latin name sharing its parent's
  genus.
- **Natural selection**: organisms with traits suited to their
  environment survive longer and reproduce more.
- **Species tracking**: population, births, deaths, peak, founding and
  extinction ticks, rolling colour, ancestry.

### Simulation
- **Self-sustaining**: runs without user intervention.
- **Adjustable speed**: 0.25× to 16×, plus single-step.
- **Pause/resume**.
- **Population cap** to guard performance.
- **Statistics**: time-series of population, births, deaths, species
  count, mean energy/age/hydration, weather, temperature, and
  per-species populations - all charted in the GUI.

### GUI (native PySide6 / Qt6)
- **Interactive world canvas**: pan, zoom (mouse wheel), click to
  select organisms, night overlay, weather tints, food density
  shading, vision-range indicator for the selected organism.
- **Control bar**: pause/play, speed slider, step, world size/seed,
  regenerate, weather trigger.
- **Inspector tab**: selected organism's full stats, genome traits,
  life-event log and current behaviour.
- **Species tab**: sortable table of all species (living and extinct)
  with colour swatches, populations, peaks, status.
- **Stats tab**: numeric summary + four live line charts.
- **Environment tab**: world info, biome distribution table, weather
  details.
- **Journal tab**: chronological log of notable events (species
  founding, extinctions, weather, population milestones).
- **Menus**: File (new/save/load), Simulation (pause/step/reset view),
  Weather (trigger any weather type), Help (about, quick help).

### Persistence
- **Save/load** to `~/.local/share/project-ecosystem/saves/` (or
  project-local `saves/`).
- **Atomic writes** (temp file + rename) so a crash never corrupts a
  save.
- **Save-as / load-from** arbitrary JSON files.
- **Versioned format** with forward-compatibility hooks.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- pip
- A Linux desktop with X11 or Wayland (PySide6 needs a display server)
- libEGL / libGL (usually pre-installed on desktop Linux)

### From source

```bash
git clone https://github.com/MagicDippyEgg/Project-Ecosystem-Z.AI.git
cd Project-Ecosystem-Z.AI
pip install -r requirements.txt
```

### As a package

```bash
pip install .
project-ecosystem
```

---

## Usage

### Running the app

```bash
./run.sh
```

or

```bash
python -m ecosystem.app
```

### Headless / offscreen mode

If you have no display server (e.g. CI, SSH):

```bash
QT_QPA_PLATFORM=offscreen python -m ecosystem.app
```

### Controls

| Action | Input |
|---|---|
| Select an organism | Left-click on it |
| Pan the world | Left-drag on empty space |
| Zoom | Mouse wheel |
| Reset view | `R` |
| Pause / play | `Space` or the Pause button |
| Step one tick (while paused) | `.` or the Step button |
| New world | `Ctrl+N` |
| Save world | `Ctrl+S` |
| Load world | `Ctrl+O` |
| Clear selection | `Esc` |
| Quit | `Ctrl+Q` |

### Tips for longer-running ecosystems

- Use larger worlds (128×80+) with more founding species (4-5).
- Lower the speed to watch individuals; raise it to see long-term
  dynamics.
- Trigger rain during a drought to save a struggling population.
- Extinction is a natural outcome - if a world dies, generate a new
  one with a different seed.

---

## Architecture

```
project_ecosystem/
├── ecosystem/
│   ├── __init__.py
│   ├── app.py                  # entry point
│   ├── core/                   # pure-Python simulation backend
│   │   ├── genome.py           # traits, mutation, recombination
│   │   ├── resource.py         # discrete resource nodes
│   │   ├── organism.py         # autonomous agents
│   │   ├── species.py          # speciation & tracking
│   │   ├── world.py            # terrain, biomes, food
│   │   ├── weather.py          # weather state machine
│   │   ├── simulation.py       # top-level controller
│   │   └── journal.py          # event log
│   ├── persistence/            # JSON save/load
│   └── gui/                    # PySide6 front-end
│       ├── theme.py            # colours & palette
│       ├── world_view.py       # custom QPainter canvas
│       ├── panels.py           # side panels & charts
│       └── main_window.py      # main window & menus
├── tests/                      # pytest suite (92 tests)
├── scripts/
│   ├── take_screenshots.py     # headless screenshot helper
│   └── lib/                    # bundled libEGL for minimal systems
├── saves/                      # save directory
├── run.sh                      # launcher
├── setup.py / pyproject.toml   # packaging
└── requirements.txt
```

The **core** package has no Qt dependency, so the simulation can be
imported, scripted and tested headlessly. The **gui** package wraps it
in a native PySide6 window.

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers:

- **Genome**: range clamping, wrapping, mutation, recombination,
  genetic distance, serialisation.
- **Organism**: metabolism, death causes, foraging, hunting, fleeing,
  reproduction, cannibalism prevention, serialisation.
- **World**: terrain determinism, biomes, food growth, water,
  resources, day/night, serialisation.
- **Weather**: event lifecycle, effects consistency, serialisation.
- **Species**: founding, clustering, extinction, naming, serialisation.
- **Simulation**: ticking, pause/speed, regeneration, population cap,
  stats sampling, offspring production, extreme weather, serialisation
  round-trip.
- **Persistence**: save/load, atomic writes, SaveManager, species &
  journal preservation.
- **GUI**: window construction, panel data, canvas painting,
  inspector, control signals, save/load (under offscreen QPA).

---

## How the simulation works

### Tick loop

Each call to `Simulation.tick()`:

1. Advances world time (tick count + time-of-day).
2. Updates the weather state machine.
3. Regrows resources (logistic plant food + water springs).
4. Rebuilds the spatial hash for neighbour queries.
5. Steps every living organism (shuffled to avoid positional bias):
   - Metabolises (energy/hydration/health, crowding-stress scaled).
   - Ages.
   - Checks for death.
   - Decides & acts: flee → drink → eat (hunt/forage) → reproduce →
     wander.
6. Processes birth requests (recombination + mutation + speciation).
7. Handles deaths, updates species populations.
8. Logs weather/species events to the journal.
9. Samples statistics every 8 ticks.

### Decision logic

Organisms use a greedy need-weighted priority:

1. **Flee** if a predator that can eat us is within vision and we have
   enough energy to run.
2. **Drink** if hydration < 25.
3. **Eat** if energy < 25: predators hunt prey, others forage plants.
4. **Reproduce** if mature, energy ≥ threshold, off cooldown, hydrated.
5. **Opportunistic hunting** for predators below 80% of threshold.
6. **Forage** if energy < 85% of max.
7. **Wander** otherwise (rest at night if well-fed).

### Speciation

When a child is born, its genome is compared to every living species'
centroid using a weighted, normalised genetic distance. If the closest
match is within the species threshold (0.18), the child joins that
species and the centroid shifts toward it. Otherwise a new species is
founded, inheriting the parent's genus name with a new species suffix.

---

## License

MIT
