# Ecosystem Simulation

This project simulates an evolving ecosystem with herbivores, predators, and plants using neural networks and genetic algorithms. The simulation now supports a web-based frontend via FastAPI, with the legacy Pygame UI available separately.

## Features

- Herbivores and predators controlled by neural networks
- Evolution via mutation and reproduction
- Real-time visualization with Pygame
- Population and fitness plots using Matplotlib
- Interactive selection and stats display for individual animals

## Getting Started

### Prerequisites

- Python 3.10+
- Required packages: `numpy`, `pandas`, `torch`, `scipy`, `fastapi`, `uvicorn`

Install dependencies with:

```sh
pip install numpy pandas torch scipy fastapi uvicorn
```

### Running the Web Simulation

Start the backend server with:

```sh
uvicorn app:app --reload
```

Open a browser and visit:

```sh
http://127.0.0.1:8000
```

### Legacy Desktop UI

The legacy Pygame desktop UI is still available via:

```sh
python main.py
```

## Files

- `main.py` — Entry point for the simulation
- `class_world.py` — Main world logic and simulation loop
- `class_herbivore_nn.py` — Neural network class for animals
- `game_functions.py` — Utility and vision functions
- `app.py` — FastAPI backend serving simulation state for a web UI
- `herbivore.png`, `predator.png` — Sprites for visualization

## FastAPI Backend

A minimal FastAPI backend is available in `app.py`.

Run it with:

```sh
uvicorn app:app --reload
```

Then use `/state`, `/chart`, and `/step` endpoints to drive the simulation from a browser UI.

## Controls

- `r` — Toggle raycast vision display
- `SPACE` — Deselect selected animal
- `s` — Spawn more plants, herbivores, and predators
- `p` — Print extinction counters
- `a` — Print neural network hidden layer statistics
- Click on an animal to view its stats
