# Ecosystem Simulation

This project simulates an evolving ecosystem with herbivores, predators, and plants using neural networks and genetic algorithms. The simulation is visualized using Pygame and includes population and fitness tracking with Matplotlib.

## Features

- Herbivores and predators controlled by neural networks
- Evolution via mutation and reproduction
- Real-time visualization with Pygame
- Population and fitness plots using Matplotlib
- Interactive selection and stats display for individual animals

## Getting Started

### Prerequisites

- Python 3.10+
- Required packages: `numpy`, `pandas`, `pygame`, `matplotlib`, `torch`, `scipy`, `tkinter`

Install dependencies with:

```sh
pip install numpy pandas pygame matplotlib torch scipy
```

### Running the Simulation

To start the simulation, run:

```sh
python main.py
```

## Files

- `main.py` — Entry point for the simulation
- `class_world.py` — Main world logic and simulation loop
- `class_herbivore_nn.py` — Neural network class for animals
- `game_functions.py` — Utility and vision functions
- `herbivore.png`, `predator.png` — Sprites for visualization

## Controls

- `r` — Toggle raycast vision display
- `SPACE` — Deselect selected animal
- `s` — Spawn more plants, herbivores, and predators
- `p` — Print extinction counters
- `a` — Print neural network hidden layer statistics
- Click on an animal to view its stats
