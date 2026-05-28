import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from class_world import World

app = FastAPI(title="Ecosystem Simulation API")
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(static_dir / "index.html")

SAVE_PATH = Path(__file__).resolve().parent / "world_save.pkl"
world = World()
world.spawn_food(150)
world.spawn_herbivore(50)
world.spawn_predator(0)


class StepRequest(BaseModel):
    dt: float = 1 / 60.0


class SpawnRequest(BaseModel):
    food: Optional[int] = 0
    herbivores: Optional[int] = 0
    predators: Optional[int] = 0


class SpeedRequest(BaseModel):
    multiplier: float

@app.post("/save")
def save_world():
    try:
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(world, f)
        return {"success": True, "path": str(SAVE_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

@app.post("/load")
def load_world():
    global world
    if not SAVE_PATH.exists():
        raise HTTPException(status_code=404, detail="No save file found")
    try:
        with open(SAVE_PATH, "rb") as f:
            world = pickle.load(f)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {e}")

@app.get("/state")
def get_state():
    return world.get_simulation_state()


@app.get("/chart")
def get_chart():
    return world.get_chart_data()


@app.post("/step")
def step(req: StepRequest):
    world.update(req.dt)
    return world.get_simulation_state()


@app.post("/spawn")
def spawn(req: SpawnRequest):
    if req.food and req.food > 0:
        world.spawn_food(req.food)
    if req.herbivores and req.herbivores > 0:
        world.spawn_herbivore(req.herbivores)
    if req.predators and req.predators > 0:
        world.spawn_predator(req.predators)
    return world.get_simulation_state()


@app.post("/speed")
def set_speed(req: SpeedRequest):
    if req.multiplier <= 0:
        raise HTTPException(status_code=400, detail="Multiplier must be positive")
    world.world_speed_multiplier = req.multiplier
    return {"world_speed_multiplier": world.world_speed_multiplier}


@app.get("/animal/{species}/{index}")
def animal_stats(species: str, index: int):
    try:
        return world.get_animal_stats(species, index)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        # Unexpected error — return 500 with detail rather than crashing
        raise HTTPException(status_code=500, detail=f"Internal error fetching animal stats: {exc}")

@app.post("/select_animal")
def select_animal(data: dict):

    species = data.get("species")
    animal_id = data.get("id")

    if species == "herbivore":
        world.selected_herbivore_index = animal_id
    else:
        world.selected_herbivore_index = None

    return {"success": True}
