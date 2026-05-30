import pickle
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from class_world import World

app = FastAPI(title="Ecosystem Simulation API")
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(static_dir / "index.html")

world = World()
world.spawn_food(200)
world.spawn_herbivore(200)
world.spawn_predator(0)


class StepRequest(BaseModel):
    dt: float = 1 / 60.0


class SpawnRequest(BaseModel):
    food: Optional[int] = 0
    herbivores: Optional[int] = 0
    predators: Optional[int] = 0

class SpeedRequest(BaseModel):
    multiplier: float
class RestartSettingsRequest(BaseModel):
    # Global Settings
    world_speed_multiplier: float
    global_mutation_rate: float
    global_mutation_strength: float
    
    # Plant Settings
    max_plant: int
    plant_size: float
    plant_nutrition_value: float
    plant_regrowth_power: float
    
    # Herbivore Settings
    max_herbivore: int
    herbivore_satiety_loss_factor: float
    herbivore_max_satiety: float
    herbivore_avg_gestation_time: float
    herbivore_gestation_time_std_dev: float
    herbivore_reproduction_minimum_satiety: float
    herbivore_reproduction_satiety_loss: float
    herbivore_max_percent_satiety_to_eat: float
    herbivore_FOV: float
    herbivore_vision_range: float
    herbivore_avg_age: float
    herbivore_age_std_dev: float
    herbivore_min_age_to_reproduce: float

@app.get("/save")
def save_world():
    buf = io.BytesIO()
    pickle.dump(world, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=world_save.pkl"}
    )

@app.post("/load")
async def load_world(file: UploadFile = File(...)):
    global world
    data = await file.read()
    world = pickle.load(io.BytesIO(data))
    return {"success": True}

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


@app.post("/restart_simulation")
def restart_simulation(req: RestartSettingsRequest):
    global world
    
    # 1. Create a fresh world object to clear old entities
    world = World()
    
    # 2. Apply all editable settings received from the frontend configuration
    # Globals
    world.world_speed_multiplier = req.world_speed_multiplier
    world.global_mutation_rate = req.global_mutation_rate
    world.global_mutation_strength = req.global_mutation_strength
    
    # Plants
    world.max_plant = req.max_plant
    world.plant_size = req.plant_size
    world.plant_nutrition_value = req.plant_nutrition_value
    world.plant_regrowth_power = req.plant_regrowth_power
    
    # Herbivores
    world.max_herbivore = req.max_herbivore
    world.herbivore_satiety_loss_factor = req.herbivore_satiety_loss_factor
    world.herbivore_max_satiety = req.herbivore_max_satiety
    world.herbivore_avg_gestation_time = req.herbivore_avg_gestation_time
    world.herbivore_gestation_time_std_dev = req.herbivore_gestation_time_std_dev
    world.herbivore_reproduction_minimum_satiety = req.herbivore_reproduction_minimum_satiety
    world.herbivore_reproduction_satiety_loss = req.herbivore_reproduction_satiety_loss
    world.herbivore_max_percent_satiety_to_eat = req.herbivore_max_percent_satiety_to_eat
    world.herbivore_FOV = req.herbivore_FOV
    world.herbivore_vision_range = req.herbivore_vision_range
    world.herbivore_avg_age = req.herbivore_avg_age
    world.herbivore_age_std_dev = req.herbivore_age_std_dev
    world.herbivore_min_age_to_reproduce = req.herbivore_min_age_to_reproduce
    
    # 3. recaulculate variables that depend on the above variables
    world.recalculate_dependent_attributes()
    
    # 4. Spawn initial entities into your freshly updated ecosystem
    world.spawn_food(150)
    world.spawn_herbivore(50)
    world.spawn_predator(0)
    
    return {"success": True, "message": "Simulation restarted with new settings"}


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
#this is the frontend click selection sending a message to the backend to set the selected animal index
def select_animal(data: dict):

    species = data.get("species")
    animal_id = data.get("id")

    if species == "herbivore":
        world.selected_herbivore_index = animal_id
    else:
        world.selected_herbivore_index = None

    return {"success": True}
