import pickle
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File
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
world.spawn_food(world.starting_plant)
world.spawn_herbivore(world.starting_herbivore)
world.spawn_predator(world.starting_predator)


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
    max_speed: float
    max_angular_velocity: float
    global_mutation_rate: float
    global_mutation_strength: float
    weight_std_for_new_neurons: float
    starting_herbivore: int
    starting_predator: int
    starting_plant: int
    
    # Plant Settings
    max_plant: int
    plant_size: float
    plant_nutrition_value: float
    plant_regrowth_power: float
    
    # Predator Settings
    max_predator: int
    predator_avg_gestation_time: float
    predator_gestation_time_std_dev: float
    predator_reproduction_minimum_satiety: float
    predator_reproduction_satiety_loss: float
    predator_max_percent_satiety_to_eat: float
    predator_FOV: float
    predator_vision_range: float
    predator_avg_age: float
    predator_age_std_dev: float
    predator_min_age_to_reproduce: float
    predators_resurrect_after_herbivores_reach: int
    predator_resurrection_count: int
    predator_resurrection_recent_count: int
    predator_resurrection_random_count: int
    
    # Herbivore Settings
    max_herbivore: int
    herbivore_size : float
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
    herbivore_nutrition_value: float
    herbivore_resurrection_count: int
    herbivore_resurrection_random_count: int
    herbivore_resurrection_recent_count: int

    predator_size: float
    predator_satiety_loss_factor: float
    predator_max_satiety: float
    min_hidden_dim_size: int
    max_hidden_dim_size: int

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
    #get info about world state from backend
    return world.get_simulation_state()

@app.post("/step")
def step(req: StepRequest):
    #push world state one step forward
    world.update(req.dt)
    return world.get_simulation_state()

@app.get("/chart")
def get_chart():
    #get data needed to draw charts from backend
    return world.get_chart_data()

@app.get("/save_chart_data")
def save_chart_data():
    # this endpoint provides the current chart snapshot. The browser stores the full history locally
    return world.get_chart_data()

@app.get("/settings")
def get_current_settings():
    return {
        "world_speed_multiplier": world.world_speed_multiplier,
        "max_speed": world.max_speed,
        "max_angular_velocity": world.max_angular_velocity,
        "global_mutation_rate": world.global_mutation_rate,
        "global_mutation_strength": world.global_mutation_strength,
        "weight_std_for_new_neurons": world.weight_std_for_new_neurons,
        "starting_herbivore": world.starting_herbivore,
        "starting_predator": world.starting_predator,
        "starting_plant": world.starting_plant,
        
        "max_plant": world.max_plant,
        "plant_size": world.plant_size,
        "plant_nutrition_value": world.plant_nutrition_value,
        "plant_regrowth_power": world.plant_regrowth_power,
        
        "max_predator": world.max_predator,
        "predator_avg_gestation_time": world.predator_avg_gestation_time,
        "predator_gestation_time_std_dev": world.predator_gestation_time_std_dev,
        "predator_reproduction_minimum_satiety": world.predator_reproduction_minimum_satiety,
        "predator_reproduction_satiety_loss": world.predator_reproduction_satiety_loss,
        "predator_max_percent_satiety_to_eat": world.predator_max_percent_satiety_to_eat,
        "predator_FOV": world.predator_FOV,
        "predator_vision_range": world.predator_vision_range,
        "predator_avg_age": world.predator_avg_age,
        "predator_age_std_dev": world.predator_age_std_dev,
        "predator_min_age_to_reproduce": world.predator_min_age_to_reproduce,
        "predators_resurrect_after_herbivores_reach": world.predators_resurrect_after_herbivores_reach,
        "predator_resurrection_count": world.predator_resurrection_count,
        "predator_resurrection_recent_count": world.predator_resurrection_recent_count,
        "predator_resurrection_random_count": world.predator_resurrection_random_count,
        
        "max_herbivore": world.max_herbivore,
        "herbivore_size": world.herbivore_size,
        "herbivore_satiety_loss_factor": world.herbivore_satiety_loss_factor,
        "herbivore_max_satiety": world.herbivore_max_satiety,
        "herbivore_avg_gestation_time": world.herbivore_avg_gestation_time,
        "herbivore_gestation_time_std_dev": world.herbivore_gestation_time_std_dev,
        "herbivore_reproduction_minimum_satiety": world.herbivore_reproduction_minimum_satiety,
        "herbivore_reproduction_satiety_loss": world.herbivore_reproduction_satiety_loss,
        "herbivore_max_percent_satiety_to_eat": world.herbivore_max_percent_satiety_to_eat,
        "herbivore_FOV": world.herbivore_FOV,
        "herbivore_vision_range": world.herbivore_vision_range,
        "herbivore_avg_age": world.herbivore_avg_age,
        "herbivore_age_std_dev": world.herbivore_age_std_dev,
        "herbivore_min_age_to_reproduce": world.herbivore_min_age_to_reproduce,
        "herbivore_nutrition_value": world.herbivore_nutrition_value,
        "herbivore_resurrection_count": world.herbivore_resurrection_count,
        "herbivore_resurrection_random_count": world.herbivore_resurrection_random_count,
        "herbivore_resurrection_recent_count": world.herbivore_resurrection_recent_count,

        "predator_size": world.predator_size,
        "predator_satiety_loss_factor": world.predator_satiety_loss_factor,
        "predator_max_satiety": world.predator_max_satiety,
        "min_hidden_dim_size": world.min_hidden_dim_size,
        "max_hidden_dim_size": world.max_hidden_dim_size,
        
    }

@app.post("/restart_simulation")
def restart_simulation(req: RestartSettingsRequest):
    global world
    
    # 1. Instantiate a clean world object context
    world = World()
    
    # 2. Overwrite configurations with frontend inputs
    world.world_speed_multiplier = req.world_speed_multiplier
    world.max_speed = req.max_speed
    world.max_angular_velocity = req.max_angular_velocity
    world.global_mutation_rate = req.global_mutation_rate
    world.global_mutation_strength = req.global_mutation_strength
    world.weight_std_for_new_neurons = req.weight_std_for_new_neurons
    world.starting_herbivore = req.starting_herbivore
    world.starting_predator = req.starting_predator
    world.starting_plant = req.starting_plant
    
    world.max_plant = req.max_plant
    world.plant_size = req.plant_size
    world.plant_nutrition_value = req.plant_nutrition_value
    world.plant_regrowth_power = req.plant_regrowth_power
    
    world.max_predator = req.max_predator
    world.predator_avg_gestation_time = req.predator_avg_gestation_time
    world.predator_gestation_time_std_dev = req.predator_gestation_time_std_dev
    world.predator_reproduction_minimum_satiety = req.predator_reproduction_minimum_satiety
    world.predator_reproduction_satiety_loss = req.predator_reproduction_satiety_loss
    world.predator_max_percent_satiety_to_eat = req.predator_max_percent_satiety_to_eat
    world.predator_FOV = req.predator_FOV
    world.predator_vision_range = req.predator_vision_range
    world.predator_avg_age = req.predator_avg_age
    world.predator_age_std_dev = req.predator_age_std_dev
    world.predator_min_age_to_reproduce = req.predator_min_age_to_reproduce
    world.predators_resurrect_after_herbivores_reach = req.predators_resurrect_after_herbivores_reach
    world.predator_resurrection_count = req.predator_resurrection_count
    world.predator_resurrection_recent_count = req.predator_resurrection_recent_count
    world.predator_resurrection_random_count = req.predator_resurrection_random_count
    
    world.max_herbivore = req.max_herbivore
    world.herbivore_size = req.herbivore_size
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
    world.herbivore_nutrition_value = req.herbivore_nutrition_value
    world.herbivore_resurrection_count = req.herbivore_resurrection_count
    world.herbivore_resurrection_random_count = req.herbivore_resurrection_random_count
    world.herbivore_resurrection_recent_count = req.herbivore_resurrection_recent_count

    
    world.predator_size = req.predator_size
    world.predator_satiety_loss_factor = req.predator_satiety_loss_factor
    world.predator_max_satiety = req.predator_max_satiety
    world.min_hidden_dim_size = req.min_hidden_dim_size
    world.max_hidden_dim_size = req.max_hidden_dim_size

    # This call updates dependent attributes such as plant_random_spawn_interval, 
    # plant_reproduction_interval, and re-allocates size arrays (e.g., herbivore_positions, 
    # predator_positions) matching updated capacity bounds before populating them.
    world.recalculate_dependent_attributes()
    
    # 3. Spawn initial populations using the configured starting constraints
    world.spawn_food(world.starting_plant)
    world.spawn_herbivore(world.starting_herbivore)
    world.spawn_predator(world.starting_predator)
    
    return {"success": True, "message": "Simulation restarted with new parameters"}

@app.post("/select_animal")
def select_animal(data: dict):
    #this is the frontend click selection sending a message to the backend to set the selected animal index
    species = data.get("species")
    animal_id = data.get("id")

    if species == "herbivore":
        world.selected_herbivore_index = animal_id
        world.selected_predator_index = None
        
    elif species == "predator":
        world.selected_predator_index = animal_id
        world.selected_herbivore_index = None

    else:
        world.selected_herbivore_index = None
        world.selected_predator_index = None
    
    return {"success": True}


@app.post("/debug_kill_selected")
def kill_selected_animal():
    success = world.debug_kill()
    return {"status": "success", "message": "Selected agent terminated successfully"}
