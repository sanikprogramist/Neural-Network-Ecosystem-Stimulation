import numpy as np
import pandas as pd
import copy
import torch
import threading

from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from game_functions import *
from class_animal_brain_nn import *

#NOTES:
# 1. fitness function doesnt work. As in, the fitness it calculates is meaningless
# 2. herbivore reproduction timer is going up even though pop is at max
# 3. current_predators are meaningless
# 4. predator preception vision is still old version
# 5. predator movement is still old
# 9. predator stats returning reproduction based on old invariable percentage
# 10. predators still use invariable gestation time and old colour change and static life expectancy and selected predator api endpoint
# 11. no predator settings in settings page
# 12. im not sure if spawn things work properly
# 13. settings sliders in settings page currently dont read default values from world itself
# 14. start them out with 1 hidden layer, then they can evolve to have more. if they want.

class World:

    def __init__(self):
        # I will mark editable settings with #M
        #globals
        self.world_speed_multiplier = 1.15 #M
        self.world_width = 800
        self.world_height = 600

        self.max_speed = 20 # max speed
        self.max_angular_velocity = 3.5 # how fast animals can turn
        self.max_acceleration = 4.0 # how fast animals can speed up and slow down
        self.weight_std_for_new_neurons = 0.35

        self.global_mutation_rate = 0.035 # stable value was 0.1 #M
        self.global_mutation_strength = 0.04 # stable value was 0.2 #M
        self.colour_change_strength = 15
        self.min_hidden_dim_size = 2 # starting bound - they might evolve smaller or larger
        self.max_hidden_dim_size = 4 # starting bound

        #plants
        self.max_plant = 300 #M
        self.plant_size = 5 #M
        self.plant_nutrition_value = 0.85 #M
        self.plant_regrowth_power = 1.0 #M already added
        self.plant_random_spawn_interval = 5 / self.plant_regrowth_power #2.2 originally
        self.plant_reproduction_interval = 10 / self.plant_regrowth_power #5.5 originally

        #predators # we dont add predator settings rn because their mechanics are very outdated
        self.max_predator = 70 
        self.predator_size = 5 
        self.predator_satiety_loss_factor = 0.006
        self.predator_max_satiety = 2 
        self.predator_gestation_time = 30
        self.predator_reproduction_minimum_satiety = 1.0
        self.predator_reproduction_satiety_loss = 0.5
        self.predator_max_percent_satiety_to_eat = 0.75 # they wont eat if their satiety is above this percentage
        self.predator_FOV = np.pi/3
        self.predator_num_of_raysections = 5
        self.predator_vision_range = 250
        self.predator_max_age = 110 
        self.predator_min_age_to_reproduce = 20

        #herbivores
        self.max_herbivore = 2000 #M
        self.herbivore_size = 4
        self.herbivore_satiety_loss_factor = 0.006 #M how fast they go hungry, this is multiplied with speed
        self.herbivore_max_satiety = 2 #M
        self.herbivore_avg_gestation_time = 25 #M
        self.herbivore_gestation_time_std_dev = 5 #M
        self.herbivore_reproduction_minimum_satiety = 1.0 #M # minimum satiety required to start gestation
        self.herbivore_reproduction_satiety_loss = 0.5 #M # how much satiety they lose when they reproduce
        self.herbivore_max_percent_satiety_to_eat = 0.75 #M # they wont eat if their satiety is above this percentage
        self.herbivore_FOV = np.pi*1.2 #M
        self.herbivore_vision_range = 170 #M
        self.herbivore_avg_age = 100 # in seconds #M
        self.herbivore_age_std_dev = 7 #M
        self.herbivore_min_age_to_reproduce = 20 #M # they wont reproduce if they are younger than this
        self.herbivore_nutrition_value = 1.0 #how much satiety predators get from eating a herbivore.

        #DO NOT EDIT
        self.plant_positions = np.zeros((self.max_plant,2))
        self.plant_reproduction_timers = np.zeros((self.max_plant,))
        self.alive_plant_array = np.zeros(self.max_plant,dtype=bool)
        self.plant_reproduction_timer_accumulator = 0.0
        self.plant_spawn_time_accumulator = 0


        self.herbivore_positions = np.zeros((self.max_herbivore,2))
        self.herbivore_angles = np.zeros((self.max_herbivore,))
        self.herbivore_speeds = np.zeros((self.max_herbivore,))
        self.herbivore_angular_velocities = np.zeros((self.max_herbivore,))
        self.herbivore_colours = np.zeros((self.max_herbivore,3))
        self.herbivore_satiety = np.zeros((self.max_herbivore,))
        self.herbivore_life_expectancy = np.zeros((self.max_herbivore,))
        self.alive_herbivore_array = np.zeros(self.max_herbivore,dtype=bool)
        self.herbivore_detectable_object_types = 3 # plant, conspecific, predator
        self.herbivore_types_of_info_about_each_object = 2 # distance and angle to it
        self.herbivore_num_external_infos = self.herbivore_detectable_object_types * self.herbivore_types_of_info_about_each_object
        self.herbivore_self_infos = 2 # speed and satiety
        self.herbivore_nn_inputs = np.zeros((self.max_herbivore, (self.herbivore_num_external_infos + self.herbivore_self_infos)),dtype=np.float32)
        #i dont know why but this is required still by app.js
        self.selected_herbivore_nn_hdim1 = None
        self.selected_herbivore_nn_hdim2 = None
        self.selected_herbivore_nn_output = None 
        self.herbivore_reproduction_timers = np.zeros((self.max_herbivore,))
        self.herbivore_gestation_time_reqs = np.zeros((self.max_herbivore,))
        self.herbivore_ages = np.zeros((self.max_herbivore,))
        self.herbivore_offsping_count = np.zeros((self.max_herbivore,))
        self.herbivore_brains = np.array([[None] * self.max_herbivore])[0]
        self.herbivore_generations = np.zeros((self.max_herbivore,))
        self.herbivore_fitnesses = np.zeros((self.max_herbivore,))
        self.herbivore_best_brain = None
        self.herbivore_current_best_fitness = 0
        self.herbivore_best_bias = 0
        self.herbivore_best_generation = 0

        self.predator_positions = np.zeros((self.max_predator,2))
        self.predator_angles = np.zeros((self.max_predator,))
        self.predator_speeds = np.zeros((self.max_predator,))
        self.predator_angular_velocities = np.zeros((self.max_predator,))
        self.predator_colours = np.zeros((self.max_predator,3))
        self.predator_satiety = np.zeros((self.max_predator,))
        self.alive_predator_array = np.zeros(self.max_predator,dtype=bool)
        self.num_types_of_visual_info = 2 # distance and desirability (which is basically whether there is a herbivore and how close it is
        self.predator_nn_inputs = np.zeros((self.max_predator, (1+self.predator_num_of_raysections)*self.num_types_of_visual_info+2),dtype=np.float32)
        self.predator_reproduction_timers = np.zeros((self.max_predator,))
        self.predator_ages = np.zeros((self.max_predator,))
        self.predator_offsping_count = np.zeros((self.max_predator,))
        self.predator_brains = np.array([[None] * self.max_predator])[0]
        self.predator_generations = np.zeros((self.max_predator,))
        self.predator_fitnesses = np.zeros((self.max_predator,))
        self.predator_best_brain = None
        self.predator_current_best_fitness = 0
        self.predator_best_bias = 0
        self.perdator_best_generation = 0

        self.current_predator = 0
        self.selected_herbivore_index = None
        self.selected_predator_index = None
        self.world_time = 0
        self.predator_death_log = pd.DataFrame(columns=["fitness", "generation"])
    
    def recalculate_dependent_attributes(self):
        #this is called when the world is restarted after new settings are applied
        self.plant_random_spawn_interval = 4 / self.plant_regrowth_power 
        self.plant_reproduction_interval = 10 / self.plant_regrowth_power 
        self.plant_positions = np.zeros((self.max_plant,2))
        self.plant_reproduction_timers = np.zeros((self.max_plant,))
        self.alive_plant_array = np.zeros(self.max_plant,dtype=bool)
        self.plant_reproduction_timer_accumulator = 0.0

        self.herbivore_positions = np.zeros((self.max_herbivore,2))
        self.herbivore_angles = np.zeros((self.max_herbivore,))
        self.herbivore_speeds = np.zeros((self.max_herbivore,))
        self.herbivore_angular_velocities = np.zeros((self.max_herbivore,))
        self.herbivore_colours = np.zeros((self.max_herbivore,3))
        self.herbivore_satiety = np.zeros((self.max_herbivore,))
        self.herbivore_life_expectancy = np.zeros((self.max_herbivore,))
        self.alive_herbivore_array = np.zeros(self.max_herbivore,dtype=bool)
        self.herbivore_nn_inputs = np.zeros((self.max_herbivore, (self.herbivore_num_external_infos + self.herbivore_self_infos)),dtype=np.float32)
        self.herbivore_reproduction_timers = np.zeros((self.max_herbivore,))
        self.herbivore_gestation_time_reqs = np.zeros((self.max_herbivore,))
        self.herbivore_ages = np.zeros((self.max_herbivore,))
        self.herbivore_offsping_count = np.zeros((self.max_herbivore,))
        self.herbivore_brains = np.array([[None] * self.max_herbivore])[0]
        self.herbivore_generations = np.zeros((self.max_herbivore,))
        self.herbivore_fitnesses = np.zeros((self.max_herbivore,))

    def update(self,dt): # super master function which updates the state of the world
        self.update_plants(dt)
        self.update_herbivores(dt)
        self.update_predators(dt)
        self.world_time += dt * self.world_speed_multiplier
    
    def get_free_indices(self, mask, slots_needed): # utility function
        free_indices = np.where(mask == False)[0][0:slots_needed]
        return free_indices
    
    def get_best_brain(self):
        return
        
    def _serialize_position(self, position):
        return [float(position[0]), float(position[1])]

    def _serialize_scalar(self, value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _to_json_compatible(self, obj):
        # Recursively convert numpy types and arrays to native Python types
        if isinstance(obj, dict):
            return {k: self._to_json_compatible(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_json_compatible(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_json_compatible(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def get_simulation_state(self):
        plant_indices = np.where(self.alive_plant_array)[0]
        herbivore_indices = np.where(self.alive_herbivore_array)[0]
        predator_indices = np.where(self.alive_predator_array)[0]

        plants = [
            {
                "x": float(self.plant_positions[i, 0]),
                "y": float(self.plant_positions[i, 1])
            }
            for i in plant_indices
        ]


        herbivores = [ 
            {
                "id": int(i),
                "species": "herbivore",
                "generation": int(self.herbivore_generations[i]),
                "x": float(self.herbivore_positions[i, 0]),
                "y": float(self.herbivore_positions[i, 1]),
                "angle": float(self.herbivore_angles[i]),
                "speed": float(self.herbivore_speeds[i]),
                "red": int(self.herbivore_colours[i,0]),
                "green": int(self.herbivore_colours[i,1]),
                "blue": int(self.herbivore_colours[i,2]),
            }
            for i in herbivore_indices
        ]


        predators = [
        {
            "id": int(i),
            "species": "predator",
            "x": float(self.predator_positions[i, 0]),
            "y": float(self.predator_positions[i, 1]),
            "angle": float(self.predator_angles[i]),
            "speed": float(self.predator_speeds[i]),
            "red": int(self.predator_colours[i,0]),
            "green": int(self.predator_colours[i,1]),
            "blue": int(self.predator_colours[i,2]),
            "satiety": float(self.predator_satiety[i]),
            "age": float(self.predator_ages[i]),
            "generation": int(self.predator_generations[i]),
            "fitness": float(self.predator_fitnesses[i]),
            "reproduction_progress": float(self.predator_reproduction_timers[i] / max(self.predator_gestation_time, 1)),
            "nn_distances" : self.predator_nn_inputs[i, 0:self.predator_num_of_raysections+1].tolist(),
            "nn_desirability_labels" : self.predator_nn_inputs[i, self.predator_num_of_raysections+1:self.predator_num_of_raysections*2+2].tolist(),
            "fov": float(self.predator_FOV),
            "vision_range": float(self.predator_vision_range),
        }
            for i in predator_indices
            ]
        selected = None
        if self.selected_herbivore_index is not None and self.alive_herbivore_array[self.selected_herbivore_index]:
            brain = self.herbivore_brains[self.selected_herbivore_index]
            weights = self._to_json_compatible(brain.get_network_weights())
            selected = {
                "species": "herbivore",
                "id": int(self.selected_herbivore_index),
                "x": float(self.herbivore_positions[self.selected_herbivore_index, 0]),
                "y": float(self.herbivore_positions[self.selected_herbivore_index, 1]),
                "speed": float(self.herbivore_speeds[self.selected_herbivore_index]),
                "face_direction": float(self.herbivore_angles[self.selected_herbivore_index]),
                "satiety": float(self.herbivore_satiety[self.selected_herbivore_index]),
                "age": float(self.herbivore_ages[self.selected_herbivore_index]),
                "generation": int(self.herbivore_generations[self.selected_herbivore_index]),
                "fitness": float(self.herbivore_fitnesses[self.selected_herbivore_index]),
                "reproduction_progress": float(self.herbivore_reproduction_timers[self.selected_herbivore_index] / self.herbivore_gestation_time_reqs[self.selected_herbivore_index]),
                "fov": float(self.herbivore_FOV),
                "vision_range": float(self.herbivore_vision_range),
                "offspring_count": int(self.herbivore_offsping_count[self.selected_herbivore_index]),

                "nn_distances_angles" : self.herbivore_nn_inputs[self.selected_herbivore_index,0:self.herbivore_num_external_infos].tolist(), #this is basically the inputs again
                "inputs": self.herbivore_nn_inputs[self.selected_herbivore_index].tolist(),
                #clean this up later: input, hidden_dim1, hidden_dim2, output already contained in weights
                "hidden_dim_1": self.selected_herbivore_nn_hdim1.tolist() if self.selected_herbivore_nn_hdim1 is not None else None,
                "hidden_dim_2": self.selected_herbivore_nn_hdim2.tolist() if self.selected_herbivore_nn_hdim2 is not None else None,
                "output": self.selected_herbivore_nn_output.tolist() if self.selected_herbivore_nn_output is not None else None,
                "weights": weights,
            }
        elif self.selected_predator_index is not None and self.alive_predator_array[self.selected_predator_index]:
            selected = {
                "species": "predator",
                "id": int(self.selected_predator_index)
            }

        return {
            "world": {
                "width": int(self.world_width),
                "height": int(self.world_height),
                "time": float(self.world_time),
                "current_plant": int(np.sum(self.alive_plant_array)),
                "current_herbivore": int(np.sum(self.alive_herbivore_array)),
                "current_predator": int(np.sum(self.alive_predator_array)),
            },
            "plants": plants,
            "herbivores": herbivores,
            "predators": predators,
            "selected": selected,
        }

    def get_chart_data(self):
        return {
            "world_time": float(self.world_time),
            "current_plant": int(np.sum(self.alive_plant_array)),
            "current_herbivore": int(np.sum(self.alive_herbivore_array)),
            "current_predator": int(np.sum(self.alive_predator_array)),
        }

    def get_animal_stats(self, species, index):
        if species == "herbivore":
                if index < 0 or index >= self.max_herbivore or not self.alive_herbivore_array[index]:
                    raise IndexError("Herbivore index is invalid or dead")
                brain = self.herbivore_brains[index]
                positions = self.herbivore_positions[index]
                stats = {
                        "species": "herbivore",
                        "id": int(index),
                        "x": float(positions[0]),
                        "y": float(positions[1]),
                        "angle": float(self.herbivore_angles[index]),
                        "speed": float(self.herbivore_speeds[index]),
                        "satiety": float(self.herbivore_satiety[index]),
                        "age": float(self.herbivore_ages[index]),
                        "generation": int(self.herbivore_generations[index]),
                        "fitness": float(self.herbivore_fitnesses[index]),
                        "reproduction_progress": float(self.herbivore_reproduction_timers[index] / max(self.herbivore_gestation_time_reqs[index], 1)),
                        "children": int(self.herbivore_offsping_count[index]),
                        "hidden_dim_1": None,
                        "hidden_dim_2": None,
                        "network_weights": None,
                    }
                # if brain exists, include hidden layer sizes and weights
                if brain is not None:
                    try:
                        dims = brain.get_dim_sizes()
                        stats["hidden_dim_1"] = int(dims[0])
                        stats["hidden_dim_2"] = int(dims[1])
                        stats["network_weights"] = brain.get_network_weights()
                    except Exception:
                        stats["hidden_dim_1"] = None
                        stats["hidden_dim_2"] = None
                        stats["network_weights"] = None
                return self._to_json_compatible(stats)
        elif species == "predator":
            if index < 0 or index >= self.max_predator or not self.alive_predator_array[index]:
                raise IndexError("Predator index is invalid or dead")
            brain = self.predator_brains[index]
            positions = self.predator_positions[index]
            stats = {
                    "species": "predator",
                    "id": int(index),
                    "x": float(positions[0]),
                    "y": float(positions[1]),
                    "angle": float(self.predator_angles[index]),
                    "speed": float(self.predator_speeds[index]),
                    "satiety": float(self.predator_satiety[index]),
                    "age": float(self.predator_ages[index]),
                    "generation": int(self.predator_generations[index]),
                    "fitness": float(self.predator_fitnesses[index]),
                    "reproduction_progress": float(self.predator_reproduction_timers[index] / max(self.predator_gestation_time, 1)),
                    "children": int(self.predator_offsping_count[index]),
                    "hidden_dim_1": None,
                    "hidden_dim_2": None,
                    "network_weights": None,
                }
            if brain is not None:
                try:
                    dims = brain.get_dim_sizes()
                    stats["hidden_dim_1"] = int(dims[0])
                    stats["hidden_dim_2"] = int(dims[1])
                    stats["network_weights"] = brain.get_network_weights()
                except Exception:
                    stats["hidden_dim_1"] = None
                    stats["hidden_dim_2"] = None
                    stats["network_weights"] = None
            return self._to_json_compatible(stats)
        else:
            raise ValueError("Species must be 'herbivore' or 'predator'")
        



############# -------------------------------------- PLANTS ----------------------------------- ###############   
############# -------------------------------------- PLANTS ----------------------------------- ###############
############# -------------------------------------- PLANTS ----------------------------------- ###############
    def update_plants(self,dt): # master plant function
        self.spawn_random_plants(dt)
        self.plants_reproduce(dt)

    def spawn_random_plants(self,dt):
        self.plant_spawn_time_accumulator += dt * self.world_speed_multiplier
        if (self.plant_spawn_time_accumulator > self.plant_random_spawn_interval):
            self.spawn_food(1)
            self.plant_spawn_time_accumulator -= self.plant_random_spawn_interval
    
    def plants_reproduce(self, dt):
        #this is weird
        self.plant_reproduction_timer_accumulator += dt * self.world_speed_multiplier
        self.plant_reproduction_timers[self.alive_plant_array] += dt * self.world_speed_multiplier
        if self.plant_reproduction_timer_accumulator > 1.5:
            reproducing_plant_indices = np.where((self.plant_reproduction_timers >= self.plant_reproduction_interval) & 
                                                    self.alive_plant_array)[0]
            self.spawn_food_from_parents(reproducing_plant_indices)
            self.plant_reproduction_timer_accumulator = 0.0
    
    def spawn_food(self,how_many_to_spawn):
            available_slots = np.sum(np.invert(self.alive_plant_array))
            spawn_count = min(how_many_to_spawn, available_slots)

            if spawn_count > 0:
                free_indeces = self.get_free_indices(self.alive_plant_array,spawn_count)


                new_plant_positions = np.random.randint(low=[1,1], 
                                                        high=[self.world_width-1,self.world_height-1],
                                                        size=(spawn_count,2))
                self.plant_positions[free_indeces] = new_plant_positions
                self.alive_plant_array[free_indeces] = True
                self.plant_reproduction_timers[free_indeces] = 0.0
                #self.current_plant += spawn_count
    
    def spawn_food_from_parents(self, parent_indices):
        available_slots = np.sum(~self.alive_plant_array)
        spawn_count = min(parent_indices.size, available_slots)

        if spawn_count == 0:
            return

        parent_indices = parent_indices[:spawn_count]
        free_indices = self.get_free_indices(self.alive_plant_array, spawn_count)

        max_attempts = 10
        placed_mask = np.zeros(spawn_count, dtype=bool)

        for attempt in range(max_attempts):
            # Generate candidate positions for all unplaced plants in batch
            unplaced_idx = np.where(~placed_mask)[0]
            if unplaced_idx.size == 0:
                break

            random_offsets = np.random.randint(
                low=[-15, -15],
                high=[15, 15],
                size=(unplaced_idx.size, 2)
            )
            candidate_positions = self.plant_positions[parent_indices[unplaced_idx]] + random_offsets

            # Clamp to boundaries
            candidate_positions[:, 0] = np.clip(candidate_positions[:, 0], self.plant_size, self.world_width - self.plant_size)
            candidate_positions[:, 1] = np.clip(candidate_positions[:, 1], self.plant_size, self.world_height - self.plant_size)

            # Check collisions in batch
            alive_positions = self.plant_positions[self.alive_plant_array]
            if alive_positions.shape[0] == 0:
                dists_ok = np.ones(unplaced_idx.size, dtype=bool)
            else:
                dists = np.linalg.norm(alive_positions[None, :, :] - candidate_positions[:, None, :], axis=2)
                dists_ok = np.all(dists > self.plant_size * 2, axis=1)

            # Assign positions for those that found valid spots
            to_place_idx = unplaced_idx[dists_ok]
            self.plant_positions[free_indices[to_place_idx]] = candidate_positions[dists_ok]
            self.alive_plant_array[free_indices[to_place_idx]] = True
            self.plant_reproduction_timers[free_indices[to_place_idx]] = 0.0

            placed_mask[to_place_idx] = True

        # Final bookkeeping
        self.plant_reproduction_timers[parent_indices] = 0.0


################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
    def update_herbivores(self,dt): # master function that includes all the rest
        self.herbivores_increment_fitness(dt)
        self.herbivores_die_of_natural_causes(dt) 
        self.herbivores_perceive() # calculate neural network inputs
        self.herbivores_process_NN(dt) # push neural network inputs to the NNs, get movement parameters out
        self.herbivores_move(dt) # update new positions using the parameters from previous function
        self.check_herbivore_plant_collisions() # check if they collided with food
        self.herbivores_reproduce(dt) # check if they reproduce 
    
    def herbivores_increment_fitness(self,dt):
        return
        #self.herbivore_fitnesses[self.alive_herbivore_array] += 0.05* (self.herbivore_satiety[self.alive_herbivore_array]/self.herbivore_max_satiety) * dt * self.world_speed_multiplier * (1/max(self.current_plant, 0.01)) * (1+0.5*self.current_predator/self.max_predator)

    def herbivores_die_of_natural_causes(self, dt):
        #check is anyone starved
        self.herbivore_satiety[self.alive_herbivore_array] -= (self.herbivore_satiety_loss_factor * 6.5 * abs(self.herbivore_speeds[self.alive_herbivore_array]/self.max_speed) + 2.5 * self.herbivore_satiety_loss_factor) * dt * self.world_speed_multiplier
        self.alive_herbivore_array &= (self.herbivore_satiety > 0)

        #check if anyone died of old age
        self.herbivore_ages[self.alive_herbivore_array] += dt * self.world_speed_multiplier
        self.alive_herbivore_array &= (self.herbivore_ages < self.herbivore_life_expectancy)
        
        if self.selected_herbivore_index != None:
            if self.alive_herbivore_array[self.selected_herbivore_index] == False:
                #check if selected herbivore died
                self.selected_herbivore_index = None
            

    def herbivores_perceive(self):
        alive_indices = np.where(self.alive_herbivore_array)[0]
        if alive_indices.size == 0:
            return
        
        #the new nn input for herbivores should be: [dist_plant, angle_plant, dist_conspecific, angle_conspecific, dist_predator, angle_predator, own_satiety, own_speed]
        output_from_new_perception_function = herbivores_perception_function(
            self_positions = self.herbivore_positions[alive_indices],
            self_angles = self.herbivore_angles[alive_indices],
            food_positions = self.plant_positions[self.alive_plant_array], 
            predator_positions = self.predator_positions[self.alive_predator_array],
            vision_range = self.herbivore_vision_range,
            vision_fov = self.herbivore_FOV
        )

        self.herbivore_nn_inputs[alive_indices,0:self.herbivore_num_external_infos] = output_from_new_perception_function
        self.herbivore_nn_inputs[alive_indices,self.herbivore_num_external_infos:self.herbivore_num_external_infos+2] = np.stack((1-(self.herbivore_satiety[alive_indices] / self.herbivore_max_satiety), self.herbivore_speeds[alive_indices] / self.max_speed), axis=1)

    def herbivores_process_NN(self, dt):
        alive_indices = np.where(self.alive_herbivore_array)[0]
        if alive_indices.size == 0:
            return

        # === Batch processing using precomputed nn_inputs ===
        input_tensor = torch.from_numpy(self.herbivore_nn_inputs[alive_indices])  # shape (N_alive, 8)

        # === Forward pass through each individual's brain ===
        outputs = []
        for i, idx in enumerate(alive_indices):
            brain = self.herbivore_brains[idx]
            if idx == self.selected_herbivore_index:
                with torch.no_grad():
                    output, h1, h2 = brain(
                        input_tensor[i].unsqueeze(0),
                        return_activations=True
                    )

                self.selected_herbivore_nn_hdim1 = h1.cpu().numpy()[0]
                self.selected_herbivore_nn_hdim2 = h2.cpu().numpy()[0]
                self.selected_herbivore_nn_output = output.cpu().numpy()[0]

                output = output.numpy()[0]

            else:
                with torch.no_grad():
                    output = brain(input_tensor[i].unsqueeze(0), return_activations=False).numpy()[0]
            outputs.append(output)

        outputs = np.array(outputs, dtype=np.float32)  # shape (N_alive, 2)

        # === assign outputs ===
         #First output is the acceleration
        #if this was a cleaner codebase, this would go in herbivores_move
        #output 0 means no acceleration. -1 means speed is decreased by max_acceleration, 1 means speed is increased by max_acceleration this tick
        self.herbivore_speeds[alive_indices] = outputs[:, 0] * self.max_speed #* self.world_speed_multiplier * dt
        #speed is clipped between 0 and max speed
        #self.herbivore_speeds[alive_indices] = np.clip(self.herbivore_speeds[alive_indices], 0, self.max_speed)

        #second output is directly assigned to angular velocity array after scaling:
        target_angular_velocities = outputs[:, 1] * self.max_angular_velocity * self.world_speed_multiplier * dt
        self.herbivore_angular_velocities[alive_indices] = target_angular_velocities

    def herbivores_move(self,dt):
        #okay so nn outputs are the speed and angular velocity. angular velocity gets added onto existing facing direction.
        self.herbivore_angles[self.alive_herbivore_array] += self.herbivore_angular_velocities[self.alive_herbivore_array] * dt * self.world_speed_multiplier
        self.herbivore_angles %= 2 * np.pi

        #using the new angle, the direction vector is calculated:
        dpos = np.zeros((np.sum(self.alive_herbivore_array),2))

        np.cos(self.herbivore_angles[self.alive_herbivore_array], out=dpos[:,0])
        np.sin(self.herbivore_angles[self.alive_herbivore_array], out=dpos[:,1])

        #then the direction vector is multiplied by the speed to get the actual movement vector:
        dpos *= self.herbivore_speeds[self.alive_herbivore_array, np.newaxis] * self.world_speed_multiplier  * dt

        self.herbivore_positions[self.alive_herbivore_array] += dpos 

        #handle world boundary
        self.herbivore_positions[:, 0] %= self.world_width
        self.herbivore_positions[:, 1] %= self.world_height


    def check_herbivore_plant_collisions(self):
        if not np.any(self.alive_herbivore_array) or not np.any(self.alive_plant_array):
            return

        d = distance_matrix(self.herbivore_positions[self.alive_herbivore_array], self.plant_positions[self.alive_plant_array])
        rows, cols = np.where(d <= self.plant_size + 2)
        if rows.size == 0:
            return

        alive_herbivores = np.where(self.alive_herbivore_array)[0]
        alive_plants = np.where(self.alive_plant_array)[0]

        valid = (rows < alive_herbivores.shape[0]) & (cols < alive_plants.shape[0])
        if not np.all(valid):
            rows = rows[valid]
            cols = cols[valid]
            if rows.size == 0:
                return

        herbivores_that_ate_indices = alive_herbivores[rows]
        plants_that_were_eaten_indices = alive_plants[cols]

        # Filter herbivores that have satiety < X% of max
        can_eat_mask = self.herbivore_satiety[herbivores_that_ate_indices] < (self.herbivore_max_percent_satiety_to_eat * self.herbivore_max_satiety)

        # Apply filter
        herbivores_that_ate_indices = herbivores_that_ate_indices[can_eat_mask]
        plants_that_were_eaten_indices = plants_that_were_eaten_indices[can_eat_mask]

        # Feed herbivores
        self.herbivore_satiety[herbivores_that_ate_indices] += self.plant_nutrition_value
        np.clip(self.herbivore_satiety, -1, self.herbivore_max_satiety, out=self.herbivore_satiety)

        # Mark plants as dead
        self.alive_plant_array[plants_that_were_eaten_indices] = False
        #self.current_plant -= plants_that_were_eaten_indices.size
    
    def herbivores_reproduce(self,dt):
        sated_herbivore_indices = np.where((self.alive_herbivore_array) & 
                                           (self.herbivore_ages >= self.herbivore_min_age_to_reproduce) &
                                           (self.herbivore_satiety >= self.herbivore_reproduction_minimum_satiety))[0]
        self.herbivore_reproduction_timers[sated_herbivore_indices] += dt*self.world_speed_multiplier
        # np.clip(self.herbivore_reproduction_timers, a_min=0, a_max=self.herbivore_gestation_time_reqs,out=self.herbivore_reproduction_timers)
        # i dont know why we're clipping it 
        reproducing_herbivore_indices = np.where((self.herbivore_reproduction_timers >= self.herbivore_gestation_time_reqs) & 
                                                 self.alive_herbivore_array & 
                                                 (self.herbivore_satiety >= self.herbivore_reproduction_minimum_satiety))[0]        
        for i in reproducing_herbivore_indices:
            self.spawn_herbivore(1,parent_index=i)


    def spawn_herbivore(self, how_many_to_spawn, parent_index=-1):
            available_slots = np.sum(np.invert(self.alive_herbivore_array))
            spawn_count = min(how_many_to_spawn, available_slots)

            if spawn_count > 0: # if there are actually free slots to spawn AND new herbivores that need spawning, then spawn
                free_indeces = self.get_free_indices(self.alive_herbivore_array,spawn_count)

                if parent_index < 0: # if parent index is less than 0, it means we are spawning randomly (not from a parent), so we will spawn in random positions and with random brains
                    new_herbivore_positions = np.random.randint(low=[1,1], 
                                                        high=[self.world_width-1,self.world_height-1],
                                                        size=(spawn_count,2))
                    new_herbivore_angles = np.random.uniform(low=-np.pi, 
                                                        high=np.pi,
                                                        size=spawn_count)
                    self.herbivore_positions[free_indeces] = new_herbivore_positions
                    self.herbivore_angles[free_indeces] = new_herbivore_angles
                    self.herbivore_speeds[free_indeces] = 0
                    self.herbivore_angular_velocities[free_indeces] = 0
                    self.herbivore_colours[free_indeces] = np.random.randint(0,255,size=(spawn_count,3))
                    self.herbivore_reproduction_timers[free_indeces] = 0.0
                    self.herbivore_ages[free_indeces] = 0
                    self.herbivore_fitnesses[free_indeces] = 0
                    self.herbivore_offsping_count[free_indeces] = 0.0
                    self.herbivore_satiety[free_indeces] = 1
                    self.alive_herbivore_array[free_indeces] = True 
                    self.herbivore_generations[free_indeces] = 0
                    self.herbivore_life_expectancy[free_indeces] = np.random.normal(loc=self.herbivore_avg_age, scale=self.herbivore_age_std_dev, size=spawn_count)
                    self.herbivore_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)
                    for idx in free_indeces:
                        self.herbivore_brains[idx] = AnimalBrain(
                            n_external_infos = self.herbivore_num_external_infos,
                            n_self_infos=self.herbivore_self_infos,
                            hidden_dim_1=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0],
                            hidden_dim_2=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0],
                            initial_weight_std = self.weight_std_for_new_neurons
                            )
                
                else: #spawning from a parent
                    self.herbivore_offsping_count[parent_index] += 1
                    #since this option will always just spawn one animal, free indeces will always be a single number,
                    #and so will spawn count
                    parent_position = self.herbivore_positions[parent_index]
                    child_position = parent_position + np.random.randint(0,1,2)
                    self.herbivore_positions[free_indeces] = child_position
                    # we need to add gaussian noise multiplied bz self.mutation_strength to parent colour
                    self.herbivore_angles[free_indeces] = np.random.randint(-np.pi,np.pi,1)
                    self.herbivore_speeds[free_indeces] = 0
                    self.herbivore_angular_velocities[free_indeces] = 0
                    self.herbivore_reproduction_timers[free_indeces] = 0
                    self.herbivore_ages[free_indeces] = 0.0
                    self.herbivore_offsping_count[free_indeces] = 0.0
                    self.herbivore_satiety[free_indeces] = 1
                    self.herbivore_fitnesses[free_indeces] = 0
                    self.alive_herbivore_array[free_indeces] = True
                    self.herbivore_generations[free_indeces] = self.herbivore_generations[parent_index]+1
                    #normal distribution for life expectancy and gestation time:
                    self.herbivore_life_expectancy[free_indeces] = np.random.normal(loc=self.herbivore_avg_age, scale=self.herbivore_age_std_dev, size=spawn_count)
                    self.herbivore_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)
                    parent_brain = self.herbivore_brains[parent_index]
                    child_brain = copy.deepcopy(parent_brain)

                    # mutate weights:
                    child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)

                    #calculate new colour based on similarity between parent and child brains:
                    brain_distance = calculate_brain_similarity(parent_brain, child_brain)

                    # possibility to add or delete a neuron in a hidden layer
                    sizes = child_brain.get_dim_sizes()
                    for dimension, index in zip(["fc1","fc2"],[0,1]):
                        if np.random.uniform(low=0.0,high=1.0,size=1)[0] <= self.global_mutation_rate:
                            child_brain = resize_layer_in_animal_brain(child_brain,
                                                                layer=dimension,
                                                                new_size=np.clip((sizes[index]+np.random.choice([-1,1])),a_min=1,a_max=20),
                                                                init_std=self.weight_std_for_new_neurons)
                            brain_distance += 0.3 # if we add or delete a neuron, we consider that a bigger mutation than just changing weights, so we add a fixed value to the brain distance to reflect that
                    
                    self.herbivore_brains[free_indeces] = child_brain
                    new_colour = self.herbivore_colours[parent_index] + np.random.normal(0, brain_distance*150, size=(spawn_count,3))
                    self.herbivore_colours[free_indeces] = np.clip(new_colour, 0, 255)

                    self.herbivore_reproduction_timers[parent_index] = 0.0
                    self.herbivore_gestation_time_reqs[parent_index] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)[0]
                    self.herbivore_satiety[parent_index] -= self.herbivore_reproduction_satiety_loss

    def resurrect_herbivores(self, spawn_count):
            return
            # outdated

            free_indeces = self.get_free_indices(self.alive_herbivore_array,spawn_count)

            new_positions = np.random.randint(low=[1,1], 
                                                high=[self.world_width-1,self.world_height-1],
                                                size=(spawn_count,2))
            new_angles = np.random.uniform(low=-np.pi, 
                                                high=np.pi,
                                                size=spawn_count)
            self.herbivore_positions[free_indeces] = new_positions
            self.herbivore_angles[free_indeces] = new_angles
            self.herbivore_speeds[free_indeces] = 0
            self.herbivore_angular_velocities[free_indeces] = 0
            self.herbivore_reproduction_timers[free_indeces] = 0.0
            self.herbivore_ages[free_indeces] = 5
            self.herbivore_offsping_count[free_indeces] = 0.0
            self.herbivore_satiety[free_indeces] = 1                
            self.herbivore_fitnesses[free_indeces] = 0
            self.alive_herbivore_array[free_indeces] = True
            #self.herbivore_nn_inputs[free_indeces,self.herbivore_num_of_raysections*2+1] = np.clip(self.herbivore_best_bias + np.random.uniform(low=-self.global_mutation_strength,high=self.global_mutation_strength,size=1),-1,1)
            self.herbivore_generations[free_indeces] = self.herbivore_best_generation+1
            for idx in free_indeces:
                new_brain = copy.deepcopy(self.herbivore_best_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                self.herbivore_brains[idx] = new_brain
            self.current_predator = spawn_count


################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
    def update_predators(self,dt): # master function that includes all the rest
        self.predators_up_fitness(dt)
        self.predators_die_of_natural_causes(dt) 
        self.predators_perceive() # calculate raysection casting outputs
        self.predators_process_NN() # push raysection casting outputs into the NNs, get movement parameters out
        self.predators_move(dt) # update new positions using the parameters from previous function
        self.check_predator_herbivore_collisions() # check if they collided with food
        self.predators_reproduce(dt) # check if they reproduce 
    
    def predators_up_fitness(self,dt):
        return
        #self.predator_fitnesses[self.alive_predator_array] += 0.05* (self.predator_satiety[self.alive_predator_array]/self.predator_max_satiety) * dt * self.world_speed_multiplier * max((0.4+1/max(self.current_herbivore, 0.01)),1)
        
    def predators_die_of_natural_causes(self, dt):
        #check is anyone starved
        array_copy_for_checking = copy.deepcopy(self.alive_predator_array)
        self.predator_satiety[self.alive_predator_array] -= self.predator_satiety_loss_factor * 8 * (self.predator_speeds[self.alive_predator_array]/self.max_speed) * dt * self.world_speed_multiplier + self.predator_satiety_loss_factor * dt * self.world_speed_multiplier
        self.alive_predator_array &= (self.predator_satiety > 0)

        #check if anyone died of old age
        self.predator_ages[self.alive_predator_array] += dt * self.world_speed_multiplier
        self.alive_predator_array &= (self.predator_ages < self.predator_max_age)

        indices_of_dead_animals = np.where((array_copy_for_checking == True) & (self.alive_predator_array == False))[0]
        if indices_of_dead_animals.size > 0:
            
            deaths_df = pd.DataFrame({
                "fitness": self.predator_fitnesses[indices_of_dead_animals],
                "generation": self.predator_generations[indices_of_dead_animals]
            })
            self.predator_death_log = pd.concat([self.predator_death_log, deaths_df], ignore_index=True)
            N = 50
            current_max_gen = self.predator_generations.max()
            min_allowed_gen = current_max_gen - N + 1
            self.predator_death_log = self.predator_death_log[self.predator_death_log["generation"] >= min_allowed_gen].reset_index(drop=True)

        self.current_predator = np.count_nonzero(self.alive_predator_array)
        if self.selected_predator_index != None:
            if self.alive_predator_array[self.selected_predator_index] == False:
                self.selected_predator_index = None
    
    def predators_perceive(self):
        alive_indices = np.where(self.alive_predator_array)[0]
        if alive_indices.size == 0:
            return
 
        distances_sections, desirabilities_sections = predator_section_vision_self_and_food(
            self_positions=self.predator_positions[alive_indices],
            self_angles=self.predator_angles[alive_indices],
            food_positions=self.herbivore_positions[self.alive_herbivore_array],
            self_num_of_raysections = self.predator_num_of_raysections,
            self_vision_range = self.predator_vision_range,
            self_fov = self.predator_FOV)
        # distances_sections: (N_alive, N_raysections+1)
        # desirabilities_sections: (N_alive, N_raysections+1)
        
        # Direct write distances to columns 0:5 (or however many raysections there are)
        self.predator_nn_inputs[alive_indices, 0:self.predator_num_of_raysections+1] = distances_sections

        # Direct write desirabilities to columns 5:10
        self.predator_nn_inputs[alive_indices, self.predator_num_of_raysections+1:self.predator_num_of_raysections*2+2] = desirabilities_sections

        # Satiety normalized to column raysection*2
        self.predator_nn_inputs[alive_indices, self.predator_num_of_raysections*2+2] = self.predator_satiety[alive_indices] / self.predator_max_satiety 
    
    def predators_process_NN(self):
        alive_indices = np.where(self.alive_predator_array)[0]
        if alive_indices.size == 0:
            return

        # === Batch processing using precomputed nn_inputs ===
        input_tensor = torch.from_numpy(self.predator_nn_inputs[alive_indices])  # shape (N_alive, 12)

        # === Forward pass through each individual's brain ===
        outputs = []
        for i, idx in enumerate(alive_indices):
            brain = self.predator_brains[idx]
            with torch.no_grad():
                output = brain(input_tensor[i].unsqueeze(0)).numpy()[0]
            outputs.append(output)
        outputs = np.array(outputs, dtype=np.float32)  # shape (N_alive, 2)

        # === Scale outputs ===
        target_speeds = (outputs[:, 0] + 1) / 2 * self.max_speed
        target_angular_velocities = outputs[:, 1] * self.max_angular_velocity

        # === Assign directly ===
        self.predator_speeds[alive_indices] = target_speeds
        self.predator_angular_velocities[alive_indices] = target_angular_velocities
    
    def predators_move(self,dt):
        #self.herbivore_angular_velocities[self.alive_herbivore_array] += self.herbivore_angular_accelerations[self.alive_herbivore_array] * dt * self.world_speed_multiplier - 1*self.herbivore_angular_velocities[self.alive_herbivore_array] * self.speed_friction_coefficient * dt * self.world_speed_multiplier
        self.predator_angles[self.alive_predator_array] += self.predator_angular_velocities[self.alive_predator_array] * dt * self.world_speed_multiplier
        self.predator_angles %= 2 * np.pi
        #np.clip(self.herbivore_angular_velocities, -self.max_angular_velocity,self.max_angular_velocity,out=self.herbivore_angular_velocities)

        #self.herbivore_speeds[self.alive_herbivore_array] += self.herbivore_accelerations[self.alive_herbivore_array] * dt * self.world_speed_multiplier - self.herbivore_speeds[self.alive_herbivore_array] * self.speed_friction_coefficient * dt * self.world_speed_multiplier
        #np.clip(self.herbivore_speeds, 0,self.max_speed,out=self.herbivore_speeds)
        
        dpos = np.zeros((np.sum(self.alive_predator_array),2))

        np.cos(self.predator_angles[self.alive_predator_array], out=dpos[:,0])
        np.sin(self.predator_angles[self.alive_predator_array], out=dpos[:,1])

        dpos *= self.predator_speeds[self.alive_predator_array, np.newaxis]

        self.predator_positions[self.alive_predator_array] += dpos * self.world_speed_multiplier  * dt

        #handle world boundary
        #self.herbivore_positions[:,0] = np.clip(self.herbivore_positions[:,0], 0+self.herbivore_size, self.world_width-self.herbivore_size)
        #self.herbivore_positions[:,1] = np.clip(self.herbivore_positions[:,1], 0+self.herbivore_size, self.world_height-self.herbivore_size)
        self.predator_positions[:, 0] %= self.world_width
        self.predator_positions[:, 1] %= self.world_height
    
    def check_predator_herbivore_collisions(self):
        d = distance_matrix(self.predator_positions[self.alive_predator_array],self.herbivore_positions[self.alive_herbivore_array])
        consumed = np.column_stack(np.where(d<=self.herbivore_size+1))
        
        #careful, these are indices of the alive animals
        #consumed_herbivores_indices = consumed[:,1]
        #predators_that_ate_indices = consumed[:,0]

        alive_predators = np.where(self.alive_predator_array)[0]
        predators_that_ate_indices = alive_predators[consumed[:,0]]

        alive_herbivores = np.where(self.alive_herbivore_array)[0]
        herbivores_that_were_eaten_indices = alive_herbivores[consumed[:,1]]

        # Filter predators that have satiety < X% of max
        can_eat_mask = self.predator_satiety[predators_that_ate_indices] < (self.predator_max_percent_satiety_to_eat * self.predator_max_satiety)

        # Apply filter
        predators_that_ate_indices = predators_that_ate_indices[can_eat_mask]
        herbivores_that_were_eaten_indices = herbivores_that_were_eaten_indices[can_eat_mask]

        # Feed predators
        self.predator_satiety[predators_that_ate_indices] += self.herbivore_nutrition_value
        np.clip(self.predator_satiety, -1, self.predator_max_satiety, out=self.predator_satiety)

        # Mark herbivores as dead
        self.alive_herbivore_array[herbivores_that_were_eaten_indices] = False
        # also need to remove selected herbivore if it was eaten:
        # THIS COULD CAUSE A BUG WHEN SELECTED HERBIVORE EATEN
        if self.selected_herbivore_index in herbivores_that_were_eaten_indices:
            self.selected_herbivore_index = None
    
    def predators_reproduce(self,dt):
        sated_predator_indices = np.where((self.alive_predator_array) & 
                                           (self.predator_ages >= self.predator_min_age_to_reproduce) &
                                           (self.predator_satiety >= self.predator_reproduction_minimum_satiety))[0]
        self.predator_reproduction_timers[sated_predator_indices] += dt*self.world_speed_multiplier
        np.clip(self.predator_reproduction_timers, a_min=0, a_max=self.predator_gestation_time,out=self.predator_reproduction_timers)


        reproducing_predator_indices = np.where((self.predator_reproduction_timers >= self.predator_gestation_time) & 
                                                 self.alive_predator_array & 
                                                 (self.predator_satiety >= self.predator_reproduction_minimum_satiety))[0]
        for i in reproducing_predator_indices:
            self.spawn_predator(1,parent_index=i)
    
    def spawn_predator(self, how_many_to_spawn, parent_index=-1):
            available_slots = np.sum(np.invert(self.alive_predator_array))
            spawn_count = min(how_many_to_spawn, available_slots)

            if spawn_count > 0:
                free_indeces = self.get_free_indices(self.alive_predator_array,spawn_count)

                if parent_index < 0:
                    new_predator_positions = np.random.randint(low=[1,1], 
                                                        high=[self.world_width-1,self.world_height-1],
                                                        size=(spawn_count,2))
                    new_predator_angles = np.random.uniform(low=-np.pi, 
                                                        high=np.pi,
                                                        size=spawn_count)
                    self.predator_positions[free_indeces] = new_predator_positions
                    self.predator_angles[free_indeces] = new_predator_angles
                    self.predator_speeds[free_indeces] = 0
                    self.predator_angular_velocities[free_indeces] = 0
                    self.predator_colours[free_indeces] = np.random.randint(0,255,size=(spawn_count,3))
                    self.predator_reproduction_timers[free_indeces] = 0.0
                    self.predator_ages[free_indeces] = 5
                    self.predator_offsping_count[free_indeces] = 0.0
                    self.predator_satiety[free_indeces] = 1
                    self.predator_fitnesses[free_indeces] = 0
                    self.alive_predator_array[free_indeces] = True
                    self.predator_nn_inputs[free_indeces,self.predator_num_of_raysections*2+1] = np.random.uniform(low= -1.0,
                                                                            high = 1.0,
                                                                            size=spawn_count)
                    self.predator_generations[free_indeces] = 0
                    for idx in free_indeces:
                        self.predator_brains[idx] = AnimalBrain(
                            n_external_infos = self.predator_num_of_raysections*2+1,
                            n_self_infos=0,
                            hidden_dim_1=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0],
                            hidden_dim_2=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0]
                            )
                    self.current_predator += spawn_count
                
                else: 
                    self.predator_offsping_count[parent_index] += 1
                    #self.predator_fitnesses[parent_index] += 0.1 *  max((0.4+1/max(self.current_herbivore, 0.01)),1)
                    #since this option will always just spawn one animals, free indeces will always be a single number,
                    #and so will spawn count
                    parent_position = self.predator_positions[parent_index]
                    child_position = parent_position + np.random.randint(-10,10,2)
                    self.predator_positions[free_indeces] = child_position
                    self.predator_angles[free_indeces] = np.random.randint(-np.pi,np.pi,1)
                    self.predator_speeds[free_indeces] = 0
                    self.predator_angular_velocities[free_indeces] = 0
                    self.predator_colours[free_indeces] = self.predator_colours[parent_index] + np.random.normal(loc=0, scale=self.global_mutation_strength*self.colour_change_strength, size=(spawn_count,3))
                    self.predator_reproduction_timers[free_indeces] = 0
                    self.predator_ages[free_indeces] = 0.0
                    self.predator_offsping_count[free_indeces] = 0.0
                    self.predator_satiety[free_indeces] = 1
                    self.predator_fitnesses[free_indeces] = 0
                    self.alive_predator_array[free_indeces] = True
                    self.predator_generations[free_indeces] = self.predator_generations[parent_index]+1
                    self.predator_nn_inputs[free_indeces,self.predator_num_of_raysections*2+1] = np.clip((self.predator_nn_inputs[parent_index,self.predator_num_of_raysections*2+1] + np.random.uniform(low=-self.global_mutation_strength,
                                                                                                                                                                                                        high= self.global_mutation_strength,
                                                                                                                                                                                                        size=1)), 
                                                                                                                                                                                                        -1,1)
                    
                    child_brain = copy.deepcopy(self.predator_brains[parent_index])
                    sizes = child_brain.get_dim_sizes()
                    for dimension, index in zip(["fc1","fc2"],[0,1]):
                        if np.random.uniform(low=0.0,high=1.0,size=1)[0] <= self.global_mutation_rate:
                            child_brain = resize_layer_in_animal_brain(child_brain,
                                                                layer=dimension,
                                                                new_size=np.clip((sizes[index]+np.random.choice([-1,1])),a_min=1,a_max=20),
                                                                init_std=0.1)
                    child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                    self.predator_brains[free_indeces] = child_brain
                    self.current_predator += spawn_count
                    self.predator_reproduction_timers[parent_index] = 0.0
                    self.predator_satiety[parent_index] -= self.predator_reproduction_satiety_loss
    
    def resurrect_predators(self, spawn_count):
        # doesnt have colours yet

            free_indeces = self.get_free_indices(self.alive_predator_array,spawn_count)

            new_predator_positions = np.random.randint(low=[1,1], 
                                                high=[self.world_width-1,self.world_height-1],
                                                size=(spawn_count,2))
            new_predator_angles = np.random.uniform(low=-np.pi, 
                                                high=np.pi,
                                                size=spawn_count)
            self.predator_positions[free_indeces] = new_predator_positions
            self.predator_angles[free_indeces] = new_predator_angles
            self.predator_speeds[free_indeces] = 0
            self.predator_angular_velocities[free_indeces] = 0
            self.predator_reproduction_timers[free_indeces] = 0.0
            self.predator_ages[free_indeces] = 5
            self.predator_offsping_count[free_indeces] = 0.0
            self.predator_satiety[free_indeces] = 1                
            self.predator_fitnesses[free_indeces] = 0
            self.alive_predator_array[free_indeces] = True
            self.predator_nn_inputs[free_indeces,self.predator_num_of_raysections*2+1] = np.clip(self.predator_best_bias + np.random.uniform(low=-self.global_mutation_strength,high=self.global_mutation_strength,size=1),-1,1)
            self.predator_generations[free_indeces] = self.perdator_best_generation+1
            for idx in free_indeces:
                new_brain = copy.deepcopy(self.predator_best_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                self.predator_brains[idx] = new_brain
            self.current_predator = spawn_count