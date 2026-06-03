import numpy as np
import pandas as pd
import copy
import torch
import threading

from scipy.spatial import distance_matrix
#from scipy.spatial import cKDTree
from game_functions import *
from class_animal_brain_nn import *

#NOTES:
# 1. fitness is exploitable but i dont know how to fix it. It probably will always be exploitable
# 19. brain designer tab? 
# 23. may be too complicated - skip connections - for a real NEAT lagorithm
# 24. new chart - brain complexity
# 25. replace speed as an input.
# 26. before brain designer tab - save/load brain pickles and inject them into existing animals. maybe even spawn new animals button 



#since last commit:
# fixed resurrect bug


class World:

    def __init__(self):
        #globals
        self.world_speed_multiplier = 1.15 #setting exists
        self.world_width = 800
        self.world_height = 600
        self.selected_herbivore_index = None
        self.selected_predator_index = None
        self.world_time = 0

        self.max_speed = 25 #setting exists
        self.max_angular_velocity = 3.5 #setting exists

        self.global_mutation_rate = 0.035 #setting exists
        self.global_mutation_strength = 0.04 #setting exists
        self.colour_change_strength = 15 
        self.min_hidden_dim_size = 1 #setting exists 
        self.max_hidden_dim_size = 10 #setting exists 
        self.weight_std_for_new_neurons = 0.35 #setting exists
        self.fitness_distance_multiplier = 0.01 # how much the distance to food when it was found should contribute to fitness. multiplied with the distance and then added to fitness, so its basically like they get extra fitness for finding food from farther away, to encourage exploration
        self.enable_training_predators = True #ADD enable or disable always spawning a random "training" predator when there are no predators left, to exert selection pressure on herbivores even in the absence of predators. 
        
        self.starting_herbivore = 100 #setting exists
        self.starting_predator = 0 #setting exists
        self.starting_plant = 100 #setting exists

        #plants
        self.max_plant = 300 #setting exists
        self.plant_size = 5  #setting exists
        self.plant_nutrition_value = 0.85 #setting exists
        self.plant_regrowth_power = 1.0 #setting exists
        self.plant_random_spawn_interval = 1 / self.plant_regrowth_power #2.2 originally
        self.plant_reproduction_interval = 15 / self.plant_regrowth_power #5.5 originally

        #predators # we dont add predator settings rn because their mechanics are very outdated
        self.max_predator = 1000 #setting exists
        self.predator_size = 5 #setting exists
        self.predator_satiety_loss_factor = 0.005 #setting exists
        self.predator_max_satiety = 2 #setting exists
        self.predator_avg_gestation_time = 32 #setting exists
        self.predator_gestation_time_std_dev = 5 #setting exists
        self.predator_reproduction_minimum_satiety = 1.0 #setting exists
        self.predator_reproduction_satiety_loss = 0.4 #setting exists
        self.predator_max_percent_satiety_to_eat = 0.75 #setting exists # they wont eat if their satiety is above this percentage
        self.predator_FOV = np.pi/3 #setting exists
        self.predator_vision_range = 220 #setting exists
        self.predator_avg_age = 120 #setting exists
        self.predator_age_std_dev = 7 #setting exists
        self.predator_min_age_to_reproduce = 24 #setting exists

        #herbivores
        self.max_herbivore = 2000 #setting exists
        self.herbivore_size = 5 #setting exists 
        self.herbivore_satiety_loss_factor = 0.006 #setting exists how fast they go hungry, this is multiplied with speed
        self.herbivore_max_satiety = 2 #setting exists
        self.herbivore_avg_gestation_time = 28 #setting exists
        self.herbivore_gestation_time_std_dev = 5 #setting exists
        self.herbivore_reproduction_minimum_satiety = 1.1 #setting exists # minimum satiety required to start gestation
        self.herbivore_reproduction_satiety_loss = 0.5 #setting exists # how much satiety they lose when they reproduce
        self.herbivore_max_percent_satiety_to_eat = 0.75 #setting exists # they wont eat if their satiety is above this percentage
        self.herbivore_FOV = np.pi*1.1 #setting exists
        self.herbivore_vision_range = 150 #setting exists
        self.herbivore_avg_age = 100 #setting exists
        self.herbivore_age_std_dev = 7 #setting exists
        self.herbivore_min_age_to_reproduce = 23 #setting exists # they wont reproduce if they are younger than this
        self.herbivore_nutrition_value = 1.0 #setting exists #how much satiety predators get from eating a herbivore.

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
        self.selected_herbivore_nn_activations = None
        self.herbivore_reproduction_timers = np.zeros((self.max_herbivore,))
        self.herbivore_gestation_time_reqs = np.zeros((self.max_herbivore,))
        self.herbivore_ages = np.zeros((self.max_herbivore,))
        self.herbivore_offsping_count = np.zeros((self.max_herbivore,))
        self.herbivore_brains = np.array([[None] * self.max_herbivore])[0]
        self.herbivore_generations = np.zeros((self.max_herbivore,))
        self.herbivore_dist_since_last_meal = np.zeros((self.max_herbivore,)) # for fitness calculations #update spawn
        self.herbivore_fitnesses = np.zeros((self.max_herbivore,)) # current fitnesses of herbivores

        self.predator_positions = np.zeros((self.max_predator,2))
        self.predator_angles = np.zeros((self.max_predator,))
        self.predator_speeds = np.zeros((self.max_predator,))
        self.predator_angular_velocities = np.zeros((self.max_predator,))
        self.predator_colours = np.zeros((self.max_predator,3))
        self.predator_satiety = np.zeros((self.max_predator,))
        self.predator_life_expectancy = np.zeros((self.max_predator,))
        self.alive_predator_array = np.zeros(self.max_predator,dtype=bool)
        self.predator_detectable_object_types = 2 # herbivore (food), conspecific
        self.predator_types_of_info_about_each_object = 2 # distance and angle to it
        self.predator_num_external_infos = self.predator_detectable_object_types * self.predator_types_of_info_about_each_object
        self.predator_self_infos = 2 # speed and satiety
        self.predator_nn_inputs = np.zeros((self.max_predator, (self.predator_num_external_infos + self.predator_self_infos)),dtype=np.float32)
        self.selected_predator_nn_hdim1 = None
        self.selected_predator_nn_hdim2 = None
        self.selected_predator_nn_output = None 
        self.selected_predator_nn_activations = None
        self.predator_reproduction_timers = np.zeros((self.max_predator,))
        self.predator_gestation_time_reqs = np.zeros((self.max_predator,))
        self.predator_ages = np.zeros((self.max_predator,))
        self.predator_offsping_count = np.zeros((self.max_predator,))
        self.predator_brains = np.array([[None] * self.max_predator])[0]
        self.predator_generations = np.zeros((self.max_predator,))
        self.predator_dist_since_last_meal = np.zeros((self.max_predator,)) # for fitness calculations #update spawn
        self.predator_fitnesses = np.zeros((self.max_predator,))
        
        #RESURRECTION:
        self.herbivore_resurrection_count = 40 #setting exists  #resurrect from fitness archive
        self.herbivore_resurrection_random_count = 40 #setting exists #resurrect from most recent animal archive
        self.herbivore_resurrection_recent_count = 40 #setting exists resurrectt random
        self.herbivore_new_archive_size = 100
        self.herbivore_top_n = 20
        self.herbivore_resurrect_after_plants_reach = 200
        self.herbivore_top_generations = np.zeros((self.herbivore_top_n,)) # stored top generations corresponding to the top fitnesses, start at 0
        self.herbivore_top_fitnesses = np.zeros((self.herbivore_top_n,)) # stored top fitnesses, start at 0
        self.herbivore_top_brains = np.array([[None] * self.herbivore_top_n])[0] # stored top brains corresponding to the top fitnesses, start at None
        self.herbivore_top_colours = np.zeros((self.herbivore_top_n,3)) # stored top colours corresponding to the top fitnesses, start at 0
        self.herbivore_new_archive_generations = np.zeros((self.herbivore_new_archive_size,))
        self.herbivore_new_archive_fitnesses = np.zeros((self.herbivore_new_archive_size,))
        self.herbivore_new_archive_brains = np.array([[None] * self.herbivore_new_archive_size])[0]
        self.herbivore_new_archive_colours = np.zeros((self.herbivore_new_archive_size,3))
        self.herbivore_random_spawn_mask = np.zeros((self.max_herbivore,))
        self.already_resurrected_herbivores = 0
        self.herbivore_resurrection_delay_counter = 0
        self.start_resurrecting_herbivores = False 

        self.predator_resurrection_count = 10 #setting exists
        self.predator_resurrection_recent_count = 10 #setting exists
        self.predator_resurrection_random_count = 10 #setting exists       
        self.predator_new_archive_size = 100 #this will store most recent animals
        self.predator_top_n = 20
        self.predators_resurrect_after_herbivores_reach = 140 #setting exists
        self.predator_top_fitnesses = np.zeros((self.predator_top_n,)) # stored top fitnesses, start at 0
        self.predator_top_brains = np.array([[None] * self.predator_top_n])[0] # stored top brains corresponding to the top fitnesses, start at None
        self.predator_top_colours = np.zeros((self.predator_top_n,3)) # stored top colours corresponding to the top fitnesses, start at 0
        self.predator_top_generations = np.zeros((self.predator_top_n,)) # stored top generations corresponding to the top fitnesses, start at 0
        self.predator_new_archive_generations = np.zeros((self.predator_new_archive_size,))
        self.predator_new_archive_fitnesses = np.zeros((self.predator_new_archive_size,))
        self.predator_new_archive_brains = np.array([[None] * self.predator_new_archive_size])[0]
        self.predator_new_archive_colours = np.zeros((self.predator_new_archive_size,3))
        self.predator_random_spawn_mask = np.zeros((self.max_predator,))
        self.already_resurrected_predators = 0
        self.predator_resurrection_delay_counter = 0  
        self.start_resurrecting_predators = False 

    def recalculate_dependent_attributes(self):
        #this is called when the world is restarted after new settings are applied
        self.plant_random_spawn_interval = 1 / self.plant_regrowth_power 
        self.plant_reproduction_interval = 15 / self.plant_regrowth_power 
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
        self.herbivore_random_spawn_mask = np.zeros((self.max_herbivore,))

        self.predator_positions = np.zeros((self.max_predator,2))
        self.predator_angles = np.zeros((self.max_predator,))
        self.predator_speeds = np.zeros((self.max_predator,))
        self.predator_angular_velocities = np.zeros((self.max_predator,))
        self.predator_colours = np.zeros((self.max_predator,3))
        self.predator_satiety = np.zeros((self.max_predator,))
        self.predator_life_expectancy = np.zeros((self.max_predator,))
        self.alive_predator_array = np.zeros(self.max_predator,dtype=bool)
        self.predator_nn_inputs = np.zeros((self.max_predator, (self.predator_num_external_infos + self.predator_self_infos)),dtype=np.float32)
        self.predator_reproduction_timers = np.zeros((self.max_predator,))
        self.predator_gestation_time_reqs = np.zeros((self.max_predator,))
        self.predator_ages = np.zeros((self.max_predator,))
        self.predator_offsping_count = np.zeros((self.max_predator,))
        self.predator_brains = np.array([[None] * self.max_predator])[0]
        self.predator_generations = np.zeros((self.max_predator,))
        self.predator_dist_since_last_meal = np.zeros((self.max_predator,)) # for fitness calculations #update spawn
        self.predator_fitnesses = np.zeros((self.max_predator,))
        self.predator_random_spawn_mask = np.zeros((self.max_predator,))

    def update(self,dt): # super master function which updates the state of the world
        self.update_plants(dt)
        self.update_herbivores(dt)
        self.update_predators(dt)
        self.world_time += dt * self.world_speed_multiplier
    
    def get_free_indices(self, mask, slots_needed): # utility function
        free_indices = np.where(mask == False)[0][0:slots_needed]
        return free_indices

    def get_simulation_state(self):
        #main api endpoint
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
                "generation": int(self.predator_generations[i]),
                "x": float(self.predator_positions[i, 0]),
                "y": float(self.predator_positions[i, 1]),
                "angle": float(self.predator_angles[i]),
                "speed": float(self.predator_speeds[i]),
                "red": int(self.predator_colours[i,0]),
                "green": int(self.predator_colours[i,1]),
                "blue": int(self.predator_colours[i,2]),
            }
            for i in predator_indices
            ]

        selected = None
        if self.selected_herbivore_index is not None and self.alive_herbivore_array[self.selected_herbivore_index]:
            brain = self.herbivore_brains[self.selected_herbivore_index]
            weights = to_json_compatible(brain.get_network_weights())
            
            # Safely look up dynamic activations arrays saved from the process loop
            hidden_activations = []
            if hasattr(self, 'selected_herbivore_nn_activations') and self.selected_herbivore_nn_activations is not None:
                hidden_activations = [act.tolist() for act in self.selected_herbivore_nn_activations]

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

                "nn_distances_angles" : self.herbivore_nn_inputs[self.selected_herbivore_index,0:self.herbivore_num_external_infos].tolist(),
                "inputs": self.herbivore_nn_inputs[self.selected_herbivore_index].tolist(),
                "hidden_layers_activations": hidden_activations, 
                "output": self.selected_herbivore_nn_output.tolist() if self.selected_herbivore_nn_output is not None else None,
                "weights": weights,
            }

        if self.selected_predator_index is not None and self.alive_predator_array[self.selected_predator_index]:
            brain = self.predator_brains[self.selected_predator_index]
            weights = to_json_compatible(brain.get_network_weights())
            
            # Safely look up dynamic activations arrays saved from the process loop
            hidden_activations = []
            if hasattr(self, 'selected_predator_nn_activations') and self.selected_predator_nn_activations is not None:
                hidden_activations = [act.tolist() for act in self.selected_predator_nn_activations]

            selected = {
                "species": "predator",
                "id": int(self.selected_predator_index),
                "x": float(self.predator_positions[self.selected_predator_index, 0]),
                "y": float(self.predator_positions[self.selected_predator_index, 1]),
                "speed": float(self.predator_speeds[self.selected_predator_index]),
                "face_direction": float(self.predator_angles[self.selected_predator_index]),
                "satiety": float(self.predator_satiety[self.selected_predator_index]),
                "age": float(self.predator_ages[self.selected_predator_index]),
                "generation": int(self.predator_generations[self.selected_predator_index]),
                "fitness": float(self.predator_fitnesses[self.selected_predator_index]),
                "reproduction_progress": float(self.predator_reproduction_timers[self.selected_predator_index] / self.predator_gestation_time_reqs[self.selected_predator_index]),
                "fov": float(self.predator_FOV),
                "vision_range": float(self.predator_vision_range),
                "offspring_count": int(self.predator_offsping_count[self.selected_predator_index]),

                "nn_distances_angles" : self.predator_nn_inputs[self.selected_predator_index,0:self.predator_num_external_infos].tolist(),
                "inputs": self.predator_nn_inputs[self.selected_predator_index].tolist(),
                "hidden_layers_activations": hidden_activations, 
                "output": self.selected_predator_nn_output.tolist() if self.selected_predator_nn_output is not None else None,
                "weights": weights,
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
        #we could just get this from world state, but this will make it easier to add more charts later:
        return {
            #for population chart:
            "world_time": float(self.world_time),
            "current_plant": int(np.sum(self.alive_plant_array)),
            "current_herbivore": int(np.sum(self.alive_herbivore_array)),
            "current_predator": int(np.sum(self.alive_predator_array)),

            #for population pyramid:
            "alive_herbivore_ages": self.herbivore_ages[self.alive_herbivore_array].tolist(),
            "alive_predator_ages": self.predator_ages[self.alive_predator_array].tolist(),
            "herbivore_maturity_age": int(self.herbivore_min_age_to_reproduce),
            "predator_maturity_age": int(self.predator_min_age_to_reproduce),
        }
    
    def debug_kill(self):
        if self.selected_herbivore_index != None:
            self.alive_herbivore_array[self.selected_herbivore_index] = False
        elif self.selected_predator_index != None:
            self.alive_predator_array[self.selected_predator_index] = False
        else:
            # no animal selected
            return False
        return True


############# -------------------------------------- PLANTS ----------------------------------- ###############   
############# -------------------------------------- PLANTS ----------------------------------- ###############
############# -------------------------------------- PLANTS ----------------------------------- ###############
    def update_plants(self,dt): # master plant function
        self.spawn_random_plants(dt)
        self.plants_reproduce(dt)

    def spawn_random_plants(self,dt):
        self.plant_spawn_time_accumulator += dt * self.world_speed_multiplier
        if (self.plant_spawn_time_accumulator > self.plant_random_spawn_interval):
            self.spawn_food(np.random.randint(1, 3)) # spawn a random number of plants between 1 and 2
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
        self.herbivores_check_resurrect(dt)
        self.herbivores_die_of_natural_causes(dt) 
        self.herbivores_perceive() # calculate neural network inputs
        self.herbivores_process_NN(dt) # push neural network inputs to the NNs, get movement parameters out
        self.herbivores_move(dt) # update new positions using the parameters from previous function
        self.check_herbivore_plant_collisions() # check if they collided with food
        self.herbivores_reproduce(dt) # check if they reproduce 
    
    def herbivores_check_resurrect(self, dt):
        if (np.sum(self.alive_herbivore_array) == 0) and (np.sum(self.alive_plant_array) > self.herbivore_resurrect_after_plants_reach):
            self.start_resurrecting_herbivores = True
        if not self.start_resurrecting_herbivores:
            return

        # every 5 seconds, spawn a random amount between 1 and 5 until that number reaches the res count
        # then spawn random ones every 5 seconds until it reaches the res count+random res count
        self.herbivore_resurrection_delay_counter += dt * self.world_speed_multiplier
        if self.herbivore_resurrection_delay_counter < 3.0:
            return
        
        self.herbivore_resurrection_delay_counter = 0.0
        if self.already_resurrected_herbivores <= self.herbivore_resurrection_count:
            to_res = np.random.randint(low=1, high=6)
            self.resurrect_herbivores(to_res, 0)
            self.already_resurrected_herbivores += to_res
        elif self.already_resurrected_herbivores <= self.herbivore_resurrection_count + self.herbivore_resurrection_recent_count: 
            to_res = np.random.randint(low=1, high=6)
            self.resurrect_herbivores(0, to_res)
            self.already_resurrected_herbivores += to_res
        elif self.already_resurrected_herbivores <= self.herbivore_resurrection_count + self.herbivore_resurrection_recent_count + self.herbivore_resurrection_random_count: 
            to_res = np.random.randint(low=1, high=6)
            self.spawn_herbivore(to_res, parent_index=-1, random_res=True)
            self.already_resurrected_herbivores += to_res
        else:
            self.start_resurrecting_herbivores = False
            self.already_resurrected_herbivores = 0

    def herbivores_die_of_natural_causes(self, dt):
        #check is anyone starved
        self.herbivore_satiety[self.alive_herbivore_array] -= (self.herbivore_satiety_loss_factor * 3 * abs(self.herbivore_speeds[self.alive_herbivore_array]/self.max_speed) + 5 * self.herbivore_satiety_loss_factor) * dt * self.world_speed_multiplier
        died_starvation = self.alive_herbivore_array & (self.herbivore_satiety <= 0)
        self.alive_herbivore_array &= (self.herbivore_satiety > 0)

        #check if anyone died of old age
        self.herbivore_ages[self.alive_herbivore_array] += dt * self.world_speed_multiplier
        died_age = self.alive_herbivore_array & (self.herbivore_ages >= self.herbivore_life_expectancy)
        self.alive_herbivore_array &= (self.herbivore_ages < self.herbivore_life_expectancy)

        died_indices = np.where(died_starvation | died_age)[0]
        self.add_herbivores_to_archives(died_indices)

        if self.selected_herbivore_index != None:
            if self.alive_herbivore_array[self.selected_herbivore_index] == False:
                #check if selected herbivore died
                self.selected_herbivore_index = None
            

    def herbivores_perceive(self):
        alive_indices = np.where(self.alive_herbivore_array)[0]
        if alive_indices.size == 0:
            return
        
        #the new nn input for herbivores should be: [dist_plant, angle_plant, dist_conspecific, angle_conspecific, dist_predator, angle_predator, own_satiety, own_speed]
        output_from_perception_function = herbivores_perception_function(
            self_positions = self.herbivore_positions[alive_indices],
            self_angles = self.herbivore_angles[alive_indices],
            food_positions = self.plant_positions[self.alive_plant_array], 
            predator_positions = self.predator_positions[self.alive_predator_array],
            vision_range = self.herbivore_vision_range,
            vision_fov = self.herbivore_FOV
        )

        self.herbivore_nn_inputs[alive_indices,0:self.herbivore_num_external_infos] = output_from_perception_function
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
                    output, activations = brain(input_tensor[i].unsqueeze(0), return_activations=True)
                
                # Store dynamic activations list safely for UI/Logging
                self.selected_herbivore_nn_activations = [act.cpu().numpy()[0] for act in activations]
                self.selected_herbivore_nn_output = output.cpu().numpy()[0]
                output = output.numpy()[0]

            else:
                with torch.no_grad(): 
                    output = brain(input_tensor[i].unsqueeze(0), return_activations=False).numpy()[0]
            outputs.append(output)

        outputs = np.array(outputs, dtype=np.float32)  # shape (N_alive, 2)

        # === assign outputs ===
        #First output is the speed
        #if this was a cleaner codebase, this would go in herbivores_move
        #output 0 means no speed. 1 means max speed forward. -1 means back up.
        self.herbivore_speeds[alive_indices] = outputs[:, 0] * self.max_speed 

        #second output is directly assigned to angular velocity array after scaling:
        self.herbivore_angular_velocities[alive_indices] = outputs[:, 1] * self.max_angular_velocity * self.world_speed_multiplier * dt

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

        #calculate travelled distance for fitness calcs:
        self.herbivore_dist_since_last_meal[self.alive_herbivore_array] += np.linalg.norm(dpos, axis=1)


    def check_herbivore_plant_collisions(self):
        if not np.any(self.alive_herbivore_array) or not np.any(self.alive_plant_array):
            return

        d = distance_matrix(self.herbivore_positions[self.alive_herbivore_array], self.plant_positions[self.alive_plant_array])
        rows, cols = np.where(d <= self.plant_size + 2)
        if rows.size == 0:
            return

        alive_herbivores = np.where(self.alive_herbivore_array)[0]
        alive_plants = np.where(self.alive_plant_array)[0]
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

        # update fitness of herbivores that ate
        # fitness += 1 + distance_to_nearest_food_when_food_was_found * 0.02 (so that animals that find food from farther away get more fitness, to encourage exploration)
        self.herbivore_fitnesses[herbivores_that_ate_indices] += (1 + self.herbivore_dist_since_last_meal[herbivores_that_ate_indices] * self.fitness_distance_multiplier)
        self.herbivore_dist_since_last_meal[herbivores_that_ate_indices] = 0.0

        # Mark plants as dead
        self.alive_plant_array[plants_that_were_eaten_indices] = False
    
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


    def spawn_herbivore(self, how_many_to_spawn, parent_index=-1, random_res=False):
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
                if random_res == True:
                    self.herbivore_random_spawn_mask[free_indeces] = True
                else:
                    self.herbivore_random_spawn_mask[free_indeces] = False
                self.alive_herbivore_array[free_indeces] = True 
                self.herbivore_generations[free_indeces] = 0
                self.herbivore_life_expectancy[free_indeces] = np.random.normal(loc=self.herbivore_avg_age, scale=self.herbivore_age_std_dev, size=spawn_count)
                self.herbivore_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)
                for idx in free_indeces:
                    # Generate a random initial layout: e.g., could be [], [8], [12, 6], etc.
                    num_initial_layers = np.random.randint(0, 3) 
                    random_dims = [
                        np.random.randint(self.min_hidden_dim_size, self.max_hidden_dim_size + 1)
                        for _ in range(num_initial_layers)
                    ]
                    
                    self.herbivore_brains[idx] = AnimalBrain(
                        n_external_infos=self.herbivore_num_external_infos,
                        n_self_infos=self.herbivore_self_infos,
                        hidden_dims=random_dims,
                        initial_weight_std=self.weight_std_for_new_neurons
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
                self.herbivore_random_spawn_mask[free_indeces] = False
                self.herbivore_fitnesses[free_indeces] = 0
                self.alive_herbivore_array[free_indeces] = True
                self.herbivore_generations[free_indeces] = self.herbivore_generations[parent_index]+1
                #normal distribution for life expectancy and gestation time:
                self.herbivore_life_expectancy[free_indeces] = np.random.normal(loc=self.herbivore_avg_age, scale=self.herbivore_age_std_dev, size=spawn_count)
                self.herbivore_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)
                
                #brains:
                parent_brain = self.herbivore_brains[parent_index]
                child_brain = copy.deepcopy(parent_brain)

                # mutate weights:
                child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)

                #calculate new colour based on similarity between parent and child brains:
                brain_distance = calculate_brain_similarity(parent_brain, child_brain)

                # 2. Mutate architecture (Dynamic layers & neurons!)
                child_brain, structural_dist = mutate_brain_architecture(
                    child_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )

                brain_distance += structural_dist

                self.herbivore_brains[free_indeces] = child_brain
                new_colour = self.herbivore_colours[parent_index] + np.random.normal(0, brain_distance*150, size=(spawn_count,3))
                self.herbivore_colours[free_indeces] = np.clip(new_colour, 0, 255)

                self.herbivore_reproduction_timers[parent_index] = 0.0
                self.herbivore_gestation_time_reqs[parent_index] = np.random.normal(loc=self.herbivore_avg_gestation_time, scale=self.herbivore_gestation_time_std_dev, size=spawn_count)[0]
                self.herbivore_satiety[parent_index] -= self.herbivore_reproduction_satiety_loss

    def resurrect_herbivores(self, spawn_count, spawn_count_recent):
        #1. from fitness archive:
        if spawn_count > 0:
            total_top_fitness = np.sum(self.herbivore_top_fitnesses)
            if total_top_fitness == 0:
                #they all died before a single one of them eating food?
                self.spawn_herbivore(spawn_count, parent_index=-1)
                return
            relative_fitnesses = self.herbivore_top_fitnesses / total_top_fitness
            #this will be used as a probability of spawning a new herbivore:
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
            self.herbivore_ages[free_indeces] = 0
            self.herbivore_offsping_count[free_indeces] = 0.0
            self.herbivore_satiety[free_indeces] = 1        
            self.herbivore_random_spawn_mask[free_indeces] = False        
            self.herbivore_fitnesses[free_indeces] = 0
            self.alive_herbivore_array[free_indeces] = True

            #stuff that will depend on the fitness of the top brains: generation number, brain, colour:
            indices_of_parents = np.random.choice(len(self.herbivore_top_fitnesses), size=spawn_count, p=relative_fitnesses)
            #this means that if a brain has higher fitness, it is more likely to be chosen as a parent for the new herbivores that are being spawned to replace the ones that died.
            self.herbivore_generations[free_indeces] = self.herbivore_top_generations[indices_of_parents]+1
            
            for i, idx in enumerate(free_indeces):
                parent_idx = indices_of_parents[i]
                parent_brain = self.herbivore_top_brains[parent_idx]
                new_brain = copy.deepcopy(parent_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                brain_distance = calculate_brain_similarity(parent_brain, new_brain)
                new_brain, structural_dist = mutate_brain_architecture(
                    new_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )

                brain_distance += structural_dist
                self.herbivore_brains[idx] = new_brain
                #calculate new colour based on similarity between parent and child brains:
                new_colour = self.herbivore_top_colours[parent_idx] + np.random.normal(0, brain_distance*150, size=3)
                self.herbivore_colours[idx] = np.clip(new_colour, 0, 255)
            
        # 2. from recent archive:
        if spawn_count_recent > 0:
            total_new_archive_fitness = np.sum(self.herbivore_new_archive_fitnesses)
            if total_new_archive_fitness <= 0:
                #they all died before a single one of them eating food?
                #or negative fitness? that shouldnt happen
                self.spawn_herbivore(spawn_count_recent, parent_index=-1)
                return

            relative_fitnesses = self.herbivore_new_archive_fitnesses / total_new_archive_fitness
            #this will be used as a probability of spawning a new herbivore:
            free_indeces = self.get_free_indices(self.alive_herbivore_array,spawn_count_recent)

            new_positions = np.random.randint(low=[1,1], 
                                                    high=[self.world_width-1,self.world_height-1],
                                                    size=(spawn_count_recent,2))
            new_angles = np.random.uniform(low=-np.pi, 
                                                    high=np.pi,
                                                    size=spawn_count_recent)
            self.herbivore_positions[free_indeces] = new_positions
            self.herbivore_angles[free_indeces] = new_angles
            self.herbivore_speeds[free_indeces] = 0
            self.herbivore_angular_velocities[free_indeces] = 0
            self.herbivore_reproduction_timers[free_indeces] = 0.0
            self.herbivore_ages[free_indeces] = 0
            self.herbivore_offsping_count[free_indeces] = 0.0
            self.herbivore_satiety[free_indeces] = 1        
            self.herbivore_random_spawn_mask[free_indeces] = False        
            self.herbivore_fitnesses[free_indeces] = 0
            self.alive_herbivore_array[free_indeces] = True

            #stuff that will depend on the fitness of the new_archive brains: generation number, brain, colour:
            indices_of_parents = np.random.choice(len(self.herbivore_new_archive_fitnesses), size=spawn_count_recent, p=relative_fitnesses)
            #this means that if a brain has higher fitness, it is more likely to be chosen as a parent for the new herbivores that are being spawned to replace the ones that died.
            self.herbivore_generations[free_indeces] = self.herbivore_new_archive_generations[indices_of_parents]+1
            #brains:
            for i, idx in enumerate(free_indeces):
                parent_idx = indices_of_parents[i]
                parent_brain = self.herbivore_new_archive_brains[parent_idx]
                new_brain = copy.deepcopy(parent_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                brain_distance = calculate_brain_similarity(parent_brain, new_brain)
                new_brain, structural_dist = mutate_brain_architecture(
                    new_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )
                brain_distance += structural_dist
                self.herbivore_brains[idx] = new_brain
                #calculate new colour based on similarity between parent and child brains:
                new_colour = self.herbivore_new_archive_colours[parent_idx] + np.random.normal(0, brain_distance*150, size=3)
                self.herbivore_colours[idx] = np.clip(new_colour, 0, 255)

    def add_herbivores_to_archives(self, dying_herbivore_indeces):
        for idx in dying_herbivore_indeces:
            #fitness based archive:
            brain = self.herbivore_brains[idx]
            min_top_idx = np.argmin(self.herbivore_top_fitnesses)
            if self.herbivore_fitnesses[idx] > self.herbivore_top_fitnesses[min_top_idx]:
                self.herbivore_top_fitnesses[min_top_idx] = self.herbivore_fitnesses[idx]
                self.herbivore_top_brains[min_top_idx] = copy.deepcopy(brain)
                self.herbivore_top_generations[min_top_idx] = self.herbivore_generations[idx]
                self.herbivore_top_colours[min_top_idx] = self.herbivore_colours[idx]

            #recency based archive:
            if self.herbivore_random_spawn_mask[idx]: 
                #if this was a random spawn from resurrect, dont add to evolutionary archive
                return

            empty_indeces = np.where(self.herbivore_new_archive_brains == None)[0]
            if len(empty_indeces) > 0:
                self.herbivore_new_archive_brains[empty_indeces[0]] = copy.deepcopy(brain)
                self.herbivore_new_archive_fitnesses[empty_indeces[0]] = self.herbivore_fitnesses[idx]
                self.herbivore_new_archive_generations[empty_indeces[0]] = self.herbivore_generations[idx]
                self.herbivore_new_archive_colours[empty_indeces[0]] = self.herbivore_colours[idx]
            else:
                #if full, shove in randomly.
                rand_idx = np.random.randint(0,self.herbivore_new_archive_size)
                self.herbivore_new_archive_brains[rand_idx] = copy.deepcopy(brain)
                self.herbivore_new_archive_fitnesses[rand_idx] = self.herbivore_fitnesses[idx]
                self.herbivore_new_archive_generations[rand_idx] = self.herbivore_generations[idx]
                self.herbivore_new_archive_colours[rand_idx] = self.herbivore_colours[idx]


################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
################ ---------------------------------------------- PREDATORS ----------------------------------------------- ################
    def update_predators(self,dt): # master function that includes all the rest
        self.predators_check_resurrect(dt) #check if they are extinct and res in some time
        self.predators_die_of_natural_causes(dt) #check if they died from starving or old age
        self.predators_perceive() # calculate what the animal sees - nn inputs
        self.predators_process_NN(dt) # push inputs from last function into the NNs, get movement parameters out
        self.predators_move(dt) # update new positions using the parameters from previous function
        self.check_predator_herbivore_collisions() # check if they collided with herbivores
        self.predators_reproduce(dt) # check if they reproduce 
    
    def predators_check_resurrect(self, dt):
        if (np.sum(self.alive_predator_array) <= 5) and (np.sum(self.alive_herbivore_array) > self.predators_resurrect_after_herbivores_reach):
            self.start_resurrecting_predators = True
        elif (np.sum(self.alive_predator_array) == 0) and self.enable_training_predators:
            #  add random "training predators" to exert selection pressure:
            self.spawn_predator(np.random.randint(low=1, high=4), parent_index=-1, random_res=True)
        if not self.start_resurrecting_predators:
            return

        # ---- Gradual resurrection:
        # every 5 seconds, spawn a random amount between 1 and 5 until that number reaches the res count
        # then spawn random ones every 5 seconds until it reaches the res count+random res count
        # this is so that there is no crazy apocalyptic event happening suddenly
        self.predator_resurrection_delay_counter += dt * self.world_speed_multiplier
        if self.predator_resurrection_delay_counter < 5.0:
            return
        
        self.predator_resurrection_delay_counter = 0.0
        if self.already_resurrected_predators <= self.predator_resurrection_count:
            to_res = np.random.randint(low=1, high=4)
            self.resurrect_predators(to_res, 0)
            self.already_resurrected_predators += to_res
        elif self.already_resurrected_predators <= self.predator_resurrection_count + self.predator_resurrection_recent_count: 
            to_res = np.random.randint(low=1, high=4)
            self.resurrect_predators(0, to_res)
            self.already_resurrected_predators += to_res
        elif self.already_resurrected_predators <= self.predator_resurrection_count + self.predator_resurrection_recent_count + self.predator_resurrection_random_count: 
            to_res = np.random.randint(low=1, high=4)
            self.spawn_predator(to_res, parent_index=-1, random_res=True)
            self.already_resurrected_predators += to_res
        else:
            self.start_resurrecting_predators = False
            self.already_resurrected_predators = 0
  
    def predators_die_of_natural_causes(self, dt): 
        #check is anyone starved
        self.predator_satiety[self.alive_predator_array] -= (self.predator_satiety_loss_factor * 6 * abs(self.predator_speeds[self.alive_predator_array]/self.max_speed) + 2 * self.predator_satiety_loss_factor) * dt * self.world_speed_multiplier
        died_starvation = self.alive_predator_array & (self.predator_satiety <= 0)
        self.alive_predator_array &= (self.predator_satiety > 0)

        #check if anyone died of old age
        self.predator_ages[self.alive_predator_array] += dt * self.world_speed_multiplier
        died_age = self.alive_predator_array & (self.predator_ages >= self.predator_life_expectancy)
        self.alive_predator_array &= (self.predator_ages < self.predator_life_expectancy)

        died_indeces = np.where(died_starvation | died_age)[0]
        # Update top brains archive
        self.add_predators_to_archives(died_indeces)

        if self.selected_predator_index != None:
            if self.alive_predator_array[self.selected_predator_index] == False:
                self.selected_predator_index = None
    
    def predators_perceive(self): 
        alive_indices = np.where(self.alive_predator_array)[0]
        if alive_indices.size == 0:
            return
        
        #the  nn input for predtors should be: [dist_herb, angle_herb, dist_conspecific, angle_conspecific own_satiety, own_speed]
        output_from_perception_function = predators_perception_function(
            self_positions = self.predator_positions[alive_indices],
            self_angles = self.predator_angles[alive_indices],
            food_positions = self.herbivore_positions[self.alive_herbivore_array], 
            vision_range = self.predator_vision_range,
            vision_fov = self.predator_FOV
        )

        self.predator_nn_inputs[alive_indices,0:self.predator_num_external_infos] = output_from_perception_function
        self.predator_nn_inputs[alive_indices,self.predator_num_external_infos:self.predator_num_external_infos+2] = np.stack((1-(self.predator_satiety[alive_indices] / self.predator_max_satiety), self.predator_speeds[alive_indices] / self.max_speed), axis=1)

    def predators_process_NN(self, dt):
        alive_indices = np.where(self.alive_predator_array)[0]
        if alive_indices.size == 0: return

        # === Batch processing using precomputed nn_inputs ===
        input_tensor = torch.from_numpy(self.predator_nn_inputs[alive_indices])  # shape (N_alive, 6)

        # === Forward pass through each individual's brain ===
        outputs = []
        for i, idx in enumerate(alive_indices):
            brain = self.predator_brains[idx]
            if idx == self.selected_predator_index:
                with torch.no_grad():
                    output, activations = brain(input_tensor[i].unsqueeze(0), return_activations=True)
                
                # Store dynamic activations list safely for UI/Logging
                self.selected_predator_nn_activations = [act.cpu().numpy()[0] for act in activations]
                self.selected_predator_nn_output = output.cpu().numpy()[0]
                output = output.numpy()[0]
            else:
                with torch.no_grad(): # INDENT THIS
                    output = brain(input_tensor[i].unsqueeze(0), return_activations=False).numpy()[0]
            outputs.append(output)
        outputs = np.array(outputs, dtype=np.float32)  # shape (N_alive, 2)

        # === assign outputs ===
        #First output is the speed
        #if this was a cleaner codebase, this would go in predators_move
        #output 0 means no speed. 1 means max speed forward. -1 means back up.
        self.predator_speeds[alive_indices] = outputs[:, 0] * self.max_speed 

        #second output is directly assigned to angular velocity array after scaling:
        self.predator_angular_velocities[alive_indices] = outputs[:, 1] * self.max_angular_velocity * self.world_speed_multiplier * dt

    
    def predators_move(self,dt): 
        #okay so nn outputs are the speed and angular velocity. angular velocity gets added onto existing facing direction.
        self.predator_angles[self.alive_predator_array] += self.predator_angular_velocities[self.alive_predator_array] * dt * self.world_speed_multiplier
        self.predator_angles %= 2 * np.pi

        #using the new angle, the direction vector is calculated:
        dpos = np.zeros((np.sum(self.alive_predator_array),2))

        np.cos(self.predator_angles[self.alive_predator_array], out=dpos[:,0])
        np.sin(self.predator_angles[self.alive_predator_array], out=dpos[:,1])

        #then the direction vector is multiplied by the speed to get the actual movement vector:
        dpos *= self.predator_speeds[self.alive_predator_array, np.newaxis] * self.world_speed_multiplier  * dt

        self.predator_positions[self.alive_predator_array] += dpos 

        #handle world boundary
        self.predator_positions[:, 0] %= self.world_width
        self.predator_positions[:, 1] %= self.world_height

        #calculate travelled distance for fitness calcs:
        self.predator_dist_since_last_meal[self.alive_predator_array] += abs(self.predator_speeds[self.alive_predator_array])

    
    def check_predator_herbivore_collisions(self):
        if not np.any(self.alive_herbivore_array) or not np.any(self.alive_predator_array):
            return
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

        # update fitness of predators that ate
        self.predator_fitnesses[predators_that_ate_indices] += (1 + self.predator_dist_since_last_meal[predators_that_ate_indices] * self.fitness_distance_multiplier)
        self.herbivore_dist_since_last_meal[predators_that_ate_indices] = 0.0

        # Mark herbivores as dead
        self.alive_herbivore_array[herbivores_that_were_eaten_indices] = False
        #also add dead herbivores to archives:
        self.add_herbivores_to_archives(herbivores_that_were_eaten_indices)


        # also need to remove selected herbivore if it was eaten:
        if self.selected_herbivore_index in herbivores_that_were_eaten_indices:
            self.selected_herbivore_index = None
    
    def predators_reproduce(self,dt): 
        sated_predator_indices = np.where((self.alive_predator_array) & 
                                           (self.predator_ages >= self.predator_min_age_to_reproduce) &
                                           (self.predator_satiety >= self.predator_reproduction_minimum_satiety))[0]
        self.predator_reproduction_timers[sated_predator_indices] += dt*self.world_speed_multiplier
        
        reproducing_predator_indices = np.where((self.predator_reproduction_timers >= self.predator_gestation_time_reqs) & 
                                                 self.alive_predator_array & 
                                                 (self.predator_satiety >= self.predator_reproduction_minimum_satiety))[0]        
        for i in reproducing_predator_indices:
            self.spawn_predator(1,parent_index=i)

    def spawn_predator(self, how_many_to_spawn, parent_index=-1, random_res=False): 
        available_slots = np.sum(np.invert(self.alive_predator_array))
        spawn_count = min(how_many_to_spawn, available_slots)

        if spawn_count > 0: # if there are actually free slots to spawn AND new predators that need spawning, then spawn
            free_indeces = self.get_free_indices(self.alive_predator_array,spawn_count)

            if parent_index < 0: # if parent index is less than 0, it means we are spawning randomly (not from a parent), so we will spawn in random positions and with random brains
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
                self.predator_ages[free_indeces] = 0
                self.predator_fitnesses[free_indeces] = 0
                self.predator_offsping_count[free_indeces] = 0.0
                self.predator_satiety[free_indeces] = 1
                self.alive_predator_array[free_indeces] = True 
                self.predator_generations[free_indeces] = 0
                if random_res == True:
                    self.predator_random_spawn_mask[free_indeces] = True
                else:
                    self.predator_random_spawn_mask[free_indeces] = False
                self.predator_life_expectancy[free_indeces] = np.random.normal(loc=self.predator_avg_age, scale=self.predator_age_std_dev, size=spawn_count)
                self.predator_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.predator_avg_gestation_time, scale=self.predator_gestation_time_std_dev, size=spawn_count)
                
                #brains:
                for idx in free_indeces:
                    # Generate a random initial layout: e.g., could be [], [8], [12, 6], etc.
                    num_initial_layers = np.random.randint(0, 3) 
                    random_dims = [
                        np.random.randint(self.min_hidden_dim_size, self.max_hidden_dim_size + 1)
                        for _ in range(num_initial_layers)
                    ]
                    
                    self.predator_brains[idx] = AnimalBrain(
                        n_external_infos=self.predator_num_external_infos,
                        n_self_infos=self.predator_self_infos,
                        hidden_dims=random_dims,
                        initial_weight_std=self.weight_std_for_new_neurons
                    )
            
            else: #spawning from a parent
                self.predator_offsping_count[parent_index] += 1
                #since this option will always just spawn one animal, free indeces will always be a single number,
                #and so will spawn count
                parent_position = self.predator_positions[parent_index]
                child_position = parent_position + np.random.randint(0,1,2)
                self.predator_positions[free_indeces] = child_position
                self.predator_angles[free_indeces] = np.random.randint(-np.pi,np.pi,1)
                self.predator_speeds[free_indeces] = 0
                self.predator_angular_velocities[free_indeces] = 0
                self.predator_reproduction_timers[free_indeces] = 0
                self.predator_ages[free_indeces] = 0.0
                self.predator_offsping_count[free_indeces] = 0.0
                self.predator_satiety[free_indeces] = 1
                self.predator_random_spawn_mask[free_indeces] = False
                self.predator_fitnesses[free_indeces] = 0
                self.alive_predator_array[free_indeces] = True
                self.predator_generations[free_indeces] = self.predator_generations[parent_index]+1
                self.predator_life_expectancy[free_indeces] = np.random.normal(loc=self.predator_avg_age, scale=self.predator_age_std_dev, size=spawn_count)
                self.predator_gestation_time_reqs[free_indeces] = np.random.normal(loc=self.predator_avg_gestation_time, scale=self.predator_gestation_time_std_dev, size=spawn_count)
                
                #brains: 
                parent_brain = self.predator_brains[parent_index]
                child_brain = copy.deepcopy(parent_brain)

                # mutate weights:
                child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)

                #calculate new colour based on similarity between parent and child brains:
                brain_distance = calculate_brain_similarity(parent_brain, child_brain)

                # 2. Mutate architecture (Dynamic layers & neurons!)
                child_brain, structural_dist = mutate_brain_architecture(
                    child_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )
                brain_distance += structural_dist

                self.predator_brains[free_indeces] = child_brain
                new_colour = self.predator_colours[parent_index] + np.random.normal(0, brain_distance*150, size=(spawn_count,3))
                self.predator_colours[free_indeces] = np.clip(new_colour, 0, 255)

                #parent stats update:
                self.predator_reproduction_timers[parent_index] = 0.0
                self.predator_gestation_time_reqs[parent_index] = np.random.normal(loc=self.predator_avg_gestation_time, scale=self.predator_gestation_time_std_dev, size=spawn_count)[0]
                self.predator_satiety[parent_index] -= self.predator_reproduction_satiety_loss

    def resurrect_predators(self, spawn_count, spawn_count_recent):
        #1. from fitness archive:
        if spawn_count > 0:
            total_top_fitness = np.sum(self.predator_top_fitnesses)
            if total_top_fitness == 0:
                #they all died before a single one of them eating food?
                self.spawn_predator(spawn_count, parent_index=-1)
                return
            relative_fitnesses = self.predator_top_fitnesses / total_top_fitness
            #this will be used as a probability of spawning a new predator:
            free_indeces = self.get_free_indices(self.alive_predator_array,spawn_count)

            new_positions = np.random.randint(low=[1,1], 
                                                    high=[self.world_width-1,self.world_height-1],
                                                    size=(spawn_count,2))
            new_angles = np.random.uniform(low=-np.pi, 
                                                    high=np.pi,
                                                    size=spawn_count)
            self.predator_positions[free_indeces] = new_positions
            self.predator_angles[free_indeces] = new_angles
            self.predator_speeds[free_indeces] = 0
            self.predator_angular_velocities[free_indeces] = 0
            self.predator_reproduction_timers[free_indeces] = 0.0
            self.predator_ages[free_indeces] = 0
            self.predator_offsping_count[free_indeces] = 0.0
            self.predator_satiety[free_indeces] = 1        
            self.predator_random_spawn_mask[free_indeces] = False        
            self.predator_fitnesses[free_indeces] = 0
            self.alive_predator_array[free_indeces] = True

            #stuff that will depend on the fitness of the top brains: generation number, brain, colour:
            indices_of_parents = np.random.choice(len(self.predator_top_fitnesses), size=spawn_count, p=relative_fitnesses)
            #this means that if a brain has higher fitness, it is more likely to be chosen as a parent for the new predators that are being spawned to replace the ones that died.
            self.predator_generations[free_indeces] = self.predator_top_generations[indices_of_parents]+1
            for i, idx in enumerate(free_indeces):
                parent_idx = indices_of_parents[i]
                parent_brain = self.predator_top_brains[parent_idx]
                new_brain = copy.deepcopy(parent_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                brain_distance = calculate_brain_similarity(parent_brain, new_brain)
                new_brain, structural_dist = mutate_brain_architecture(
                    new_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )

                brain_distance += structural_dist
                self.predator_brains[idx] = new_brain
                #calculate new colour based on similarity between parent and child brains:
                new_colour = self.predator_top_colours[parent_idx] + np.random.normal(0, brain_distance*150, size=3)
                self.predator_colours[idx] = np.clip(new_colour, 0, 255)
            
        # 2. from recent archive:
        if spawn_count_recent > 0:
            total_new_archive_fitness = np.sum(self.predator_new_archive_fitnesses)
            if total_new_archive_fitness <= 0:
                #they all died before a single one of them eating food?
                self.spawn_predator(spawn_count_recent, parent_index=-1)
                return

            relative_fitnesses = self.predator_new_archive_fitnesses / total_new_archive_fitness
            #this will be used as a probability of spawning a new predator:
            free_indeces = self.get_free_indices(self.alive_predator_array,spawn_count_recent)

            new_positions = np.random.randint(low=[1,1], 
                                                    high=[self.world_width-1,self.world_height-1],
                                                    size=(spawn_count_recent,2))
            new_angles = np.random.uniform(low=-np.pi, 
                                                    high=np.pi,
                                                    size=spawn_count_recent)
            self.predator_positions[free_indeces] = new_positions
            self.predator_angles[free_indeces] = new_angles
            self.predator_speeds[free_indeces] = 0
            self.predator_angular_velocities[free_indeces] = 0
            self.predator_reproduction_timers[free_indeces] = 0.0
            self.predator_ages[free_indeces] = 0
            self.predator_offsping_count[free_indeces] = 0.0
            self.predator_satiety[free_indeces] = 1        
            self.predator_random_spawn_mask[free_indeces] = False        
            self.predator_fitnesses[free_indeces] = 0
            self.alive_predator_array[free_indeces] = True

            #stuff that will depend on the fitness of the new_archive brains: generation number, brain, colour:
            indices_of_parents = np.random.choice(len(self.predator_new_archive_fitnesses), size=spawn_count_recent, p=relative_fitnesses)
            #this means that if a brain has higher fitness, it is more likely to be chosen as a parent for the new predators that are being spawned to replace the ones that died.
            self.predator_generations[free_indeces] = self.predator_new_archive_generations[indices_of_parents]+1
            #brains:
            for i, idx in enumerate(free_indeces):
                parent_idx = indices_of_parents[i]
                parent_brain = self.predator_new_archive_brains[parent_idx]
                new_brain = copy.deepcopy(parent_brain)
                new_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                brain_distance = calculate_brain_similarity(parent_brain, new_brain)
                new_brain, structural_dist = mutate_brain_architecture(
                    new_brain, 
                    self.global_mutation_rate,
                    init_std=self.weight_std_for_new_neurons,
                    min_size = self.min_hidden_dim_size,
                    max_size = self.max_hidden_dim_size
                )
                brain_distance += structural_dist
                self.predator_brains[idx] = new_brain
                #calculate new colour based on similarity between parent and child brains:
                new_colour = self.predator_new_archive_colours[parent_idx] + np.random.normal(0, brain_distance*150, size=3)
                self.predator_colours[idx] = np.clip(new_colour, 0, 255)

        # 3. random fresh spawns:
        #this can be handled without using this function just by calling spawn_predators(to_res,parent_idx=-1)

    def add_predators_to_archives(self, dying_predator_indeces):
        for idx in dying_predator_indeces:
            
            #fitness based archive:
            brain = self.predator_brains[idx]
            min_top_idx = np.argmin(self.predator_top_fitnesses)
            if self.predator_fitnesses[idx] > self.predator_top_fitnesses[min_top_idx]:
                self.predator_top_fitnesses[min_top_idx] = self.predator_fitnesses[idx]
                self.predator_top_brains[min_top_idx] = copy.deepcopy(brain)
                self.predator_top_generations[min_top_idx] = self.predator_generations[idx]
                self.predator_top_colours[min_top_idx] = self.predator_colours[idx]

            #recency based archive:
            if self.predator_random_spawn_mask[idx]: 
                #if this was a random spawn from resurrect, dont add to evolutionary archive
                return

            #recency based archive:
            empty_indeces = np.where(self.predator_new_archive_brains == None)[0]
            if len(empty_indeces) > 0:
                self.predator_new_archive_fitnesses[empty_indeces[0]] = self.predator_fitnesses[idx]
                self.predator_new_archive_brains[empty_indeces[0]] = copy.deepcopy(brain)
                self.predator_new_archive_generations[empty_indeces[0]] = self.predator_generations[idx]
                self.predator_new_archive_colours[empty_indeces[0]] = self.predator_colours[idx]
            else:
                rand_idx = np.random.randint(0,self.predator_new_archive_size)
                self.predator_new_archive_fitnesses[rand_idx] = self.predator_fitnesses[idx]
                self.predator_new_archive_brains[rand_idx] = copy.deepcopy(brain)
                self.predator_new_archive_generations[rand_idx] = self.predator_generations[idx]
                self.predator_new_archive_colours[rand_idx] = self.predator_colours[idx]
