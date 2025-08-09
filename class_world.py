import numpy as np
import pygame as pg
import pandas as pd
import tkinter as tk
from tkinter import ttk
import copy
import torch

from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from game_functions import *
from class_herbivore_nn import *




#need boundary handling

class World:


    def __init__(self):
        #globals
        self.world_speed_multiplier = 5
        self.world_width = 800
        self.world_height = 600

        self.speed_friction_coefficient = 0.15
        self.max_speed = 25
        self.max_angular_velocity = 0.8

        self.global_mutation_rate = 0.1
        self.global_mutation_strength = 0.2
        self.min_hidden_dim_size = 6
        self.max_hidden_dim_size = 12

        self.show_raycast = False

        #plants
        self.starting_plant = 100
        self.max_plant = 300
        self.plant_size = 5
        self.plant_nutrition_value = 0.9
        self.plant_random_spawn_interval = 2.2
        self.plant_reproduction_interval = 5.5

        #predators
        self.starting_predator = 25
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
        self.starting_herbivore = 100
        self.max_herbivore = 200
        self.herbivore_size = 4
        self.herbivore_satiety_loss_factor = 0.006
        self.herbivore_max_satiety = 2
        self.herbivore_gestation_time = 20
        self.herbivore_reproduction_minimum_satiety = 1.0
        self.herbivore_reproduction_satiety_loss = 0.5
        self.herbivore_max_percent_satiety_to_eat = 0.75
        self.herbivore_FOV = np.pi*1.2
        self.herbivore_num_of_raysections = 10
        self.herbivore_vision_range = 170
        self.herbivore_max_age = 100
        self.herbivore_min_age_to_reproduce = 12
        self.herbivore_nutrition_value = 1.0
        

        #DO NOT EDIT
        self.plant_positions = np.zeros((self.max_plant,2))
        self.plant_reproduction_timers = np.zeros((self.max_plant,))
        self.alive_plant_array = np.zeros(self.max_plant,dtype=bool)
        self.plant_reproduction_timer_accumulator = 0.0

        self.herbivore_positions = np.zeros((self.max_herbivore,2))
        self.herbivore_angles = np.zeros((self.max_herbivore,))
        self.herbivore_speeds = np.zeros((self.max_herbivore,))
        #self.herbivore_accelerations = np.zeros((self.max_herbivore,))
        #self.herbivore_angular_accelerations = np.zeros((self.max_herbivore,))
        self.herbivore_angular_velocities = np.zeros((self.max_herbivore,))
        self.herbivore_satiety = np.zeros((self.max_herbivore,))
        self.alive_herbivore_array = np.zeros(self.max_herbivore,dtype=bool)
        self.num_types_of_visual_info = 2
        self.herbivore_nn_inputs = np.zeros((self.max_herbivore, (1+self.herbivore_num_of_raysections)*self.num_types_of_visual_info+2),dtype=np.float32)
        self.herbivore_reproduction_timers = np.zeros((self.max_herbivore,))
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
        #self.herbivore_accelerations = np.zeros((self.max_predator,))
        #self.herbivore_angular_accelerations = np.zeros((self.max_predator,))
        self.predator_angular_velocities = np.zeros((self.max_predator,))
        self.predator_satiety = np.zeros((self.max_predator,))
        self.alive_predator_array = np.zeros(self.max_predator,dtype=bool)
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
        self.current_herbivore = 0
        self.current_plant = 0
        self.plant_spawn_time_accumulator = 0
        self.selected_herbivore_index = None
        self.selected_predator_index = None
        self.world_time = 0
        self.herbivore_sprite = pg.image.load("herbivore.png").convert_alpha()
        self.herbivore_sprite = pg.transform.scale(self.herbivore_sprite, (self.herbivore_size*2, self.herbivore_size*2))
        self.predator_sprite = pg.image.load("predator.png").convert_alpha()
        self.predator_sprite = pg.transform.scale(self.predator_sprite, (self.predator_size*2, self.predator_size*2))
        self.predator_death_log = pd.DataFrame(columns=["fitness", "generation"])

    def update(self,dt): # super master function which updates the state of the world
        self.update_plants(dt)
        self.update_herbivores(dt)
        self.update_predators(dt)
    
    def get_free_indices(self, mask, slots_needed): # utility function
        free_indices = np.where(mask == False)[0][0:slots_needed]
        return free_indices
    
    
    def get_best_brain(self):
        if np.max(self.predator_fitnesses) >= self.predator_current_best_fitness:
            self.predator_best_brain = self.predator_brains[np.argmax(self.predator_fitnesses)]
            self.predator_current_best_fitness = np.max(self.predator_fitnesses)
            self.predator_best_bias = self.predator_nn_inputs[np.argmax(self.predator_fitnesses),self.predator_num_of_raysections*2+1]
            self.perdator_best_generation = self.predator_generations[np.argmax(self.predator_fitnesses)]
    
        if np.max(self.herbivore_fitnesses) >= self.herbivore_current_best_fitness:
            self.herbivore_best_brain = self.herbivore_brains[np.argmax(self.herbivore_fitnesses)]
            self.herbivore_current_best_fitness = np.max(self.herbivore_fitnesses)
            self.herbivore_best_bias = self.herbivore_nn_inputs[np.argmax(self.herbivore_fitnesses),self.herbivore_num_of_raysections*2+1]
            self.herbivore_best_generation = self.herbivore_generations[np.argmax(self.herbivore_fitnesses)]
            
        



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
            self.current_plant += spawn_count
    
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
        successful_spawns = np.sum(placed_mask)
        self.current_plant += successful_spawns
        self.plant_reproduction_timers[parent_indices] = 0.0
    
    def spawn_food_from_parents_NEW_DOESNTWORK(self, parent_indices):
        available_slots = np.sum(~self.alive_plant_array)
        spawn_count = min(parent_indices.size, available_slots)

        if spawn_count == 0:
            return

        parent_indices = parent_indices[:spawn_count]
        free_indices = self.get_free_indices(self.alive_plant_array, spawn_count)

        # Fixed offsets around parent for candidate positions (8 directions)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        offsets = np.stack([np.cos(angles), np.sin(angles)], axis=1) * (self.plant_size * 2.5)  # Adjustable distance
        K = offsets.shape[0]

        # Get parent 
        parent_positions = self.plant_positions[parent_indices]  # (spawn_count, 2)

        # Generate candidate positions for each parent in 8 directions
        candidate_positions = (parent_positions[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 2)  # (spawn_count * 8, 2)

        # Clamp to world boundaries
        candidate_positions[:, 0] = np.clip(candidate_positions[:, 0], self.plant_size, self.world_width - self.plant_size)
        candidate_positions[:, 1] = np.clip(candidate_positions[:, 1], self.plant_size, self.world_height - self.plant_size)

        # Collision check using KDTree
        alive_positions = self.plant_positions[self.alive_plant_array]
        if alive_positions.shape[0] > 0:
            tree = cKDTree(alive_positions)
            collisions = tree.query_ball_point(candidate_positions, r=10)
            free_mask = np.array([len(c) == 0 for c in collisions])
        else:
            # No alive plants, all candidates are free
            free_mask = np.ones(candidate_positions.shape[0], dtype=bool)

        # Map candidate indices back to parent indices
        valid_candidate_indices = np.nonzero(free_mask)[0]
        parent_idx_from_candidate = valid_candidate_indices // K

        # Keep first valid candidate per parent
        unique_parents, first_valid_idx = np.unique(parent_idx_from_candidate, return_index=True)
        chosen_candidates = candidate_positions[valid_candidate_indices[first_valid_idx]]

        # Determine how many we can place based on free slots
        n_to_place = min(chosen_candidates.shape[0], free_indices.shape[0])

        if n_to_place > 0:
            # Place new plants
            self.plant_positions[free_indices[:n_to_place]] = chosen_candidates[:n_to_place]
            self.alive_plant_array[free_indices[:n_to_place]] = True
            self.plant_reproduction_timers[free_indices[:n_to_place]] = 0.0

            # Increment plant count
            self.current_plant += n_to_place

        # Reset reproduction timers of parents
        self.plant_reproduction_timers[parent_indices] = 0.0




################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
################## ----------------------------------- HEBIVORES ----------------------------------------------- ####################
    def update_herbivores(self,dt): # master function that includes all the rest
        self.herbivores_increment_fitness(dt)
        self.herbivores_die_of_natural_causes(dt) 
        self.herbivores_perceive() # calculate raysection casting outputs
        self.herbivores_process_NN() # push raysection casting outputs into the NNs, get movement parameters out
        self.herbivores_move(dt) # update new positions using the parameters from previous function
        self.check_herbivore_plant_collisions() # check if they collided with food
        self.herbivores_reproduce(dt) # check if they reproduce 
    
    def herbivores_increment_fitness(self,dt):
        self.herbivore_fitnesses[self.alive_herbivore_array] += 0.05* (self.herbivore_satiety[self.alive_herbivore_array]/self.herbivore_max_satiety) * dt * self.world_speed_multiplier * (1/max(self.current_plant, 0.01)) * (1+0.5*self.current_predator/self.max_predator)

    def herbivores_die_of_natural_causes(self, dt):
        #check is anyone starved
        self.herbivore_satiety[self.alive_herbivore_array] -= self.herbivore_satiety_loss_factor * 8 * (self.herbivore_speeds[self.alive_herbivore_array]/self.max_speed) * dt * self.world_speed_multiplier + self.herbivore_satiety_loss_factor * dt * self.world_speed_multiplier
        self.alive_herbivore_array &= (self.herbivore_satiety > 0)

        #check if anyone died of old age
        self.herbivore_ages[self.alive_herbivore_array] += dt * self.world_speed_multiplier
        self.alive_herbivore_array &= (self.herbivore_ages < self.herbivore_max_age)
        
        self.current_herbivore = np.count_nonzero(self.alive_herbivore_array)#
        if self.selected_herbivore_index != None:
            if self.alive_herbivore_array[self.selected_herbivore_index] == False:
                self.selected_herbivore_index = None
            

    def herbivores_perceive(self):
        alive_indices = np.where(self.alive_herbivore_array)[0]
        if alive_indices.size == 0:
            return
        #ray_angles = self.herbivore_angles[self.alive_herbivore_array][:,np.newaxis]+self.herbivore_relative_ray_angles
        #now we have an array of dimension [alive_herb, num_raycasts] with the correct raycast angles
        #lets make an array of all the end points of the ray casts. it should have dimensions (alive_herb, num_raycasts, 2)
        #self.herbivore_ray_directions[alive_indices,:,0] = np.cos(ray_angles)*self.herbivore_vision_range
        #self.herbivore_ray_directions[alive_indices,:,1] = np.sin(ray_angles)*self.herbivore_vision_range
 
        distances_sections, desirabilities_sections = herbivores_section_vision_self_food_and_predators(
            self_positions=self.herbivore_positions[alive_indices],
            self_angles=self.herbivore_angles[alive_indices],
            food_positions=self.plant_positions[self.alive_plant_array],
            predator_positions = self.predator_positions[self.alive_predator_array],
            self_num_of_raysections = self.herbivore_num_of_raysections,
            self_vision_range = self.herbivore_vision_range,
            self_fov = self.herbivore_FOV)
        # disarabilities are 1 for food, 0 for otehr herbivores, -0.5 for nothing and -1 for predators
        # distances_sections: (N_alive, 5)
        # desirabilities_sections: (N_alive, 5)
        
        # Direct write distances to columns 0:5
        self.herbivore_nn_inputs[alive_indices, 0:self.herbivore_num_of_raysections+1] = distances_sections

        # Direct write desirabilities to columns 5:10
        self.herbivore_nn_inputs[alive_indices, self.herbivore_num_of_raysections+1:self.herbivore_num_of_raysections*2+2] = desirabilities_sections

        # Satiety normalized to column 10
        self.herbivore_nn_inputs[alive_indices, self.herbivore_num_of_raysections*2+2] = self.herbivore_satiety[alive_indices] / self.herbivore_max_satiety 
    
    def herbivores_process_NN(self):
        alive_indices = np.where(self.alive_herbivore_array)[0]
        if alive_indices.size == 0:
            return

        # === Batch processing using precomputed nn_inputs ===
        input_tensor = torch.from_numpy(self.herbivore_nn_inputs[alive_indices])  # shape (N_alive, 12)

        # === Forward pass through each individual's brain ===
        outputs = []
        for i, idx in enumerate(alive_indices):
            brain = self.herbivore_brains[idx]
            with torch.no_grad():
                output = brain(input_tensor[i].unsqueeze(0)).numpy()[0]
            outputs.append(output)
        outputs = np.array(outputs, dtype=np.float32)  # shape (N_alive, 2)

        # === Scale outputs ===
        target_speeds = (outputs[:, 0] + 1) / 2 * self.max_speed
        target_angular_velocities = outputs[:, 1] * self.max_angular_velocity

        # === Assign directly ===
        self.herbivore_speeds[alive_indices] = target_speeds
        self.herbivore_angular_velocities[alive_indices] = target_angular_velocities

 
    def herbivores_move(self,dt):
        #self.herbivore_angular_velocities[self.alive_herbivore_array] += self.herbivore_angular_accelerations[self.alive_herbivore_array] * dt * self.world_speed_multiplier - 1*self.herbivore_angular_velocities[self.alive_herbivore_array] * self.speed_friction_coefficient * dt * self.world_speed_multiplier
        self.herbivore_angles[self.alive_herbivore_array] += self.herbivore_angular_velocities[self.alive_herbivore_array] * dt * self.world_speed_multiplier
        self.herbivore_angles %= 2 * np.pi
        #np.clip(self.herbivore_angular_velocities, -self.max_angular_velocity,self.max_angular_velocity,out=self.herbivore_angular_velocities)

        #self.herbivore_speeds[self.alive_herbivore_array] += self.herbivore_accelerations[self.alive_herbivore_array] * dt * self.world_speed_multiplier - self.herbivore_speeds[self.alive_herbivore_array] * self.speed_friction_coefficient * dt * self.world_speed_multiplier
        #np.clip(self.herbivore_speeds, 0,self.max_speed,out=self.herbivore_speeds)
        
        dpos = np.zeros((np.sum(self.alive_herbivore_array),2))

        np.cos(self.herbivore_angles[self.alive_herbivore_array], out=dpos[:,0])
        np.sin(self.herbivore_angles[self.alive_herbivore_array], out=dpos[:,1])

        dpos *= self.herbivore_speeds[self.alive_herbivore_array, np.newaxis]

        self.herbivore_positions[self.alive_herbivore_array] += dpos * self.world_speed_multiplier  * dt

        #handle world boundary
        #self.herbivore_positions[:,0] = np.clip(self.herbivore_positions[:,0], 0+self.herbivore_size, self.world_width-self.herbivore_size)
        #self.herbivore_positions[:,1] = np.clip(self.herbivore_positions[:,1], 0+self.herbivore_size, self.world_height-self.herbivore_size)
        self.herbivore_positions[:, 0] %= self.world_width
        self.herbivore_positions[:, 1] %= self.world_height


    def check_herbivore_plant_collisions(self):
        d = distance_matrix(self.herbivore_positions[self.alive_herbivore_array],self.plant_positions[self.alive_plant_array])
        consumed = np.column_stack(np.where(d<=self.plant_size+2))
        
        #careful, these are indices of the alive plants and animals
        #consumed_plant_indices = consumed[:,1]
        #herbivores_that_ate_indices = consumed[:,0]

        alive_herbivores = np.where(self.alive_herbivore_array)[0]
        herbivores_that_ate_indices = alive_herbivores[consumed[:,0]]

        alive_plants = np.where(self.alive_plant_array)[0]
        plants_that_were_eaten_indices = alive_plants[consumed[:,1]]

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
        self.current_plant -= plants_that_were_eaten_indices.size
    
    def herbivores_reproduce(self,dt):
        sated_herbivore_indices = np.where((self.alive_herbivore_array) & 
                                           (self.herbivore_ages >= self.herbivore_min_age_to_reproduce) &
                                           (self.herbivore_satiety >= self.herbivore_reproduction_minimum_satiety))[0]
        self.herbivore_reproduction_timers[sated_herbivore_indices] += dt*self.world_speed_multiplier
        np.clip(self.herbivore_reproduction_timers, a_min=0, a_max=self.herbivore_gestation_time,out=self.herbivore_reproduction_timers)

        reproducing_herbivore_indices = np.where((self.herbivore_reproduction_timers >= self.herbivore_gestation_time) & 
                                                 self.alive_herbivore_array & 
                                                 (self.herbivore_satiety >= self.herbivore_reproduction_minimum_satiety))[0]        
        for i in reproducing_herbivore_indices:
            self.spawn_herbivore(1,parent_index=i)


    def spawn_herbivore(self, how_many_to_spawn, parent_index=-1):
        available_slots = np.sum(np.invert(self.alive_herbivore_array))
        spawn_count = min(how_many_to_spawn, available_slots)

        if spawn_count > 0:
            free_indeces = self.get_free_indices(self.alive_herbivore_array,spawn_count)

            if parent_index < 0:
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
                self.herbivore_reproduction_timers[free_indeces] = 0.0
                self.herbivore_ages[free_indeces] = 5
                self.herbivore_fitnesses[free_indeces] = 0
                self.herbivore_offsping_count[free_indeces] = 0.0
                self.herbivore_satiety[free_indeces] = 1
                self.alive_herbivore_array[free_indeces] = True
                self.herbivore_nn_inputs[free_indeces,self.herbivore_num_of_raysections*2+1] = np.random.uniform(low= -1.0,
                                                                        high = 1.0,
                                                                        size=spawn_count)
                self.herbivore_generations[free_indeces] = 0
                for idx in free_indeces:
                    self.herbivore_brains[idx] = AnimalBrain(
                        n_ray_sections=self.herbivore_num_of_raysections,
                        n_types_of_info_in_each_section=self.num_types_of_visual_info,
                        hidden_dim_1=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0],
                        hidden_dim_2=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0]
                        )
                self.current_herbivore += spawn_count
            
            else: 
                self.herbivore_offsping_count[parent_index] += 1
                #since this option will always just spawn one animals, free indeces will always be a single number,
                #and so will spawn count
                parent_position = self.herbivore_positions[parent_index]
                child_position = parent_position + np.random.randint(0,1,2)
                self.herbivore_positions[free_indeces] = child_position
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
                self.herbivore_nn_inputs[free_indeces,self.herbivore_num_of_raysections*2+1] = self.herbivore_nn_inputs[parent_index,self.herbivore_num_of_raysections*2+1] + np.clip(np.random.uniform(low=-self.global_mutation_strength,
                                                                                                                                                                                                        high= self.global_mutation_strength,
                                                                                                                                                                                                        size=1), 
                                                                                                                                                                                                        -1,1)

                child_brain = copy.deepcopy(self.herbivore_brains[parent_index])
                sizes = child_brain.get_dim_sizes()
                for dimension, index in zip(["fc1","fc2"],[0,1]):
                    if np.random.uniform(low=0.0,high=1.0,size=1)[0] <= self.global_mutation_rate:
                        child_brain = resize_layer_in_animal_brain(child_brain,
                                                                layer=dimension,
                                                                new_size=np.clip((sizes[index]+np.random.choice([-1,1])),a_min=1,a_max=20),
                                                                init_std=0.5)
                child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                self.herbivore_brains[free_indeces] = child_brain
                self.current_herbivore += spawn_count
                self.herbivore_reproduction_timers[parent_index] = 0.0
                self.herbivore_satiety[parent_index] -= self.herbivore_reproduction_satiety_loss
                self.herbivore_fitnesses[parent_index] += 2 * (1/max(self.current_plant, 0.01)) * (1+0.5*self.current_predator/self.max_predator)

    def resurrect_herbivores(self, spawn_count):

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
            self.herbivore_nn_inputs[free_indeces,self.herbivore_num_of_raysections*2+1] = np.clip(self.herbivore_best_bias + np.random.uniform(low=-self.global_mutation_strength,high=self.global_mutation_strength,size=1),-1,1)
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
        self.predator_fitnesses[self.alive_predator_array] += 0.05* (self.predator_satiety[self.alive_predator_array]/self.predator_max_satiety) * dt * self.world_speed_multiplier * max((0.4+1/max(self.current_herbivore, 0.01)),1)
        
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
                        n_ray_sections=self.predator_num_of_raysections,
                        n_types_of_info_in_each_section = self.num_types_of_visual_info,
                        hidden_dim_1=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0],
                        hidden_dim_2=np.random.randint(low=self.min_hidden_dim_size,high=self.max_hidden_dim_size,size=1)[0]
                        )
                self.current_predator += spawn_count
            
            else: 
                self.predator_offsping_count[parent_index] += 1
                self.predator_fitnesses[parent_index] += 0.1 *  max((0.4+1/max(self.current_herbivore, 0.01)),1)
                #since this option will always just spawn one animals, free indeces will always be a single number,
                #and so will spawn count
                parent_position = self.predator_positions[parent_index]
                child_position = parent_position + np.random.randint(-10,10,2)
                self.predator_positions[free_indeces] = child_position
                self.predator_angles[free_indeces] = np.random.randint(-np.pi,np.pi,1)
                self.predator_speeds[free_indeces] = 0
                self.predator_angular_velocities[free_indeces] = 0
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
                                                                init_std=0.5)
                child_brain.mutate(self.global_mutation_rate, self.global_mutation_strength)
                self.predator_brains[free_indeces] = child_brain
                self.current_predator += spawn_count
                self.predator_reproduction_timers[parent_index] = 0.0
                self.predator_satiety[parent_index] -= self.predator_reproduction_satiety_loss
    
    def resurrect_predators(self, spawn_count):

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


########## ----------------------------------------------------------- DRAWING ---------------------------------------------------------###########
########## ----------------------------------------------------------- DRAWING ---------------------------------------------------------###########
########## ----------------------------------------------------------- DRAWING ---------------------------------------------------------###########
    def draw(self,display_window):
        self.draw_plants(display_window)
        self.draw_herbivores(display_window)
        self.draw_predators(display_window)
        self.draw_section_vision_for_selected_animal(display_window)
                                                     
    def draw_plants(self, display_window):
        alive_plant_indices = np.where(self.alive_plant_array)[0]
        for index in alive_plant_indices:
            plant_position = self.plant_positions[index]
            pg.draw.circle(display_window, (17, 50, 174), plant_position.astype(int), radius = self.plant_size)

    def draw_herbivores(self, display_window):
        alive_herbivore_indices = np.where(self.alive_herbivore_array)[0]
        

        angles_degrees = -np.degrees(self.herbivore_angles[alive_herbivore_indices]) - 90  # (N_alive,)
        satiety_ratios = self.herbivore_satiety[alive_herbivore_indices] / self.herbivore_max_satiety
        tint_greens = np.clip((satiety_ratios * 255).astype(int), 0, 255)

        # === Draw each alive herbivore ===
        for pos, angle_deg, green_value in zip(self.herbivore_positions[alive_herbivore_indices], angles_degrees, tint_greens):
            # Draw satiety tint ring
            #pg.draw.circle(display_window, (50, green_value, 30), pos, radius=self.herbivore_size + 2, width=2)

            # Rotate sprite to face direction
            rotated_sprite = pg.transform.rotate(self.herbivore_sprite, angle_deg)
            sprite_rect = rotated_sprite.get_rect(center=pos)

            # Blit rotated sprite
            display_window.blit(rotated_sprite, sprite_rect)

    def draw_herbivores_light(self, display_window):
        alive_herbivore_indices = np.where(self.alive_herbivore_array)[0]
        for index in alive_herbivore_indices:
            herbivore_position = self.herbivore_positions[index]
            pg.draw.circle(display_window, (50, clamp(self.herbivore_satiety[index]*255, 0,255), 30), herbivore_position.astype(int), radius = self.herbivore_size)

    
    def draw_predators(self, display_window):
        alive_predator_indices = np.where(self.alive_predator_array)[0]
        
        
        angles_degrees = -np.degrees(self.predator_angles[alive_predator_indices]) - 90  # (N_alive,)
        satiety_ratios = self.predator_satiety[alive_predator_indices] / self.predator_max_satiety
        tint_reds = np.clip((satiety_ratios * 255).astype(int), 0, 255)

        # === Draw each alive herbivore ===
        for pos, angle_deg, red_value in zip(self.predator_positions[alive_predator_indices], angles_degrees, tint_reds):
            # Draw satiety tint ring
            #pg.draw.circle(display_window, (red_value, 50, 30), pos, radius=self.predator_size + 2, width=2)

            # Rotate sprite to face direction
            rotated_sprite = pg.transform.rotate(self.predator_sprite, angle_deg)
            sprite_rect = rotated_sprite.get_rect(center=pos)

            # Blit rotated sprite
            display_window.blit(rotated_sprite, sprite_rect)

    
    def draw_predators_light(self, display_window):
        alive_predator_indices = np.where(self.alive_predator_array)[0]
        for index in alive_predator_indices:
            position = self.predator_positions[index]
            pg.draw.circle(display_window,(clamp(self.predator_satiety[index]*255, 0,255),50,30) , position.astype(int), radius = self.predator_size)
    
    def draw_section_vision_for_selected_animal(
    self,
    display_window,
    overlap_percent=0.08,
    closeby_percent=0.1,
    draw_sectors_under_closeby=True
    ):


        # Determine which animal is selected
        if self.selected_herbivore_index is not None and self.alive_herbivore_array[self.selected_herbivore_index]:
            idx = self.selected_herbivore_index
            pos = self.herbivore_positions[idx].astype(int)
            heading = self.herbivore_angles[idx]
            vision_range = self.herbivore_vision_range
            fov = self.herbivore_FOV
            num_sections = self.herbivore_num_of_raysections
            distances = self.herbivore_nn_inputs[idx, 0:num_sections + 1]
            labels = self.herbivore_nn_inputs[idx, num_sections + 1:(num_sections + 1) * 2]

        elif self.selected_predator_index is not None and self.alive_predator_array[self.selected_predator_index]:
            idx = self.selected_predator_index
            pos = self.predator_positions[idx].astype(int)
            heading = self.predator_angles[idx]
            vision_range = self.predator_vision_range
            fov = self.predator_FOV
            num_sections = self.predator_num_of_raysections
            distances = self.predator_nn_inputs[idx, 0:num_sections + 1]
            labels = self.predator_nn_inputs[idx, num_sections + 1:(num_sections + 1) * 2]
        else:
            return  # Nothing selected

        # Draw yellow circle outline
        pg.draw.circle(display_window, (255, 255, 0), pos, 15, width=2)

        if not self.show_raycast:
            return

        s = pg.Surface(display_window.get_size(), pg.SRCALPHA)

        # === Draw closeby zone first if underlay, else later ===
        closeby_radius = closeby_percent * vision_range
        dist_closeby = distances[0]
        label_closeby = labels[0]

        

        if draw_sectors_under_closeby:
            # Draw closeby first so sectors appear under
            color_closeby = get_color_from_label(label_closeby)
            pg.draw.circle(s, color_closeby, pos, int(closeby_radius))

        # === Sector drawing ===
        effective_fov = fov * (1 + overlap_percent)
        section_edges = np.linspace(-effective_fov / 2, effective_fov / 2, num_sections + 1)

        cos_edges = np.cos(heading + section_edges)
        sin_edges = np.sin(heading + section_edges)

        min_radius_frac = 0.05  # minimal visible radius

        for sec in range(1, num_sections + 1):  # skip 0, which is the closeby
            dist = distances[sec]
            label = labels[sec]

            color = get_color_from_label(label)
            radius = max(min_radius_frac * vision_range, (1 - dist) * vision_range)

            # If skipping sectors inside closeby zone:
            if not draw_sectors_under_closeby and radius <= closeby_radius:
                continue

            angle_left_cos = cos_edges[sec - 1]
            angle_left_sin = sin_edges[sec - 1]
            angle_right_cos = cos_edges[sec]
            angle_right_sin = sin_edges[sec]

            p0 = pos
            p1 = pos + radius * np.array([angle_left_cos, angle_left_sin])
            p2 = pos + radius * np.array([angle_right_cos, angle_right_sin])

            pg.draw.polygon(s, color, [p0, p1.astype(int), p2.astype(int)])

        # Draw closeby zone last if overlay
        if not draw_sectors_under_closeby:
            color_closeby = get_color_from_label(label_closeby)
            pg.draw.circle(s, color_closeby, pos, int(closeby_radius))

        display_window.blit(s, (0, 0))
    
    def show_herbivore_stats(self, herb_idx, root):
        brain = self.herbivore_brains[herb_idx]
        sizes = brain.get_dim_sizes()
        stats = f"""
        Herbivore {herb_idx}
        ---------------------------
        Age: {round(self.herbivore_ages[herb_idx],2)}
        Generation: {round(self.herbivore_generations[herb_idx])}
        Children: {round(self.herbivore_offsping_count[herb_idx])}
        Angle: {round(self.herbivore_angles[herb_idx],2)}
        Satiety: {round(self.herbivore_satiety[herb_idx],2)}
        Gestation: {round(100*self.herbivore_reproduction_timers[herb_idx]/self.herbivore_gestation_time,2)} %

        Speed: {round(self.herbivore_speeds[herb_idx],2)}
        Angle: {round(self.herbivore_angles[herb_idx],2)}
        
        Hidden dimension 1 size: {round(sizes[0])}
        Hidden dimension 2 size: {round(sizes[1])}
        """
        if hasattr(self, 'stats_window') and self.stats_window.winfo_exists():
            self.stats_label.config(text=stats)
        else:
            self.stats_window = tk.Toplevel(root)  # Use Toplevel instead of Tk()
            self.stats_window.title(f"Herbivore {herb_idx} Stats")
            self.stats_label = tk.Label(self.stats_window, text=stats, justify='left', font=("Courier", 10))
            self.stats_label.pack(padx=10, pady=10)

            close_button = tk.Button(self.stats_window, text="Close", command=self.stats_window.destroy)
            close_button.pack(pady=5)
    
    def show_predator_stats(self, herb_idx, root): #too laze to rename herb_idx here sorry
        brain = self.predator_brains[herb_idx]
        sizes = brain.get_dim_sizes()
        stats = f"""
        Predator {herb_idx}
        ---------------------------
        Age: {round(self.predator_ages[herb_idx],2)}
        Generation: {round(self.predator_generations[herb_idx])}
        Children: {round(self.predator_offsping_count[herb_idx])}
        Angle: {round(self.predator_angles[herb_idx],2)}
        Satiety: {round(self.predator_satiety[herb_idx],2)}
        Gestation: {round(100*self.predator_reproduction_timers[herb_idx]/self.predator_gestation_time,2)} %

        Speed: {round(self.predator_speeds[herb_idx],2)}
        Angle: {round(self.predator_angles[herb_idx],2)}

        Hidden dimension 1 size: {round(sizes[0])}
        Hidden dimension 2 size: {round(sizes[1])}
        """
        if hasattr(self, 'stats_window') and self.stats_window.winfo_exists():
            self.stats_label.config(text=stats)
        else:
            self.stats_window = tk.Toplevel(root)  # Use Toplevel instead of Tk()
            self.stats_window.title("Animal Stats")
            self.stats_label = tk.Label(self.stats_window, text=stats, justify='left', font=("Courier", 10))
            self.stats_label.pack(padx=10, pady=10)

            close_button = tk.Button(self.stats_window, text="Close", command=self.stats_window.destroy)
            close_button.pack(pady=5)

    