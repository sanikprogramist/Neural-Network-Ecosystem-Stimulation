import numpy as np
import pandas as p
import pygame as pg
import time


from game_functions import *
from class_world import *

import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#TASK LIST
# X Create World
# X Create FPS independent timing control
# X Spawn food system
# X spawn herbivore system
# X animal movement system
# X herbivore eat plant 
# X satiety based death of herbivores
# X boundary handling
# X angular velocity
# X herbivores raycast to detect walls and food, output with shape (N_herbivore, N_ray, 2) (first number stores distance to overlapping piece of food, second to wall, 0 if nothing)
# X Herbivores die of old age
# X herbivores reproduce if they have enough food and enough time has passed from last reproduction
# display herbivore stats
# initial data

# FIX REPRODUCTION GESTATION TIME OVER 100%

#----------------------------------------------------
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(0)  # DPI_AWARENESS_SYSTEM_AWARE
except:
    pass

# Initialize Pygame
pg.init()

# Set up the game window
screen = pg.display.set_mode((800 , 600))
pg.display.set_caption("Evolving Ecosystem")
clock = pg.time.Clock()

root = tk.Tk()
root.withdraw()  # Hide the root window

herbivore_extinction_counter = 0
predator_extinction_counter = 0

print("testing message to test git")


# ----------------------INITIALISE PLOTS -------------------------
plot_window = tk.Tk()
plot_window.wm_title("Herbivore Population Over Time")
plot_window.resizable(False,False)

plot_update_interval = 5.0
last_update_time = 0.0
max_points = 1000

time_data = []
herbivore_data = []
predator_data = []
extra_data = []

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 10), dpi=50)

line1, = ax1.plot([], [], color='green')
ax1.set_ylabel("Herbivore Population")
ax1.grid(True)

line2, = ax2.plot([], [], color='red')
ax2.set_ylabel("Predator Population")
ax2.set_xlabel("World Time (s)")
ax2.grid(True)

scatter = ax3.scatter([], [], color='blue',s=10,alpha=0.6)
ax3.set_xlabel("Generation")
ax3.set_ylabel("Predator Fitness")
ax3.grid(True)

fig.tight_layout()

canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=plot_window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# ----------------------INITIALISE PLOTS -------------------------
 

#create world
world = World()
world.spawn_food(200)
world.spawn_herbivore(100)
world.spawn_predator(10)
    
# Game loop
running = True
while running:
    dt = clock.tick(60)/1000
    world.world_time += dt*world.world_speed_multiplier
    screen.fill("white")

    #update states of everything in the world, plants, animals, etc
    world.update(dt)    
    if world.current_herbivore == 0:
        world.resurrect_herbivores(50)
        world.spawn_food(60)
        herbivore_extinction_counter += 1
    elif world.current_predator == 0:
        world.resurrect_predators(10)
        predator_extinction_counter += 1


    #draw everything onto the screen
    world.draw(screen)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_r:
                world.show_raycast = not world.show_raycast
            if event.key == pg.K_SPACE:
                world.show_raycast = False
                world.selected_herbivore_index = None
                world.selected_predator_index = None
            if event.key == pg.K_s:
                world.spawn_food(150)
                world.spawn_herbivore(75)
                world.spawn_predator(10)
            if event.key == pg.K_p:
                print(herbivore_extinction_counter)
                print(predator_extinction_counter)
            if event.key == pg.K_a:
                herb_dim_sizes_1 = []
                herb_dim_sizes_2 = []
                for brain in world.herbivore_brains[world.alive_herbivore_array]:
                    size = brain.get_dim_sizes()
                    herb_dim_sizes_1.append(size[0])
                    herb_dim_sizes_2.append(size[1])
                pred_dim_sizes_1 = []
                pred_dim_sizes_2 = []
                for brain in world.predator_brains[world.alive_predator_array]:
                    size = brain.get_dim_sizes()
                    pred_dim_sizes_1.append(size[0])
                    pred_dim_sizes_2.append(size[1])
                stats = f"""
                Herbivore hdim 1 mean size: {np.mean(herb_dim_sizes_1)}
                Herbivore hdim 2 mean size: {np.mean(herb_dim_sizes_2)}
                ---------------------------
                Predator hdim 1 mean size: {np.mean(pred_dim_sizes_1)}
                Predator hdim 2 mean size: {np.mean(pred_dim_sizes_2)}
                """
                print(stats)

        elif event.type == pg.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pg.mouse.get_pos())
            click_radius = 8
            click_radius_sq = click_radius ** 2

            # Check herbivores
            alive_herb_indices = np.where(world.alive_herbivore_array)[0]
            herb_positions = world.herbivore_positions[alive_herb_indices]
            herb_distances_sq = np.sum((herb_positions - mouse_pos) ** 2, axis=1)

            # Check predators
            alive_pred_indices = np.where(world.alive_predator_array)[0]
            pred_positions = world.predator_positions[alive_pred_indices]
            pred_distances_sq = np.sum((pred_positions - mouse_pos) ** 2, axis=1)

            # Determine closest clicked entity
            clicked_herbs = alive_herb_indices[herb_distances_sq <= click_radius_sq]
            clicked_preds = alive_pred_indices[pred_distances_sq <= click_radius_sq]

            selected_idx = None
            selected_type = None

            if clicked_herbs.size > 0:
                closest_idx = np.argmin(herb_distances_sq[herb_distances_sq <= click_radius_sq])
                selected_idx = clicked_herbs[closest_idx]
                selected_type = 'herbivore'
            elif clicked_preds.size > 0:
                closest_idx = np.argmin(pred_distances_sq[pred_distances_sq <= click_radius_sq])
                selected_idx = clicked_preds[closest_idx]
                selected_type = 'predator'

            if selected_type == 'herbivore':
                world.selected_herbivore_index = selected_idx
                world.selected_predator_index = None
                world.show_herbivore_stats(selected_idx, root)

            elif selected_type == 'predator':
                world.selected_predator_index = selected_idx
                world.selected_herbivore_index = None
                # world.show_predator_stats(selected_idx, root)  # Add this later if desired
    

    now = time.time()
    if now - last_update_time > plot_update_interval:
        world.get_best_brain()
        update_plots(world,time_data,herbivore_data,predator_data,line1,line2,scatter,ax1,ax2,ax3,max_points,canvas)
        if world.selected_herbivore_index != None:
            world.show_herbivore_stats(world.selected_herbivore_index,root)
        elif world.selected_predator_index != None:
            world.show_predator_stats(world.selected_predator_index,root)
        last_update_time = now
        print(dt)

    # Update our window
    pg.display.flip()
    #root.update()
# Quit Pygame   
pg.quit()