import numpy as np
import time
from class_animal_brain_nn import *

FOOD_LABEL = 1.0
CONSPECIFIC_LABEL = 0.1
PREDATOR_LABEL = -1.0
EMPTY_LABEL = -0.5

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

def predator_section_vision_self_and_food(
    self_positions,
    self_angles,            # (N_predators,) in radians
    food_positions,
    self_num_of_raysections,
    self_vision_range,
    self_fov,                # in radians
    overlap_factor = 0.08,
    closeby_zone_factor = 0.1
):
    N_self = self_positions.shape[0]
    N_food = food_positions.shape[0]

    # Combine objects
    all_objects = np.vstack([food_positions, self_positions])
    object_labels = np.hstack([np.ones(N_food), np.full(N_self, 0.1)])

    # Outputs
    distances_output = np.zeros((N_self, self_num_of_raysections + 1), dtype=np.float32)  # +1 for closeby zone
    labels_output = np.full((N_self, self_num_of_raysections + 1), -0.5, dtype=np.float32)  # -0.5 for "nothing"
    
    effective_fov = self_fov * (1 + overlap_factor)
    section_edges = np.linspace(-effective_fov/2, effective_fov/2, self_num_of_raysections + 1)

    closeby_zone_radius = self_vision_range * closeby_zone_factor  # 10% of vision range considered "closeby zone"

    for i in range(N_self):
        diffs = all_objects - self_positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        angles = np.arctan2(diffs[:,1], diffs[:,0]) - self_angles[i]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        within_range = dists <= self_vision_range
        within_fov = np.abs(angles) <= effective_fov / 2
        visible = within_range & within_fov

        # Exclude self
        exclude_self = np.ones(len(all_objects), dtype=bool)
        exclude_self[N_food + i] = False
        visible_objects_mask = visible & exclude_self

        # --- CLOSEBY ZONE (index -1) ---
        closeby_mask = (dists <= closeby_zone_radius) & visible_objects_mask
        if np.any(closeby_mask):
            section_dists = dists[closeby_mask]
            min_idx = np.argmin(section_dists)
            closest_dist = max(section_dists[min_idx], 1e-2)  # Clamp to avoid 0
            obj_label = object_labels[closeby_mask][min_idx]

            normalized_dist = 1 - (closest_dist / closeby_zone_radius)
            distances_output[i, 0] = normalized_dist
            labels_output[i, 0] = obj_label
        else:
            distances_output[i, 0] = 0.0
            labels_output[i, 0] = -0.5  # nothing

        # --- SECTOR PROCESSING ---
        for sec in range(self_num_of_raysections):
            angle_min = section_edges[sec]
            angle_max = section_edges[sec + 1]

            in_section = (angles >= angle_min) & (angles < angle_max) & visible_objects_mask
            if np.any(in_section):
                section_dists = dists[in_section]
                min_idx = np.argmin(section_dists)
                closest_dist = max(section_dists[min_idx], 1e-2)  # Clamp
                obj_label = object_labels[in_section][min_idx]

                normalized_dist = 1 - (closest_dist / self_vision_range)
                distances_output[i, sec + 1] = normalized_dist
                labels_output[i, sec + 1] = obj_label
            else:
                distances_output[i, sec + 1] = 0.0
                labels_output[i, sec + 1] = -0.5  # nothing

    return distances_output, labels_output

def herbivores_section_vision_self_food_and_predators(
    self_positions,
    self_angles,              # (N_herbivores,)
    food_positions,
    predator_positions,
    self_num_of_raysections,
    self_vision_range,
    self_fov,
    closeby_zone_fraction=0.1,      # e.g., 0.1 = 10% of vision range
    sector_overlap_fraction=0.08     # e.g., 0.1 = 10% overlap
):
    N_self = self_positions.shape[0]
    N_food = food_positions.shape[0]
    N_pred = predator_positions.shape[0]

    closeby_zone_radius = closeby_zone_fraction * self_vision_range
    per_sector_width = self_fov / self_num_of_raysections
    overlap_radians = per_sector_width * sector_overlap_fraction

    all_objects = np.vstack([
        food_positions,               # N_food
        predator_positions,           # N_pred
        self_positions                # N_self
    ])

    # Labels:
    # +1.0: food
    # -1.0: predator
    #  0.1: conspecific
    object_labels = np.hstack([
        np.ones(N_food),
        -1.0 * np.ones(N_pred),
        np.full(N_self, 0.1)
    ])

    distances_output = np.zeros((N_self, self_num_of_raysections + 1), dtype=np.float32)
    labels_output = np.full((N_self, self_num_of_raysections + 1), -0.5, dtype=np.float32)  # -0.5 = nothing seen

    section_edges = np.linspace(-self_fov / 2, self_fov / 2, self_num_of_raysections + 1)
    section_edges[0] -= overlap_radians / 2
    section_edges[-1] += overlap_radians / 2

    for i in range(N_self):
        diffs = all_objects - self_positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0]) - self_angles[i]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        within_range = dists <= self_vision_range
        within_fov = np.abs(angles) <= self_fov / 2
        visible = within_range & within_fov

        exclude_self = np.ones(len(all_objects), dtype=bool)
        exclude_self[N_food + N_pred + i] = False
        visible &= exclude_self

        # === Closeby Zone (sector 0) ===
        in_closeby = (dists <= closeby_zone_radius) & visible
        if np.any(in_closeby):
            section_dists = dists[in_closeby]
            min_idx = np.argmin(section_dists)
            closest_dist = max(section_dists[min_idx], 1e-4)
            obj_label = object_labels[in_closeby][min_idx]
            normalized_dist = 1 - (closest_dist / closeby_zone_radius)
            distances_output[i, 0] = normalized_dist
            labels_output[i, 0] = obj_label
        else:
            distances_output[i, 0] = 0
            labels_output[i, 0] = -0.5

        # === Angular sectors (1..num_raysections) ===
        for sec in range(self_num_of_raysections):
            angle_min = section_edges[sec] - overlap_radians / 2
            angle_max = section_edges[sec + 1] + overlap_radians / 2

            in_sector = (angles >= angle_min) & (angles < angle_max) & visible & (dists > closeby_zone_radius)
            col = sec + 1

            if np.any(in_sector):
                section_dists = dists[in_sector]
                min_idx = np.argmin(section_dists)
                closest_dist = max(section_dists[min_idx], 1e-4)
                obj_label = object_labels[in_sector][min_idx]
                normalized_dist = 1 - (closest_dist / self_vision_range)
                distances_output[i, col] = normalized_dist
                labels_output[i, col] = obj_label
            else:
                distances_output[i, col] = 0
                labels_output[i, col] = -0.5

    return distances_output, labels_output


def update_plots(world,time_data,herbivore_data,predator_data,line1,line2,scatter,ax1,ax2,ax3,max_points,canvas):
    # add data point
    time_data.append(world.world_time)
    herbivore_data.append(world.current_herbivore)
    predator_data.append(world.current_predator)
    
    generations = world.predator_death_log["generation"].values
    fitnesses = world.predator_death_log["fitness"].values
        
    #clip to max number of points
    if len(time_data) > max_points:
        time_data = time_data[-max_points:]
        herbivore_data = herbivore_data[-max_points:]
        predator_data = predator_data[-max_points:]
        #extra_data = extra_data[-max_points:]

        #make lines        
    line1.set_data(time_data, herbivore_data)
    line2.set_data(time_data, predator_data)
    scatter.set_offsets(np.c_[generations, fitnesses])

    # Update axis limits
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    if generations.size > 0 and fitnesses.size > 0:
        ax3.set_xlim(generations.min() - 1, generations.max() + 1)
        ax3.set_ylim(fitnesses.min() - 0.02, fitnesses.max() + 0.02)
        # Redraw
    canvas.draw()
    canvas.flush_events()

def get_color_from_label(label):
        if np.isclose(label, FOOD_LABEL):
            return (0, 255, 0, 100)
        elif np.isclose(label, PREDATOR_LABEL):
            return (255, 0, 0, 100)
        elif np.isclose(label, CONSPECIFIC_LABEL):
            return (0, 0, 255, 100)
        else:
            return (100, 100, 100, 80)
        
def resize_layer_in_animal_brain(
    brain: AnimalBrain,
    layer: str,                # "fc1" or "fc2"
    new_size: int,             # desired new size of that layer's output dimension
    init_std: float = 0.1,     # std for initializing new weights
    clip_value: float = 1.0    # clip range [-clip_value, clip_value]
    ) -> AnimalBrain:
    """
    Returns a new AnimalBrain with the specified layer resized (increase or decrease),
    preserving existing weights where possible, initializing new weights, and clipping all weights.
    """
    # Extract initialization parameters
    n_ray_sections = (brain.fc1.in_features - 2) // 1 - 1  # since n_types_of_info_in_each_section = 1
    n_types_of_info_in_each_section = 1
    hidden_dim_1 = brain.fc1.out_features
    hidden_dim_2 = brain.fc2.out_features

    # Update target layer dimension
    if layer == "fc1":
        hidden_dim_1 = new_size
    elif layer == "fc2":
        hidden_dim_2 = new_size
    else:
        raise ValueError("Layer must be 'fc1' or 'fc2'")

    # Create new brain
    new_brain = AnimalBrain(
        n_ray_sections=n_ray_sections,
        n_types_of_info_in_each_section=n_types_of_info_in_each_section,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2
    )

    with torch.no_grad():
        # Resize and copy fc1
        old_w = brain.fc1.weight
        new_w = new_brain.fc1.weight
        min_o, min_i = min(old_w.shape[0], new_w.shape[0]), min(old_w.shape[1], new_w.shape[1])
        new_w[:min_o, :min_i] = old_w[:min_o, :min_i]
        if new_w.shape[0] > old_w.shape[0]:
            nn.init.normal_(new_w[min_o:, :], mean=0.0, std=init_std)
        if new_w.shape[1] > old_w.shape[1]:
            nn.init.normal_(new_w[:, min_i:], mean=0.0, std=init_std)
        new_w.clamp_(-clip_value, clip_value)

        # Resize and copy fc2
        old_w = brain.fc2.weight
        new_w = new_brain.fc2.weight
        min_o, min_i = min(old_w.shape[0], new_w.shape[0]), min(old_w.shape[1], new_w.shape[1])
        new_w[:min_o, :min_i] = old_w[:min_o, :min_i]
        if new_w.shape[0] > old_w.shape[0]:
            nn.init.normal_(new_w[min_o:, :], mean=0.0, std=init_std)
        if new_w.shape[1] > old_w.shape[1]:
            nn.init.normal_(new_w[:, min_i:], mean=0.0, std=init_std)
        new_w.clamp_(-clip_value, clip_value)

        # Resize and copy out
        old_w = brain.out.weight
        new_w = new_brain.out.weight
        min_o, min_i = min(old_w.shape[0], new_w.shape[0]), min(old_w.shape[1], new_w.shape[1])
        new_w[:min_o, :min_i] = old_w[:min_o, :min_i]
        if new_w.shape[0] > old_w.shape[0]:
            nn.init.normal_(new_w[min_o:, :], mean=0.0, std=init_std)
        if new_w.shape[1] > old_w.shape[1]:
            nn.init.normal_(new_w[:, min_i:], mean=0.0, std=init_std)
        new_w.clamp_(-clip_value, clip_value)

    return new_brain