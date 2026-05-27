import numpy as np
import time
from class_animal_brain_nn import *

FOOD_LABEL = 1.0
CONSPECIFIC_LABEL = 0.1
PREDATOR_LABEL = -1.0
EMPTY_LABEL = -0.5

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

def find_nearest_visible_target(
    observer_positions,    # (N, 2) — positions of N observers
    observer_angles,       # (N,) — heading angles in radians
    target_positions,      # (T, 2) — positions of T potential targets
    vision_range,          # scalar
    vision_fov,            # scalar, radians
    exclude_self=False     # bool, whether to exclude targets at the same position as observers (for conspecific detection)
):
    """
    For each observer, find the nearest visible target.
    
    Args:
        observer_positions: (N, 2)
        observer_angles: (N,) heading in radians
        target_positions: (T, 2)
        vision_range: scalar
        vision_fov: scalar, full field of view in radians
    
    Returns:
        norm_dist: (N,) — normalised distance 1 - dist/vision_range, or -1 if nothing visible
        norm_angle: (N,) — normalised angle relative to heading, in range [-1, 1], or 0 if nothing visible
    """
    N = observer_positions.shape[0]
    half_fov = vision_fov / 2.0
 
    norm_dist  = np.full(N, -1.0, dtype=np.float32)
    norm_angle = np.zeros(N, dtype=np.float32)
 
    if target_positions.shape[0] == 0:
        return norm_dist, norm_angle
 
    # Pairwise differences: (N, T, 2)
    diffs = target_positions[np.newaxis, :, :] - observer_positions[:, np.newaxis, :]
 
    # Pairwise distances: (N, T)
    dists = np.linalg.norm(diffs, axis=2)
 
    # Pairwise angles relative to each observer's heading: (N, T)
    abs_angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])
    rel_angles = abs_angles - observer_angles[:, np.newaxis]
    
    # Wrap to [-pi, pi]
    rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi
 
    # Visibility: within range and within FOV
    visible = (dists <= vision_range) & (np.abs(rel_angles) <= half_fov)
    
    if exclude_self:
        np.fill_diagonal(visible, False)
 
    # Mask out non-visible with inf so argmin ignores them
    masked_dists = np.where(visible, dists, np.inf)
 
    # Find nearest visible target for each observer
    nearest_idx = np.argmin(masked_dists, axis=1)  # (N,)
    nearest_dist = masked_dists[np.arange(N), nearest_idx]  # (N,)
 
    # Check which observers found anything
    any_visible = nearest_dist < np.inf
 
    # Normalised distance: 1 - dist/vision_range, or -1 if nothing found
    norm_dist = np.where(
        any_visible,
        1.0 - nearest_dist / vision_range,
        -1.0
    ).astype(np.float32)
 
    # Normalised angle: relative angle / half_fov, clamped to [-1, 1], or 0 if nothing found
    nearest_angle = rel_angles[np.arange(N), nearest_idx]  # (N,)
    norm_angle = np.where(
        any_visible,
        np.clip(nearest_angle / half_fov, -1.0, 1.0),
        0.0
    ).astype(np.float32)
 
    return norm_dist, norm_angle

def herbivores_perception_function(
    self_positions,       # (N, 2)
    self_angles,          # (N,)
    food_positions,       # (F, 2) — already filtered to alive plants
    predator_positions,   # (P, 2) — already filtered to alive predators
    vision_range,         # scalar
    vision_fov,           # scalar, radians
):
    """
    Compute perception inputs for N herbivores.
    
    Returns array of shape (N, 6):
        [dist_plant, angle_plant,
         dist_conspecific, angle_conspecific,
         dist_predator, angle_predator]
    
    dist values: 1 - dist/vision_range if detected, -1 if nothing seen
    angle values: relative to heading, normalised to [-1, 1] over FOV
                  0.0 if nothing seen
    """
    N = self_positions.shape[0]
    output = np.zeros((N, 6), dtype=np.float32)
 
    # Plants
    d, a = find_nearest_visible_target(
        self_positions, self_angles, food_positions,
        vision_range, vision_fov
    )
    output[:, 0] = d
    output[:, 1] = a
 
    # Conspecifics (exclude self)
    d, a = find_nearest_visible_target(
        self_positions, self_angles, self_positions,
        vision_range, vision_fov,
        exclude_self=True
    )
    output[:, 2] = d
    output[:, 3] = a
 
    # Predators
    d, a = find_nearest_visible_target(
        self_positions, self_angles, predator_positions,
        vision_range, vision_fov
    )
    output[:, 4] = d
    output[:, 5] = a
 
    return output


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
    input_dim = brain.fc1.in_features
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
        n_external_infos=input_dim,
        n_self_infos=0,
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