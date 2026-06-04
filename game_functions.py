import numpy as np
from class_animal_brain_nn import *

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
        norm_dist: (N,) — normalised distance 1 - dist/vision_range, or 0 if nothing visible
        norm_angle: (N,) — normalised angle relative to heading, in range [-1, 1], or 0 if nothing visible
    """
    N = observer_positions.shape[0]
    half_fov = vision_fov / 2.0
 
    norm_dist  = np.full(N, 0.0, dtype=np.float32)
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
 
    # Normalised distance: 1 - dist/vision_range, or 0 if nothing found
    norm_dist = np.where(
        any_visible,
        1.0 - nearest_dist / vision_range,
        0.0
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

def predators_perception_function(
    self_positions,       # (N, 2)
    self_angles,          # (N,)
    food_positions,       # (F, 2) — already filtered to alive herbivores
    vision_range,         # scalar
    vision_fov,           # scalar, radians
):
    """
    Compute perception inputs for N predators.
    
    Returns array of shape (N, 4):
        [dist_herbivore, angle_herbivore,
         dist_conspecific, angle_conspecific]
    
    dist values: 1 - dist/vision_range if detected, 0 if nothing seen
    angle values: relative to heading, normalised to [-1, 1] over FOV
                  0.0 if nothing seen
    """

    N = self_positions.shape[0]
    output = np.zeros((N, 4), dtype=np.float32)
 
    # Herbivores (food for predators)
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
 
    return output

def calculate_brain_similarity(brain_1: AnimalBrain, brain_2: AnimalBrain):
    total_diff = 0.0
    total_params = 0

    for b1_param, b2_param in zip(brain_1.parameters(),brain_2.parameters()):

        diff = b2_param.data - b1_param.data

        total_diff += torch.sum(diff * diff).item()
        total_params += diff.numel()

    return np.sqrt(total_diff / total_params) 

def mutate_brain_architecture(
    brain: AnimalBrain,
    global_mutation_rate: float,
    min_size: int = 1,
    max_size: int = 20,
    init_std: float = 0.1
) -> tuple[AnimalBrain, float]:
    """
    Handles neuron addition/subtraction, layer addition, and layer deletion.
    Guarantees that neuron deletion targets random indices safely without index crashes.
    """
    hidden_dims = list(brain.get_dim_sizes())
    structural_change_score = 0.0
    species = brain.species
    
    # 1. Determine New Dimensions and explicitly save lists of surviving indices
    # We keep track of the *exact* original indices that survived for each layer
    surviving_neuron_maps = []
    new_dims = []
    
    for dim in hidden_dims:
        if np.random.rand() <= global_mutation_rate:
            change = np.random.choice([-1, 1])
            new_dim = dim + change
            structural_change_score += 0.15
            
            if new_dim >= min_size:
                final_dim = min(new_dim, max_size)
                new_dims.append(final_dim)
                
                if final_dim < dim:
                    # Randomly choose which indices survive this layer's shrinking
                    indices_to_keep = np.sort(np.random.choice(dim, final_dim, replace=False))
                    surviving_neuron_maps.append(indices_to_keep)
                else:
                    # Grew or stayed same -> all original indices survive
                    surviving_neuron_maps.append(np.arange(dim))
            else:
                structural_change_score += 0.15 #if layer deleted
        else:
            new_dims.append(dim)
            surviving_neuron_maps.append(np.arange(dim))
            
    # 2. Randomly Mutate Layer Count (Sprouting a brand new layer)
    if np.random.rand() <= (global_mutation_rate): 
        new_dims.append(np.random.randint(2, 6))
        surviving_neuron_maps.append(np.arange(new_dims[-1]))
        structural_change_score += 0.3

    # Create the new shell brain architecture
    new_brain = AnimalBrain(
        n_external_infos=brain.input_dim,
        n_self_infos=0, 
        hidden_dims=new_dims,
        initial_weight_std=init_std,
        species=species
    )
    
    # 3. Transfer Weights step-by-step
    with torch.no_grad():
        old_layers = list(brain.layers)
        new_layers = list(new_brain.layers)
        
        # We will match the input/output sizes layer by layer
        # 'prev_keep_cols' keeps track of which neurons survived the *previous* layer's structural changes
        prev_keep_cols = np.arange(brain.input_dim) 
        
        # Process all hidden layers that can be matched up
        num_hidden_to_copy = min(len(old_layers), len(new_layers))
        for i in range(num_hidden_to_copy):
            old_l = old_layers[i]
            new_l = new_layers[i]
            
            # Rows = output neurons of current layer
            current_keep_rows = surviving_neuron_maps[i]
            
            # Safeguard limits so we never grab slices larger than what physically exists in the new/old shapes
            max_rows = min(len(current_keep_rows), old_l.weight.shape[0], new_l.weight.shape[0])
            max_cols = min(len(prev_keep_cols), old_l.weight.shape[1], new_l.weight.shape[1])
            
            selected_rows = current_keep_rows[:max_rows]
            selected_cols = prev_keep_cols[:max_cols]
            
            # Map the checkerboard matrix indices safely
            grid_indices = np.ix_(selected_rows, selected_cols)
            new_l.weight[:max_rows, :max_cols] = old_l.weight[grid_indices]
            
            if new_l.bias is not None and old_l.bias is not None:
                new_l.bias[:max_rows] = old_l.bias[selected_rows]
                
            # Update history for the next layer's inputs
            prev_keep_cols = current_keep_rows
            
        # 4. Handle the final output connection layer ('brain.out')
        # This prevents the exact bug you ran into when layers shrink or disappear!
        old_out = brain.out
        new_out = new_brain.out
        
        # The output layer always keeps both of its output dimensions (speed, turning)
        max_rows = min(old_out.weight.shape[0], new_out.weight.shape[0]) # usually 2
        max_cols = min(len(prev_keep_cols), old_out.weight.shape[1], new_out.weight.shape[1])
        
        selected_cols = prev_keep_cols[:max_cols]
        selected_rows = np.arange(max_rows)
        
        grid_indices = np.ix_(selected_rows, selected_cols)
        new_out.weight[:max_rows, :max_cols] = old_out.weight[grid_indices]
        
        if new_out.bias is not None and old_out.bias is not None:
            new_out.bias[:max_rows] = old_out.bias[:max_rows]
                
    return new_brain, structural_change_score

def to_json_compatible(obj):
        # Recursively convert numpy types and arrays to native Python types
        if isinstance(obj, dict):
            return {k: to_json_compatible(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_compatible(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_json_compatible(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj