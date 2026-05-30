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

def resize_layer_in_animal_brain(
    brain: AnimalBrain,
    layer: str,
    new_size: int,
    init_std: float = 0.1,
) -> AnimalBrain:
    """
    Returns a new AnimalBrain with the specified layer resized.
    When shrinking, neurons to remove are chosen randomly.
    When growing, new neurons are randomly initialised.
    The corresponding input columns in the next layer are updated to match.
    """
    input_dim = brain.fc1.in_features
    old_h1    = brain.fc1.out_features
    old_h2    = brain.fc2.out_features

    new_h1 = new_size if layer == "fc1" else old_h1
    new_h2 = new_size if layer == "fc2" else old_h2

    if layer not in ("fc1", "fc2"):
        raise ValueError("Layer must be 'fc1' or 'fc2'")

    new_brain = AnimalBrain(
        n_external_infos=input_dim,
        n_self_infos=0,
        hidden_dim_1=new_h1,
        hidden_dim_2=new_h2
    )

    with torch.no_grad():

        # --- Which neurons survive? ---
        # keep_h1: indices of fc1 output neurons to keep (relevant when fc1 shrinks)
        # keep_h2: indices of fc2 output neurons to keep (relevant when fc2 shrinks)

        if new_h1 < old_h1:
            keep_h1 = np.sort(np.random.choice(old_h1, new_h1, replace=False))
        else:
            keep_h1 = np.arange(old_h1)

        if new_h2 < old_h2:
            keep_h2 = np.sort(np.random.choice(old_h2, new_h2, replace=False))
        else:
            keep_h2 = np.arange(old_h2)

        # ── fc1 ─────────────────────────────────────────────────────────────
        # weight shape: (h1, input_dim)  — rows = output neurons

        old_w = brain.fc1.weight.data
        if new_h1 <= old_h1:
            new_brain.fc1.weight.data[:, :] = old_w[keep_h1, :]
        else:
            # Growing: copy all old rows, init new rows
            new_brain.fc1.weight.data[:old_h1, :] = old_w
            nn.init.normal_(new_brain.fc1.weight.data[old_h1:, :], std=init_std)

        # bias shape: (h1,) — same indexing as rows
        if brain.fc1.bias is not None:
            old_b = brain.fc1.bias.data
            if new_h1 <= old_h1:
                new_brain.fc1.bias.data[:] = old_b[keep_h1]
            else:
                new_brain.fc1.bias.data[:old_h1] = old_b
                nn.init.normal_(new_brain.fc1.bias.data[old_h1:], std=init_std)

        # ── fc2 ─────────────────────────────────────────────────────────────
        # weight shape: (h2, h1) — rows = output neurons, cols = inputs from fc1

        old_w = brain.fc2.weight.data  # (old_h2, old_h1)

        # Step 1: trim/expand columns to match new fc1 output size
        if new_h1 <= old_h1:
            old_w = old_w[:, keep_h1]          # (old_h2, new_h1)
        # if fc1 grew, new columns will be inited below

        # Step 2: trim/expand rows for fc2 resize
        min_h2     = min(old_h2, new_h2)
        min_h1_cols = min(old_h1, new_h1)

        if new_h2 <= old_h2:
            new_brain.fc2.weight.data[:, :min_h1_cols] = old_w[keep_h2, :]
        else:
            new_brain.fc2.weight.data[:old_h2, :min_h1_cols] = old_w
            nn.init.normal_(new_brain.fc2.weight.data[old_h2:, :min_h1_cols], std=init_std)

        # If fc1 grew: init the new input columns in fc2
        if new_h1 > old_h1:
            nn.init.normal_(new_brain.fc2.weight.data[:min_h2, old_h1:], std=init_std)


        # bias shape: (h2,)
        if brain.fc2.bias is not None:
            old_b = brain.fc2.bias.data
            if new_h2 <= old_h2:
                new_brain.fc2.bias.data[:] = old_b[keep_h2]
            else:
                new_brain.fc2.bias.data[:old_h2] = old_b
                nn.init.normal_(new_brain.fc2.bias.data[old_h2:], std=init_std)

        # ── out ──────────────────────────────────────────────────────────────
        # weight shape: (output_dim=2, h2) — cols = inputs from fc2

        old_w = brain.out.weight.data  # (2, old_h2)

        if new_h2 <= old_h2:
            new_brain.out.weight.data[:, :] = old_w[:, keep_h2]
        else:
            new_brain.out.weight.data[:, :old_h2] = old_w
            nn.init.normal_(new_brain.out.weight.data[:, old_h2:], std=init_std)

        # bias shape: (output_dim=2,) — always same size, always copy directly
        if brain.out.bias is not None:
            new_brain.out.bias.data[:] = brain.out.bias.data

    return new_brain