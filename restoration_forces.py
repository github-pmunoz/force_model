# restoration_forces.py - Integration module for existing forces.py

import numpy as np
from numba import jit
from shapely.geometry import LineString

@jit(nopython=True)
def identify_safe_vertices(original_coords, other_segment_coords, distance_threshold):
    """
    Pre-compute which vertices are "safe" based on original polygon positions.
    This is called once at the beginning and never changes.
    
    Args:
        original_coords: Original positions (n, 2)
        other_segment_coords: Other segment coordinates (m, 2)
        distance_threshold: Safe distance threshold
    
    Returns:
        safe_mask: Boolean array indicating which vertices are safe (n,)
    """
    n = original_coords.shape[0]
    safe_mask = np.zeros(n, dtype=np.bool_)
    
    for i in range(1, n-1):  # Skip endpoints
        original_pos = original_coords[i]
        
        # Find minimum distance to other segment
        min_dist = np.inf
        for j in range(other_segment_coords.shape[0]):
            other_pos = other_segment_coords[j]
            dx = original_pos[0] - other_pos[0]
            dy = original_pos[1] - other_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
        
        # Mark as safe if distance is greater than threshold
        safe_mask[i] = min_dist > distance_threshold
    
    return safe_mask

@jit(nopython=True)
def calculate_restoration_forces_precomputed(current_coords, original_coords, safe_mask, restoration_strength=1.0):
    """
    Calculate restoration forces only for pre-determined safe vertices.
    Much more efficient since we don't recalculate distances every iteration.
    
    Args:
        current_coords: Current positions (n, 2)
        original_coords: Original positions (n, 2)
        safe_mask: Boolean array indicating which vertices are safe (n,)
        restoration_strength: Force strength multiplier
    
    Returns:
        forces: Restoration forces array (n, 2)
    """
    n = current_coords.shape[0]
    forces = np.zeros((n, 2))
    
    for i in range(1, n-1):  # Skip endpoints
        if safe_mask[i]:  # Only apply to safe vertices
            current_pos = current_coords[i]
            original_pos = original_coords[i]
            
            displacement = original_pos - current_pos
            displacement_mag = np.sqrt(displacement[0]**2 + displacement[1]**2)
            
            if displacement_mag > 1e-6:
                # Force proportional to displacement
                force_magnitude = displacement_mag * restoration_strength
                force_direction = displacement / displacement_mag
                forces[i] = force_direction * force_magnitude
    
    return forces

def add_restoration_to_smoothing(old_segment, original_segment, other_segment, distance_threshold, 
                                restoration_strength=0.5, smoothing_weight=1.0, restoration_weight=0.3,
                                safe_mask=None):
    """
    Enhanced smoothing that includes restoration forces.
    This function can replace your current smoothing logic.
    
    Args:
        old_segment: Current LineString
        original_segment: Original LineString (for restoration)
        other_segment: Other LineString (for distance calculation)
        distance_threshold: Safety distance
        restoration_strength: How strong restoration forces are
        smoothing_weight: Weight for smoothing forces
        restoration_weight: Weight for restoration forces
        safe_mask: Pre-computed boolean array of safe vertices (optional, computed if None)
    
    Returns:
        new_segment: Updated LineString
        force_info: Dictionary with force information for debugging
        safe_mask: Boolean array of safe vertices (for reuse in next iteration)
    """
    # Convert to numpy arrays
    current_coords = np.array(old_segment.coords, dtype=np.float64)
    original_coords = np.array(original_segment.coords, dtype=np.float64)
    other_coords = np.array(other_segment.coords, dtype=np.float64)
    
    # Pre-compute safe vertices if not provided (only needed once)
    if safe_mask is None:
        safe_mask = identify_safe_vertices(original_coords, other_coords, distance_threshold)
    
    # Apply smoothing (your existing logic)
    from forces import simple_moving_average_fast  # Import your optimized function
    
    smoothed_x = simple_moving_average_fast(current_coords[:, 0])
    smoothed_y = simple_moving_average_fast(current_coords[:, 1])
    
    # Preserve endpoints
    smoothed_x[0] = current_coords[0, 0]
    smoothed_x[-1] = current_coords[-1, 0]
    smoothed_y[0] = current_coords[0, 1]
    smoothed_y[-1] = current_coords[-1, 1]
    
    smoothed_coords = np.column_stack((smoothed_x, smoothed_y))
    
    # Calculate restoration forces using pre-computed safe vertices
    restoration_forces = calculate_restoration_forces_precomputed(
        current_coords, original_coords, safe_mask, restoration_strength
    )
    
    # Combine forces
    smoothing_displacement = smoothed_coords - current_coords
    total_displacement = (smoothing_weight * smoothing_displacement + 
                         restoration_weight * restoration_forces)
    
    # Apply combined displacement
    new_coords = current_coords + total_displacement
    
    # Create new LineString
    new_segment = LineString(new_coords)
    
    # Return force information for debugging
    force_info = {
        'smoothing_displacement': smoothing_displacement,
        'restoration_forces': restoration_forces,
        'total_displacement': total_displacement,
        'safe_vertices': np.sum(safe_mask)
    }
    
    return new_segment, force_info, safe_mask

# Example integration into your existing main loop:
"""
To integrate into your existing forces.py, replace your smoothing section with:

# Store original segments at the beginning
original_segment_A = segment_A
original_segment_B = segment_B

# Pre-compute safe vertices once (this is the key optimization!)
safe_mask_A = identify_safe_vertices(
    np.array(original_segment_A.coords, dtype=np.float64),
    np.array(original_segment_B.coords, dtype=np.float64),
    distance_threshold
)
safe_mask_B = identify_safe_vertices(
    np.array(original_segment_B.coords, dtype=np.float64),
    np.array(original_segment_A.coords, dtype=np.float64),
    distance_threshold
)

# In your iteration loop, replace the smoothing logic with:
if hull.distance(new_A) <= 0:
    new_A, force_info_A, _ = add_restoration_to_smoothing(
        old_A, original_segment_A, old_B, distance_threshold,
        restoration_strength=0.5, restoration_weight=0.3,
        safe_mask=safe_mask_A  # Pass pre-computed safe vertices
    )
    if iteration % 50 == 0:  # Debug info every 50 iterations
        print(f"Segment A: {force_info_A['safe_vertices']} safe vertices")

if hull.distance(new_B) <= 0:
    new_B, force_info_B, _ = add_restoration_to_smoothing(
        old_B, original_segment_B, old_A, distance_threshold,
        restoration_strength=0.5, restoration_weight=0.3,
        safe_mask=safe_mask_B  # Pass pre-computed safe vertices
    )
"""

def create_safe_masks(segment_A, segment_B, distance_threshold):
    """
    Convenience function to create safe masks for both segments.
    Call this once at the beginning of your simulation.
    
    Args:
        segment_A: Original LineString A
        segment_B: Original LineString B
        distance_threshold: Safety distance
    
    Returns:
        safe_mask_A: Boolean array for segment A
        safe_mask_B: Boolean array for segment B
    """
    coords_A = np.array(segment_A.coords, dtype=np.float64)
    coords_B = np.array(segment_B.coords, dtype=np.float64)
    
    safe_mask_A = identify_safe_vertices(coords_A, coords_B, distance_threshold)
    safe_mask_B = identify_safe_vertices(coords_B, coords_A, distance_threshold)
    
    return safe_mask_A, safe_mask_B
