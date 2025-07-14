#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import time
from numba import jit

@jit(nopython=True)
def simple_moving_average_fast(arr, window_size=20):
    """
    Numba optimized simple moving average (faster alternative to pure Python)
    """
    n = len(arr)
    smoothed = np.zeros(n)
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        count = end - start
        sum_val = 0.0
        for j in range(start, end):
            sum_val += arr[j]
        smoothed[i] = sum_val / count
    
    return smoothed

@jit(nopython=True)
def calculate_restoration_forces(current_coords, original_coords, other_segment_coords, distance_threshold, restoration_strength=1.0):
    """
    Calculate restoration forces for vertices that are "safe" (far from other segment).
    Safe vertices get a force pulling them back to their original position.
    Force strength increases with distance from original position.
    
    Args:
        current_coords: Current positions of vertices (n, 2)
        original_coords: Original positions of vertices (n, 2) 
        other_segment_coords: Coordinates of the other segment (m, 2)
        distance_threshold: Minimum safe distance
        restoration_strength: Multiplier for restoration force strength
    
    Returns:
        forces: Array of restoration forces (n, 2)
    """
    n = current_coords.shape[0]
    forces = np.zeros((n, 2))
    
    for i in range(n):
        current_pos = current_coords[i]
        original_pos = original_coords[i]
        
        # Check if this vertex is "safe" (far enough from other segment)
        min_distance_to_other = np.inf
        for j in range(other_segment_coords.shape[0]):
            other_pos = other_segment_coords[j]
            dx = current_pos[0] - other_pos[0]
            dy = current_pos[1] - other_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_distance_to_other:
                min_distance_to_other = dist
        
        # If vertex is safe (far from other segment), apply restoration force
        if min_distance_to_other > distance_threshold:
            # Calculate displacement from original position
            displacement = original_pos - current_pos
            displacement_magnitude = np.sqrt(displacement[0]**2 + displacement[1]**2)
            
            # Force strength increases with displacement (stronger pull when farther away)
            if displacement_magnitude > 1e-6:  # Avoid division by zero
                # Normalized direction towards original position
                force_direction = displacement / displacement_magnitude
                # Force magnitude proportional to displacement
                force_magnitude = displacement_magnitude * restoration_strength
                forces[i] = force_direction * force_magnitude
    
    return forces

@jit(nopython=True) 
def apply_combined_forces(coords, smoothing_forces, restoration_forces, smoothing_weight=1.0, restoration_weight=0.5, step_size=0.01):
    """
    Apply combined smoothing and restoration forces to coordinates
    
    Args:
        coords: Current coordinates (n, 2)
        smoothing_forces: Forces from smoothing/laplacian (n, 2)
        restoration_forces: Forces pulling towards original positions (n, 2)
        smoothing_weight: Weight for smoothing forces
        restoration_weight: Weight for restoration forces
        step_size: Integration step size
        
    Returns:
        new_coords: Updated coordinates (n, 2)
    """
    n = coords.shape[0]
    new_coords = np.zeros((n, 2))
    
    for i in range(n):
        # Skip first and last points (keep endpoints fixed)
        if i == 0 or i == n - 1:
            new_coords[i] = coords[i]
            continue
            
        # Combine forces
        total_force = (smoothing_weight * smoothing_forces[i] + 
                      restoration_weight * restoration_forces[i])
        
        # Apply force with step size
        new_coords[i] = coords[i] + total_force * step_size
    
    return new_coords

def enhanced_force_model(current_segment, original_segment, other_segment, distance_threshold, 
                        restoration_strength=1.0, smoothing_window=20):
    """
    Enhanced force model that combines smoothing with restoration forces
    
    Args:
        current_segment: Current LineString
        original_segment: Original LineString (for restoration targets)
        other_segment: Other LineString (for safety distance calculation)
        distance_threshold: Minimum safe distance
        restoration_strength: Strength of restoration forces
        smoothing_window: Window size for smoothing
        
    Returns:
        smoothing_forces: Forces from smoothing operation
        restoration_forces: Forces pulling towards original positions
    """
    # Convert to numpy arrays
    current_coords = np.array(current_segment.coords, dtype=np.float64)
    original_coords = np.array(original_segment.coords, dtype=np.float64)
    other_coords = np.array(other_segment.coords, dtype=np.float64)
    
    # Calculate smoothing forces (approximated as displacement from smoothed position)
    smoothed_x = simple_moving_average_fast(current_coords[:, 0], smoothing_window)
    smoothed_y = simple_moving_average_fast(current_coords[:, 1], smoothing_window)
    
    # Preserve endpoints in smoothing
    smoothed_x[0] = current_coords[0, 0]
    smoothed_x[-1] = current_coords[-1, 0]
    smoothed_y[0] = current_coords[0, 1]
    smoothed_y[-1] = current_coords[-1, 1]
    
    smoothed_coords = np.column_stack((smoothed_x, smoothed_y))
    smoothing_forces = smoothed_coords - current_coords
    
    # Calculate restoration forces
    restoration_forces = calculate_restoration_forces(
        current_coords, original_coords, other_coords, 
        distance_threshold, restoration_strength
    )
    
    return smoothing_forces, restoration_forces

# Copy existing functions from forces.py (keeping the core structure)
def regularizing_function(n):
    """creates a normal distribution with peak at n/2"""
    cutoff = 0.1
    if n <= 0:
        raise ValueError("n must be a positive integer")    
    
    x = np.arange(n)
    half = n / 2
    regularizing_func = np.exp(-np.abs((x - half)/(n/4))**2)
    regularizing_func -= np.min(regularizing_func)
    regularizing_func /= np.max(regularizing_func)
    return regularizing_func

def create_ellipse(a, b, cx, cy, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points)
    amplitude = 0.5
    frequency = 6
    x = cx + (a + amplitude * np.sin(frequency * theta)) * np.cos(theta)
    y = cy + (b + amplitude * np.sin(frequency * theta)) * np.sin(theta)
    return Polygon(zip(x, y))

def find_infracting_segments(polygonA, polygonB, distance_threshold):
    start = None
    end = None
    for i in range(len(polygonA.exterior.coords)):
        p1 = Point(polygonA.exterior.coords[i])
        p2 = Point(polygonA.exterior.coords[(i + 1) % len(polygonA.exterior.coords)])
        p1_is_infracting = polygonB.distance(p1) < distance_threshold
        p2_is_infracting = polygonB.distance(p2) < distance_threshold
        if(p1_is_infracting and not p2_is_infracting):
            if end is None:
                end = i
        if(not p1_is_infracting and p2_is_infracting):
            if start is None:
                start = i
    return (start, end)

def get_segment_length(polygon, start, end):
    num_points = len(polygon.exterior.coords)
    if start < 0 or end >= num_points:
        raise ValueError("Start or end index out of bounds.")
    if start <= end:
        segment = polygon.exterior.coords[start:end + 1]
    else:
        segment = polygon.exterior.coords[start:] + polygon.exterior.coords[:end + 1]
    return len(segment)

def get_pre_post_segment(polygonA, polygonB, distance_threshold):
    indices_A = find_infracting_segments(polygonA, polygonB, distance_threshold)
    length_A = get_segment_length(polygonA, indices_A[0], indices_A[1])
    if indices_A[0] is not None and indices_A[1] is not None:
        pre = int(indices_A[0] - length_A / 2)
        post = int(indices_A[1] + length_A / 2)
    else:
        pre = 0
        post = 0
    num_points = len(polygonA.exterior.coords)
    if pre <= post:
        segment_points = [polygonA.exterior.coords[i % num_points] for i in range(pre, post + 1)]
    else:
        segment_points = [polygonA.exterior.coords[i % num_points] for i in range(pre, pre + (post - pre + num_points) + 1)]
    return LineString(segment_points)

def find_close_points(polygonA, polygonB, distance_threshold):
    close_points = []
    for point in polygonA.exterior.coords:
        p = Point(point)
        if polygonB.distance(p) < distance_threshold:
            close_points.append(p)
    for point in polygonB.exterior.coords:
        p = Point(point)
        if polygonA.distance(p) < distance_threshold:
            close_points.append(p)
    return close_points

def convex_hull(points):
    if len(points) < 3:
        return Polygon(points)
    return Polygon(points).convex_hull

def main():
    # Ellipse parameters
    a1, b1, cx1, cy1 = 24, 11, 0, 0
    a2, b2, cx2, cy2 = 15, 20, 40, -10
    distance_threshold = 5.0

    # Create ellipses
    ellipse1 = create_ellipse(a1, b1, cx1, cy1)
    ellipse2 = create_ellipse(a2, b2, cx2, cy2)

    # Find close points and create segments
    close_points = find_close_points(ellipse1, ellipse2, distance_threshold)
    segment_A = get_pre_post_segment(ellipse1, ellipse2, distance_threshold)
    segment_B = get_pre_post_segment(ellipse2, ellipse1, distance_threshold)

    # Store original segments for restoration forces
    original_segment_A = segment_A
    original_segment_B = segment_B
    
    # Current working segments
    current_A = segment_A
    current_B = segment_B

    # Enhanced simulation parameters
    max_iterations = 500
    restoration_strength = 0.8  # How strong the pull back to original position is
    smoothing_weight = 1.0      # Weight for smoothing forces
    restoration_weight = 0.3    # Weight for restoration forces
    step_size = 0.02           # Integration step size

    print(f"Starting enhanced simulation with {len(current_A.coords)} and {len(current_B.coords)} points")
    print(f"Parameters: restoration_strength={restoration_strength}, restoration_weight={restoration_weight}")

    total_time = 0.0
    iteration = 0  # Initialize iteration counter
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Calculate forces for segment A
        smoothing_forces_A, restoration_forces_A = enhanced_force_model(
            current_A, original_segment_A, current_B, distance_threshold, restoration_strength
        )
        
        # Calculate forces for segment B  
        smoothing_forces_B, restoration_forces_B = enhanced_force_model(
            current_B, original_segment_B, current_A, distance_threshold, restoration_strength
        )
        
        # Apply combined forces
        coords_A = np.array(current_A.coords, dtype=np.float64)
        coords_B = np.array(current_B.coords, dtype=np.float64)
        
        new_coords_A = apply_combined_forces(
            coords_A, smoothing_forces_A, restoration_forces_A,
            smoothing_weight, restoration_weight, step_size
        )
        
        new_coords_B = apply_combined_forces(
            coords_B, smoothing_forces_B, restoration_forces_B, 
            smoothing_weight, restoration_weight, step_size
        )
        
        # Update segments
        current_A = LineString(new_coords_A)
        current_B = LineString(new_coords_B)
        
        iteration_time = time.time() - start_time
        total_time += iteration_time
        
        # Check convergence (simplified)
        if iteration % 50 == 0:
            min_dist = current_A.distance(current_B)
            print(f"Iteration {iteration}: min_distance = {min_dist:.3f}, time = {iteration_time:.4f}s")
            
            if min_dist > distance_threshold * 1.2:  # Some margin above threshold
                print(f"Converged at iteration {iteration}")
                break

    print(f"Total simulation time: {total_time:.4f} seconds")
    print(f"Average time per iteration: {total_time/max(1, iteration+1):.6f} seconds")

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original configuration
    ax = axs[0, 0]
    x1, y1 = ellipse1.exterior.xy
    x2, y2 = ellipse2.exterior.xy
    ax.plot(x1, y1, 'b-', label='Ellipse 1', alpha=0.7)
    ax.plot(x2, y2, 'b-', label='Ellipse 2', alpha=0.7)
    ax.plot(*segment_A.xy, 'r-', linewidth=3, label='Original Segment A')
    ax.plot(*segment_B.xy, 'r-', linewidth=3, label='Original Segment B')
    ax.set_title('Original Configuration')
    ax.axis('equal')
    ax.legend()
    
    # Final result
    ax = axs[0, 1]
    ax.plot(x1, y1, 'b--', alpha=0.3, label='Original Ellipses')
    ax.plot(x2, y2, 'b--', alpha=0.3)
    ax.plot(*current_A.xy, 'g-', linewidth=3, label='Final Segment A')
    ax.plot(*current_B.xy, 'g-', linewidth=3, label='Final Segment B')
    ax.plot(*segment_A.xy, 'r:', alpha=0.5, label='Original Segments')
    ax.plot(*segment_B.xy, 'r:', alpha=0.5)
    ax.set_title('Final Result with Restoration Forces')
    ax.axis('equal')
    ax.legend()
    
    # Force visualization for final iteration
    ax = axs[1, 0]
    smoothing_forces_A, restoration_forces_A = enhanced_force_model(
        current_A, original_segment_A, current_B, distance_threshold, restoration_strength
    )
    coords_A = np.array(current_A.coords)
    
    # Plot segments
    ax.plot(*current_A.xy, 'g-', linewidth=2, label='Current Segment A')
    ax.plot(*original_segment_A.xy, 'r:', alpha=0.5, label='Original Segment A')
    
    # Plot forces as arrows (subsample for clarity)
    step = max(1, len(coords_A) // 20)
    for i in range(0, len(coords_A), step):
        if i == 0 or i == len(coords_A)-1:
            continue  # Skip endpoints
        pos = coords_A[i]
        smooth_force = smoothing_forces_A[i] * 20  # Scale for visibility
        restore_force = restoration_forces_A[i] * 20  # Scale for visibility
        
        # Smoothing force (blue arrows)
        if np.linalg.norm(smooth_force) > 0.1:
            ax.arrow(pos[0], pos[1], smooth_force[0], smooth_force[1], 
                    head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.7)
        
        # Restoration force (red arrows)
        if np.linalg.norm(restore_force) > 0.1:
            ax.arrow(pos[0], pos[1], restore_force[0], restore_force[1],
                    head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.7)
    
    ax.set_title('Force Visualization\n(Blue: Smoothing, Red: Restoration)')
    ax.axis('equal')
    ax.legend()
    
    # Distance analysis
    ax = axs[1, 1]
    original_coords_A = np.array(original_segment_A.coords)
    current_coords_A = np.array(current_A.coords)
    
    # Calculate displacement from original positions
    displacements = np.linalg.norm(current_coords_A - original_coords_A, axis=1)
    vertex_indices = np.arange(len(displacements))
    
    ax.plot(vertex_indices, displacements, 'o-', label='Displacement from Original')
    ax.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold ({distance_threshold})')
    ax.set_xlabel('Vertex Index')
    ax.set_ylabel('Displacement')
    ax.set_title('Vertex Displacement Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
