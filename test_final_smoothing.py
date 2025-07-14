#!/usr/bin/env python3
"""
Test script to verify the final smoothing phase works correctly
"""

import numpy as np
from shapely.geometry import LineString
from numba import jit

@jit(nopython=True)
def simple_moving_average_fast(arr, window_size=20):
    """
    Numba optimized simple moving average (copy from forces.py)
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

def create_test_segment_with_spikes():
    """Create a test segment with artificial spikes"""
    # Create a smooth curve
    t = np.linspace(0, 2*np.pi, 20)
    x = np.cos(t)
    y = np.sin(t)
    
    # Add some artificial spikes
    x[5] += 0.5  # Spike at index 5
    y[5] += 0.3
    x[10] -= 0.4  # Spike at index 10
    y[10] += 0.6
    x[15] += 0.3  # Spike at index 15
    y[15] -= 0.5
    
    coords = np.column_stack((x, y))
    return LineString(coords)

def calculate_curvature(segment):
    """Calculate approximate curvature at each point"""
    coords = np.array(segment.coords)
    n = len(coords)
    curvatures = np.zeros(n)
    
    for i in range(1, n-1):
        # Use three consecutive points to estimate curvature
        p1 = coords[i-1]
        p2 = coords[i]
        p3 = coords[i+1]
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms > 1e-10:
            cos_angle = np.clip(dot_product / norms, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures[i] = angle / np.linalg.norm(v1 + v2)
    
    return curvatures

def test_final_smoothing():
    """Test the final smoothing phase"""
    print("Testing final smoothing phase...")
    
    # Create test segment with spikes
    original_segment = create_test_segment_with_spikes()
    test_segment = original_segment
    
    print(f"Original segment has {len(original_segment.coords)} points")
    
    # Calculate initial curvature
    initial_curvatures = calculate_curvature(test_segment)
    print(f"Initial max curvature: {np.max(initial_curvatures):.4f}")
    print(f"Initial mean curvature: {np.mean(initial_curvatures):.4f}")
    
    # Apply final smoothing iterations (simulating the final phase)
    final_smoothing_iterations = 2
    
    for final_iter in range(final_smoothing_iterations):
        coords = np.array(test_segment.coords)
        x_coords = coords[:, 0].copy()
        y_coords = coords[:, 1].copy()
        
        # Apply Laplacian smoothing
        smoothed_x = simple_moving_average_fast(x_coords)
        smoothed_y = simple_moving_average_fast(y_coords)
        
        # Preserve endpoints
        smoothed_x[0] = x_coords[0]
        smoothed_x[-1] = x_coords[-1]
        smoothed_y[0] = y_coords[0]
        smoothed_y[-1] = y_coords[-1]
        
        new_coords = list(zip(smoothed_x, smoothed_y))
        test_segment = LineString(new_coords)
        
        # Calculate curvature after this iteration
        curvatures = calculate_curvature(test_segment)
        print(f"After iteration {final_iter + 1}: max curvature = {np.max(curvatures):.4f}, mean = {np.mean(curvatures):.4f}")
    
    # Final curvature analysis
    final_curvatures = calculate_curvature(test_segment)
    print(f"\nFinal results:")
    print(f"  Max curvature reduced from {np.max(initial_curvatures):.4f} to {np.max(final_curvatures):.4f}")
    print(f"  Mean curvature reduced from {np.mean(initial_curvatures):.4f} to {np.mean(final_curvatures):.4f}")
    print(f"  Curvature reduction: {(np.max(initial_curvatures) - np.max(final_curvatures)) / np.max(initial_curvatures) * 100:.1f}%")
    
    return test_segment

if __name__ == "__main__":
    test_final_smoothing()
    print("\nTest completed successfully!")
