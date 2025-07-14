# optimized_forces.py
import numpy as np
import time
from numba import jit, njit, prange
from shapely.geometry import LineString

@njit(parallel=True, fastmath=True)
def smooth_coordinates_optimized(coords, window_size=20):
    """
    Highly optimized coordinate smoothing using Numba with parallel processing
    """
    n = coords.shape[0]
    smoothed = np.zeros_like(coords)
    half_window = window_size // 2
    
    # Process x and y coordinates in parallel
    for dim in prange(2):  # 0 for x, 1 for y
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            sum_val = 0.0
            count = end - start
            
            for j in range(start, end):
                sum_val += coords[j, dim]
            
            smoothed[i, dim] = sum_val / count
    
    # Preserve endpoints
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    
    return smoothed

@njit(fastmath=True)
def distance_to_hull_fast(point, hull_coords):
    """
    Fast approximate distance calculation to convex hull
    """
    min_dist = np.inf
    for i in range(len(hull_coords)):
        dx = point[0] - hull_coords[i, 0]
        dy = point[1] - hull_coords[i, 1]
        dist = dx*dx + dy*dy  # Skip sqrt for comparison
        if dist < min_dist:
            min_dist = dist
    return np.sqrt(min_dist)

@njit(fastmath=True)
def check_hull_intersection_fast(coords, hull_coords, threshold=0.1):
    """
    Fast check if any coordinate is inside/near the hull
    """
    for i in range(len(coords)):
        if distance_to_hull_fast(coords[i], hull_coords) <= threshold:
            return True
    return False

def optimized_smoothing_loop(segment_A, segment_B, hull, max_iterations=1000):
    """
    Optimized version of the smoothing loop
    """
    # Pre-convert to numpy arrays to avoid repeated conversions
    coords_A = np.array(segment_A.coords, dtype=np.float64)
    coords_B = np.array(segment_B.coords, dtype=np.float64)
    hull_coords = np.array(hull.exterior.coords, dtype=np.float64) if hasattr(hull, 'exterior') else np.array([[0, 0]])
    
    # Pre-allocate arrays to avoid memory allocations
    smoothed_A = np.zeros_like(coords_A)
    smoothed_B = np.zeros_like(coords_B)
    
    total_time = 0.0
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Check if smoothing is needed (fast approximate check)
        need_smooth_A = check_hull_intersection_fast(coords_A, hull_coords)
        need_smooth_B = check_hull_intersection_fast(coords_B, hull_coords)
        
        if need_smooth_A:
            coords_A = smooth_coordinates_optimized(coords_A, window_size=20)
        
        if need_smooth_B:
            coords_B = smooth_coordinates_optimized(coords_B, window_size=20)
            # Preserve more endpoints for stability
            original_B = np.array(segment_B.coords, dtype=np.float64)
            for i in range(min(3, len(coords_B))):
                coords_B[i] = original_B[i]
                coords_B[-(i+1)] = original_B[-(i+1)]
        
        total_time += time.time() - start_time
        
        # Check convergence with fast distance approximation
        if not need_smooth_A and not need_smooth_B:
            print(f"Converged at iteration {iteration}")
            break
    
    return LineString(coords_A), LineString(coords_B), total_time

def benchmark_approaches(segment_A, segment_B, hull):
    """
    Benchmark different optimization approaches
    """
    print("Benchmarking different approaches...")
    
    # Test optimized version
    start_time = time.time()
    new_A, new_B, smoothing_time = optimized_smoothing_loop(segment_A, segment_B, hull, 100)
    total_time = time.time() - start_time
    
    print(f"Optimized approach:")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Smoothing time: {smoothing_time:.4f} seconds")
    print(f"  Overhead: {total_time - smoothing_time:.4f} seconds")
    
    return new_A, new_B
