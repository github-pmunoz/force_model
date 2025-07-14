# forces_optimized.py
import numpy as np
import time
from numba import jit

@jit(nopython=True)
def simple_moving_average_numba(arr, window_size=20):
    """
    Numba optimized simple moving average (alternative to Cython)
    """
    n = len(arr)
    smoothed = np.zeros(n)
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        smoothed[i] = np.mean(arr[start:end])
    
    return smoothed

@jit(nopython=True)
def smooth_coordinates_numba(coords, window_size=20):
    """
    Numba optimized coordinate smoothing
    """
    n = coords.shape[0]
    smoothed_coords = np.zeros((n, 2))
    
    # Smooth x coordinates
    smoothed_coords[:, 0] = simple_moving_average_numba(coords[:, 0], window_size)
    # Smooth y coordinates  
    smoothed_coords[:, 1] = simple_moving_average_numba(coords[:, 1], window_size)
    
    # Preserve endpoints
    smoothed_coords[0] = coords[0]
    smoothed_coords[-1] = coords[-1]
    
    return smoothed_coords

def time_smoothing_operations(coords_A, coords_B, iterations=100):
    """
    Time different smoothing approaches
    """
    print(f"Timing smoothing operations with {len(coords_A)} points over {iterations} iterations...")
    
    # Time original Python version
    start_time = time.time()
    for _ in range(iterations):
        # Original approach
        x_coords_A = coords_A[:, 0].copy()
        y_coords_A = coords_A[:, 1].copy()
        from forces import simple_moving_average
        smoothed_x_A = simple_moving_average(x_coords_A)
        smoothed_y_A = simple_moving_average(y_coords_A)
        smoothed_x_A[0] = x_coords_A[0]
        smoothed_x_A[-1] = x_coords_A[-1]
        smoothed_y_A[0] = y_coords_A[0]
        smoothed_y_A[-1] = y_coords_A[-1]
    python_time = time.time() - start_time
    
    # Time Numba version
    start_time = time.time()
    for _ in range(iterations):
        smoothed_coords_A = smooth_coordinates_numba(coords_A)
    numba_time = time.time() - start_time
    
    print(f"Python version: {python_time:.4f} seconds")
    print(f"Numba version: {numba_time:.4f} seconds")
    print(f"Speedup: {python_time / numba_time:.2f}x")
    
    return python_time, numba_time
