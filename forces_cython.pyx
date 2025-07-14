# forces_cython.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def simple_moving_average_cy(double[:] arr, int window_size=20):
    """
    Cython optimized simple moving average
    """
    cdef int n = arr.shape[0]
    cdef np.ndarray[double, ndim=1] smoothed = np.zeros(n, dtype=np.float64)
    cdef int half_window = window_size // 2
    cdef int start, end, i, j
    cdef double sum_val
    cdef int count
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        sum_val = 0.0
        count = 0
        for j in range(start, end):
            sum_val += arr[j]
            count += 1
        smoothed[i] = sum_val / count
    
    return smoothed

@cython.boundscheck(False)
@cython.wraparound(False)
def smooth_coordinates_cy(double[:, :] coords, int window_size=20):
    """
    Cython optimized coordinate smoothing that preserves endpoints
    """
    cdef int n = coords.shape[0]
    cdef np.ndarray[double, ndim=2] smoothed_coords = np.zeros((n, 2), dtype=np.float64)
    
    # Extract x and y coordinates
    cdef np.ndarray[double, ndim=1] x_coords = np.ascontiguousarray(coords[:, 0])
    cdef np.ndarray[double, ndim=1] y_coords = np.ascontiguousarray(coords[:, 1])
    
    # Smooth coordinates
    cdef np.ndarray[double, ndim=1] smoothed_x = simple_moving_average_cy(x_coords, window_size)
    cdef np.ndarray[double, ndim=1] smoothed_y = simple_moving_average_cy(y_coords, window_size)
    
    # Preserve endpoints
    smoothed_x[0] = x_coords[0]
    smoothed_x[n-1] = x_coords[n-1]
    smoothed_y[0] = y_coords[0]
    smoothed_y[n-1] = y_coords[n-1]
    
    # Combine back into coordinate array
    smoothed_coords[:, 0] = smoothed_x
    smoothed_coords[:, 1] = smoothed_y
    
    return smoothed_coords

@cython.boundscheck(False)
@cython.wraparound(False)
def preserve_endpoints_cy(double[:, :] coords, int preserve_count=3):
    """
    Preserve more endpoint coordinates for stability
    """
    cdef int n = coords.shape[0]
    cdef np.ndarray[double, ndim=2] result = np.array(coords)
    
    # This function can be extended to preserve more endpoints
    # For now, it's a placeholder that returns the input
    return result
