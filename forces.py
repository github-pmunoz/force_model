#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import time
from numba import jit
from restoration_forces import add_restoration_to_smoothing, create_safe_masks

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

def regularizing_function(n):
    """
    creates a normal distribution with peak at n/2
    """

    cutoff = 0.1  # Define a cutoff value for the regularizing function
    if n <= 0:
        raise ValueError("n must be a positive integer")    
    
    result = np.zeros(n)
    x = np.arange(n)
    half = n / 2
    # Create a bell-shaped curve with a plateau in the middle
    regularizing_func = np.exp( -np.abs((x - half)/(n/4))**2)
    
    # Apply a cutoff to the function

    #sustract the minimum value to ensure the function starts at 0
    regularizing_func -= np.min(regularizing_func)
    # Normalize the function to ensure it starts at 0 and ends at 1
    regularizing_func /= np.max(regularizing_func)
    return regularizing_func


def create_ellipse(a, b, cx, cy, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points)
    # Add sinusoidal perturbation for curvyness
    amplitude = 0.5  # Adjust amplitude for curvyness
    frequency = 6    # Adjust frequency for number of waves
    x = cx + (a + amplitude * np.sin(frequency * theta)) * np.cos(theta)
    y = cy + (b + amplitude * np.sin(frequency * theta)) * np.sin(theta)
    return Polygon(zip(x, y))

def get_lengths(lineSegment):
    """
    return an array with the length of each edge v(i),v(i+1)
    """
    lengths = []
    for i in range(len(lineSegment.coords) - 1):
        p1 = np.array(lineSegment.coords[i])
        p2 = np.array(lineSegment.coords[i + 1])
        lengths.append(np.linalg.norm(p2 - p1))
    return np.array(lengths)

def get_segment_length(polygon, start, end):
    """
    Calculate the number of points in the polygon's exterior and return the length of the segment
    """
    num_points = len(polygon.exterior.coords)
    if start < 0 or end >= num_points:
        raise ValueError("Start or end index out of bounds.")
    if start <= end:
        segment = polygon.exterior.coords[start:end + 1]
    else:
        # Wrap around the polygon
        segment = polygon.exterior.coords[start:] + polygon.exterior.coords[:end + 1]
    return len(segment)

def find_infracting_segments(polygonA, polygonB, distance_threshold):
    # the start of the segment is the point "i" in polygonA such that
    # v(i-1) is not infracting, v(i) is infracting
    # the end of the segment is the point "i" in polygonB such that
    # v(i-1) is infracting, v(i) is not infracting
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

def move_points(line_string, forces, step_size=0.01):
    """
    Moves each point in the LineString by its corresponding force vector, scaled by step_size.
    Returns a new LineString with updated coordinates.
    """
    new_coords = []
    for coord, force in zip(line_string.coords, forces):
        force = np.array(force)
        if np.linalg.norm(force) > 0:
            force = force / np.linalg.norm(force)
        new_coord = (coord[0] + force[0] * step_size, coord[1] + force[1] * step_size)
        new_coords.append(new_coord)
    return LineString(new_coords)

def find_close_points_indices(polygonA, polygonB, distance_threshold):
    close_points_indices_A = []
    close_points_indices_B = []
    for i, point in enumerate(polygonA.exterior.coords):
        p = Point(point)
        if polygonB.distance(p) < distance_threshold:
            close_points_indices_A.append(i)
    for i, point in enumerate(polygonB.exterior.coords):
        p = Point(point)
        if polygonA.distance(p) < distance_threshold:
            close_points_indices_B.append(i)
    return close_points_indices_A, close_points_indices_B

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

def get_pre_post_segment(polygonA, polygonB, distance_threshold):
    indices_A = find_infracting_segments(polygonA, polygonB, distance_threshold)
    length_A = get_segment_length(polygonA, indices_A[0], indices_A[1])
    if indices_A[0] is not None and indices_A[1] is not None:
        pre = int(indices_A[0] - length_A/2 )
        post = int(indices_A[1] + length_A/2 )
    else:
        pre = 0
        post = 0
    num_points = len(polygonA.exterior.coords)
    # Collect all points from pre to post (inclusive), handling wrap-around
    if pre <= post:
        segment_points = [polygonA.exterior.coords[i % num_points] for i in range(pre, post + 1)]
    else:
        # Wrap around the polygon
        segment_points = [polygonA.exterior.coords[i % num_points] for i in range(pre, pre + (post - pre + num_points) + 1)]
    return LineString(segment_points)

def simple_moving_average(arr, window_size=20):
    """
    Calculate the simple moving average of an array with a given window size.
    leaving unaffected the first and last points.
    """
    if window_size <= 0 or window_size > len(arr):
        raise ValueError("Window size must be a positive integer less than or equal to the length of the array.")
    
    smoothed = np.zeros_like(arr)
    half_window = window_size // 2
    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        smoothed[i] = np.mean(arr[start:end])
    
    return smoothed

def force_model(stringA, stringB, original_lengths, distance_threshold, spring_factor=1.0, repulsion_factor=1.0):
    """
    simpler force model.
    each con point can only move through the bisector of the segment connecting it to the left and right neighbors.
    first and last points are not moved.
    and the forces are scaled with the regularizing function.
    """
    n = len(stringA.coords)
    if n < 3:
        return np.zeros((n, 2))
    forces = np.zeros((n, 2))
    regularizing_func = regularizing_function(n)
    centroid_A = np.mean(np.array(stringA.coords), axis=0)
    # plt.plot(regularizing_func)
    # plt.show()
    for i in range(1, n - 1):
        p_prev = np.array(stringA.coords[i - 1])
        p_curr = np.array(stringA.coords[i])
        p_next = np.array(stringA.coords[i + 1])

        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            # in this case we default to the direction towards the centroid
            lineB = LineString(stringB.coords)
            projected = lineB.interpolate(lineB.project(Point(p_curr)))
            bisector = np.array(projected.coords[0]) - p_curr
            if np.linalg.norm(bisector) == 0:
                bisector = np.array([0, 0])
        else:
            bisector = (v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2))

        if np.linalg.norm(bisector) == 0:
            lineB = LineString(stringB.coords)
            projected = lineB.interpolate(lineB.project(Point(p_curr)))
            bisector = np.array(projected.coords[0]) - p_curr
            if np.linalg.norm(bisector) == 0:
                bisector = np.array([0, 0])
        else:
            bisector /= np.linalg.norm(bisector)
        forces[i] = bisector * regularizing_func[i]

    return forces


def main():
    # Ellipse parameters
    a1, b1, cx1, cy1 = 24, 11, 0, 0
    a2, b2, cx2, cy2 = 15, 20, 40, -10  # shifted so they overlap
    distance_threshold = 5.0

    # Create ellipses
    ellipse1 = create_ellipse(a1, b1, cx1, cy1)
    ellipse2 = create_ellipse(a2, b2, cx2, cy2)

    # Find close points
    close_points = find_close_points(ellipse1, ellipse2, distance_threshold)

    # Plot ellipses and convex hull
    x1, y1 = ellipse1.exterior.xy
    x2, y2 = ellipse2.exterior.xy
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax = axs[0, 0]

    ax.plot(x1, y1, label='Ellipse 1', color='blue')
    ax.plot(x2, y2, label='Ellipse 2', color='blue')
    hull = convex_hull([p.coords[0] for p in close_points])
    # Ensure hull is a Polygon before accessing exterior
    if isinstance(hull, Polygon):
        hull_x, hull_y = hull.exterior.xy
        ax.fill(hull_x, hull_y, color='red', alpha=0.3, label='Convex Hull')
    else:
        hull_x, hull_y = None, None  # Explicitly set to None if hull is not a Polygon
    ax.axis('equal')

    # Use the function to get the segment
    segment_A = get_pre_post_segment(ellipse1, ellipse2, distance_threshold)
    segment_B = get_pre_post_segment(ellipse2, ellipse1, distance_threshold)    
    ax.plot(*segment_A.xy, color='red', linewidth=2, label='Pre-Post Line')
    ax.plot(*segment_B.xy, color='red', linewidth=2, label='Pre-Post Line B')
    
    # Draw lines connecting the endpoints of segment_A and segment_B to close the segments
    ax.plot(
        [segment_A.coords[-1][0], segment_A.coords[0][0]],
        [segment_A.coords[-1][1], segment_A.coords[0][1]],
        color='red', linewidth=2, linestyle='--', label='Segment A Closure'
    )
    ax.plot(
        [segment_B.coords[-1][0], segment_B.coords[0][0]],
        [segment_B.coords[-1][1], segment_B.coords[0][1]],
        color='red', linewidth=2, linestyle='--', label='Segment B Closure'
    )

    old_A = segment_A
    old_B = segment_B

    # Store original segments for restoration forces
    original_segment_A = segment_A
    original_segment_B = segment_B

    # Pre-compute safe vertices once (key optimization - safeness doesn't change!)
    safe_mask_A, safe_mask_B = create_safe_masks(segment_A, segment_B, distance_threshold)
    print(f"Pre-computed safe vertices: Segment A has {np.sum(safe_mask_A)} safe vertices, Segment B has {np.sum(safe_mask_B)} safe vertices")

    # Apply iterative smoothing while preserving endpoints
    max_iterations = 500
    final_smoothing_iterations = 0  # Pure Laplacian iterations at the end to eliminate spikes
    print(len(old_A.coords), len(old_B.coords))
    
    # Timing variables
    total_smoothing_time = 0.0
    use_fast_version = True  # Set to True to use Numba optimization
    use_restoration_forces = False  # Set to True to use restoration forces
    restoration_strength = 0.20 # How strong the pull back to original is
    restoration_weight = 0.3   # Weight of restoration vs smoothing
    window_size = 20  # Window size for smoothing (affects both smoothing and endpoint clamping)
    iteration = 0  # Initialize iteration counter
    
    print(f"Starting simulation: {max_iterations} iterations with restoration forces, then {final_smoothing_iterations} pure Laplacian iterations")
    print("Phase 1: Restoration forces maintain shape while allowing deformation")
    print(f"Window size: {window_size}, Endpoint clamping: {window_size // 4} points from each end (both segments)")
    
    for iteration in range(max_iterations):
        # Apply forces to move points
        new_A = old_A
        new_B = old_B
        
        start_time = time.time()
        
        if new_B.distance(new_A) <= distance_threshold and hull.distance(new_A) <= 0:
            if use_restoration_forces:
                # Use enhanced smoothing with restoration forces
                new_A, force_info_A, _ = add_restoration_to_smoothing(
                    old_A, original_segment_A, old_B, distance_threshold,
                    restoration_strength=restoration_strength, 
                    restoration_weight=restoration_weight,
                    safe_mask=safe_mask_A
                )
                new_coords_A = list(new_A.coords)
            else:
                # Original smoothing logic
                coords_A = np.array(new_A.coords)
                x_coords_A = coords_A[:, 0].copy()
                y_coords_A = coords_A[:, 1].copy()
                
                if use_fast_version:
                    smoothed_x_A = simple_moving_average_fast(x_coords_A, window_size)
                    smoothed_y_A = simple_moving_average_fast(y_coords_A, window_size)
                else:
                    smoothed_x_A = simple_moving_average(x_coords_A, window_size)
                    smoothed_y_A = simple_moving_average(y_coords_A, window_size)
                    
                # Clamp endpoints and their neighbors for better stability
                # Use 1/4 of window size for clamping (less aggressive than half window)
                quarter_window = window_size // 10
                clamp_count_A = max(1,min(quarter_window, len(x_coords_A) // 2))  # Don't clamp more than half the array
                
                for i in range(clamp_count_A):
                    smoothed_x_A[i] = x_coords_A[i]
                    smoothed_x_A[-(i+1)] = x_coords_A[-(i+1)]
                    smoothed_y_A[i] = y_coords_A[i]
                    smoothed_y_A[-(i+1)] = y_coords_A[-(i+1)]
                new_coords_A = list(zip(smoothed_x_A, smoothed_y_A))
        else:
            new_coords_A = list(old_A.coords)

        # Apply smoothing separately to x and y coordinates while preserving endpoints
        if new_A.distance(new_B) <= distance_threshold and hull.distance(new_B) <= 0:
            if use_restoration_forces:
                # Use enhanced smoothing with restoration forces
                new_B, force_info_B, _ = add_restoration_to_smoothing(
                    old_B, original_segment_B, old_A, distance_threshold,
                    restoration_strength=restoration_strength, 
                    restoration_weight=restoration_weight,
                    safe_mask=safe_mask_B
                )
                new_coords_B = list(new_B.coords)
            else:
                # Original smoothing logic
                coords_B = np.array(new_B.coords)
                x_coords_B = coords_B[:, 0].copy()
                y_coords_B = coords_B[:, 1].copy()
                
                if use_fast_version:
                    smoothed_x_B = simple_moving_average_fast(x_coords_B, window_size)
                    smoothed_y_B = simple_moving_average_fast(y_coords_B, window_size)
                else:
                    smoothed_x_B = simple_moving_average(x_coords_B, window_size)
                    smoothed_y_B = simple_moving_average(y_coords_B, window_size)
                    
                # Clamp endpoints and their neighbors for better stability
                # Use 1/4 of window size for clamping (less aggressive than half window)
                quarter_window = window_size // 10
                clamp_count_B = max(1,min(quarter_window, len(x_coords_B) // 2))  # Don't clamp more than half the array
                
                for i in range(clamp_count_B):
                    smoothed_x_B[i] = x_coords_B[i]
                    smoothed_x_B[-(i+1)] = x_coords_B[-(i+1)]
                    smoothed_y_B[i] = y_coords_B[i]
                    smoothed_y_B[-(i+1)] = y_coords_B[-(i+1)]
                new_coords_B = list(zip(smoothed_x_B, smoothed_y_B))
        else:
            new_coords_B = list(old_B.coords)
        
        smoothing_time = time.time() - start_time
        total_smoothing_time += smoothing_time
        
        if not use_restoration_forces:
            old_A = LineString(new_coords_A)
            old_B = LineString(new_coords_B)
        else:
            old_A = new_A
            old_B = new_B

        # Debug information every 50 iterations
        if iteration % 50 == 0 and use_restoration_forces:
            print(f"Iteration {iteration}: Segment A has {np.sum(safe_mask_A)} safe vertices, Segment B has {np.sum(safe_mask_B)} safe vertices")

        # Check convergence - if segments are far enough apart, break
        if hull.distance(old_A) > 0 and hull.distance(old_B) > 0:
            print(f"Converged at iteration {iteration}")
            break
    
    # Final pure Laplacian smoothing phase to eliminate spikes
    print(f"\nPhase 2: Applying {final_smoothing_iterations} final pure Laplacian iterations to eliminate spikes...")
    final_smoothing_time = 0.0
    
    for final_iter in range(final_smoothing_iterations):
        start_time = time.time()
        
        # Apply pure Laplacian smoothing (no restoration forces)
        if hull.distance(old_A) <= 0:
            coords_A = np.array(old_A.coords)
            x_coords_A = coords_A[:, 0].copy()
            y_coords_A = coords_A[:, 1].copy()
            
            if use_fast_version:
                smoothed_x_A = simple_moving_average_fast(x_coords_A, window_size)
                smoothed_y_A = simple_moving_average_fast(y_coords_A, window_size)
            else:
                smoothed_x_A = simple_moving_average(x_coords_A, window_size)
                smoothed_y_A = simple_moving_average(y_coords_A, window_size)
                
            # Clamp endpoints and their neighbors for better stability
            # Use 1/4 of window size for clamping (less aggressive than half window)
            quarter_window = window_size // 4
            clamp_count_A = min(quarter_window, len(x_coords_A) // 2)  # Don't clamp more than half the array
            
            for i in range(clamp_count_A):
                smoothed_x_A[i] = x_coords_A[i]
                smoothed_x_A[-(i+1)] = x_coords_A[-(i+1)]
                smoothed_y_A[i] = y_coords_A[i]
                smoothed_y_A[-(i+1)] = y_coords_A[-(i+1)]
            new_coords_A = list(zip(smoothed_x_A, smoothed_y_A))
            old_A = LineString(new_coords_A)
        
        if hull.distance(old_B) <= 0:
            coords_B = np.array(old_B.coords)
            x_coords_B = coords_B[:, 0].copy()
            y_coords_B = coords_B[:, 1].copy()
            
            if use_fast_version:
                smoothed_x_B = simple_moving_average_fast(x_coords_B, window_size)
                smoothed_y_B = simple_moving_average_fast(y_coords_B, window_size)
            else:
                smoothed_x_B = simple_moving_average(x_coords_B, window_size)
                smoothed_y_B = simple_moving_average(y_coords_B, window_size)
                
            # Clamp endpoints and their neighbors for better stability
            # Use 1/4 of window size for clamping (less aggressive than half window)
            quarter_window = window_size // 4
            clamp_count_B = min(quarter_window, len(x_coords_B) // 2)  # Don't clamp more than half the array
            
            for i in range(clamp_count_B):
                smoothed_x_B[i] = x_coords_B[i]
                smoothed_x_B[-(i+1)] = x_coords_B[-(i+1)]
                smoothed_y_B[i] = y_coords_B[i]
                smoothed_y_B[-(i+1)] = y_coords_B[-(i+1)]
            new_coords_B = list(zip(smoothed_x_B, smoothed_y_B))
            old_B = LineString(new_coords_B)
        
        final_smoothing_time += time.time() - start_time
        print(f"Final iteration {final_iter + 1}/{final_smoothing_iterations} completed")
    
    print(f"Final smoothing time: {final_smoothing_time:.4f} seconds")
    print(f"Total smoothing time: {total_smoothing_time:.4f} seconds")
    print(f"Average time per iteration: {total_smoothing_time/max(1, iteration+1):.6f} seconds")
    restoration_status = "with restoration forces + final Laplacian" if use_restoration_forces else "without restoration forces"
    optimization_status = "Numba optimized" if use_fast_version else "Pure Python"
    print(f"Using {optimization_status} version {restoration_status}")
    print(f"Final result: {iteration+1} restoration iterations + {final_smoothing_iterations} pure Laplacian iterations")

    ax = axs[0, 1]
    ax.plot(*old_A.xy, color='green', linewidth=2, label='Moved Segment A')
    ax.plot(*old_B.xy, color='green', linewidth=2, label='Moved Segment B')
    # Plot original ellipses in pointed (dotted) line style
    ax.plot(segment_A.xy[0], segment_A.xy[1], linestyle=':', color='blue', linewidth=1.5, label='Original Segment A')
    ax.plot(segment_B.xy[0], segment_B.xy[1], linestyle=':', color='blue', linewidth=1.5, label='Original Segment B')
    ax.fill(hull_x, hull_y, color='red', alpha=0.3, label='Convex Hull')
    ax.axis('equal')

    ax = axs[1, 1]
    ax.plot(x1, y1, label='Ellipse 1', color='green', linewidth=0.25)
    ax.plot(x2, y2, label='Ellipse 2', color='green', linewidth=0.25)


    #use shapely to subtract the ellipse withe the convex hull of segment_A, then adding the convex hull of old_A
    hull_segment_A = segment_A.convex_hull
    unchanged_A = ellipse1.difference(hull_segment_A)
    repaired = old_A.convex_hull.union(unchanged_A)
    # Check if repaired is a Polygon or MultiPolygon before accessing exterior
    from shapely.geometry import MultiPolygon
    if isinstance(repaired, Polygon):
        x_repaired, y_repaired = repaired.exterior.xy
        ax.plot(x_repaired, y_repaired, color='blue', linewidth=2, label='Repaired Segment A')
    elif isinstance(repaired, MultiPolygon):
        for poly in repaired.geoms:
            x_repaired, y_repaired = poly.exterior.xy
            ax.plot(x_repaired, y_repaired, color='blue', linewidth=2, label='Repaired Segment A')

    hull_segment_B = segment_B.convex_hull
    unchanged_B = ellipse2.difference(hull_segment_B)
    repaired_B = old_B.convex_hull.union(unchanged_B)
    from shapely.geometry import MultiPolygon
    if isinstance(repaired_B, Polygon):
        x_repaired_B, y_repaired_B = repaired_B.exterior.xy
        ax.plot(x_repaired_B, y_repaired_B, color='blue', linewidth=2, label='Repaired Segment B')
    elif isinstance(repaired_B, MultiPolygon):
        for poly in repaired_B.geoms:
            x_repaired_B, y_repaired_B = poly.exterior.xy
            ax.plot(x_repaired_B, y_repaired_B, color='blue', linewidth=2, label='Repaired Segment B')

    # Calculate and print the area that was lost (difference between original and repaired)
    area_loss_A = ellipse1.area - repaired.area
    area_loss_B = ellipse2.area - repaired_B.area
    print(f"\nArea Analysis:")
    print(f"Original Ellipse 1 area: {ellipse1.area:.4f}")
    print(f"Repaired Ellipse 1 area: {repaired.area:.4f}")
    print(f"Area lost in Ellipse 1: {area_loss_A:.4f} ({area_loss_A/ellipse1.area*100:.2f}%)")
    print(f"Original Ellipse 2 area: {ellipse2.area:.4f}")
    print(f"Repaired Ellipse 2 area: {repaired_B.area:.4f}")
    print(f"Area lost in Ellipse 2: {area_loss_B:.4f} ({area_loss_B/ellipse2.area*100:.2f}%)")
    print(f"Total area lost: {area_loss_A + area_loss_B:.4f}")
    

    # use ax 1 0 to plot only the original ellipses
    ax = axs[1, 0]
    ax.plot(x1, y1, label='Ellipse 1', color='blue')
    ax.plot(x2, y2, label='Ellipse 2', color='blue')
    

    plt.show()
if __name__ == "__main__":
    main()