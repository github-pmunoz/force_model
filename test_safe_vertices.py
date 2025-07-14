#!/usr/bin/env python3
"""
Test script to verify the pre-computed safe vertices optimization
"""

import numpy as np
from shapely.geometry import LineString
from restoration_forces import create_safe_masks, identify_safe_vertices

# Create some test segments
coords_A = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
coords_B = np.array([[0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 10]])

segment_A = LineString(coords_A)
segment_B = LineString(coords_B)

distance_threshold = 8.0

print("Testing pre-computed safe vertices...")
print(f"Segment A coordinates: {coords_A}")
print(f"Segment B coordinates: {coords_B}")
print(f"Distance threshold: {distance_threshold}")

# Test the create_safe_masks function
safe_mask_A, safe_mask_B = create_safe_masks(segment_A, segment_B, distance_threshold)

print(f"\nSafe vertices for segment A: {safe_mask_A}")
print(f"Safe vertices for segment B: {safe_mask_B}")
print(f"Number of safe vertices A: {np.sum(safe_mask_A)}")
print(f"Number of safe vertices B: {np.sum(safe_mask_B)}")

# Test the identify_safe_vertices function directly
safe_mask_A_direct = identify_safe_vertices(coords_A, coords_B, distance_threshold)
safe_mask_B_direct = identify_safe_vertices(coords_B, coords_A, distance_threshold)

print(f"\nDirect calculation - Safe vertices A: {safe_mask_A_direct}")
print(f"Direct calculation - Safe vertices B: {safe_mask_B_direct}")

# Verify they match
print(f"\nResults match: {np.array_equal(safe_mask_A, safe_mask_A_direct) and np.array_equal(safe_mask_B, safe_mask_B_direct)}")

print("\nTest completed successfully!")
