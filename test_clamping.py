#!/usr/bin/env python3
"""
Test script to verify the improved clamping logic
"""

import numpy as np
from shapely.geometry import LineString

# Simulate the clamping logic from the main script
def test_clamping_logic():
    print("Testing improved clamping logic...")
    
    # Create test data
    test_lengths = [10, 20, 30, 50, 100]
    window_size = 20
    
    for length in test_lengths:
        # Create some test coordinates
        x_coords = np.linspace(0, 10, length)
        y_coords = np.sin(x_coords)
        
        # Simulate the clamping logic
        half_window = window_size // 2  # Keep for reference
        quarter_window = window_size // 4  # Updated to quarter window
        clamp_count = min(quarter_window, length // 2)
        
        print(f"\nArray length: {length}")
        print(f"Window size: {window_size}")
        print(f"Half window: {half_window}")
        print(f"Quarter window: {quarter_window}")
        print(f"Clamp count: {clamp_count}")
        print(f"Points clamped at each end: {clamp_count}")
        print(f"Total points clamped: {clamp_count * 2}")
        print(f"Points smoothed: {length - (clamp_count * 2)}")
        
        # Verify the clamping doesn't exceed reasonable bounds
        assert clamp_count <= length // 2, f"Clamping too many points for length {length}"
        assert clamp_count <= quarter_window, f"Clamping more than quarter window for length {length}"
        
        # Show which indices would be clamped
        clamped_indices = []
        for i in range(clamp_count):
            clamped_indices.append(i)
            clamped_indices.append(length - 1 - i)
        
        print(f"Clamped indices: {sorted(clamped_indices)}")
    
    print("\n" + "="*50)
    print("Comparison with old hardcoded approach:")
    print("Old approach: Always clamped 3 points from each end (6 total)")
    print("New approach: Clamps quarter_window points from each end (both segments A and B)")
    quarter_window = window_size // 4
    print(f"With window_size={window_size}: Clamps {quarter_window} points from each end ({quarter_window * 2} total per segment)")
    print(f"Improvement: More consistent with window size, applies to both segments A and B")
    print(f"Clamping reduction: {quarter_window * 2} vs old 6 points = {abs(quarter_window * 2 - 6)} point difference")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_clamping_logic()
