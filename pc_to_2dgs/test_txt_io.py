#!/usr/bin/env python3
"""Test script for txt_io.py"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.txt_io import load_xyzrgb_txt, validate_format, save_xyzrgb_txt
import numpy as np

def test_load_sample():
    """Test loading the sample point cloud."""
    print("Testing load_xyzrgb_txt...")
    points, colors = load_xyzrgb_txt("data/input/auditorium_1.txt")
    print(f"  Loaded {points.shape[0]} points")
    print(f"  Points dtype: {points.dtype}")
    print(f"  Colors dtype: {colors.dtype}")
    print(f"  First point: {points[0]}")
    print(f"  First color: {colors[0]}")
    # Just verify we got a valid point cloud (not empty, correct shape)
    assert points.shape[0] > 0, "No points loaded"
    assert points.shape[1] == 3, f"Expected 3 columns, got {points.shape[1]}"
    assert colors.shape[0] == points.shape[0], "Colors shape doesn't match points"
    assert colors.shape[1] == 3, f"Expected 3 color columns, got {colors.shape[1]}"
    print("  [PASSED]")

def test_validate():
    """Test format validation."""
    print("\nTesting validate_format...")
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    result = validate_format(points, colors)
    assert result == True, "Validation should return True"
    print("  [PASSED]")

def test_roundtrip():
    """Test save and reload."""
    print("\nTesting roundtrip (save/load)...")
    # Create test data
    points = np.array([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], dtype=np.float64)
    colors = np.array([[128, 64, 32], [255, 128, 0]], dtype=np.uint8)
    
    # Save
    save_xyzrgb_txt("data/output/test_roundtrip.txt", points, colors)
    
    # Load
    points2, colors2 = load_xyzrgb_txt("data/output/test_roundtrip.txt")
    
    # Compare (with tolerance for float formatting)
    assert points2.shape == points.shape, "Point shapes don't match"
    assert colors2.shape == colors.shape, "Color shapes don't match"
    assert np.allclose(points2, points, atol=1e-5), "Points don't match"
    assert np.array_equal(colors2, colors), "Colors don't match"
    print("  [PASSED]")

if __name__ == "__main__":
    test_load_sample()
    test_validate()
    test_roundtrip()
    print("\n[ALL TESTS PASSED]")
