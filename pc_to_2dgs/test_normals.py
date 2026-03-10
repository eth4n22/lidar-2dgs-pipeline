#!/usr/bin/env python3
"""Test script for normals.py"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.normals import estimate_normals_knn, orient_normals_consistently, refine_normals
import numpy as np

def test_estimate_normals():
    """Test normal estimation from KNN."""
    print("Testing estimate_normals_knn...")
    # Create a planar point cloud (XY plane)
    np.random.seed(42)
    n_points = 100
    points = np.random.randn(n_points, 3)
    points[:, 2] = 0  # Planar in XY
    
    normals, curvatures = estimate_normals_knn(points, k=10)
    print(f"  Normals shape: {normals.shape}")
    print(f"  Curvatures: min={curvatures.min():.4f}, max={curvatures.max():.4f}")
    
    # Check normals are unit length
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6), "Normals not unit length"
    
    # For XY plane, normals should point along Z
    z_component = np.abs(normals[:, 2])
    assert np.mean(z_component) > 0.9, "Normals not aligned with Z"
    
    print("  [PASSED]")

def test_orient_normals():
    """Test normal orientation consistency."""
    print("\nTesting orient_normals_consistently...")
    # Create two planar regions with opposite normals
    points1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    points2 = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)
    points = np.vstack([points1, points2])
    
    normals = np.zeros((6, 3), dtype=np.float64)
    normals[:3] = [0, 0, 1]   # Pointing up
    normals[3:] = [0, 0, -1]  # Pointing down (opposite)
    
    oriented = orient_normals_consistently(points, normals)
    
    # After orientation, all normals should point in similar direction
    dot_products = oriented[:3] @ oriented[3:].T
    assert np.all(dot_products > 0), "Normals not consistently oriented"
    
    print("  [PASSED]")

def test_refine_normals():
    """Test normal refinement."""
    print("\nTesting refine_normals...")
    # Create noisy planar points
    np.random.seed(42)
    n_points = 50
    points = np.random.randn(n_points, 3)
    points[:, 2] = np.random.randn(n_points) * 0.01  # Add small noise
    
    # Initial normals
    normals = np.zeros((n_points, 3), dtype=np.float64)
    normals[:, 2] = 1.0  # All pointing up
    
    # Refine
    refined = refine_normals(points, normals, iterations=2)
    
    # Should still be mostly aligned with Z
    z_component = np.abs(refined[:, 2])
    assert np.mean(z_component) > 0.8, "Refined normals not planar"
    
    print("  [PASSED]")

def test_full_pipeline():
    """Test full normal estimation pipeline."""
    print("\nTesting full normal estimation pipeline...")
    from src.txt_io import load_xyzrgb_txt
    from src.preprocess import normalize_points
    
    # Load and normalize
    points, _ = load_xyzrgb_txt("data/input/auditorium_1.txt")
    normalized, _ = normalize_points(points)
    
    # Estimate normals
    normals, curvatures = estimate_normals_knn(normalized, k=5)
    print(f"  Normals shape: {normals.shape}")
    
    # Orient normals
    oriented = orient_normals_consistently(normalized, normals)
    
    # Refine
    refined = refine_normals(normalized, oriented, iterations=1)
    
    print("  [PASSED]")

if __name__ == "__main__":
    test_estimate_normals()
    test_orient_normals()
    test_refine_normals()
    test_full_pipeline()
    print("\n[ALL NORMALS TESTS PASSED]")
