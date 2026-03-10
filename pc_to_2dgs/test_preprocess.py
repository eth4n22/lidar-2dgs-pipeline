#!/usr/bin/env python3
"""Test script for preprocess.py"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.preprocess import (
    remove_outliers, voxel_downsample, normalize_points, denormalize_points
)
import numpy as np

def test_voxel_downsample():
    """Test voxel grid downsampling."""
    print("Testing voxel_downsample...")
    # Create a grid of points
    points = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.01, 0.01],
        [0.09, 0.09, 0.09],
        [0.11, 0.09, 0.09],
    ], dtype=np.float64)
    
    centroids, indices = voxel_downsample(points, voxel_size=0.1)
    print(f"  Input: {points.shape[0]} points, Output: {centroids.shape[0]} centroids")
    assert centroids.shape[0] == 2, f"Expected 2 centroids, got {centroids.shape[0]}"
    print("  [PASSED]")

def test_normalize():
    """Test point cloud normalization."""
    print("\nTesting normalize_points...")
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [0.5, 1.0, 1.5],
    ], dtype=np.float64)
    
    normalized, (mean, scale) = normalize_points(points)
    print(f"  Mean: {mean}, Scale: {scale}")
    
    # Check that points are centered
    assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10), "Not centered"
    
    # Check scale
    extents = np.max(normalized, axis=0) - np.min(normalized, axis=0)
    assert np.max(extents) <= 1.0, "Scale too large"
    assert np.max(extents) > 0.5, "Scale too small"
    
    # Test denormalization
    denormalized = denormalize_points(normalized, (mean, scale))
    assert np.allclose(denormalized, points, atol=1e-10), "Denormalization failed"
    
    print("  [PASSED]")

def test_remove_outliers():
    """Test statistical outlier removal."""
    print("\nTesting remove_outliers...")
    # Create inlier points in a cluster
    inliers = np.random.randn(100, 3) * 0.1
    # Add outliers far away
    outliers = np.array([[10, 10, 10], [-10, -10, -10], [5, 5, 5]])
    points = np.vstack([inliers, outliers])
    
    filtered, mask = remove_outliers(points, k=10, std_ratio=2.0)
    print(f"  Input: {points.shape[0]} points, Output: {filtered.shape[0]} points")
    
    # Outliers should be removed
    assert filtered.shape[0] < points.shape[0], "Outliers not removed"
    print("  [PASSED]")

def test_full_pipeline():
    """Test full preprocessing pipeline."""
    print("\nTesting full preprocessing pipeline...")
    # Load sample data
    from src.txt_io import load_xyzrgb_txt
    points, _ = load_xyzrgb_txt("data/input/auditorium_1.txt")
    print(f"  Loaded {points.shape[0]} points")
    
    # Normalize
    normalized, params = normalize_points(points)
    print(f"  Normalized shape: {normalized.shape}")
    
    # Voxel downsample
    voxels, _ = voxel_downsample(normalized, voxel_size=0.1)
    print(f"  Voxel downsampled: {voxels.shape[0]} points")
    
    print("  [PASSED]")

if __name__ == "__main__":
    test_voxel_downsample()
    test_normalize()
    test_remove_outliers()
    test_full_pipeline()
    print("\n[ALL PREPROCESS TESTS PASSED]")
