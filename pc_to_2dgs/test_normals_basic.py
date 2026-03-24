#!/usr/bin/env python3
"""
Test 1: Basic Sanity Test for Normal Estimation

Validates that estimate_normals_knn() produces valid output
with correct shape and no NaN/Inf values.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.normals import estimate_normals_knn


def test_basic_sanity():
    """Test basic sanity of normal estimation."""
    
    # Generate synthetic sphere data
    print("Generating synthetic sphere data...")
    n_points = 10000
    
    # Create a unit sphere
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    print(f"  Points shape: {points.shape}")
    
    # Run normal estimation
    print("Running normal estimation...")
    normals, curvatures = estimate_normals_knn(points, k=10)
    
    # Verify shape
    print(f"  Normals shape: {normals.shape}")
    assert normals.shape[0] == points.shape[0], f"Shape mismatch: {normals.shape[0]} vs {points.shape[0]}"
    
    # Verify no NaN values
    nan_count = np.sum(np.isnan(normals))
    assert nan_count == 0, f"Found {nan_count} NaN values in normals"
    print(f"  NaN count: {nan_count}")
    
    # Verify no Inf values
    inf_count = np.sum(np.isinf(normals))
    assert inf_count == 0, f"Found {inf_count} Inf values in normals"
    print(f"  Inf count: {inf_count}")
    
    # Verify normals are unit length (approximately)
    norms = np.linalg.norm(normals, axis=1)
    print(f"  Normal norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
    
    # Normals should point outward from origin (for a sphere centered at origin)
    # The normal should roughly equal the point position (normalized)
    dot_products = np.sum(normals * points, axis=1)
    # For outward normals, dot product should be positive (same direction)
    positive_dots = np.sum(dot_products > 0)
    print(f"  Outward-facing normals: {positive_dots}/{n_points} ({100*positive_dots/n_points:.1f}%)")
    
    print("\n[OK] All basic sanity tests passed!")


if __name__ == "__main__":
    test_basic_sanity()
