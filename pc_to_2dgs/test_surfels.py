#!/usr/bin/env python3
"""Test script for surfels.py"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.surfels import (
    tangent_basis_from_normal, build_surfels, 
    compute_covar_from_knn, scales_from_covariance,
    quaternion_from_rotation_matrix
)
import numpy as np

def test_tangent_basis():
    """Test tangent/bitangent computation from normals."""
    print("Testing tangent_basis_from_normal...")
    # Create simple normals
    normals = np.array([
        [0, 0, 1],   # Z up
        [0, 1, 0],   # Y up
        [1, 0, 0],   # X up
    ], dtype=np.float64)
    
    tangents, bitangents = tangent_basis_from_normal(normals)
    print(f"  Tangents shape: {tangents.shape}")
    print(f"  Bitangents shape: {bitangents.shape}")
    
    # Check orthogonality: dot(tangent, normal) ~= 0
    for i in range(len(normals)):
        dot_t = np.dot(tangents[i], normals[i])
        dot_b = np.dot(bitangents[i], normals[i])
        dot_tb = np.dot(tangents[i], bitangents[i])
        assert abs(dot_t) < 1e-6, f"Tangent not orthogonal to normal[{i}]"
        assert abs(dot_b) < 1e-6, f"Bitangent not orthogonal to normal[{i}]"
        assert abs(dot_tb) < 1e-6, f"Tangent not orthogonal to bitangent[{i}]"
    
    # Check unit length
    t_lengths = np.linalg.norm(tangents, axis=1)
    b_lengths = np.linalg.norm(bitangents, axis=1)
    assert np.allclose(t_lengths, 1.0, atol=1e-5), "Tangents not unit length"
    assert np.allclose(b_lengths, 1.0, atol=1e-5), "Bitangents not unit length"
    
    print("  [PASSED]")

def test_quaternion():
    """Test quaternion from rotation matrix."""
    print("\nTesting quaternion_from_rotation_matrix...")
    # Identity rotation
    R_identity = np.eye(3)
    q = quaternion_from_rotation_matrix(R_identity)
    print(f"  Identity quaternion: {q}")
    assert np.allclose(q[3], 1.0, atol=1e-5), "Identity quaternion w should be 1"
    
    # 90 degree rotation around Z
    R_z90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    q = quaternion_from_rotation_matrix(R_z90)
    print(f"  Z90 quaternion: {q}")
    # Should have w = 1/sqrt(2), z = 1/sqrt(2)
    
    print("  [PASSED]")

def test_build_surfels():
    """Test surfel construction."""
    print("\nTesting build_surfels...")
    # Create simple point cloud
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    
    normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=np.float64)
    
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)
    
    surfels = build_surfels(points, normals, colors)
    
    # Check all keys exist
    expected_keys = ["position", "normal", "tangent", "bitangent", 
                     "opacity", "scale", "rotation", "color"]
    for key in expected_keys:
        assert key in surfels, f"Missing key: {key}"
        print(f"  {key}: {surfels[key].shape}")
    
    # Check shapes
    assert surfels["position"].shape == (3, 3)
    assert surfels["normal"].shape == (3, 3)
    assert surfels["color"].shape == (3, 3)
    assert surfels["opacity"].shape == (3,)
    assert surfels["rotation"].shape == (3, 4)
    
    # Check color range [0, 1]
    assert np.all(surfels["color"] >= 0) and np.all(surfels["color"] <= 1)
    
    print("  [PASSED]")

def test_covar_and_scales():
    """Test covariance and scale computation."""
    print("\nTesting compute_covar_from_knn and scales_from_covariance...")
    # Create scattered points
    np.random.seed(42)
    points = np.random.randn(50, 3) * 0.1
    
    covariances = compute_covar_from_knn(points, k=10)
    print(f"  Covariances shape: {covariances.shape}")
    
    # Check shape
    assert covariances.shape == (50, 3, 3)
    
    # Check symmetry
    for i in range(len(covariances)):
        assert np.allclose(covariances[i], covariances[i].T), f"Covariance[{i}] not symmetric"
    
    # Compute scales
    scales = scales_from_covariance(covariances, base_scale=1.0)
    print(f"  Scales shape: {scales.shape}")
    assert scales.shape == (50, 3)
    
    # Check positive scales
    assert np.all(scales > 0), "Scales should be positive"
    
    print("  [PASSED]")

def test_full_pipeline():
    """Test surfels with full preprocessing pipeline."""
    print("\nTesting surfels with full pipeline...")
    from src.txt_io import load_xyzrgb_txt
    from src.preprocess import normalize_points
    from src.normals import estimate_normals_knn, orient_normals_consistently
    
    # Load and preprocess
    points, colors = load_xyzrgb_txt("data/input/auditorium_1.txt")
    points_norm, _ = normalize_points(points)
    
    # Estimate normals
    normals, _ = estimate_normals_knn(points_norm, k=5)
    normals = orient_normals_consistently(points_norm, normals)
    
    # Build surfels
    surfels = build_surfels(points_norm, normals, colors)
    
    print(f"  Built {len(surfels['position'])} surfels")
    
    # Verify tangent/bitangent are orthogonal to normal
    for i in range(len(surfels["normal"])):
        n = surfels["normal"][i]
        t = surfels["tangent"][i]
        b = surfels["bitangent"][i]
        assert abs(np.dot(n, t)) < 1e-5, "Tangent not orthogonal to normal"
        assert abs(np.dot(n, b)) < 1e-5, "Bitangent not orthogonal to normal"
        assert abs(np.dot(t, b)) < 1e-5, "Tangent not orthogonal to bitangent"
    
    print("  [PASSED]")

if __name__ == "__main__":
    test_tangent_basis()
    test_quaternion()
    test_build_surfels()
    test_covar_and_scales()
    test_full_pipeline()
    print("\n[ALL SURFELS TESTS PASSED]")
