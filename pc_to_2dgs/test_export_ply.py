#!/usr/bin/env python3
"""Test script for export_ply.py"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.export_ply import write_ply, read_ply, ply_header_from_surfels
import numpy as np

def test_write_ply():
    """Test writing surfels to PLY."""
    print("Testing write_ply...")
    
    # Create test surfels
    surfels = {
        "position": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
        "normal": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64),
        "tangent": np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float64),
        "bitangent": np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float64),
        "opacity": np.array([0.5, 0.8], dtype=np.float64),
        "scale": np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]], dtype=np.float64),
        "rotation": np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float64),
        "color": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
    }
    
    write_ply("data/output/test_surfels.ply", surfels)
    
    # Verify file exists
    import os
    assert os.path.exists("data/output/test_surfels.ply"), "PLY file not created"
    
    print("  [PASSED]")

def test_read_ply():
    """Test reading surfels from PLY."""
    print("\nTesting read_ply...")
    
    surfels = read_ply("data/output/test_surfels.ply")
    
    expected_keys = ["position", "normal", "tangent", "bitangent", 
                     "opacity", "scale", "rotation", "color"]
    for key in expected_keys:
        assert key in surfels, f"Missing key: {key}"
        print(f"  {key}: {surfels[key].shape}")
    
    # Check shapes
    assert surfels["position"].shape == (2, 3)
    assert surfels["normal"].shape == (2, 3)
    assert surfels["color"].shape == (2, 3)
    assert surfels["opacity"].shape == (2,)
    assert surfels["rotation"].shape == (2, 4)
    
    print("  [PASSED]")

def test_roundtrip():
    """Test write/read roundtrip."""
    print("\nTesting write/read roundtrip...")
    
    # Create test surfels
    surfels_original = {
        "position": np.random.randn(10, 3).astype(np.float64),
        "normal": np.random.randn(10, 3).astype(np.float64),
        "tangent": np.random.randn(10, 3).astype(np.float64),
        "bitangent": np.random.randn(10, 3).astype(np.float64),
        "opacity": np.random.rand(10).astype(np.float64),
        "scale": np.random.rand(10, 3).astype(np.float64) * 0.1,
        "rotation": np.random.randn(10, 4).astype(np.float64),
        "color": np.random.rand(10, 3).astype(np.float64),
    }
    
    # Normalize rotation to unit quaternion
    for i in range(10):
        q = surfels_original["rotation"][i]
        q = q / (np.linalg.norm(q) + 1e-10)
        surfels_original["rotation"][i] = q
    
    # Write
    write_ply("data/output/roundtrip.ply", surfels_original)
    
    # Read
    surfels_loaded = read_ply("data/output/roundtrip.ply")
    
    # Compare
    for key in surfels_original:
        original = surfels_original[key]
        loaded = surfels_loaded[key]
        if np.issubdtype(original.dtype, np.floating):
            assert np.allclose(original, loaded, atol=1e-5), f"Mismatch in {key}"
        else:
            assert np.array_equal(original, loaded), f"Mismatch in {key}"
    
    print("  [PASSED]")

def test_header():
    """Test PLY header generation."""
    print("\nTesting ply_header_from_surfels...")
    
    header = ply_header_from_surfels(100, binary=False)
    print(f"  Header lines: {len(header.split(chr(10)))}")
    
    assert "element vertex 100" in header
    assert "property float x" in header
    assert "property float opacity" in header
    assert "property float rx" in header  # quaternion x
    assert "end_header" in header
    
    print("  [PASSED]")

def test_full_pipeline():
    """Test export with full preprocessing pipeline."""
    print("\nTesting full pipeline to PLY...")
    from src.txt_io import load_xyzrgb_txt
    from src.preprocess import normalize_points
    from src.normals import estimate_normals_knn, orient_normals_consistent
    from src.surfels import build_surfels
    
    # Load
    points, colors = load_xyzrgb_txt("data/input/auditorium_1.txt")
    
    # Preprocess
    points_norm, _ = normalize_points(points)
    
    # Normals
    normals, _ = estimate_normals_knn(points_norm, k=5)
    normals = orient_normals_consistent(points_norm, normals)
    
    # Build surfels
    surfels = build_surfels(points_norm, normals, colors)
    
    # Export to PLY
    write_ply("data/output/surfels_output.ply", surfels)
    
    # Read back
    surfels_loaded = read_ply("data/output/surfels_output.ply")
    
    assert surfels_loaded["position"].shape[0] == points_norm.shape[0]
    
    print("  [PASSED]")

if __name__ == "__main__":
    test_write_ply()
    test_read_ply()
    test_roundtrip()
    test_header()
    test_full_pipeline()
    print("\n[ALL EXPORT PLY TESTS PASSED]")
