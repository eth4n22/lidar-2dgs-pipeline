#!/usr/bin/env python3
"""Integration test: Full pipeline from TXT to normals"""

import sys
sys.path.insert(0, str(__file__.rsplit('/', 1)[0] if '/' in __file__ else '.'))

from src.txt_io import load_xyzrgb_txt, save_xyzrgb_txt
from src.preprocess import normalize_points, voxel_downsample, remove_outliers
from src.normals import estimate_normals_knn, orient_normals_consistent, refine_normals
import numpy as np

def test_full_pipeline():
    """Test complete preprocessing and normal estimation pipeline."""
    print("=" * 60)
    print("INTEGRATION TEST: TXT -> PREPROCESS -> NORMALS")
    print("=" * 60)
    
    # Stage 1: Load TXT
    print("\n[Stage 1] Loading TXT...")
    points, colors = load_xyzrgb_txt("data/input/auditorium_1.txt")
    print(f"  Loaded {points.shape[0]} points")
    print(f"  Points dtype: {points.dtype}")
    print(f"  Colors dtype: {colors.dtype}")
    
    # Stage 2: Preprocess
    print("\n[Stage 2] Preprocessing...")
    # Remove outliers
    points_clean, mask = remove_outliers(points, k=5, std_ratio=2.0)
    colors_clean = colors[mask]
    print(f"  After outlier removal: {points_clean.shape[0]} points")
    
    # Normalize
    points_norm, (mean, scale) = normalize_points(points_clean)
    print(f"  Normalized: mean={mean}, scale={scale}")
    
    # Voxel downsample (if enough points)
    if points_norm.shape[0] > 10:
        points_vox, _ = voxel_downsample(points_norm, voxel_size=0.1)
        colors_vox = colors_clean[:points_vox.shape[0]]  # Truncate colors
        print(f"  After voxel downsample: {points_vox.shape[0]} points")
    else:
        points_vox = points_norm
        colors_vox = colors_clean
        print(f"  Skipped voxel downsample (too few points)")
    
    # Stage 3: Normals
    print("\n[Stage 3] Normal Estimation...")
    normals, curvatures = estimate_normals_knn(points_vox, k=min(5, points_vox.shape[0]-1))
    print(f"  Estimated {normals.shape[0]} normals")
    
    # Orient normals
    normals_oriented = orient_normals_consistent(points_vox, normals)
    print(f"  Oriented normals")
    
    # Refine normals
    normals_refined = refine_normals(points_vox, normals_oriented, iterations=1)
    print(f"  Refined normals")
    
    # Verify outputs
    print("\n[Verification]")
    print(f"  Points shape: {points_vox.shape}")
    print(f"  Colors shape: {colors_vox.shape}")
    print(f"  Normals shape: {normals_refined.shape}")
    
    # Check normal unit length
    normal_lengths = np.linalg.norm(normals_refined, axis=1)
    print(f"  Normal lengths: min={normal_lengths.min():.4f}, max={normal_lengths.max():.4f}")
    assert np.allclose(normal_lengths, 1.0, atol=1e-5), "Normals not unit length"
    
    # Check colors in valid range
    assert np.all(colors_vox >= 0) and np.all(colors_vox <= 255), "Colors out of range"
    
    # Save intermediate results
    print("\n[Saving Results]")
    save_xyzrgb_txt("data/output/preprocessed.txt", points_vox, colors_vox)
    np.save("data/output/normals.npy", normals_refined)
    print("  Saved: data/output/preprocessed.txt")
    print("  Saved: data/output/normals.npy")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Full pipeline test passed!")
    print("=" * 60)
    
    return points_vox, colors_vox, normals_refined

if __name__ == "__main__":
    test_full_pipeline()
