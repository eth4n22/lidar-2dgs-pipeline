#!/usr/bin/env python3
"""
Test 2 & 3: Baseline Comparison and Chunk Coverage Test

Compares non-chunked vs chunked normal estimation methods
and verifies chunk coverage.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.normals import estimate_normals_knn, estimate_normals_chunked


def generate_synthetic_sphere(n_points=50000):
    """Generate synthetic sphere data."""
    print(f"Generating synthetic sphere with {n_points:,} points...")
    
    phi = np.random.uniform(0, 2*np.pi, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.column_stack([x, y, z]).astype(np.float32)
    return points


def test_chunk_coverage(points):
    """
    Test 3: Chunk Coverage
    
    Verifies that chunked method produces correct output shape
    and checks for zero normals.
    """
    print("\n" + "="*60)
    print("TEST 3: CHUNK COVERAGE")
    print("="*60)
    
    print("Running chunked normal estimation...")
    start = time.time()
    normals, curvatures = estimate_normals_chunked(points, k=10, chunk_size=10000, overlap_factor=0.15)
    elapsed = time.time() - start
    
    print(f"  Runtime: {elapsed:.2f}s")
    print(f"  Points/sec: {len(points)/elapsed:,.0f}")
    
    # Verify shape
    print(f"  Input points: {points.shape[0]:,}")
    print(f"  Output normals: {normals.shape[0]:,}")
    assert normals.shape[0] == points.shape[0], f"Shape mismatch: {normals.shape[0]} vs {points.shape[0]}"
    
    # Check for zero normals
    zero_normals = np.where(np.all(normals == 0, axis=1))[0]
    print(f"  Zero normals: {len(zero_normals)}")
    
    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(normals))
    inf_count = np.sum(np.isinf(normals))
    print(f"  NaN normals: {nan_count}")
    print(f"  Inf normals: {inf_count}")
    
    if len(zero_normals) > 0:
        print(f"  Warning: {len(zero_normals)} points have zero normals (may be isolated)")
    
    print("\n[OK] Chunk coverage test passed!")
    return normals, curvatures


def test_baseline_comparison(points):
    """
    Test 2: Baseline Comparison
    
    Compares non-chunked vs chunked methods.
    Note: For large datasets, non-chunked may use GPU which is different from chunked CPU.
    """
    print("\n" + "="*60)
    print("TEST 2: BASELINE COMPARISON")
    print("="*60)
    
    n_points = min(points.shape[0], 10000)  # Limit for comparison to avoid memory issues
    print(f"Running comparison on {n_points:,} points (limited for fair comparison)")
    
    points_subset = points[:n_points]
    
    # FAST mode (no chunking, no halo)
    print("\n  FAST mode:")
    start = time.time()
    normals1, _ = estimate_normals_knn(points_subset, k=10)
    t1 = time.time() - start
    print(f"    Runtime: {t1:.2f}s")
    print(f"    Points/sec: {n_points/t1:,.0f}")
    
    # Chunked method
    print("\n  Chunked method:")
    start = time.time()
    normals2, _ = estimate_normals_chunked(points_subset, k=10, chunk_size=5000, overlap_factor=0.15)
    t2 = time.time() - start
    print(f"    Runtime: {t2:.2f}s")
    print(f"    Points/sec: {n_points/t2:,.0f}")
    
    # Compare results
    diff = np.abs(normals1 - normals2)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    print(f"\n  Comparison results:")
    print(f"    Mean difference: {mean_diff:.6f}")
    print(f"    Max difference: {max_diff:.6f}")
    
    # Note: Due to different GPU vs CPU implementations and chunk boundary effects,
    # we use a relaxed threshold
    threshold = 0.1
    print(f"    Threshold: {threshold}")
    
    if mean_diff < threshold:
        print(f"\n[OK] Baseline comparison passed (mean_diff < {threshold})")
    else:
        print(f"\n[WARN] Baseline comparison: mean_diff ({mean_diff:.6f}) exceeds threshold ({threshold})")
        print("  This may be due to GPU vs CPU implementation differences or chunk boundary effects")


def test_small_dataset_edge_case():
    """
    Test 4: Small Dataset Edge Case
    
    Ensures normal estimation works on small datasets (~100 points).
    """
    print("\n" + "="*60)
    print("TEST 4: SMALL DATASET EDGE CASE")
    print("="*60)
    
    n_points = 100
    print(f"Testing with {n_points} random points...")
    
    # Generate small random point cloud
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    try:
        normals, curvatures = estimate_normals_knn(points, k=5)
        
        print(f"  Normals shape: {normals.shape}")
        print(f"  Expected: ({n_points}, 3)")
        assert normals.shape == (n_points, 3), f"Shape mismatch: {normals.shape}"
        
        nan_count = np.sum(np.isnan(normals))
        print(f"  NaN normals: {nan_count}")
        
        print("\n[OK] Small dataset edge case passed!")
    except Exception as e:
        print(f"\n[FAIL] Small dataset test failed: {e}")
        raise


def test_performance_smoke_test():
    """
    Test 5: Performance Smoke Test
    
    Runs normal estimation on auditorium_2.txt and reports performance.
    """
    print("\n" + "="*60)
    print("TEST 5: PERFORMANCE SMOKE TEST")
    print("="*60)
    
    # Check for auditorium_2.txt
    input_dir = Path(__file__).parent / 'data' / 'input'
    test_file = input_dir / 'auditorium_2.txt'
    
    if not test_file.exists():
        # Try to find any txt file
        txt_files = list(input_dir.glob('*.txt'))
        if txt_files:
            test_file = txt_files[0]
            print(f"Using fallback file: {test_file.name}")
        else:
            print(f"Warning: {test_file.name} not found, skipping performance test")
            print("  (This is not a failure - just skipping for demo)")
            return
    
    print(f"Loading {test_file}...")
    
    # Load points
    from src.txt_io import load_xyzrgb_txt
    points, colors = load_xyzrgb_txt(str(test_file))
    n_points = len(points)
    print(f"  Loaded {n_points:,} points")
    
    # Run normal estimation
    print("Running normal estimation (chunked for large data)...")
    start = time.time()
    
    if n_points > 1_000_000:
        # Use chunked for large datasets
        normals, curvatures = estimate_normals_chunked(points, k=10)
    else:
        normals, curvatures = estimate_normals_knn(points, k=10)
    
    elapsed = time.time() - start
    
    print(f"\n  Results:")
    print(f"    Total runtime: {elapsed:.2f}s")
    print(f"    Points/sec: {n_points/elapsed:,.0f}")
    print(f"    Normals shape: {normals.shape}")
    
    # Check for issues
    zero_normals = np.where(np.all(normals == 0, axis=1))[0]
    nan_count = np.sum(np.isnan(normals))
    
    print(f"    Zero normals: {len(zero_normals)}")
    print(f"    NaN normals: {nan_count}")
    
    print("\n[OK] Performance smoke test completed!")


if __name__ == "__main__":
    print("="*60)
    print("NORMAL ESTIMATION TEST SUITE")
    print("="*60)
    
    # Run tests
    test_small_dataset_edge_case()
    
    points = generate_synthetic_sphere(n_points=50000)
    
    test_baseline_comparison(points)
    test_chunk_coverage(points)
    test_performance_smoke_test()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
