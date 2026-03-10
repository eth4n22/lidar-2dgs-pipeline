"""Performance and benchmark tests."""

import numpy as np
import pytest
import time

from src.normals import estimate_normals_knn
from src.preprocess import voxel_downsample, remove_outliers_statistical
from src.surfels import build_surfels


class TestPerformance:
    """Performance benchmarks and regression tests."""
    
    def test_normal_estimation_scaling(self):
        """Test that normal estimation scales reasonably."""
        sizes = [1000, 5000, 10000]
        times = []
        
        for size in sizes:
            points = np.random.randn(size, 3).astype(np.float32) * 10
            
            start = time.time()
            normals = estimate_normals_knn(points, k_neighbors=20)
            elapsed = time.time() - start
            
            times.append(elapsed)
            assert normals.shape == points.shape
        
        # Check that time scales sub-quadratically (should be roughly O(N log N))
        # Time ratio should be less than size ratio squared
        ratio_1k_to_5k = times[1] / times[0] if times[0] > 0 else 1
        ratio_5k_to_10k = times[2] / times[1] if times[1] > 0 else 1
        
        # Should scale better than O(N²)
        size_ratio_1 = 5.0
        size_ratio_2 = 2.0
        
        assert ratio_1k_to_5k < size_ratio_1 ** 2, \
            f"Scaling too slow: {ratio_1k_to_5k:.2f}x for {size_ratio_1}x size increase"
        assert ratio_5k_to_10k < size_ratio_2 ** 2, \
            f"Scaling too slow: {ratio_5k_to_10k:.2f}x for {size_ratio_2}x size increase"
    
    def test_voxel_downsampling_performance(self):
        """Test voxel downsampling performance."""
        points = np.random.randn(50000, 3).astype(np.float32) * 10
        
        start = time.time()
        result = voxel_downsample(points, voxel_size=0.5)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 50K points)
        assert elapsed < 5.0, f"Voxel downsampling too slow: {elapsed:.2f}s"
        assert len(result["position"]) < len(points)  # Should reduce points
    
    def test_surfel_construction_performance(self):
        """Test surfel construction performance."""
        points = np.random.randn(10000, 3).astype(np.float32) * 10
        normals = np.random.randn(10000, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        start = time.time()
        surfels = build_surfels(points, normals)
        elapsed = time.time() - start
        
        # Should complete quickly (< 1 second for 10K points)
        assert elapsed < 1.0, f"Surfel construction too slow: {elapsed:.2f}s"
        assert len(surfels["position"]) == len(points)
    
    def test_full_pipeline_performance(self):
        """Test full pipeline performance."""
        points = np.random.randn(10000, 3).astype(np.float32) * 10
        colors = np.random.randint(0, 255, (10000, 3), dtype=np.uint8)
        
        start = time.time()
        
        # Preprocessing
        outlier_result = remove_outliers_statistical(points, k=20, std_multiplier=2.0)
        points_clean = outlier_result["position"]
        
        # Normal estimation
        normals = estimate_normals_knn(points_clean, k_neighbors=20)
        
        # Surfel construction
        colors_clean = colors[:len(points_clean)]
        surfels = build_surfels(points_clean, normals, colors=colors_clean)
        
        elapsed = time.time() - start
        
        # Full pipeline should complete in reasonable time (< 10 seconds for 10K points)
        assert elapsed < 10.0, f"Full pipeline too slow: {elapsed:.2f}s"
        assert len(surfels["position"]) > 0
    
    def test_memory_efficiency(self):
        """Test that operations don't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Process large dataset
        points = np.random.randn(50000, 3).astype(np.float32) * 10
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline
        normals = estimate_normals_knn(points, k_neighbors=20)
        surfels = build_surfels(points, normals)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Should use reasonable memory (< 1GB for 50K points)
        assert memory_used < 1024, \
            f"Memory usage too high: {memory_used:.1f}MB for 50K points"
    
    def test_throughput_benchmark(self):
        """Benchmark points per second throughput."""
        points = np.random.randn(20000, 3).astype(np.float32) * 10
        
        start = time.time()
        normals = estimate_normals_knn(points, k_neighbors=20)
        elapsed = time.time() - start
        
        throughput = len(points) / elapsed if elapsed > 0 else 0
        
        # Should achieve reasonable throughput (> 1K points/sec)
        assert throughput > 1000, \
            f"Throughput too low: {throughput:.0f} points/sec"
        
        print(f"\nThroughput: {throughput:.0f} points/sec")


class TestPerformanceRegression:
    """Regression tests to catch performance degradations."""
    
    def test_normal_estimation_speed_regression(self):
        """Ensure normal estimation doesn't get slower."""
        points = np.random.randn(5000, 3).astype(np.float32) * 10
        
        start = time.time()
        normals = estimate_normals_knn(points, k_neighbors=20)
        elapsed = time.time() - start
        
        # Should complete in < 2 seconds for 5K points
        # (Adjust threshold based on your hardware)
        assert elapsed < 2.0, \
            f"Performance regression: {elapsed:.2f}s for 5K points (expected < 2s)"
    
    def test_surfel_construction_speed_regression(self):
        """Ensure surfel construction doesn't get slower."""
        points = np.random.randn(5000, 3).astype(np.float32) * 10
        normals = np.random.randn(5000, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        start = time.time()
        surfels = build_surfels(points, normals)
        elapsed = time.time() - start
        
        # Should complete in < 0.5 seconds for 5K points
        assert elapsed < 0.5, \
            f"Performance regression: {elapsed:.2f}s for 5K surfels (expected < 0.5s)"
