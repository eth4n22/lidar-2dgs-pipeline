"""Integration tests for full pipeline."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from src.txt_io import load_xyzrgb_txt, save_xyzrgb_txt
from src.preprocess import remove_outliers_statistical, voxel_downsample
from src.normals import estimate_normals_knn
from src.surfels import build_surfels
from src.export_ply import write_ply


class TestFullPipeline:
    """Test complete LiDAR to 2DGS conversion pipeline."""
    
    @pytest.fixture
    def sample_point_cloud(self):
        """Generate a sample point cloud."""
        np.random.seed(42)
        n_points = 10000
        points = np.random.randn(n_points, 3).astype(np.float32) * 10
        colors = np.random.randint(0, 255, (n_points, 3), dtype=np.uint8)
        return points, colors
    
    def test_pipeline_basic(self, sample_point_cloud):
        """Test basic pipeline without preprocessing."""
        points, colors = sample_point_cloud
        
        # Step 1: Normal estimation
        normals = estimate_normals_knn(points, k_neighbors=20)
        assert normals.shape == points.shape
        
        # Step 2: Surfel construction
        surfels = build_surfels(points, normals, colors=colors)
        assert len(surfels["position"]) == len(points)
        
        # Step 3: Export
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            output_path = f.name
        
        try:
            write_ply(output_path, surfels, binary=True, verbose=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_pipeline_with_preprocessing(self, sample_point_cloud):
        """Test pipeline with outlier removal and downsampling."""
        points, colors = sample_point_cloud
        
        # Step 1: Outlier removal
        outlier_result = remove_outliers_statistical(points, k=20, std_multiplier=2.0)
        points_clean = outlier_result["position"]
        assert len(points_clean) <= len(points)
        
        # Step 2: Voxel downsampling
        voxel_result = voxel_downsample(points_clean, voxel_size=0.5, colors=colors[:len(points_clean)])
        points_down = voxel_result["position"]
        assert len(points_down) <= len(points_clean)
        
        # Step 3: Normal estimation
        normals = estimate_normals_knn(points_down, k_neighbors=20)
        assert normals.shape == points_down.shape
        
        # Step 4: Surfel construction
        colors_down = voxel_result.get("color")
        surfels = build_surfels(points_down, normals, colors=colors_down)
        assert len(surfels["position"]) == len(points_down)
    
    def test_pipeline_file_io(self, sample_point_cloud):
        """Test pipeline with file I/O."""
        points, colors = sample_point_cloud
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            input_path = f.name
        
        try:
            save_xyzrgb_txt(input_path, points, colors)
            
            # Load from file
            data = load_xyzrgb_txt(input_path)
            loaded_points = data["position"]
            loaded_colors = data.get("color")
            
            assert np.allclose(loaded_points, points, atol=1e-5)
            if loaded_colors is not None:
                assert np.array_equal(loaded_colors, colors)
            
            # Process loaded data
            normals = estimate_normals_knn(loaded_points, k_neighbors=20)
            surfels = build_surfels(loaded_points, normals, colors=loaded_colors)
            
            assert len(surfels["position"]) == len(loaded_points)
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
    
    def test_pipeline_empty_result(self):
        """Test pipeline handles empty results gracefully."""
        # Create points that will all be removed by outlier removal
        points = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.float32)  # Isolated outliers
        
        # Outlier removal should remove most/all points
        outlier_result = remove_outliers_statistical(points, k=1, std_multiplier=0.1)
        points_clean = outlier_result["position"]
        
        if len(points_clean) == 0:
            # Should handle empty gracefully
            normals = np.empty((0, 3), dtype=np.float32)
            surfels = build_surfels(points_clean, normals)
            assert len(surfels["position"]) == 0
        else:
            # If some points remain, process normally
            normals = estimate_normals_knn(points_clean, k_neighbors=min(3, len(points_clean)))
            surfels = build_surfels(points_clean, normals)
            assert len(surfels["position"]) == len(points_clean)


class TestStreamingPipeline:
    """Test streaming mode for large datasets."""
    
    def test_chunked_processing(self):
        """Test processing in chunks."""
        # Generate large dataset
        np.random.seed(42)
        n_total = 50000
        chunk_size = 10000
        
        all_surfels = []
        
        for i in range(0, n_total, chunk_size):
            chunk_points = np.random.randn(chunk_size, 3).astype(np.float32) * 10
            
            # Process chunk
            normals = estimate_normals_knn(chunk_points, k_neighbors=20)
            surfels = build_surfels(chunk_points, normals)
            
            all_surfels.append(surfels)
        
        # Verify all chunks processed
        assert len(all_surfels) == (n_total // chunk_size)
        total_points = sum(len(s["position"]) for s in all_surfels)
        assert total_points == n_total
    
    def test_memory_efficient_processing(self):
        """Test that processing doesn't accumulate memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple chunks
        for _ in range(5):
            points = np.random.randn(10000, 3).astype(np.float32) * 10
            normals = estimate_normals_knn(points, k_neighbors=20)
            surfels = build_surfels(points, normals)
            del points, normals, surfels  # Explicit cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 5 chunks)
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB"


class TestPipelineRobustness:
    """Test pipeline robustness with various inputs."""
    
    def test_pipeline_with_no_colors(self):
        """Test pipeline without color information."""
        points = np.random.randn(1000, 3).astype(np.float32) * 10
        
        normals = estimate_normals_knn(points, k_neighbors=20)
        surfels = build_surfels(points, normals, colors=None)
        
        assert "color" in surfels
        assert surfels["color"].shape == (len(points), 3)
        # Should default to white
        assert np.allclose(surfels["color"], 1.0)
    
    def test_pipeline_with_different_k_values(self):
        """Test pipeline with different k_neighbors values."""
        points = np.random.randn(5000, 3).astype(np.float32) * 10
        
        for k in [10, 20, 30]:
            normals = estimate_normals_knn(points, k_neighbors=k)
            surfels = build_surfels(points, normals)
            
            assert len(surfels["position"]) == len(points)
            assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    
    def test_pipeline_preserves_accuracy(self):
        """Test that pipeline preserves geometric accuracy."""
        # Create a known surface (plane)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X + 3 * Y  # Plane: z = 2x + 3y
        
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)
        
        # Process
        normals = estimate_normals_knn(points, k_neighbors=20)
        surfels = build_surfels(points, normals)
        
        # Check that normals are approximately correct (plane normal should be [-2, -3, 1] / ||[-2, -3, 1]||)
        expected_normal = np.array([-2, -3, 1], dtype=np.float32)
        expected_normal = expected_normal / np.linalg.norm(expected_normal)
        
        # Most normals should point in similar direction
        mean_normal = np.mean(normals, axis=0)
        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        
        # Check alignment (dot product should be close to 1 or -1)
        alignment = np.abs(np.dot(mean_normal, expected_normal))
        assert alignment > 0.8, f"Normals not aligned with expected plane normal: {alignment}"
