"""Tests for edge cases and error handling."""

import numpy as np
import pytest

from src.surfels import build_surfels
from src.normals import estimate_normals_knn
from src.preprocess import voxel_downsample, remove_outliers_statistical


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_points_surfels(self):
        """Test surfel construction with empty point cloud."""
        points = np.empty((0, 3), dtype=np.float32)
        normals = np.empty((0, 3), dtype=np.float32)
        
        surfels = build_surfels(points, normals)
        
        assert len(surfels["position"]) == 0
        assert len(surfels["normal"]) == 0
    
    def test_empty_points_normals(self):
        """Test normal estimation with empty point cloud."""
        points = np.empty((0, 3), dtype=np.float32)
        
        normals = estimate_normals_knn(points)
        
        assert len(normals) == 0
    
    def test_nan_points_surfels(self):
        """Test that NaN points are rejected."""
        points = np.array([[0, 0, 0], [np.nan, 1, 1], [2, 2, 2]], dtype=np.float32)
        normals = np.ones((3, 3), dtype=np.float32)
        
        with pytest.raises(ValueError, match="non-finite values"):
            build_surfels(points, normals)
    
    def test_inf_points_surfels(self):
        """Test that Inf points are rejected."""
        points = np.array([[0, 0, 0], [np.inf, 1, 1], [2, 2, 2]], dtype=np.float32)
        normals = np.ones((3, 3), dtype=np.float32)
        
        with pytest.raises(ValueError, match="non-finite values"):
            build_surfels(points, normals)
    
    def test_nan_normals_surfels(self):
        """Test that NaN normals are rejected."""
        points = np.ones((3, 3), dtype=np.float32)
        normals = np.array([[0, 0, 1], [np.nan, 0, 1], [0, 0, 1]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="non-finite values"):
            build_surfels(points, normals)
    
    def test_shape_mismatch_surfels(self):
        """Test that shape mismatches are caught."""
        points = np.ones((3, 3), dtype=np.float32)
        normals = np.ones((2, 3), dtype=np.float32)
        
        with pytest.raises(ValueError, match="same shape"):
            build_surfels(points, normals)
    
    def test_invalid_k_neighbors(self):
        """Test that invalid k_neighbors is rejected."""
        points = np.random.randn(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="at least 3"):
            estimate_normals_knn(points, k_neighbors=2)
    
    def test_too_few_points_for_k(self):
        """Test that insufficient points for k is caught."""
        points = np.random.randn(5, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="Not enough points"):
            estimate_normals_knn(points, k_neighbors=10)
    
    def test_invalid_voxel_size(self):
        """Test that invalid voxel size is handled."""
        points = np.random.randn(10, 3).astype(np.float32)
        
        # Negative voxel size should return original points
        result = voxel_downsample(points, voxel_size=-1.0)
        assert len(result["position"]) == len(points)
        
        # Zero voxel size should return original points
        result = voxel_downsample(points, voxel_size=0.0)
        assert len(result["position"]) == len(points)
    
    def test_nan_points_voxel_downsample(self):
        """Test that NaN points are rejected in voxel downsampling."""
        points = np.array([[0, 0, 0], [np.nan, 1, 1]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="non-finite values"):
            voxel_downsample(points)
    
    def test_invalid_outlier_k(self):
        """Test that invalid k for outlier removal is caught."""
        points = np.random.randn(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="at least 1"):
            remove_outliers_statistical(points, k=0)
    
    def test_invalid_std_multiplier(self):
        """Test that invalid std_multiplier is caught."""
        points = np.random.randn(10, 3).astype(np.float32)
        
        with pytest.raises(ValueError, match="positive"):
            remove_outliers_statistical(points, std_multiplier=-1.0)
    
    def test_invalid_sigma_values(self):
        """Test that invalid sigma values are rejected."""
        points = np.random.randn(10, 3).astype(np.float32)
        normals = np.random.randn(10, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="positive"):
            build_surfels(points, normals, sigma_tangent=-0.05)
        
        with pytest.raises(ValueError, match="positive"):
            build_surfels(points, normals, sigma_normal=0.0)
    
    def test_invalid_opacity(self):
        """Test that invalid opacity is rejected."""
        points = np.random.randn(10, 3).astype(np.float32)
        normals = np.random.randn(10, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="Opacity must be in"):
            build_surfels(points, normals, opacity=1.5)
        
        with pytest.raises(ValueError, match="Opacity must be in"):
            build_surfels(points, normals, opacity=-0.1)
    
    def test_color_length_mismatch(self):
        """Test that color length mismatch is caught."""
        points = np.random.randn(10, 3).astype(np.float32)
        normals = np.random.randn(10, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        colors = np.random.randint(0, 255, (5, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Colors length must match"):
            build_surfels(points, normals, colors=colors)
    
    def test_wrong_dimension_points(self):
        """Test that wrong dimension points are caught."""
        points = np.random.randn(10, 2).astype(np.float32)  # 2D instead of 3D
        
        with pytest.raises(ValueError, match="Expected \\(N, 3\\)"):
            estimate_normals_knn(points)
        
        with pytest.raises(ValueError, match="Expected \\(N, 3\\)"):
            voxel_downsample(points)
    
    def test_single_point(self):
        """Test handling of single point (edge case)."""
        points = np.array([[0, 0, 0]], dtype=np.float32)
        normals = np.array([[0, 0, 1]], dtype=np.float32)
        
        # Should handle gracefully
        surfels = build_surfels(points, normals)
        assert len(surfels["position"]) == 1
        
        # Normal estimation should fail (need at least k+1 points)
        with pytest.raises(ValueError, match="Not enough points"):
            estimate_normals_knn(points, k_neighbors=3)
