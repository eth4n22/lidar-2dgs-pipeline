"""Tests for surfels module."""

import numpy as np
import pytest

from src.surfels import build_surfels


class TestBuildSurfels:
    """Tests for build_surfels function."""

    def test_build_surfels_basic(self):
        """Test basic surfel construction."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels(points, normals)
        
        assert "position" in surfels
        assert "normal" in surfels
        assert "tangent" in surfels
        assert "bitangent" in surfels
        assert "scale" in surfels
        assert "rotation" in surfels
        assert len(surfels["position"]) == len(points)

    def test_build_surfels_with_colors(self):
        """Test surfel construction with colors."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.float32)
        
        surfels = build_surfels(points, normals, colors=colors)
        
        assert "color" in surfels
        assert len(surfels["color"]) == len(points)

    def test_build_surfels_with_custom_sigma(self):
        """Test surfel construction with custom sigma values."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels(
            points, normals,
            sigma_tangent=0.1,
            sigma_normal=0.01
        )
        
        assert "scale" in surfels
        # Scale should reflect the sigma values
        np.testing.assert_array_almost_equal(
            surfels["scale"][:, :2],  # Tangent plane
            np.full((len(points), 2), 0.1),
            decimal=5
        )
        np.testing.assert_array_almost_equal(
            surfels["scale"][:, 2],  # Normal direction
            np.full(len(points), 0.01),
            decimal=5
        )

    def test_surfels_rotation_quaternion(self):
        """Test that rotation is represented as quaternion."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels(points, normals)
        
        assert "rotation" in surfels
        # Quaternion should have 4 components (w, x, y, z)
        assert surfels["rotation"].shape[1] == 4

    def test_surfels_covariance_matrices(self):
        """Test that covariance matrices are correctly constructed."""
        from src.surfels import build_surfels_with_covariance
        
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels_with_covariance(points, normals)
        
        assert "covariance" in surfels
        # Each covariance should be 3x3 symmetric matrix
        assert surfels["covariance"].shape == (len(points), 3, 3)

    def test_build_surfels_empty_points(self):
        """Test surfel construction with empty point cloud."""
        points = np.zeros((0, 3), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)
        
        surfels = build_surfels(points, normals)
        
        assert len(surfels["position"]) == 0
        assert len(surfels["normal"]) == 0

    def test_build_surfels_preserves_point_count(self):
        """Test that surfel count matches input point count."""
        n_points = 100
        points = np.random.randn(n_points, 3).astype(np.float32)
        normals = np.zeros((n_points, 3), dtype=np.float32)
        normals[:, 2] = 1  # All pointing up
        
        surfels = build_surfels(points, normals)
        
        assert len(surfels["position"]) == n_points
        assert len(surfels["normal"]) == n_points

    def test_build_surfels_with_opacity(self):
        """Test surfel construction with custom opacity."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels(points, normals, opacity=0.5)
        
        assert "opacity" in surfels
        np.testing.assert_array_almost_equal(
            surfels["opacity"],
            np.full(len(points), 0.5)
        )

    def test_build_surfels_data_types(self):
        """Test that surfel data types are correct."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.float32)
        
        normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        surfels = build_surfels(points, normals)
        
        assert surfels["position"].dtype == np.float32
        assert surfels["normal"].dtype == np.float32
        assert surfels["scale"].dtype == np.float32
        assert surfels["rotation"].dtype == np.float32
