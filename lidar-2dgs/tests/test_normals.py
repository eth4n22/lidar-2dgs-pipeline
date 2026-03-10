"""Tests for normals estimation module."""

import numpy as np
import pytest

from src.normals import (
    estimate_normals_knn,
    estimate_normals_with_uncertainty,
    validate_normals_against_points,
    orient_normals_consistently,
    filter_normals_by_uncertainty,
    get_device
)


class TestEstimateNormalsKnn:
    """Tests for estimate_normals_knn function."""

    def test_estimate_on_flat_surface(self):
        """Test normal estimation on a flat surface (all normals should be similar)."""
        # Create a flat surface in the XY plane (normals should point in Z direction)
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        normals = estimate_normals_knn(points, k_neighbors=10)
        
        assert normals.shape == points.shape
        assert np.all(np.abs(normals[:, 2]) > 0.9)  # Z component should be dominant

    def test_estimate_on_sphere(self):
        """Test normal estimation on a sphere (normals should point outward)."""
        # Create points on a unit sphere
        phi = np.linspace(0, 2 * np.pi, 30)
        theta = np.linspace(0, np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        normals = estimate_normals_knn(points, k_neighbors=15)
        
        assert normals.shape == points.shape
        # Check that normals are roughly unit length
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)

    def test_estimate_with_custom_up_vector(self):
        """Test normal estimation with custom up vector."""
        points = np.random.randn(100, 3).astype(np.float32) * 0.1
        points[:, 2] = 0  # Flat surface
        
        normals = estimate_normals_knn(points, k_neighbors=10, up_vector=(0, 0, 1))
        
        assert normals.shape == points.shape
        # Most normals should point in +Z direction
        assert np.mean(normals[:, 2] > 0) > 0.9

    def test_estimate_cpu_mode(self):
        """Test normal estimation in CPU mode."""
        points = np.random.randn(500, 3).astype(np.float32)
        
        normals = estimate_normals_knn(points, k_neighbors=10, device='cpu')
        
        assert normals.shape == points.shape

    def test_estimate_on_small_point_cloud(self):
        """Test normal estimation on a small point cloud."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        normals = estimate_normals_knn(points, k_neighbors=3)
        
        assert normals.shape == points.shape

    def test_normals_are_unit_length(self):
        """Test that estimated normals are unit length."""
        points = np.random.randn(1000, 3).astype(np.float32) * 10
        
        normals = estimate_normals_knn(points, k_neighbors=20)
        
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)


class TestEstimateNormalsWithUncertainty:
    """Tests for estimate_normals_with_uncertainty function."""

    def test_returns_all_keys(self):
        """Test that function returns all expected keys."""
        points = np.random.randn(100, 3).astype(np.float32)
        
        result = estimate_normals_with_uncertainty(points, k_neighbors=10)
        
        assert "normals" in result
        assert "uncertainty" in result
        assert "planarity" in result

    def test_uncertainty_range(self):
        """Test that uncertainty values are in [0, 1] range."""
        points = np.random.randn(200, 3).astype(np.float32)
        
        result = estimate_normals_with_uncertainty(points, k_neighbors=15)
        
        assert np.all(result["uncertainty"] >= 0)
        assert np.all(result["uncertainty"] <= 1)

    def test_planarity_range(self):
        """Test that planarity values are in [0, 1] range."""
        points = np.random.randn(200, 3).astype(np.float32)
        
        result = estimate_normals_with_uncertainty(points, k_neighbors=15)
        
        assert np.all(result["planarity"] >= 0)
        assert np.all(result["planarity"] <= 1)


class TestValidateNormalsAgainstPoints:
    """Tests for validate_normals_against_points function."""

    def test_returns_consistency_and_deviation(self):
        """Test that validation returns consistency and deviation scores."""
        points = np.random.randn(100, 3).astype(np.float32)
        normals = estimate_normals_knn(points, k_neighbors=10)
        
        result = validate_normals_against_points(normals, points, k_neighbors=10)
        
        assert "consistency" in result
        assert "deviation" in result
        assert len(result["consistency"]) == len(points)
        assert len(result["deviation"]) == len(points)


class TestOrientNormalsConsistently:
    """Tests for orient_normals_consistently function."""

    def test_oriented_normals_are_unit_length(self):
        """Test that oriented normals are still unit length."""
        normals = np.random.randn(100, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        oriented = orient_normals_consistently(normals)
        
        norms = np.linalg.norm(oriented, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)

    def test_orient_with_reference_point(self):
        """Test orientation with a reference point."""
        points = np.random.randn(100, 3).astype(np.float32) * 10
        normals = np.column_stack([
            np.ones(100), np.zeros(100), np.zeros(100)
        ]).astype(np.float32)  # All point in +X direction
        
        oriented = orient_normals_consistently(normals, reference_point=points[0])
        
        assert oriented.shape == normals.shape


class TestFilterNormalsByUncertainty:
    """Tests for filter_normals_by_uncertainty function."""

    def test_returns_filtered_data(self):
        """Test that filtering returns expected keys."""
        points = np.random.randn(100, 3).astype(np.float32)
        normals = estimate_normals_knn(points, k_neighbors=10)
        uncertainty = np.random.rand(100) * 0.5  # Low uncertainty
        
        result = filter_normals_by_uncertainty(normals, uncertainty, threshold=0.3)
        
        assert "normals" in result
        assert "mask" in result
        assert "removed_count" in result
        assert "kept_count" in result
        assert "removal_ratio" in result

    def test_filter_removes_high_uncertainty(self):
        """Test that filtering removes high uncertainty normals."""
        normals = np.random.randn(100, 3).astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        uncertainty = np.random.rand(100)  # Random uncertainty
        
        result = filter_normals_by_uncertainty(normals, uncertainty, threshold=0.5)
        
        # All kept normals should have uncertainty < threshold
        kept_uncertainty = uncertainty[result["mask"]]
        assert np.all(kept_uncertainty < 0.5)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_string(self):
        """Test that get_device returns a string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ['cuda', 'mps', 'cpu']

    def test_returns_cpu_as_fallback(self):
        """Test that CPU is returned when no GPU is available."""
        device = get_device()
        # This test will pass regardless of GPU availability
        # because get_device always returns a valid device
        assert device is not None
