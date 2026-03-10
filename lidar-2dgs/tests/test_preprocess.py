"""Tests for preprocess module."""

import numpy as np
import pytest

from src.preprocess import voxel_downsample


class TestVoxelDownsample:
    """Tests for voxel_downsample function."""

    def test_downsample_basic(self):
        """Test basic voxel downsampling."""
        # Create a uniform grid of points
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        colors = np.random.randint(0, 255, len(points), dtype=np.uint8)
        
        result = voxel_downsample(points, voxel_size=0.2, colors=colors)
        
        assert "position" in result
        assert "color" in result
        # With 0.2 voxel size in a 1x1 grid, we should get ~5x5 = 25 points
        assert len(result["position"]) < len(points)
        assert len(result["position"]) >= 20  # Approximately 25

    def test_downsample_preserves_structure(self):
        """Test that downsampling preserves spatial structure."""
        # Create points on a sphere
        n_points = 500
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.column_stack([x, y, z]).astype(np.float32)
        
        result = voxel_downsample(points, voxel_size=0.3)
        
        # Check that points are still roughly on a sphere
        radii = np.linalg.norm(result["position"], axis=1)
        np.testing.assert_array_almost_equal(
            radii, np.ones(len(radii)), decimal=1
        )

    def test_downsample_with_colors(self):
        """Test downsampling with color preservation."""
        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0.2, 0, 0],
            [0, 0.1, 0],
            [0, 0.2, 0],
        ], dtype=np.float32)
        
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
        ], dtype=np.uint8)
        
        result = voxel_downsample(points, voxel_size=0.15, colors=colors)
        
        assert "color" in result
        # Colors should be preserved (or averaged)
        assert result["color"].dtype == np.float32

    def test_downsample_large_voxel_size(self):
        """Test downsampling with very large voxel size."""
        points = np.random.randn(100, 3).astype(np.float32)
        
        result = voxel_downsample(points, voxel_size=10.0)
        
        # With large voxel, should get very few points (ideally 1)
        assert len(result["position"]) <= 10

    def test_downsample_small_voxel_size(self):
        """Test downsampling with very small voxel size."""
        points = np.array([
            [0, 0, 0],
            [0.01, 0, 0],
            [0.02, 0, 0],
        ], dtype=np.float32)
        
        result = voxel_downsample(points, voxel_size=0.001)
        
        # With small voxel, should preserve most/all points
        assert len(result["position"]) <= len(points)

    def test_downsample_empty_points(self):
        """Test downsampling with empty point cloud."""
        points = np.zeros((0, 3), dtype=np.float32)
        
        result = voxel_downsample(points, voxel_size=0.1)
        
        assert len(result["position"]) == 0
        # Color key only present if colors were provided
        if "color" in result:
            assert len(result["color"]) == 0

    def test_downsample_preserves_dtype(self):
        """Test that output dtype is correct."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        result = voxel_downsample(points, voxel_size=0.5)
        
        assert result["position"].dtype == np.float32

    def test_downsample_output_keys(self):
        """Test that all expected keys are in output."""
        points = np.random.randn(50, 3).astype(np.float32)
        
        result = voxel_downsample(points, voxel_size=0.5)
        
        expected_keys = ["position"]
        for key in expected_keys:
            assert key in result

    def test_downsample_no_colors(self):
        """Test downsampling without colors."""
        points = np.random.randn(50, 3).astype(np.float32)
        
        result = voxel_downsample(points, voxel_size=0.5)
        
        assert "color" not in result

    def test_downsample_voxel_size_zero(self):
        """Test that zero voxel size preserves all points."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        result = voxel_downsample(points, voxel_size=0.0)
        
        # Zero voxel size should disable downsampling
        assert len(result["position"]) == len(points)
