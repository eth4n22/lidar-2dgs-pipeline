"""
Preprocessing Module

Point cloud preprocessing operations: outlier removal, downsampling, etc.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree


def remove_outliers(points: np.ndarray, 
                   colors: Optional[np.ndarray] = None,
                   k: int = 20,
                   std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers from point cloud using distance-based method.
    
    Args:
        points: (N, 3) xyz coordinates
        colors: (N, 3) rgb values, or None
        k: Number of neighbors to use for distance computation
        std_ratio: Threshold multiplier for outlier detection
        
    Returns:
        Tuple of (filtered_points, mask) where mask is boolean array
    """
    if points.shape[0] <= k:
        return points, np.ones(points.shape[0], dtype=bool)
    
    # Build KDTree for efficient neighbor search
    tree = KDTree(points)
    
    # Compute mean distance to k nearest neighbors for each point
    distances, _ = tree.query(points, k=k+1)  # k+1 because point itself is included
    mean_distances = distances[:, 1:].mean(axis=1)  # exclude self
    
    # Compute statistics of mean distances
    mean_dist = np.mean(mean_distances)
    std_dist = np.std(mean_distances)
    
    # Create mask for inliers
    threshold = mean_dist + std_ratio * std_dist
    mask = mean_distances <= threshold
    
    filtered_points = points[mask]
    
    return filtered_points, mask


def voxel_downsample(points: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     voxel_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid filtering.
    
    Args:
        points: (N, 3) xyz coordinates
        colors: (N, 3) rgb values, or None
        voxel_size: Size of voxels in meters
        
    Returns:
        Tuple of (centroids, indices) where centroids are voxel centers
        and indices are original point indices
    """
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64)
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    
    # Find unique voxels
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    # Compute centroids of each voxel
    centroids = points[unique_indices]
    
    return centroids, unique_indices


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Center and scale points to unit sphere.
    
    Args:
        points: (N, 3) xyz coordinates
        
    Returns:
        Tuple of (normalized_points, (mean, scale)) where:
        - normalized_points: (N, 3) centered and scaled xyz coordinates
        - mean: (3,) original centroid
        - scale: float scale factor
    """
    if points.shape[0] == 0:
        return points, (np.zeros(3), 1.0)
    
    # Compute centroid
    mean = np.mean(points, axis=0)
    
    # Center points
    centered = points - mean
    
    # Compute scale based on max extent
    extents = np.max(centered, axis=0) - np.min(centered, axis=0)
    max_extent = np.max(extents)
    
    if max_extent > 0:
        scale = 1.0 / max_extent
        normalized = centered * scale
    else:
        scale = 1.0
        normalized = centered
    
    return normalized, (mean, scale)


def denormalize_points(normalized: np.ndarray, 
                       params: Tuple[np.ndarray, float]) -> np.ndarray:
    """
    Denormalize points back to original coordinates.
    
    Args:
        normalized: (N, 3) normalized xyz coordinates
        params: Tuple of (mean, scale) from normalize_points
        
    Returns:
        Denormalized (N, 3) xyz coordinates
    """
    mean, scale = params
    return normalized / scale + mean
