"""
Preprocessing Module

Point cloud preprocessing operations: outlier removal, downsampling, etc.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# Optional FAISS support for large datasets
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def calculate_voxel_size_for_ratio(points: np.ndarray, target_ratio: float = 100.0) -> float:
    """
    Calculate voxel size to achieve target downsampling ratio.
    
    Args:
        points: (N, 3) xyz coordinates
        target_ratio: Desired downsampling ratio (e.g., 100.0 for 100:1)
    
    Returns:
        Voxel size in meters
    """
    if len(points) == 0:
        return 0.01
    
    # Calculate bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    # Volume of bounding box
    volume = np.prod(bbox_size)
    
    # Target number of voxels for desired ratio
    target_voxels = len(points) / target_ratio
    
    # Voxel size = cube root of (volume / target_voxels)
    if target_voxels > 0 and volume > 0:
        voxel_size = np.cbrt(volume / target_voxels)
        # Clamp to reasonable range (0.001m to 1.0m)
        voxel_size = np.clip(voxel_size, 0.001, 1.0)
    else:
        voxel_size = 0.01
    
    return float(voxel_size)


def remove_outliers_zscore(points: np.ndarray,
                           z_score_threshold: float = 3.0) -> Dict[str, np.ndarray]:
    """
    Remove statistical outliers from point cloud using z-score method.

    For each point, computes mean distance to k neighbors.
    Points with z-score above threshold are removed.

    Args:
        points: (N, 3) xyz coordinates
        z_score_threshold: Number of standard deviations for outlier detection

    Returns:
        Dictionary with 'position' and 'mask' keys
    """
    N = points.shape[0]

    # Build kd-tree
    tree = cKDTree(points)

    # Find k neighbors for each point (including self)
    k = min(20, N - 1)
    _, indices = tree.query(points, k=k + 1)
    # Use int32 indices to save memory
    indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices

    # Compute mean distance to neighbors (vectorized)
    # Get all neighbors: (N, k, 3)
    neighbors = points[indices[:, 1:]]
    # Compute distances: (N, k)
    distances = np.linalg.norm(neighbors - points[:, np.newaxis, :], axis=2)
    # Mean distance excluding self (axis=1)
    mean_distances = distances.mean(axis=1).astype(np.float32)

    # Compute z-scores
    mean_d = np.mean(mean_distances)
    std_d = np.std(mean_distances)

    if std_d < 1e-8:
        # All points have same distance - keep all
        z_scores = np.zeros(N)
    else:
        z_scores = (mean_distances - mean_d) / std_d

    # Keep points below threshold
    mask = z_scores < z_score_threshold

    return {
        "position": points[mask],
        "mask": mask,
        "removed_count": N - np.sum(mask)
    }


def remove_outliers_statistical(points: np.ndarray,
                                k: int = 20,
                                std_multiplier: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Remove outliers using statistical outlier removal.

    For each point, analyzes distances to k nearest neighbors.
    Points with mean distance > std_multiplier * global_std are removed.

    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors to analyze
        std_multiplier: Threshold multiplier for global std

    Returns:
        Dictionary with 'position' and 'mask' keys
    
    Raises:
        ValueError: If inputs are invalid
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points array, got shape {points.shape}")
    
    if len(points) == 0:
        return {"position": np.empty((0, 3), dtype=np.float32), "mask": np.array([], dtype=bool), "removed_count": 0}
    
    if not np.isfinite(points).all():
        nan_count = np.sum(~np.isfinite(points))
        raise ValueError(f"Input points contain {nan_count} non-finite values. "
                        f"Please clean your point cloud before outlier removal.")
    
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    if std_multiplier <= 0:
        raise ValueError(f"std_multiplier must be positive, got {std_multiplier}")
    
    N = points.shape[0]
    k = min(k, N - 1)
    
    if k == 0:
        # Not enough points for outlier removal
        return {"position": points.copy(), "mask": np.ones(N, dtype=bool), "removed_count": 0}

    # For large datasets, use FAISS or chunked processing to avoid memory issues
    LARGE_DATASET_THRESHOLD = 5000000  # 5M points
    
    if N > LARGE_DATASET_THRESHOLD and HAS_FAISS:
        return _remove_outliers_faiss(points, k, std_multiplier)
    elif N > LARGE_DATASET_THRESHOLD:
        return _remove_outliers_chunked(points, k, std_multiplier)
    else:
        return _remove_outliers_small(points, k, std_multiplier)


def _remove_outliers_small(points: np.ndarray,
                          k: int,
                          std_multiplier: float) -> Dict[str, np.ndarray]:
    """Small dataset outlier removal using full cKDTree."""
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)
    # Use int32 indices to save memory
    indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices
    
    # Compute mean distance to neighbors
    neighbors = points[indices[:, 1:]]
    distances = np.linalg.norm(neighbors - points[:, np.newaxis, :], axis=2)
    mean_dists = distances.mean(axis=1).astype(np.float32)
    
    # Compute global statistics
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_multiplier * global_std
    
    mask = mean_dists < threshold
    N = len(points)

    return {
        "position": points[mask],
        "mask": mask,
        "removed_count": N - np.sum(mask)
    }


def _remove_outliers_chunked(points: np.ndarray,
                             k: int,
                             std_multiplier: float) -> Dict[str, np.ndarray]:
    """
    Chunked outlier removal for large datasets.
    
    Processes points in chunks to avoid loading entire dataset into memory.
    """
    N = points.shape[0]
    chunk_size = 50000  # 50K points per chunk
    
    # Build KD-tree once on full dataset (can't be avoided for neighbor search)
    tree = cKDTree(points)
    
    # Compute local densities in chunks
    local_densities = np.zeros(N, dtype=np.float32)
    
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        
        # Query neighbors for this chunk
        _, indices = tree.query(points[chunk_start:chunk_end], k=k + 1)
        indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices
        
        # Compute mean distances
        neighbors = points[indices[:, 1:]]
        distances = np.linalg.norm(neighbors - points[chunk_start:chunk_end, np.newaxis, :], axis=2)
        mean_dists = distances.mean(axis=1)
        
        local_densities[chunk_start:chunk_end] = mean_dists
    
    # Compute global statistics
    global_mean = np.mean(local_densities)
    global_std = np.std(local_densities)
    threshold = global_mean + std_multiplier * global_std
    
    mask = local_densities < threshold
    
    return {
        "position": points[mask],
        "mask": mask,
        "removed_count": N - np.sum(mask)
    }


def _remove_outliers_faiss(points: np.ndarray,
                            k: int,
                            std_multiplier: float) -> Dict[str, np.ndarray]:
    """
    Outlier removal using FAISS for O(N log N) scaling.
    
    Much faster and more memory-efficient for large datasets.
    """
    N = points.shape[0]
    points_f32 = np.ascontiguousarray(points, dtype=np.float32)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(3)
    index.add(points_f32)
    
    # Search for k+1 neighbors (including self)
    distances, indices = index.search(points_f32, k + 1)
    indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices
    
    # Clean up FAISS index
    del index
    
    # Exclude self and compute mean distances
    mean_dists = distances[:, 1:].mean(axis=1).astype(np.float32)
    
    # Compute global statistics
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_multiplier * global_std
    
    mask = mean_dists < threshold
    N = len(points)

    return {
        "position": points[mask],
        "mask": mask,
        "removed_count": N - np.sum(mask)
    }


def voxel_downsample(points: np.ndarray,
                     voxel_size: float = 0.01,
                     colors: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Downsample point cloud using voxel grid filtering.

    For each voxel, computes centroid of all points within it.

    Args:
        points: (N, 3) xyz coordinates
        voxel_size: Size of voxels in meters
        colors: Optional (N, 3) rgb values (0-255)

    Returns:
        Dictionary with 'position' and optionally 'color' keys
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Input validation
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points array, got shape {points.shape}")
    
    if not np.isfinite(points).all():
        nan_count = np.sum(~np.isfinite(points))
        raise ValueError(f"Input points contain {nan_count} non-finite values (NaN or Inf). "
                        f"Please clean your point cloud before downsampling.")
    
    if colors is not None:
        if colors.shape[0] != points.shape[0]:
            raise ValueError(f"Colors must have same length as points: {colors.shape[0]} vs {points.shape[0]}")
        if not np.isfinite(colors).all():
            nan_count = np.sum(~np.isfinite(colors))
            raise ValueError(f"Colors contain {nan_count} non-finite values.")
    
    if voxel_size <= 0:
        if colors is not None:
            return {"position": points.copy(), "color": colors.copy() if hasattr(colors, 'copy') else colors}
        return {"position": points.copy()}
    
    if len(points) == 0:
        result = {"position": np.empty((0, 3), dtype=np.float32)}
        if colors is not None:
            result["color"] = np.empty((0, 3), dtype=np.float32)
        return result
    
    # Compute voxel indices
    voxel_min = points.min(axis=0)
    voxel_indices = ((points - voxel_min) / voxel_size).astype(np.int32)
    
    # Optimized voxel key computation using integer encoding (avoids slow Python tuples)
    # Using large prime multipliers for hash-based voxel key
    PRIMES = np.array([73856093, 19349663, 83492791], dtype=np.int64)
    voxel_keys = np.dot(voxel_indices, PRIMES)
    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    
    # Compute centroids efficiently using vectorized operations
    n_voxels = len(unique_keys)
    centroids = np.zeros((n_voxels, 3), dtype=np.float32)
    counts = np.zeros(n_voxels, dtype=np.int32)
    
    # Use bincount for efficient aggregation
    for dim in range(3):
        np.add.at(centroids[:, dim], inverse, points[:, dim])
    np.add.at(counts, inverse, 1)
    centroids = centroids / counts[:, np.newaxis]
    
    result = {"position": centroids.astype(np.float32)}
    
    if colors is not None:
        # Ensure colors is 2D (N, 3)
        colors = np.atleast_2d(colors)
        if colors.ndim == 1:
            # 1D array - treat as grayscale per point
            colors = colors.reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)
        elif colors.shape[1] != 3:
            colors = colors[:, :3]  # Take first 3 channels
        
        centroids_colors = np.zeros((n_voxels, 3), dtype=np.float32)
        for dim in range(3):
            np.add.at(centroids_colors[:, dim], inverse, colors[:, dim])
        centroids_colors = (centroids_colors / counts[:, np.newaxis]).astype(np.float32)
        result["color"] = centroids_colors
    
    return result


def compute_point_density(points: np.ndarray,
                          k: int = 20) -> np.ndarray:
    """
    Compute local density for each point.

    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors

    Returns:
        (N,) density values (mean distance to k neighbors, inverted)
    """
    N = points.shape[0]
    k = min(k, N - 1)

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)

    # Mean distance to neighbors (inverted to get density)
    mean_dists = distances[:, 1:].mean(axis=1)  # Exclude self

    # Invert: higher density = smaller distances
    density = 1.0 / (mean_dists + 1e-8)

    return density


def normal_filter_by_density(points: np.ndarray,
                             normals: np.ndarray,
                             k: int = 20,
                             density_threshold: float = 0.1) -> np.ndarray:
    """
    Filter normals based on local point density.

    Low-density areas (edges, sparse regions) get flagged.

    Args:
        points: (N, 3) xyz coordinates
        normals: (N, 3) surface normals
        k: Number of neighbors for density computation
        density_threshold: Minimum relative density

    Returns:
        (N,) boolean mask for reliable normals
    """
    density = compute_point_density(points, k)

    # Normalize density
    density_norm = density / (density.max() + 1e-8)

    return density_norm >= density_threshold


def preprocess_point_cloud(points: np.ndarray,
                           colors: Optional[np.ndarray] = None,
                           voxel_size: Optional[float] = None,
                           outlier_threshold: Optional[float] = 3.0,
                           outlier_k: int = 20) -> Dict[str, np.ndarray]:
    """
    Complete preprocessing pipeline for point cloud.

    Args:
        points: (N, 3) xyz coordinates
        colors: Optional (N, 3) rgb values (0-255)
        voxel_size: Optional voxel size for downsampling
        outlier_threshold: Optional z-score threshold for outlier removal
        outlier_k: Number of neighbors for outlier detection

    Returns:
        Dictionary with 'position', 'color' (if provided), and 'stats'
    """
    result = {
        "position": points.copy(),
        "stats": {}
    }

    if colors is not None:
        result["color"] = colors.copy()

    original_count = points.shape[0]

    # Step 1: Outlier removal
    if outlier_threshold is not None:
        outlier_result = remove_outliers_statistical(
            points, k=outlier_k, std_multiplier=outlier_threshold
        )
        result["position"] = outlier_result["position"]
        if colors is not None:
            result["color"] = outlier_result["color"][outlier_result["mask"]]
        result["stats"]["outliers_removed"] = outlier_result["removed_count"]

    # Step 2: Voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        voxel_result = voxel_downsample(
            result["position"],
            voxel_size=voxel_size,
            colors=result.get("color")
        )
        result["position"] = voxel_result["position"]
        if "color" in voxel_result:
            result["color"] = voxel_result["color"]

    final_count = result["position"].shape[0]
    result["stats"]["original_count"] = original_count
    result["stats"]["final_count"] = final_count
    result["stats"]["reduction_ratio"] = 1.0 - final_count / original_count

    return result
