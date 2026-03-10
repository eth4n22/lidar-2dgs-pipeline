"""
Metrics and Pruning Module

Quality control for Gaussian surfels: planarity scoring, normal consistency,
and pruning of low-quality surfels.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


def planarity_score(covariances: np.ndarray) -> np.ndarray:
    """
    Compute planarity score for each surfel based on covariance eigenvalues.

    Planarity = (λ2 - λ1) / λ3
    - High score (≈1): Points lie on a plane (good for surfels)
    - Low score (≈0): Points are isotropic or linear

    Args:
        covariances: (N, 3, 3) covariance matrices

    Returns:
        (N,) scores in [0, 1]
    """
    # Vectorized eigenvalue computation for all matrices at once
    # eigvalsh returns (N, 3) with eigenvalues sorted ascending
    eigvals = np.linalg.eigvalsh(covariances)  # (N, 3)
    
    l1 = eigvals[:, 0]  # Smallest eigenvalue
    l2 = eigvals[:, 1]  # Middle eigenvalue
    l3 = eigvals[:, 2]  # Largest eigenvalue
    
    # Vectorized planarity computation
    # Avoid division by zero
    scores = np.where(l3 > 1e-8, (l2 - l1) / l3, 0.0)
    
    # Clamp to [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores.astype(np.float32)


def compute_covariances_from_surfels(surfels: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute covariance matrices from surfel rotation/scale representation.

    Args:
        surfels: Dictionary with 'rotation' (N, 4 quat) and 'scale' (N, 3)

    Returns:
        (N, 3, 3) covariance matrices
    """
    quats = surfels["rotation"]  # (N, 4) [x, y, z, w]
    scales = surfels["scale"]  # (N, 3) [sx, sy, sz]
    N = quats.shape[0]
    
    # Normalize quaternions (vectorized)
    q_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    q_norms = np.where(q_norms < 1e-8, 1.0, q_norms)  # Avoid division by zero
    quats_norm = quats / q_norms  # (N, 4)
    
    qx, qy, qz, qw = quats_norm[:, 0], quats_norm[:, 1], quats_norm[:, 2], quats_norm[:, 3]
    
    # Vectorized rotation matrix construction
    # Each element is (N,) array
    R = np.zeros((N, 3, 3), dtype=np.float32)
    
    # Row 0
    R[:, 0, 0] = 1 - 2*(qy**2 + qz**2)
    R[:, 0, 1] = 2*(qx*qy - qz*qw)
    R[:, 0, 2] = 2*(qx*qz + qy*qw)
    
    # Row 1
    R[:, 1, 0] = 2*(qx*qy + qz*qw)
    R[:, 1, 1] = 1 - 2*(qx**2 + qz**2)
    R[:, 1, 2] = 2*(qy*qz - qx*qw)
    
    # Row 2
    R[:, 2, 0] = 2*(qx*qz - qy*qw)
    R[:, 2, 1] = 2*(qy*qz + qx*qw)
    R[:, 2, 2] = 1 - 2*(qx**2 + qy**2)
    
    # Vectorized scale matrix construction
    # S is diagonal with scale^2 on diagonal
    scales_sq = scales ** 2  # (N, 3)
    
    # Covariance = R @ S @ R.T for each surfel
    # Use einsum for efficient batched matrix multiplication
    # R: (N, 3, 3), S: (N, 3) -> need to make S (N, 3, 3) diagonal
    S_diag = np.zeros((N, 3, 3), dtype=np.float32)
    S_diag[:, 0, 0] = scales_sq[:, 0]
    S_diag[:, 1, 1] = scales_sq[:, 1]
    S_diag[:, 2, 2] = scales_sq[:, 2]
    
    # Batch matrix multiplication: R @ S @ R.T
    # einsum: 'nij,njk->nik' for R @ S, then 'nij,nkj->nik' for (R@S) @ R.T
    RS = np.einsum('nij,njk->nik', R, S_diag)  # (N, 3, 3)
    covariances = np.einsum('nij,nkj->nik', RS, R)  # (N, 3, 3)
    
    return covariances


def normal_consistency(normals: np.ndarray,
                      neighbor_indices: np.ndarray) -> np.ndarray:
    """
    Measure angular consistency between normals and their neighbors.

    Args:
        normals: (N, 3) surface normals
        neighbor_indices: (N, K) indices of neighbors for each point

    Returns:
        (N,) consistency scores in [0, 1]
    """
    N, K = neighbor_indices.shape
    K = K - 1  # Exclude self (first index)
    
    # Normalize normals (vectorized)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normals_norm = normals / norms  # (N, 3)
    
    # Get neighbor normals for all points at once
    # neighbor_indices: (N, K+1) where first is self
    neighbor_idx = neighbor_indices[:, 1:]  # (N, K) exclude self
    neighbor_normals = normals_norm[neighbor_idx]  # (N, K, 3)
    
    # Normalize neighbor normals (vectorized)
    neighbor_norms = np.linalg.norm(neighbor_normals, axis=2, keepdims=True)
    neighbor_norms = np.where(neighbor_norms < 1e-8, 1.0, neighbor_norms)
    neighbor_normals = neighbor_normals / neighbor_norms  # (N, K, 3)
    
    # Compute dot products: normals_norm[i] @ neighbor_normals[i] for all i
    # Expand normals_norm to (N, 1, 3) for broadcasting
    normals_expanded = normals_norm[:, np.newaxis, :]  # (N, 1, 3)
    dots = np.sum(normals_expanded * neighbor_normals, axis=2)  # (N, K)
    
    # Clip and compute angles (vectorized)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (N, K)
    
    # Mean angular deviation for each point (vectorized)
    mean_angles = np.mean(angles, axis=1)  # (N,)
    
    # Convert to consistency scores (vectorized)
    consistency = 1.0 - mean_angles / np.pi
    
    return np.clip(consistency, 0.0, 1.0).astype(np.float32)


def build_neighbor_graph(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Build k-nearest neighbor graph for point cloud.

    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors

    Returns:
        (N, k+1) neighbor indices (including self)
    """
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k + 1)
    # Use int32 indices to save memory
    indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices
    return indices


def compute_all_metrics(surfels: Dict[str, np.ndarray],
                       points: np.ndarray,
                       k_neighbors: int = 20) -> Dict[str, np.ndarray]:
    """
    Compute all quality metrics for a surfel cloud.

    Args:
        surfels: Surfel dictionary
        points: Original point cloud (N, 3)
        k_neighbors: Number of neighbors for local analysis

    Returns:
        Dictionary with all metrics
    """
    N = surfels["position"].shape[0]

    # Build neighbor graph
    neighbor_graph = build_neighbor_graph(points, k_neighbors)

    # Compute covariances
    covariances = compute_covariances_from_surfels(surfels)

    # Compute metrics
    planarity = planarity_score(covariances)
    normal_cons = normal_consistency(surfels["normal"], neighbor_graph)

    return {
        "planarity": planarity,
        "normal_consistency": normal_cons,
        "covariances": covariances
    }


def filter_by_planarity(covariances: np.ndarray,
                        threshold: float = 0.4) -> np.ndarray:
    """
    Boolean mask selecting sufficiently planar surfels.

    Args:
        covariances: (N, 3, 3) covariance matrices
        threshold: Minimum planarity score (0-1)

    Returns:
        (N,) boolean mask
    """
    scores = planarity_score(covariances)
    return scores >= threshold


def filter_by_opacity(opacity: np.ndarray,
                      min_opacity: float = 0.01) -> np.ndarray:
    """
    Boolean mask for surfels with sufficient opacity.

    Args:
        opacity: (N,) opacity values
        min_opacity: Minimum opacity threshold

    Returns:
        (N,) boolean mask
    """
    return opacity >= min_opacity


def filter_by_normal_thickness(covariances: np.ndarray,
                               max_normal_variance: float = 1e-3) -> np.ndarray:
    """
    Boolean mask for surfels with appropriate normal thickness.

    Ensures the smallest eigenvalue (normal direction) is within bounds.

    Args:
        covariances: (N, 3, 3) covariance matrices
        max_normal_variance: Maximum allowed variance along normal

    Returns:
        (N,) boolean mask
    """
    # Vectorized eigenvalue computation
    eigvals = np.linalg.eigvalsh(covariances)  # (N, 3)
    min_eig = eigvals[:, 0]  # Smallest eigenvalue for each surfel
    
    # Vectorized mask: valid if 0 < min_eig <= max_normal_variance
    mask = (min_eig > 0) & (min_eig <= max_normal_variance)
    
    return mask


def prune_surfels(surfels: Dict[str, np.ndarray],
                  points: np.ndarray,
                  min_opacity: float = 0.01,
                  min_planarity: float = 0.4,
                  max_normal_variance: float = 1e-3) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Prune low-quality surfels based on multiple criteria.

    Args:
        surfels: Surfel dictionary
        points: Original point cloud (N, 3)
        min_opacity: Minimum opacity threshold
        min_planarity: Minimum planarity score
        max_normal_variance: Maximum variance along normal direction

    Returns:
        Tuple of (pruned_surfels, pruning_stats)
    """
    N = surfels["position"].shape[0]

    # Compute covariances
    covariances = compute_covariances_from_surfels(surfels)

    # Apply all filters
    mask_opacity = filter_by_opacity(surfels["opacity"], min_opacity)
    mask_planarity = filter_by_planarity(covariances, min_planarity)
    mask_normal = filter_by_normal_thickness(covariances, max_normal_variance)

    # Combine masks
    combined_mask = mask_opacity & mask_planarity & mask_normal

    # Count before pruning
    n_before = N
    n_after = np.sum(combined_mask)

    # Print statistics
    stats = {
        "n_before": n_before,
        "n_after": n_after,
        "removed": n_before - n_after,
        "opacity_filtered": np.sum(~mask_opacity),
        "planarity_filtered": np.sum(~mask_planarity & mask_opacity),
        "normal_filtered": np.sum(~mask_normal & mask_opacity & mask_planarity)
    }

    # Apply mask
    pruned_surfels = {}
    for key, value in surfels.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                pruned_surfels[key] = value
            else:
                pruned_surfels[key] = value[combined_mask]
        else:
            pruned_surfels[key] = value

    return pruned_surfels, stats


def quality_report(surfels: Dict[str, np.ndarray],
                   points: np.ndarray,
                   k_neighbors: int = 20) -> str:
    """
    Generate a text quality report for surfel cloud.

    Args:
        surfels: Surfel dictionary
        points: Original point cloud
        k_neighbors: Neighbors for local analysis

    Returns:
        Formatted quality report string
    """
    metrics = compute_all_metrics(surfels, points, k_neighbors)

    N = surfels["position"].shape[0]

    report = []
    report.append("=" * 50)
    report.append("SURFEL QUALITY REPORT")
    report.append("=" * 50)
    report.append(f"Total surfels: {N}")
    report.append("")
    report.append("Planarity Statistics:")
    report.append(f"  Mean:   {metrics['planarity'].mean():.4f}")
    report.append(f"  Std:    {metrics['planarity'].std():.4f}")
    report.append(f"  Min:    {metrics['planarity'].min():.4f}")
    report.append(f"  Max:    {metrics['planarity'].max():.4f}")
    report.append(f"  Median: {np.median(metrics['planarity']):.4f}")
    report.append("")
    report.append("Normal Consistency Statistics:")
    report.append(f"  Mean:   {metrics['normal_consistency'].mean():.4f}")
    report.append(f"  Std:    {metrics['normal_consistency'].std():.4f}")
    report.append(f"  Min:    {metrics['normal_consistency'].min():.4f}")
    report.append(f"  Max:    {metrics['normal_consistency'].max():.4f}")
    report.append("")
    report.append("=" * 50)

    return "\n".join(report)
