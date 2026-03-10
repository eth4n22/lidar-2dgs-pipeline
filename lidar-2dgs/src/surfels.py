"""
Surfel Construction Module

Build 2D Gaussian surfels from points, normals, and colors.
Ported from Daya's proven implementation with enhancements.
"""

from typing import Dict, Tuple, Optional
import numpy as np


def tangent_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tangent and bitangent vectors orthogonal to a normal.

    Uses Gram-Schmidt orthogonalization to ensure a right-handed
    coordinate system with the normal as the Z-axis.

    Args:
        normal: (3,) unit normal vector

    Returns:
        Tuple of (tangent, bitangent), both (3,) vectors
    """
    if normal.shape != (3,):
        raise ValueError(f"Expected (3,) normal, got {normal.shape}")

    # Ensure unit normal
    n = normal / (np.linalg.norm(normal) + 1e-8)

    # Pick an arbitrary vector not parallel to n
    t1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.abs(np.dot(t1, n)) > 0.9:
        t1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Gram-Schmidt orthogonalization
    t1 = t1 - np.dot(t1, n) * n
    t1 = t1 / (np.linalg.norm(t1) + 1e-8)

    # Bitangent = cross(normal, tangent)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)

    return t1, t2


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    """
    Build rotation matrix from normal vector.

    Creates a 3x3 rotation matrix where columns are:
    [tangent, bitangent, normal]

    Args:
        normal: (3,) unit normal vector

    Returns:
        (3, 3) rotation matrix
    """
    tangent, bitangent = tangent_basis_from_normal(normal)
    R = np.column_stack([tangent, bitangent, normal])
    return R


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Uses the Sheffield method for numerical stability.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (4,) quaternion [x, y, z, w]
    """
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) matrix, got {R.shape}")

    # Ensure proper rotation matrix
    if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError("Matrix must be a proper rotation (det=1)")

    trace = np.trace(R)
    q = np.zeros(4, dtype=np.float32)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        # Find largest diagonal element
        i = np.argmax(np.diag(R))
        j, k = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][i]

        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q[i] = 0.5 * s

        if i == 0:
            q[1] = (R[1, 0] + R[0, 1]) / s
            q[2] = (R[2, 0] + R[0, 2]) / s
            q[3] = (R[1, 2] - R[2, 1]) / s
        elif i == 1:
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = (R[2, 0] - R[0, 2]) / s
        else:
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[3] = (R[0, 1] - R[1, 0]) / s

    # Normalize quaternion
    q = q / (np.linalg.norm(q) + 1e-8)

    return q


def build_surfel_covariance(normal: np.ndarray,
                            sigma_tangent: float = 0.05,
                            sigma_normal: float = 0.002) -> np.ndarray:
    """
    Build an anisotropic covariance matrix for a 3D Gaussian surfel.

    Creates a "pancake" Gaussian that is thin along the normal direction
    and wider along the tangent plane.

    Args:
        normal: (3,) unit normal vector
        sigma_tangent: Standard deviation along tangent plane (meters)
        sigma_normal: Standard deviation along normal direction (meters)

    Returns:
        (3, 3) covariance matrix
    """
    # Build rotation matrix
    R = rotation_matrix_from_normal(normal)

    # Diagonal covariance in tangent-normal space
    S = np.diag([
        sigma_tangent ** 2,
        sigma_tangent ** 2,
        sigma_normal ** 2
    ]).astype(np.float32)

    # Rotate to world coordinates: cov = R @ S @ R^T
    cov = R @ S @ R.T

    return cov


def build_surfels(points: np.ndarray,
                  normals: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  sigma_tangent: float = 0.05,
                  sigma_normal: float = 0.002,
                  opacity: float = 0.8) -> Dict[str, np.ndarray]:
    """
    Build 2D Gaussian surfels from points, normals, and colors.

    Each surfel represents a 2D Gaussian in the tangent plane at each point,
    producing surface-aligned Gaussians suitable for 2DGS rendering.

    Args:
        points: (N, 3) xyz coordinates
        normals: (N, 3) surface normals
        colors: Optional (N, 3) rgb values (0-255)
        sigma_tangent: Spread along tangent plane (meters)
        sigma_normal: Thickness along normal (meters)
        opacity: Default opacity for all surfels

    Returns:
        Dictionary with surfel attributes:
            - position: (N, 3) xyz
            - normal: (N, 3)
            - tangent: (N, 3)
            - bitangent: (N, 3)
            - opacity: (N,)
            - scale: (N, 3)
            - rotation: (N, 4) quaternion [x, y, z, w]
            - color: (N, 3) rgb (0-1)
    """
    N = points.shape[0]

    # Validate inputs
    if N == 0:
        return {
            "position": np.empty((0, 3), dtype=np.float32),
            "normal": np.empty((0, 3), dtype=np.float32),
            "tangent": np.empty((0, 3), dtype=np.float32),
            "bitangent": np.empty((0, 3), dtype=np.float32),
            "opacity": np.empty((0,), dtype=np.float32),
            "scale": np.empty((0, 3), dtype=np.float32),
            "rotation": np.empty((0, 4), dtype=np.float32),
            "color": np.empty((0, 3), dtype=np.float32)
        }
    
    if points.shape != normals.shape:
        raise ValueError(f"Points and normals must have same shape: {points.shape} vs {normals.shape}")
    
    if not np.isfinite(points).all():
        nan_count = np.sum(~np.isfinite(points))
        raise ValueError(f"Points contain {nan_count} non-finite values (NaN or Inf). "
                        f"Please clean your point cloud before building surfels.")
    
    if not np.isfinite(normals).all():
        nan_count = np.sum(~np.isfinite(normals))
        raise ValueError(f"Normals contain {nan_count} non-finite values (NaN or Inf). "
                        f"Please recompute normals with valid values.")
    
    if colors is not None:
        if len(colors) != N:
            raise ValueError(f"Colors length must match points: {len(colors)} vs {N}")
        if not np.isfinite(colors).all():
            nan_count = np.sum(~np.isfinite(colors))
            raise ValueError(f"Colors contain {nan_count} non-finite values.")
        if colors.ndim == 2 and colors.shape[1] != 3:
            raise ValueError(f"Colors must be (N, 3) RGB values, got shape {colors.shape}")
    
    if sigma_tangent <= 0 or sigma_normal <= 0:
        raise ValueError(f"Sigma values must be positive: sigma_tangent={sigma_tangent}, sigma_normal={sigma_normal}")
    
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"Opacity must be in [0, 1], got {opacity}")

    # Normalize normals
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm < 1e-8] = 1.0
    normals = normals / normals_norm

    # Initialize arrays
    tangents = np.zeros((N, 3), dtype=np.float32)
    bitangents = np.zeros((N, 3), dtype=np.float32)
    rotations = np.zeros((N, 4), dtype=np.float32)
    scales = np.full((N, 3), [sigma_tangent, sigma_tangent, sigma_normal], dtype=np.float32)

    # Default opacity
    opacity_array = np.full(N, opacity, dtype=np.float32)

    # Default color (white)
    if colors is None:
        color_array = np.ones((N, 3), dtype=np.float32)
    else:
        color_array = (colors / 255.0).astype(np.float32)

    # Compute tangent basis and rotation for all points (vectorized)
    # Vectorized computation for better performance
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    # Pick arbitrary vectors not parallel to normals
    # Shape: (N, 3)
    t1_candidates = np.tile([1.0, 0.0, 0.0], (N, 1))
    parallel_mask = np.abs(np.sum(t1_candidates * n, axis=1)) > 0.9
    t1_candidates[parallel_mask] = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt orthogonalization (vectorized)
    # t1 = t1 - (t1 · n) * n
    dot_t1_n = np.sum(t1_candidates * n, axis=1, keepdims=True)
    t1 = t1_candidates - dot_t1_n * n
    t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-8)
    
    # Bitangent = n × t1 (vectorized)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-8)
    
    tangents = t1
    bitangents = t2
    
    # Build rotation matrices (vectorized)
    # R = [t1, t2, n] as columns
    R = np.stack([t1, t2, n], axis=1)  # Shape: (N, 3, 3)
    
    # Convert rotation matrices to quaternions (vectorized)
    # Using Sheffield method for numerical stability
    trace = np.trace(R, axis1=1, axis2=2)  # Shape: (N,)
    
    # Initialize quaternions
    rotations = np.zeros((N, 4), dtype=np.float32)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = 0.5 / np.sqrt(trace[mask1] + 1.0)
    R_mask1 = R[mask1]
    rotations[mask1, 0] = (R_mask1[:, 2, 1] - R_mask1[:, 1, 2]) * s1
    rotations[mask1, 1] = (R_mask1[:, 0, 2] - R_mask1[:, 2, 0]) * s1
    rotations[mask1, 2] = (R_mask1[:, 1, 0] - R_mask1[:, 0, 1]) * s1
    rotations[mask1, 3] = 0.25 / s1
    
    # Case 2: trace <= 0
    mask2 = ~mask1
    # Extract diagonal elements manually (np.diag doesn't support axis params in older NumPy)
    R_mask2 = R[mask2]
    diags = np.array([R_mask2[:, 0, 0], R_mask2[:, 1, 1], R_mask2[:, 2, 2]]).T  # shape: (n_mask2, 3)
    i = np.argmax(diags, axis=1)
    # Lookup table for (j, k) based on i value
    jk_lookup = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)  # rows: i, cols: j, k
    j = jk_lookup[i, 0]
    k = jk_lookup[i, 1]
    
    # Use row indices for proper NumPy advanced indexing
    row_indices = np.arange(len(i))
    s2 = 2.0 * np.sqrt(1.0 + R_mask2[row_indices, i, i] - R_mask2[row_indices, j, j] - R_mask2[row_indices, k, k])
    rotations[mask2, i] = 0.5 * s2
    rotations[mask2, j] = (R_mask2[row_indices, i, j] + R_mask2[row_indices, j, i]) / s2
    rotations[mask2, k] = (R_mask2[row_indices, k, i] + R_mask2[row_indices, i, k]) / s2
    
    # Set w component (index 3) based on which diagonal element is largest
    # If i == 0: q[3] = (R[1,2] - R[2,1]) / s
    # If i == 1: q[3] = (R[2,0] - R[0,2]) / s
    # If i == 2: q[3] = (R[0,1] - R[1,0]) / s
    w_values = np.zeros(len(i), dtype=np.float32)
    mask_i0 = (i == 0)
    mask_i1 = (i == 1)
    mask_i2 = (i == 2)
    
    if np.any(mask_i0):
        w_values[mask_i0] = (R_mask2[mask_i0, 1, 2] - R_mask2[mask_i0, 2, 1]) / s2[mask_i0]
    if np.any(mask_i1):
        w_values[mask_i1] = (R_mask2[mask_i1, 2, 0] - R_mask2[mask_i1, 0, 2]) / s2[mask_i1]
    if np.any(mask_i2):
        w_values[mask_i2] = (R_mask2[mask_i2, 0, 1] - R_mask2[mask_i2, 1, 0]) / s2[mask_i2]
    
    rotations[mask2, 3] = w_values
    
    # Normalize quaternions (add epsilon to prevent division by zero)
    q_norm = np.sqrt(np.sum(rotations**2, axis=1)) + 1e-8
    rotations = rotations / q_norm[:, np.newaxis]

    return {
        "position": points.astype(np.float32),
        "normal": normals.astype(np.float32),
        "tangent": tangents,
        "bitangent": bitangents,
        "opacity": opacity_array,
        "scale": scales,
        "rotation": rotations,
        "color": color_array
    }


def build_surfels_with_covariance(points: np.ndarray,
                                   normals: np.ndarray,
                                   colors: Optional[np.ndarray] = None,
                                   sigma_tangent: float = 0.05,
                                   sigma_normal: float = 0.002,
                                   opacity: float = 0.8) -> Dict[str, np.ndarray]:
    """
    Build surfels with explicit covariance matrices.

    Alternative to build_surfels that returns full covariance matrices
    instead of rotation/scale decomposition. Useful for compatibility
    with certain renderers.

    Args:
        points: (N, 3) xyz coordinates
        normals: (N, 3) surface normals
        colors: Optional (N, 3) rgb values (0-255)
        sigma_tangent: Spread along tangent plane (meters)
        sigma_normal: Thickness along normal (meters)
        opacity: Default opacity for all surfels

    Returns:
        Dictionary with surfel attributes including 'covariance' (N, 3, 3)
    """
    N = points.shape[0]

    # Normalize normals
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm < 1e-8] = 1.0
    normals = normals / normals_norm

    # Initialize arrays
    covs = np.zeros((N, 3, 3), dtype=np.float32)

    # Default opacity
    opacity_array = np.full(N, opacity, dtype=np.float32)

    # Default color (white)
    if colors is None:
        color_array = np.ones((N, 3), dtype=np.float32)
    else:
        color_array = (colors / 255.0).astype(np.float32)

    # Compute covariance for each point
    for i in range(N):
        covs[i] = build_surfel_covariance(normals[i], sigma_tangent, sigma_normal)

    return {
        "position": points.astype(np.float32),
        "normal": normals.astype(np.float32),
        "covariance": covs,
        "opacity": opacity_array,
        "color": color_array
    }


def merge_surfels(surfels_list: list) -> Dict[str, np.ndarray]:
    """
    Merge multiple surfel dictionaries into one.

    Args:
        surfels_list: List of surfel dictionaries with same keys

    Returns:
        Merged surfel dictionary
    """
    if not surfels_list:
        raise ValueError("Cannot merge empty surfel list")

    # Check all dicts have same keys
    keys = set(surfels_list[0].keys())
    for s in surfels_list[1:]:
        if set(s.keys()) != keys:
            raise ValueError("All surfel dictionaries must have same keys")

    merged = {}
    for key in keys:
        if isinstance(surfels_list[0][key], np.ndarray):
            # Stack arrays
            if surfels_list[0][key].ndim == 0:
                # Scalar array
                merged[key] = np.concatenate([s[key].reshape(-1) for s in surfels_list])
            else:
                merged[key] = np.vstack([s[key] for s in surfels_list])
        elif isinstance(surfels_list[0][key], (int, float)):
            # Scalar values - keep first
            merged[key] = surfels_list[0][key]

    return merged
