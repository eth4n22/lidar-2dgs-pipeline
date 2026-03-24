"""
Surfel Construction Module

Build 2D Gaussian surfels from points, normals, and colors.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh


def tangent_basis_from_normal(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tangent and bitangent vectors orthogonal to a normal.
    
    For each normal, finds an orthogonal tangent and bitangent pair.
    VECTORIZED implementation for massive speedup.
    
    Args:
        normals: (N, 3) unit normal vectors
        
    Returns:
        Tuple of (tangents, bitangents), both (N, 3) arrays
    """
    n = normals.shape[0]
    normals = normals.astype(np.float64)
    
    # Handle empty input
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    
    # Normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    normals_normed = normals / norms
    
    # Find arbitrary vectors not parallel to each normal
    # Use different strategies based on normal direction
    arbitrary = np.zeros((n, 3), dtype=np.float64)
    
    # For normals with |x| < 0.9, use [1,0,0]; otherwise use [0,1,0]
    mask_x = np.abs(normals_normed[:, 0]) < 0.9
    arbitrary[mask_x] = np.array([1.0, 0.0, 0.0])
    arbitrary[~mask_x] = np.array([0.0, 1.0, 0.0])
    
    # Vectorized cross product: tangent = normal × arbitrary
    tangents = np.cross(normals_normed, arbitrary)
    
    # Normalize tangents
    t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    t_norms[t_norms < 1e-8] = 1.0
    tangents = tangents / t_norms
    
    # Vectorized cross product: bitangent = normal × tangent
    bitangents = np.cross(normals_normed, tangents)
    
    # Normalize bitangents (should already be unit, but ensure)
    bt_norms = np.linalg.norm(bitangents, axis=1, keepdims=True)
    bt_norms[bt_norms < 1e-8] = 1.0
    bitangents = bitangents / bt_norms
    
    return tangents, bitangents


def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: (3, 3) rotation matrix
        
    Returns:
        (4,) quaternion [x, y, z, w]
    """
    # Based on Shewchuk's method for numerical stability
    m = R.astype(np.float64)
    
    # Compute trace of matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([
            (m[2, 1] - m[1, 2]) * s,
            (m[0, 2] - m[2, 0]) * s,
            (m[1, 0] - m[0, 1]) * s,
            0.25 / s
        ])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s
        ])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s
        ])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s
        ])
    
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    return q


def compute_covar_from_knn(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute local covariance matrices using KNN.
    
    For each point, finds k nearest neighbors and computes the
    covariance matrix of the local neighborhood.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors
        
    Returns:
        (N, 3, 3) covariance matrices
    """
    n = points.shape[0]
    covariances = np.zeros((n, 3, 3), dtype=np.float64)
    
    if n < 2:
        return covariances
    
    tree = KDTree(points)
    k_eff = min(k + 1, n)  # +1 to include self
    
    for i in range(n):
        distances, indices = tree.query(points[i], k=k_eff)
        neighbors = points[indices[1:]]  # Exclude self
        
        # Compute centroid
        centroid = np.mean(neighbors, axis=0)
        
        # Center the data
        centered = neighbors - centroid
        
        # Compute covariance matrix
        cov = centered.T @ centered / (len(neighbors) - 1)
        
        # Ensure symmetry (numerical stability)
        covariances[i] = (cov + cov.T) / 2.0
    
    return covariances


def scales_from_covariance(covariances: np.ndarray, base_scale: float = 1.0) -> np.ndarray:
    """
    Compute scale parameters from covariance matrices.
    
    Extracts the sqrt of eigenvalues as scale factors.
    
    Args:
        covariances: (N, 3, 3) covariance matrices
        base_scale: Base scale multiplier
        
    Returns:
        (N, 3) scale factors
    """
    n = covariances.shape[0]
    
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    # Vectorized: compute all eigenvalues at once
    # Reshape to (-1, 3, 3) for batch processing
    covariances_flat = covariances.reshape(-1, 3, 3)
    all_eigenvalues = np.linalg.eigvalsh(covariances_flat)  # (N, 3)
    
    # Ensure positive and compute sqrt
    all_eigenvalues = np.maximum(all_eigenvalues, 1e-10)
    scales = np.sqrt(all_eigenvalues) * base_scale
    
    return scales


def build_surfels(points: np.ndarray,
                  normals: np.ndarray,
                  colors: np.ndarray,
                  scales: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Build 2D Gaussian surfels from points, normals, and colors.
    
    Each surfel represents a 2D Gaussian in the tangent plane at each point.
    
    Args:
        points: (N, 3) xyz coordinates
        normals: (N, 3) surface normals
        colors: (N, 3) rgb values (0-255 or 0-1)
        scales: Optional (N, 3) scale factors
        
    Returns:
        Dictionary with surfel attributes:
            - position: (N, 3) xyz
            - normal: (N, 3) 
            - tangent: (N, 3)
            - bitangent: (N, 3)
            - opacity: (N,) default 1.0
            - scale: (N, 3)
            - rotation: (N, 4) quaternion [x,y,z,w]
            - color: (N, 3) rgb (0-1)
    """
    n = points.shape[0]
    
    # Normalize normals
    normals = normals.astype(np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    normals = normals / norms
    
    # Compute tangent and bitangent
    tangents, bitangents = tangent_basis_from_normal(normals)
    
    # Build rotation matrix from normal, tangent, bitangent
    # The rotation matrix maps from local frame to world frame
    # Columns are: tangent, bitangent, normal (or similar)
    # VECTORIZED: Compute all quaternions at once
    rotations = np.zeros((n, 4), dtype=np.float64)
    
    # Stack rotation matrices: R = [tangent, bitangent, normal]
    # Each row i contains the rotation matrix for surfel i
    R_matrices = np.stack([tangents, bitangents, normals], axis=2)  # (n, 3, 3)
    
    # Vectorized quaternion conversion for all matrices
    # Based on Shewchuk's method
    m = R_matrices
    
    # Compute trace of each matrix
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]  # (n,)
    
    # Case 1: trace > 0
    q = np.zeros((n, 4), dtype=np.float64)
    mask1 = trace > 0
    
    if np.any(mask1):
        s = np.zeros(n, dtype=np.float64)
        s[mask1] = 0.5 / np.sqrt(trace[mask1] + 1.0)
        q[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) * s[mask1]
        q[mask1, 1] = (m[mask1, 0, 2] - m[mask1, 2, 0]) * s[mask1]
        q[mask1, 2] = (m[mask1, 1, 0] - m[mask1, 0, 1]) * s[mask1]
        q[mask1, 3] = 0.25 / s[mask1]
    
    # Case 2: m00 is largest diagonal
    mask2 = ~mask1 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    if np.any(mask2):
        s = np.zeros(n, dtype=np.float64)
        s[mask2] = 2.0 * np.sqrt(1.0 + m[mask2, 0, 0] - m[mask2, 1, 1] - m[mask2, 2, 2])
        q[mask2, 0] = 0.25 * s[mask2]
        q[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s[mask2]
        q[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s[mask2]
        q[mask2, 3] = (m[mask2, 1, 2] - m[mask2, 2, 1]) / s[mask2]
    
    # Case 3: m11 is largest diagonal
    mask3 = ~mask1 & ~mask2 & (m[:, 1, 1] > m[:, 2, 2])
    if np.any(mask3):
        s = np.zeros(n, dtype=np.float64)
        s[mask3] = 2.0 * np.sqrt(1.0 + m[mask3, 1, 1] - m[mask3, 0, 0] - m[mask3, 2, 2])
        q[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s[mask3]
        q[mask3, 1] = 0.25 * s[mask3]
        q[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s[mask3]
        q[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s[mask3]
    
    # Case 4: m22 is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if np.any(mask4):
        s = np.zeros(n, dtype=np.float64)
        s[mask4] = 2.0 * np.sqrt(1.0 + m[mask4, 2, 2] - m[mask4, 0, 0] - m[mask4, 1, 1])
        q[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s[mask4]
        q[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s[mask4]
        q[mask4, 2] = 0.25 * s[mask4]
        q[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s[mask4]
    
    # Normalize all quaternions
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_norms[q_norms < 1e-10] = 1.0
    rotations = q / q_norms
    
    # Handle colors - normalize to [0, 1]
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float64) / 255.0
    elif colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0
    
    # Default scales if not provided
    if scales is None:
        scales = np.ones((n, 3), dtype=np.float64) * 0.01
    
    # Build surfel dictionary
    surfels = {
        "position": points.astype(np.float64),
        "normal": normals,
        "tangent": tangents,
        "bitangent": bitangents,
        "opacity": np.ones(n, dtype=np.float64),
        "scale": scales.astype(np.float64),
        "rotation": rotations,
        "color": colors.astype(np.float64)
    }
    
    return surfels
