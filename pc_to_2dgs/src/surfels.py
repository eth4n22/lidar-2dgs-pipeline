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
    
    Args:
        normals: (N, 3) unit normal vectors
        
    Returns:
        Tuple of (tangents, bitangents), both (N, 3) arrays
    """
    n = normals.shape[0]
    tangents = np.zeros((n, 3), dtype=np.float64)
    bitangents = np.zeros((n, 3), dtype=np.float64)
    
    # Handle edge case of very small norms
    eps = 1e-8
    
    for i in range(n):
        normal = normals[i]
        norm = np.linalg.norm(normal)
        if norm < eps:
            # Fallback to arbitrary basis
            tangents[i] = [1, 0, 0]
            bitangents[i] = [0, 1, 0]
            continue
        normal = normal / norm
        
        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
        
        # Tangent = normal × arbitrary (cross product)
        tangent = np.cross(normal, arbitrary)
        tangent = tangent / np.linalg.norm(tangent)
        
        # Bitangent = normal × tangent
        bitangent = np.cross(normal, tangent)
        bitangent = bitangent / np.linalg.norm(bitangent)
        
        tangents[i] = tangent
        bitangents[i] = bitangent
    
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
    scales = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(covariances[i])
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
        scales[i] = np.sqrt(eigenvalues) * base_scale
    
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
    rotations = np.zeros((n, 4), dtype=np.float64)
    
    for i in range(n):
        R = np.column_stack([tangents[i], bitangents[i], normals[i]])
        rotations[i] = quaternion_from_rotation_matrix(R)
    
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
