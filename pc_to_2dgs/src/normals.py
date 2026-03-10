"""
Normals Estimation Module

Estimate surface normals from point clouds using KNN.
Memory-safe version for large point clouds (100M+ points).

Supports:
- GPU acceleration via PyTorch (NVIDIA RTX 3060+ compatible)
- FAISS-accelerated neighbor search for large datasets
- Vectorized CPU fallback for systems without GPU
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.linalg import eigh
import gc

# Optional GPU acceleration via PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional FAISS for fast neighbor search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def get_device():
    """
    Get the best available device for computation.
    
    Returns:
        str: 'cuda', 'mps' (Apple Silicon), or 'cpu'
    """
    if not HAS_TORCH:
        return 'cpu'
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def estimate_normals_gpu(points: np.ndarray,
                            k: int = 10,
                            batch_size: int = 500000,
                            device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated normal estimation using PyTorch.
    
    This is the fastest option for NVIDIA GPUs (RTX 3060+).
    Uses batched matrix operations for PCA computation.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood
        batch_size: Points to process per GPU batch (default: 500,000)
        device: 'cuda', 'mps', or 'cpu'. Auto-detects if None.
        
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length, float32)
        - curvatures: (N,) curvature values (float32)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration. Install with: pip install torch")
    
    n_points = points.shape[0]
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(n_points, dtype=np.float32)
    
    # Determine device
    if device is None:
        device = get_device()
    
    # Cap k
    k = min(k, 20)  # Allow higher k on GPU
    k_effective = min(k + 1, n_points)
    
    # Convert to float32 if needed
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    
    print(f"  GPU Normal Estimation ({device})")
    print(f"    Processing {n_points:,} points, k={k}, batch_size={batch_size:,}")
    
    # Move points to device
    if device == 'cuda':
        points_gpu = torch.from_numpy(points).cuda()
    elif device == 'mps':
        points_gpu = torch.from_numpy(points).to('mps')
    else:
        points_gpu = torch.from_numpy(points)
    
    # Preallocate outputs
    normals = np.empty((n_points, 3), dtype=np.float32)
    curvatures = np.empty((n_points,), dtype=np.float32)
    
    # Process in batches
    n_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_points)
        batch_n = end - start
        
        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx + 1}/{n_batches}: {start:,}-{end:,}")
        
        batch_points = points_gpu[start:end]
        
        # Use FAISS if available for fast neighbor search, otherwise use cKDTree
        if HAS_FAISS and device == 'cuda':
            # FAISS GPU search
            points_np = batch_points.cpu().numpy()
            index = faiss.IndexFlatL2(3)
            index.add(points.cpu().numpy())  # Full dataset index
            
            # Search in batches to avoid memory issues
            batch_search_size = min(batch_n, 100000)
            all_indices = []
            for search_start in range(0, batch_n, batch_search_size):
                search_end = min(search_start + batch_search_size, batch_n)
                _, indices_batch = index.search(points_np[search_start:search_end], k_effective)
                all_indices.append(indices_batch)
            indices = np.vstack(all_indices)
        else:
            # Use scipy cKDTree for neighbor search (still fast)
            tree = cKDTree(points)
            indices = tree.query(points[start:end], k=k_effective)[1]
        
        # Get neighbors: (batch_n, k_effective, 3)
        # Use advanced indexing on GPU
        indices_tensor = torch.from_numpy(indices[:, 1:]).long()
        if device == 'cuda':
            indices_tensor = indices_tensor.cuda()
        elif device == 'mps':
            indices_tensor = indices_tensor.to('mps')
        
        # Gather neighbors - use CPU for this part to save GPU memory
        indices_cpu = torch.from_numpy(indices[:, 1:]).long()
        neighbors = points_gpu.cpu()[indices_cpu]  # (batch_n, k, 3)
        
        # Move to device for computation
        if device == 'cuda':
            neighbors = neighbors.cuda()
        elif device == 'mps':
            neighbors = neighbors.to('mps')
        
        # Vectorized PCA computation on GPU
        # Compute centroids: (batch_n, 3)
        centroids = neighbors.mean(dim=1, keepdim=True)  # (batch_n, 1, 3)
        
        # Center the data: (batch_n, k, 3)
        centered = neighbors - centroids
        
        # Compute covariance matrices: (batch_n, 3, 3)
        # cov[i] = (centered[i].T @ centered[i]) / (k - 1)
        cov = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
        
        # Eigendecomposition for all covariance matrices at once
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Smallest eigenvector is the normal direction: (batch_n, 3)
        batch_normals = eigenvectors[:, :, 0]
        
        # Curvature: ratio of smallest to total eigenvalues
        total_eig = eigenvalues.sum(dim=1)  # (batch_n,)
        batch_curvatures = eigenvalues[:, 0] / (total_eig + 1e-10)
        
        # Ensure normals point away from origin
        batch_points_np = points[start:end]
        batch_normals_np = batch_normals.cpu().numpy()
        dot_products = np.sum(batch_normals_np * batch_points_np, axis=1, keepdims=True)
        flip_mask = dot_products < 0
        batch_normals_np[flip_mask.flatten()] *= -1
        
        # Store results
        normals[start:end] = batch_normals_np.astype(np.float32)
        curvatures[start:end] = batch_curvatures.cpu().numpy().astype(np.float32)
        
        # Cleanup GPU memory
        del neighbors, centered, cov, eigenvalues, eigenvectors, batch_normals
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # Final normalization
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    
    return normals, curvatures


def estimate_normals_vectorized_cpu(points: np.ndarray,
                                    k: int = 10,
                                    batch_size: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized CPU normal estimation - faster than the loop-based approach.
    
    Uses batched matrix operations instead of per-point loops.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood
        batch_size: Points to process per batch
        
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length, float32)
        - curvatures: (N,) curvature values (float32)
    """
    n_points = points.shape[0]
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(n_points, dtype=np.float32)
    
    k = min(k, 12)
    k_effective = min(k + 1, n_points)
    
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    
    normals = np.empty((n_points, 3), dtype=np.float32)
    curvatures = np.empty((n_points,), dtype=np.float32)
    
    print(f"  Vectorized CPU Normal Estimation")
    print(f"    Processing {n_points:,} points, k={k}, batch_size={batch_size:,}")
    
    # Build KDTree once
    tree = cKDTree(points)
    
    n_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_points)
        batch_n = end - start
        batch_points = points[start:end]
        
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx + 1}/{n_batches}: {start:,}-{end:,}")
        
        # Query neighbors
        distances, indices = tree.query(batch_points, k=k_effective)
        del distances
        
        # Get neighbor indices excluding self: (batch_n, k)
        neighbor_idx = indices[:, 1:]
        
        # Gather all neighbors at once: (batch_n, k, 3)
        # This is the key vectorization - gather all at once instead of loop
        neighbors = points[neighbor_idx]  # Broadcasting magic
        
        # Vectorized centroid computation: (batch_n, 3)
        centroids = neighbors.mean(axis=1)  # (batch_n, 3)
        
        # Center all neighbors at once: (batch_n, k, 3)
        centered = neighbors - centroids[:, np.newaxis, :]
        
        # Vectorized covariance: (batch_n, 3, 3)
        # Use einsum for efficiency: cov[i] = centered[i].T @ centered[i] / (k-1)
        cov = np.einsum('bij,bik->bjk', centered, centered) / (k - 1)
        
        # Batch eigendecomposition
        batch_normals = np.empty((batch_n, 3), dtype=np.float32)
        batch_curvatures = np.empty((batch_n,), dtype=np.float32)
        
        for i in range(batch_n):
            eigenvalues, eigenvectors = eigh(cov[i])
            batch_normals[i] = eigenvectors[:, 0].astype(np.float32)
            total_eig = eigenvalues.sum()
            batch_curvatures[i] = float(eigenvalues[0] / total_eig) if total_eig > 0 else 0.0
        
        # Flip normals to point away from origin
        dot_products = np.sum(batch_normals * batch_points, axis=1)
        flip_mask = dot_products < 0
        batch_normals[flip_mask] *= -1
        
        normals[start:end] = batch_normals
        curvatures[start:end] = batch_curvatures
        
        del indices, neighbors, centered, cov
        gc.collect()
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    
    return normals, curvatures


def estimate_normals_knn(points: np.ndarray,
                         k: int = 10,
                         batch_size: int = 100000,
                         workers: int = 1,
                         use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate surface normals using K-Nearest Neighbors.
    
    Automatically selects the best available method:
    1. GPU (PyTorch) - fastest for NVIDIA GPUs
    2. Vectorized CPU - faster than loop-based
    3. Legacy loop-based - fallback for compatibility
    
    For each point, finds k nearest neighbors, computes PCA of local
    neighborhood, and uses the smallest eigenvector as normal direction.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood (default: 10, capped at 20 for GPU)
        batch_size: Points to process per batch (default: 100,000 for CPU, 500,000 for GPU)
        workers: Number of parallel workers for KDTree query (default: 1, use -1 for all CPUs)
        use_gpu: Whether to use GPU if available (default: True)
        
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length, float32)
        - curvatures: (N,) curvature values (float32)
    """
    n_points = points.shape[0]
    
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(n_points, dtype=np.float32)
    
    # Auto-select best method based on availability
    device = get_device() if use_gpu else 'cpu'
    
    # Try GPU first if requested and available
    if use_gpu and device != 'cpu' and HAS_TORCH:
        print(f"  Using GPU acceleration ({device})")
        try:
            # Use larger batch size for GPU
            gpu_batch_size = min(batch_size * 5, 500000)
            return estimate_normals_gpu(points, k=k, batch_size=gpu_batch_size, device=device)
        except Exception as e:
            print(f"    GPU computation failed: {e}")
            print(f"    Falling back to vectorized CPU...")
    
    # Fall back to vectorized CPU
    print(f"  Using vectorized CPU normal estimation")
    return estimate_normals_vectorized_cpu(points, k=k, batch_size=batch_size)


def estimate_normals_knn_legacy(points: np.ndarray,
                                 k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy implementation for small point clouds.
    
    WARNING: This can consume huge amounts of memory for large point clouds.
    Use estimate_normals_knn() instead for large datasets.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood
        
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length)
        - curvatures: (N,) curvature values
    """
    if points.shape[0] < 3:
        return np.zeros_like(points), np.zeros(points.shape[0])
    
    # Build KDTree for efficient neighbor search
    tree = KDTree(points)
    
    # Find k+1 neighbors (include self)
    k_effective = min(k + 1, points.shape[0])
    distances, indices = tree.query(points, k=k_effective)
    
    normals = np.zeros_like(points)
    curvatures = np.zeros(points.shape[0])
    
    for i in range(points.shape[0]):
        # Get neighbors (excluding self)
        neighbors = points[indices[i, 1:]]
        
        # Compute centroid of neighbors
        centroid = np.mean(neighbors, axis=0)
        
        # Compute covariance matrix
        centered = neighbors - centroid
        cov = centered.T @ centered / (k - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(cov)
        
        # Smallest eigenvector is the normal direction
        normal = eigenvectors[:, 0]
        
        # Smallest eigenvalue relates to curvature
        curvature = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
        
        # Ensure normal points away from origin (or consistent direction)
        if np.dot(normal, points[i]) < 0:
            normal = -normal
        
        normals[i] = normal
        curvatures[i] = curvature
    
    # Normalize to unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms
    
    return normals, curvatures


def orient_normals_consistently(points: np.ndarray,
                                normals: np.ndarray,
                                reference_point: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Orient normals consistently so they all point in the same general direction.
    
    Uses the mean normal direction to determine the majority orientation
    and flips normals as needed.
    
    Args:
        points: (N, 3) xyz coordinates (used if reference_point is None)
        normals: (N, 3) surface normals
        reference_point: Optional custom reference point for orientation
        
    Returns:
        (N, 3) oriented normals
    """
    if normals.shape[0] == 0:
        return normals
    
    # Use provided reference_point or compute from points
    if reference_point is None:
        centroid = np.mean(points, axis=0)
        reference_normal = np.mean(normals, axis=0)
        
        # If mean is zero (equal UP and DOWN), use z-component direction
        norm = np.linalg.norm(reference_normal)
        if norm < 1e-10:
            reference_normal = np.array([0.0, 0.0, 1.0])
        else:
            reference_normal = reference_normal / norm
    else:
        # Use custom reference point - compute direction from centroid
        centroid = np.mean(points, axis=0)
        to_reference = reference_point - centroid
        if np.linalg.norm(to_reference) > 1e-10:
            reference_normal = to_reference / np.linalg.norm(to_reference)
        else:
            reference_normal = np.array([0.0, 0.0, 1.0])
    
    # Flip normals that point opposite to the reference direction
    for i in range(normals.shape[0]):
        if np.dot(normals[i], reference_normal) < 0:
            normals[i] = -normals[i]
    
    return normals


def refine_normals(points: np.ndarray,
                   normals: np.ndarray,
                   iterations: int = 1) -> np.ndarray:
    """
    Refine normals using iterative smoothing.
    
    For each point, computes a new normal as the average of normals
    from nearby points, weighted by distance.
    
    Args:
        points: (N, 3) xyz coordinates
        normals: (N, 3) surface normals to refine
        iterations: Number of refinement iterations
        
    Returns:
        (N, 3) refined normals
    """
    if iterations <= 0:
        return normals
    
    tree = cKDTree(points)
    refined = normals.copy()
    
    for _ in range(iterations):
        new_normals = np.zeros_like(refined)
        
        for i in range(points.shape[0]):
            # Find nearby points
            distances, indices = tree.query(points[i], k=16)
            
            # Weight by inverse distance
            weights = 1.0 / (distances + 1e-6)
            weights = weights / np.sum(weights)
            
            # Average normals
            new_normal = np.sum(refined[indices].T * weights, axis=1)
            new_normals[i] = new_normal
        
        # Reorient
        new_normals = orient_normals_consistently(points, new_normals, reference_point=np.mean(points, axis=0))
        
        # Normalize
        norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        refined = new_normals / norms
    
    return refined
