"""
Normals Estimation Module

Estimate surface normals from point clouds using KNN.
Memory-safe version for large point clouds (100M+ points).

Supports:
- GPU acceleration via PyTorch (NVIDIA RTX 3060+ compatible)
- Chunked spatial partitioning for large datasets
- Vectorized CPU fallback for systems without GPU
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.linalg import eigh
import gc
import os

# Optional GPU acceleration via PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def partition_point_cloud_spatial(points: np.ndarray, 
                                  chunk_size: int = 500000,
                                  overlap_factor: float = 0.15) -> List[Dict]:
    """
    Partition point cloud into spatial chunks with overlap.
    
    Args:
        points: (N, 3) xyz coordinates
        chunk_size: Target points per chunk (default: 500K)
        overlap_factor: Overlap as fraction of chunk size (default: 0.15)
    
    Returns:
        List of chunk dicts with:
        - 'indices': global indices of core points
        - 'points': all points in chunk (core + halo)
        - 'local_to_global': mapping from local to global indices
    """
    n_points = points.shape[0]
    
    # Calculate number of chunks needed
    n_chunks = max(1, int(np.ceil(n_points / chunk_size)))
    
    # Calculate bounding box
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    ranges = max_coords - min_coords
    
    # Find optimal grid dimensions
    dim = 3
    grid_size = int(np.ceil(n_chunks ** (1.0 / dim)))
    
    # Calculate chunk sizes per dimension
    chunk_ranges = ranges / grid_size
    overlap = chunk_ranges * overlap_factor
    
    chunks = []
    global_indices = np.arange(n_points)
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Calculate bounds for this chunk
                chunk_min = min_coords + np.array([i, j, k]) * chunk_ranges - overlap
                chunk_max = min_coords + np.array([i+1, j+1, k+1]) * chunk_ranges + overlap
                
                # Find points in this region
                mask = ((points >= chunk_min) & (points <= chunk_max)).all(axis=1)
                chunk_indices = np.where(mask)[0]
                
                if len(chunk_indices) == 0:
                    continue
                
                # Add to chunks list
                chunks.append({
                    'indices': chunk_indices,  # Global indices
                    'points': points[chunk_indices],  # Points for this chunk
                    'local_to_global': chunk_indices  # Identity mapping
                })
    
    return chunks


def estimate_normals_chunked(points: np.ndarray,
                            k: int = 10,
                            chunk_size: int = 500000,
                            overlap_factor: float = 0.15,
                            device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chunked normal estimation for very large point clouds.
    
    Partitions point cloud into spatial chunks and processes each independently.
    This reduces memory usage and allows processing of 100M+ point clouds.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood
        chunk_size: Target points per chunk (default: 500K)
        overlap_factor: Overlap between chunks (default: 0.15)
        device: 'cuda', 'mps', or 'cpu'. Auto-detects if None.
    
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length, float32)
        - curvatures: (N,) curvature values (float32)
    """
    n_points = points.shape[0]
    
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(n_points, dtype=np.float32)
    
    if device is None:
        device = get_device()
    
    print(f"  Chunked Normal Estimation ({device})")
    print(f"    Total points: {n_points:,}, k={k}, chunk_size={chunk_size:,}")
    
    # Partition into spatial chunks
    chunks = partition_point_cloud_spatial(points, chunk_size, overlap_factor)
    n_chunks = len(chunks)
    print(f"    Number of chunks: {n_chunks}")
    
    # Assert: All points should be covered by chunks
    covered_indices = set()
    for chunk in chunks:
        covered_indices.update(chunk['indices'])
    assert len(covered_indices) == n_points, f"Chunk coverage error: {len(covered_indices)}/{n_points} points covered"
    
    # Preallocate output arrays
    normals = np.zeros((n_points, 3), dtype=np.float32)
    curvatures = np.zeros(n_points, dtype=np.float32)
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        chunk_points = chunk['points']
        global_indices = chunk['indices']
        chunk_n = len(chunk_points)
        
        if chunk_n < 3:
            continue
        
        if chunk_idx % 5 == 0 or chunk_idx == n_chunks - 1:
            print(f"    Chunk {chunk_idx + 1}/{n_chunks}: {chunk_n:,} points")
        
        # Build local cKDTree for this chunk
        local_tree = cKDTree(chunk_points)
        
        # Query neighbors for all points in chunk
        k_eff = min(k + 1, chunk_n)
        distances, local_indices = local_tree.query(chunk_points, k=k_eff)
        
        # Convert to float32
        if chunk_points.dtype != np.float32:
            chunk_points = chunk_points.astype(np.float32)
        
        # Move to device
        if device == 'cuda' and HAS_TORCH:
            points_gpu = torch.from_numpy(chunk_points).cuda()
        elif device == 'mps' and HAS_TORCH:
            points_gpu = torch.from_numpy(chunk_points).to('mps')
        elif HAS_TORCH:
            points_gpu = torch.from_numpy(chunk_points)
        else:
            points_gpu = chunk_points
        
        # GPU PCA computation
        if HAS_TORCH and device != 'cpu':
            # Get neighbors
            neighbor_indices = torch.from_numpy(local_indices[:, 1:]).long()
            if device == 'cuda':
                neighbor_indices = neighbor_indices.cuda()
                points_gpu = points_gpu.cuda()
            elif device == 'mps':
                neighbor_indices = neighbor_indices.to('mps')
                points_gpu = points_gpu.to('mps')
            
            neighbors = points_gpu[neighbor_indices]
            
            # PCA
            centroids = neighbors.mean(dim=1, keepdim=True)
            centered = neighbors - centroids
            cov = torch.matmul(centered.transpose(1, 2), centered) / (k - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            chunk_normals = eigenvectors[:, :, 0].cpu().numpy()
            chunk_curvatures = (eigenvalues[:, 0] / (eigenvalues.sum(dim=1) + 1e-10)).cpu().numpy()
            
            # Cleanup
            del neighbors, centered, cov, eigenvalues, eigenvectors
            if device == 'cuda':
                torch.cuda.empty_cache()
        else:
            # CPU fallback
            neighbor_indices = local_indices[:, 1:]
            neighbors = chunk_points[neighbor_indices]
            
            centroids = neighbors.mean(axis=1, keepdims=True)
            centered = neighbors - centroids
            cov = np.einsum('bij,bik->bjk', centered, centered) / (k - 1)
            
            chunk_normals = np.empty((chunk_n, 3), dtype=np.float32)
            chunk_curvatures = np.empty(chunk_n, dtype=np.float32)
            
            for i in range(chunk_n):
                eigvals, eigvecs = eigh(cov[i])
                chunk_normals[i] = eigvecs[:, 0].astype(np.float32)
                total = eigvals.sum()
                chunk_curvatures[i] = eigvals[0] / total if total > 0 else 0.0
        
        # Flip normals to point away from origin
        dot_products = np.sum(chunk_normals * chunk_points, axis=1)
        flip_mask = dot_products < 0
        chunk_normals[flip_mask] *= -1
        
        # Normalize
        norms = np.linalg.norm(chunk_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        chunk_normals = chunk_normals / norms
        
        # Store in global arrays
        normals[global_indices] = chunk_normals
        curvatures[global_indices] = chunk_curvatures.astype(np.float32)
        
        # Cleanup
        del chunk_points, local_tree, local_indices
        gc.collect()
    
    print(f"    Completed {n_chunks} chunks")
    
    # Assert: All normals should be computed (no zero normals)
    zero_normals = np.where(np.all(normals == 0, axis=1))[0]
    if len(zero_normals) > 0:
        print(f"    Warning: {len(zero_normals)} points have zero normals")
        # Don't fail - zero normals might be from isolated points with no neighbors
    
    # Assert: Output shape matches input
    assert normals.shape[0] == n_points, f"Output normals shape mismatch: {normals.shape[0]} vs {n_points}"
    assert curvatures.shape[0] == n_points, f"Output curvatures shape mismatch: {curvatures.shape[0]} vs {n_points}"
    
    return normals, curvatures


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
    
    # Build KNN index ONCE before batch loop using scipy cKDTree
    # Note: FAISS is too slow on Windows for this workload
    print("[INFO] Using scipy cKDTree KNN")
    tree = cKDTree(points)
    
    print("[INFO] Points:", points.shape[0])
    
    # Process in batches
    n_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_points)
        batch_n = end - start
        
        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx + 1}/{n_batches}: {start:,}-{end:,}")
        
        batch_points = points_gpu[start:end]
        
        # KNN search using cKDTree (built once outside loop)
        indices = tree.query(points[start:end], k=k_effective)[1]
        
        # Get neighbors: (batch_n, k_effective, 3)
        # Use GPU indices directly - keep all computation on GPU!
        indices_gpu = torch.from_numpy(indices[:, 1:]).long().to(device)
        neighbors = points_gpu[indices_gpu]  # All GPU, no CPU round-trip
        
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


def _process_chunk_cpu(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single chunk on CPU (for parallel processing).
    
    Args:
        args: Tuple of (chunk_points, global_indices, k)
    
    Returns:
        Tuple of (chunk_normals, chunk_curvatures, global_indices)
    """
    chunk_points, global_indices, k = args
    chunk_n = len(chunk_points)
    
    if chunk_n < 3:
        normals = np.zeros((chunk_n, 3), dtype=np.float32)
        curvatures = np.zeros(chunk_n, dtype=np.float32)
        return normals, curvatures, global_indices
    
    # Ensure float32
    if chunk_points.dtype != np.float32:
        chunk_points = chunk_points.astype(np.float32)
    
    # Build local cKDTree
    tree = cKDTree(chunk_points)
    k_eff = min(k + 1, chunk_n)
    distances, local_indices = tree.query(chunk_points, k=k_eff)
    
    # Get neighbors
    neighbor_indices = local_indices[:, 1:]
    neighbors = chunk_points[neighbor_indices]
    
    # PCA
    centroids = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroids
    cov = np.einsum('bij,bik->bjk', centered, centered) / (k - 1)
    
    chunk_normals = np.empty((chunk_n, 3), dtype=np.float32)
    chunk_curvatures = np.empty(chunk_n, dtype=np.float32)
    
    for i in range(chunk_n):
        eigvals, eigvecs = eigh(cov[i])
        chunk_normals[i] = eigvecs[:, 0].astype(np.float32)
        total = eigvals.sum()
        chunk_curvatures[i] = eigvals[0] / total if total > 0 else 0.0
    
    # Flip normals
    dot_products = np.sum(chunk_normals * chunk_points, axis=1)
    flip_mask = dot_products < 0
    chunk_normals[flip_mask] *= -1
    
    # Normalize
    norms = np.linalg.norm(chunk_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    chunk_normals = chunk_normals / norms
    
    return chunk_normals, chunk_curvatures.astype(np.float32), global_indices


def estimate_normals_chunked_parallel(points: np.ndarray,
                                    k: int = 10,
                                    chunk_size: int = 500000,
                                    overlap_factor: float = 0.15,
                                    n_jobs: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel chunked normal estimation for large point clouds.
    
    Uses CPU parallelization for faster processing on multi-core systems.
    Note: Uses CPU for all chunks (no GPU) but parallelizes across chunks.
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood
        chunk_size: Target points per chunk (default: 500K)
        overlap_factor: Overlap between chunks (default: 0.15)
        n_jobs: Number of parallel workers (default: 2)
    
    Returns:
        Tuple of:
        - normals: (N, 3) surface normals (unit length, float32)
        - curvatures: (N,) curvature values
    """
    n_points = points.shape[0]
    
    if n_points < 3:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros(n_points, dtype=np.float32)
    
    print(f"  Parallel Chunked Normal Estimation (CPU, n_jobs={n_jobs})")
    print(f"    Total points: {n_points:,}, k={k}, chunk_size={chunk_size:,}")
    
    # Partition into spatial chunks
    chunks = partition_point_cloud_spatial(points, chunk_size, overlap_factor)
    n_chunks = len(chunks)
    print(f"    Number of chunks: {n_chunks}")
    
    # Assert: All points should be covered by chunks
    covered_indices = set()
    for chunk in chunks:
        covered_indices.update(chunk['indices'])
    assert len(covered_indices) == n_points, f"Chunk coverage error: {len(covered_indices)}/{n_points} points covered"
    
    # Prepare arguments for parallel processing
    chunk_args = [(chunk['points'], chunk['indices'], k) for chunk in chunks]
    
    # Process chunks in parallel using joblib
    try:
        from joblib import Parallel, delayed
        print(f"    Processing {n_chunks} chunks with {n_jobs} workers...")
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_process_chunk_cpu)(args) for args in chunk_args
        )
    except ImportError:
        print(f"    joblib not available, processing sequentially...")
        results = [_process_chunk_cpu(args) for args in chunk_args]
    
    # Combine results
    normals = np.zeros((n_points, 3), dtype=np.float32)
    curvatures = np.zeros(n_points, dtype=np.float32)
    
    for chunk_normals, chunk_curvatures, global_indices in results:
        normals[global_indices] = chunk_normals
        curvatures[global_indices] = chunk_curvatures
    
    print(f"    Completed {n_chunks} chunks")
    
    # Assert: All normals should be computed (no zero normals)
    zero_normals = np.where(np.all(normals == 0, axis=1))[0]
    if len(zero_normals) > 0:
        print(f"    Warning: {len(zero_normals)} points have zero normals")
    
    # Assert: Output shape matches input
    assert normals.shape[0] == n_points, f"Output normals shape mismatch: {normals.shape[0]} vs {n_points}"
    assert curvatures.shape[0] == n_points, f"Output curvatures shape mismatch: {curvatures.shape[0]} vs {n_points}"
    
    return normals, curvatures


def estimate_normals_knn(points: np.ndarray,
                         k: int = 10,
                         batch_size: int = 100000,
                         workers: int = 1,
                         use_gpu: bool = True,
                         use_chunked: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate surface normals using K-Nearest Neighbors.
    
    Automatically selects the best available method:
    1. Chunked GPU - for large datasets (>1M points)
    2. GPU (PyTorch) - for medium datasets
    3. Vectorized CPU - fallback
    4. Legacy loop-based - compatibility
    
    Args:
        points: (N, 3) xyz coordinates
        k: Number of neighbors for local neighborhood (default: 10, capped at 20 for GPU)
        batch_size: Points per batch for non-chunked methods
        workers: Number of parallel workers
        use_gpu: Whether to use GPU if available (default: True)
        use_chunked: Use chunked approach for large datasets (default: True)
        
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
    
    # Use chunked approach for large datasets
    CHUNK_THRESHOLD = 1_000_000  # Use chunked for > 1M points
    
    if use_chunked and n_points >= CHUNK_THRESHOLD and HAS_TORCH:
        print(f"  Using chunked normal estimation")
        return estimate_normals_chunked(points, k=k, device=device)
    
    # Try GPU first if requested and available
    if use_gpu and device != 'cpu' and HAS_TORCH:
        print(f"  Using GPU acceleration ({device})")
        try:
            # GPU batch size calculation - adaptive based on k and available memory
            # Memory formula: ~100 * batch_size * (k+1) bytes
            # For RTX 3060 (6GB), conservatively use ~4GB max
            MAX_GPU_BATCH = 2_000_000  # Absolute maximum
            MIN_GPU_BATCH = 100_000     # Minimum batch size
            MEMORY_BUDGET_GB = 4.0       # Conservative 4GB for computation
            
            # Base: 100 * batch_size * (k+1) bytes = 100 * batch_size * (k+1) / 1e9 GB
            max_by_k = int(MEMORY_BUDGET_GB * 1e9 / (100 * (k + 1)))
            
            # Scale down for very large point clouds (more memory pressure)
            if n_points > 5_000_000:
                max_by_k = int(max_by_k * 0.5)
            elif n_points > 2_000_000:
                max_by_k = int(max_by_k * 0.7)
            
            # Apply limits
            gpu_batch_size = min(max_by_k, MAX_GPU_BATCH)
            gpu_batch_size = max(gpu_batch_size, MIN_GPU_BATCH)
            
            # User can override with explicit batch_size parameter
            if batch_size < n_points:
                gpu_batch_size = min(batch_size * 20, gpu_batch_size)
            
            print(f"    Adaptive GPU batch size: {gpu_batch_size:,} (k={k}, n_points={n_points:,})")
            
            # Try GPU with adaptive batch size, with retry on OOM
            retry_count = 0
            max_retries = 3
            current_batch = gpu_batch_size
            
            while retry_count < max_retries:
                try:
                    return estimate_normals_gpu(points, k=k, batch_size=current_batch, device=device)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                        retry_count += 1
                        old_batch = current_batch
                        current_batch = max(MIN_GPU_BATCH, current_batch // 2)
                        print(f"    GPU OOM: reducing batch from {old_batch:,} to {current_batch:,} (retry {retry_count}/{max_retries})")
                        # Clear GPU memory
                        if HAS_TORCH and device == 'cuda':
                            import torch
                            torch.cuda.empty_cache()
                        if current_batch <= MIN_GPU_BATCH:
                            print(f"    GPU batch size too small, falling back to CPU...")
                            break
                    else:
                        raise
            
            # Fallback to CPU if GPU failed after retries
            print(f"    Falling back to vectorized CPU...")
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
