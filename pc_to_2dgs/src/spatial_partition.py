"""
Spatial partitioning using voxel grid for improved KNN performance.
"""

import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional


def compute_voxel_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of points."""
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    return min_coords, max_coords


def streaming_bounding_box(input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box by streaming through file."""
    min_coords = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_coords = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    
    with open(input_path, 'r') as f:
        for line in f:
            # Try comma first, then space
            if ',' in line:
                parts = line.strip().split(',')
            else:
                parts = line.strip().split()
            
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    min_coords[0] = min(min_coords[0], x)
                    min_coords[1] = min(min_coords[1], y)
                    min_coords[2] = min(min_coords[2], z)
                    max_coords[0] = max(max_coords[0], x)
                    max_coords[1] = max(max_coords[1], y)
                    max_coords[2] = max(max_coords[2], z)
                except ValueError:
                    continue
    
    return min_coords.astype(np.float32), max_coords.astype(np.float32)


def streaming_voxel_partition(input_path: str, 
                              output_dir: str,
                              target_points_per_chunk: int = 500000,
                              min_voxels: int = 8,
                              max_voxels: int = 64) -> List[str]:
    """
    Partition points into spatial chunks using voxel grid.
    
    1. Compute bounding box (streaming)
    2. Determine voxel grid size based on target chunk size
    3. Assign points to voxels and write chunk files
    
    Returns:
        List of chunk file paths
    """
    print(f"[SPATIAL] Computing bounding box...")
    min_coords, max_coords = streaming_bounding_box(input_path)
    
    extents = max_coords - min_coords
    max_extent = max(extents)
    
    # Handle edge case where all points are in a line or at same location
    if max_extent < 1e-6:
        print(f"[SPATIAL] Warning: very small extent ({max_extent}), using small voxel grid")
        max_extent = 1.0
        min_coords = min_coords - 0.5
        max_coords = max_coords + 0.5
        extents = max_coords - min_coords
        max_extent = max(extents)
    
    # Read all points for voxel assignment (streaming partition is complex)
    print(f"[SPATIAL] Loading points...")
    # Try comma first, fall back to space
    try:
        points = np.loadtxt(input_path, delimiter=',', dtype=np.float32, usecols=(0, 1, 2))
    except ValueError:
        points = np.loadtxt(input_path, dtype=np.float32, usecols=(0, 1, 2))
    n_points = len(points)
    print(f"[SPATIAL] Total points: {n_points:,}")
    
    # Calculate number of voxels per dimension
    # Target: ~4x more voxels than target chunks to allow for uneven distribution
    target_chunks = max(1, n_points // target_points_per_chunk)
    n_voxels = max(min_voxels, min(max_voxels, int(target_chunks ** (1/3)) + 2))
    
    # Ensure odd number for centering
    n_voxels = n_voxels if n_voxels % 2 == 1 else n_voxels + 1
    
    voxel_size = max_extent / n_voxels
    
    print(f"[SPATIAL] Grid: {n_voxels}x{n_voxels}x{n_voxels} voxels")
    print(f"[SPATIAL] Voxel size: {voxel_size:.3f}")
    
    # Assign each point to a voxel
    # Use broadcasting for efficiency
    relative = points - min_coords
    ix = np.clip((relative[:, 0] / voxel_size).astype(np.int32), 0, n_voxels - 1)
    iy = np.clip((relative[:, 1] / voxel_size).astype(np.int32), 0, n_voxels - 1)
    iz = np.clip((relative[:, 2] / voxel_size).astype(np.int32), 0, n_voxels - 1)
    
    # Flatten voxel index
    voxel_ids = (ix * n_voxels * n_voxels + iy * n_voxels + iz).astype(np.uint32)
    
    # Sort by voxel ID for efficient grouping
    sort_idx = np.argsort(voxel_ids)
    points_sorted = points[sort_idx]
    voxel_ids_sorted = voxel_ids[sort_idx]
    
    # Find voxel boundaries
    unique_voxels, boundaries = np.unique(voxel_ids_sorted, return_index=True)
    boundaries = np.append(boundaries, len(points_sorted))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write chunk files
    chunk_files = []
    chunk_stats = []
    
    for i, (voxel_id, start) in enumerate(zip(unique_voxels, boundaries[:-1])):
        end = boundaries[i + 1]
        chunk_points = points_sorted[start:end]
        
        # Compute voxel coordinates for filename
        ix_v = voxel_id // (n_voxels * n_voxels)
        iy_v = (voxel_id % (n_voxels * n_voxels)) // n_voxels
        iz_v = voxel_id % n_voxels
        
        chunk_name = f"chunk_{ix_v:02d}_{iy_v:02d}_{iz_v:02d}.bin"
        chunk_path = output_path / chunk_name
        
        # Write as binary (float32 xyz)
        chunk_points.astype(np.float32).tofile(chunk_path)
        chunk_files.append(str(chunk_path))
        chunk_stats.append(len(chunk_points))
    
    # Sort chunks by filename for deterministic ordering
    chunk_files = sorted(chunk_files)
    chunk_stats = [len(np.fromfile(f, dtype=np.float32).reshape(-1, 3)) for f in chunk_files]
    
    n_chunks = len(chunk_files)
    avg_points = n_points / n_chunks if n_chunks > 0 else 0
    
    print(f"[SPATIAL] Number of chunks: {n_chunks}")
    print(f"[SPATIAL] Avg points per chunk: {avg_points:,.0f}")
    print(f"[SPATIAL] Min points per chunk: {min(chunk_stats) if chunk_stats else 0:,}")
    print(f"[SPATIAL] Max points per chunk: {max(chunk_stats) if chunk_stats else 0:,}")
    
    return chunk_files


def streaming_voxel_partition_streaming(input_path: str,
                                        output_dir: str,
                                        target_points_per_chunk: int = 500000,
                                        chunk_buffer_size: int = 100000) -> List[str]:
    """
    Memory-efficient spatial partitioning using streaming.
    
    First pass: compute bounding box
    Second pass: assign to voxels and write chunks
    
    Uses buffering to limit memory usage.
    """
    # First pass: compute bounding box
    print(f"[SPATIAL] First pass: computing bounding box...")
    min_coords, max_coords = streaming_bounding_box(input_path)
    
    extents = max_coords - min_coords
    max_extent = max(extents)
    
    print(f"[SPATIAL] Bounding box: min={min_coords}, max={max_coords}")
    print(f"[SPATIAL] Extent: {extents}, max_extent: {max_extent}")
    
    # Estimate number of voxels
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # subtract header if exists
    
    target_chunks = max(1, total_lines // target_points_per_chunk)
    n_voxels = max(8, min(64, int(target_chunks ** (1/3)) + 2))
    n_voxels = n_voxels if n_voxels % 2 == 1 else n_voxels + 1
    voxel_size = max_extent / n_voxels
    
    print(f"[SPATIAL] Grid: {n_voxels}x{n_voxels}x{n_voxels}")
    print(f"[SPATIAL] Voxel size: {voxel_size:.3f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize chunk buffers
    max_chunks = n_voxels ** 3
    chunk_buffers = {}  # voxel_id -> list of points
    chunk_counts = np.zeros(max_chunks, dtype=np.int64)
    
    # Second pass: assign points to voxels
    print(f"[SPATIAL] Second pass: partitioning points...")
    
    buffer = []
    buffer_voxel_ids = []
    
    with open(input_path, 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    buffer.append([x, y, z])
                    
                    # Compute voxel index
                    ix = int((x - min_coords[0]) / voxel_size)
                    iy = int((y - min_coords[1]) / voxel_size)
                    iz = int((z - min_coords[2]) / voxel_size)
                    ix = np.clip(ix, 0, n_voxels - 1)
                    iy = np.clip(iy, 0, n_voxels - 1)
                    iz = np.clip(iz, 0, n_voxels - 1)
                    voxel_id = ix * n_voxels * n_voxels + iy * n_voxels + iz
                    buffer_voxel_ids.append(voxel_id)
                    
                except (ValueError, IndexError):
                    continue
            
            # Flush buffer when full
            if len(buffer) >= chunk_buffer_size:
                # Group by voxel
                for pt, vid in zip(buffer, buffer_voxel_ids):
                    if vid not in chunk_buffers:
                        chunk_buffers[vid] = []
                    chunk_buffers[vid].append(pt)
                    chunk_counts[vid] += 1
                
                buffer = []
                buffer_voxel_ids = []
    
    # Flush remaining
    for pt, vid in zip(buffer, buffer_voxel_ids):
        if vid not in chunk_buffers:
            chunk_buffers[vid] = []
        chunk_buffers[vid].append(pt)
        chunk_counts[vid] += 1
    
    # Write chunk files
    print(f"[SPATIAL] Writing chunk files...")
    chunk_files = []
    
    for voxel_id in sorted(chunk_buffers.keys()):
        points = np.array(chunk_buffers[voxel_id], dtype=np.float32)
        
        # Compute voxel coordinates
        ix_v = voxel_id // (n_voxels * n_voxels)
        iy_v = (voxel_id % (n_voxels * n_voxels)) // n_voxels
        iz_v = voxel_id % n_voxels
        
        chunk_name = f"chunk_{ix_v:02d}_{iy_v:02d}_{iz_v:02d}.bin"
        chunk_path = output_path / chunk_name
        
        points.tofile(chunk_path)
        chunk_files.append(str(chunk_path))
    
    n_chunks = len(chunk_files)
    n_points = total_lines
    avg_points = n_points / n_chunks if n_chunks > 0 else 0
    
    print(f"[SPATIAL] Number of chunks: {n_chunks}")
    print(f"[SPATIAL] Avg points per chunk: {avg_points:,.0f}")
    
    return chunk_files


def load_spatial_chunk(chunk_path: str) -> np.ndarray:
    """Load a spatial chunk from binary file."""
    points = np.fromfile(chunk_path, dtype=np.float32).reshape(-1, 3)
    return points


def cleanup_spatial_chunks(output_dir: str):
    """Remove temporary spatial chunk files."""
    import shutil
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
        print(f"[SPATIAL] Cleaned up: {output_dir}")
