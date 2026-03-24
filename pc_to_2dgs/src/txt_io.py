"""
TXT I/O Module

Load and save LiDAR point cloud data from/to TXT files.
Includes streaming support for large files (100M+ points).
"""

from pathlib import Path
from typing import Dict, Generator, Tuple
import numpy as np
import os
import time


def load_xyzrgb_txt(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud data from a TXT file.

    Expected format: one point per line, "x y z r g b" (space-separated)

    Memory-efficient single-pass loading using numpy.

    Args:
        filepath: Path to the TXT file

    Returns:
        Tuple of:
            - points: (N, 3) xyz coordinates (float32)
            - colors: (N, 3) rgb values (uint8)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"  Loading {os.path.basename(filepath)} (memory-efficient)...")
    
    # Read file content - for very large files, this still needs RAM
    # but with float32 it's about half the memory of float64
    try:
        # Try numpy's loadtxt which is faster and uses less memory than genfromtxt
        # Use float32 to halve memory usage
        data = np.loadtxt(
            filepath,
            dtype=np.float32,
            comments='#',
            delimiter=None,  # whitespace
            usecols=(0, 1, 2, 3, 4, 5),  # x y z r g b
            unpack=False
        )
        
        # Split into positions and colors
        points = data[:, 0:3]
        colors = (data[:, 3:6]).astype(np.uint8)
        
        return points, colors
        
    except ValueError as e:
        # If usecols fails, try reading all and splitting
        # Fall back to line-by-line for compatibility
        return _load_xyzrgb_txt_fallback(filepath)


def _load_xyzrgb_txt_fallback(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback line-by-line loader for unusual formats."""
    import os
    
    # Count lines first
    point_count = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    point_count += 1

    print(f"  Loading {point_count:,} points (fallback mode)...")
    
    # Pre-allocate
    points = np.empty((point_count, 3), dtype=np.float32)
    colors = np.empty((point_count, 3), dtype=np.uint8)
    
    idx = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                points[idx, 0] = float(parts[0])
                points[idx, 1] = float(parts[1])
                points[idx, 2] = float(parts[2])
                
                if len(parts) >= 6:
                    colors[idx, 0] = min(255, max(0, int(float(parts[3]))))
                    colors[idx, 1] = min(255, max(0, int(float(parts[4]))))
                    colors[idx, 2] = min(255, max(0, int(float(parts[5]))))
                else:
                    colors[idx] = [255, 255, 255]
                    
                idx += 1
            except (ValueError, IndexError):
                continue

    return points[:idx], colors[:idx]


def validate_format(points: np.ndarray, colors: np.ndarray) -> bool:
    """
    Validate that point cloud data has the expected format.

    Args:
        points: (N, 3) xyz coordinates
        colors: (N, 3) rgb values

    Returns:
        True if format is valid, False otherwise
    """
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    if colors.ndim != 2 or colors.shape[1] != 3:
        return False
    if points.shape[0] != colors.shape[0]:
        return False
    if not np.issubdtype(points.dtype, np.floating):
        return False
    if not np.issubdtype(colors.dtype, np.integer):
        return False
    return True


def save_xyzrgb_txt(filepath: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save point cloud data to a TXT file.

    Args:
        filepath: Path to output TXT file
        points: (N, 3) xyz coordinates
        colors: (N, 3) rgb values (0-255)
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")


class StreamingTXTReader:
    """
    Memory-efficient streaming reader for large TXT point cloud files.
    
    Reads the file in chunks without loading everything into memory.
    Supports "x y z r g b" format (space-separated, one point per line).
    """
    
    def __init__(self, filepath: str, chunk_size: int = 500000, count_points: bool = True):
        """
        Initialize streaming reader.
        
        Args:
            filepath: Path to the TXT file
            chunk_size: Number of lines to read per chunk (default: 500K)
            count_points: Whether to count total points (default: True). Set False to skip.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.total_points = None
        self.processed_points = 0
        
        # Count total lines first (for progress tracking)
        if count_points:
            print(f"  Counting points in {os.path.basename(filepath)}...")
            self._count_lines()
    
    def _count_lines(self):
        """Count total number of valid points in the file."""
        count = 0
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        count += 1
        self.total_points = count
        print(f"  Total points: {self.total_points:,}")
    
    def stream_chunks(self) -> Generator[Tuple[np.ndarray, np.ndarray, int, int], None, None]:
        """
        Stream file in chunks.
        
        Yields:
            Tuple of (points, colors, start_idx, chunk_idx) where:
            - points: (N, 3) xyz coordinates (float32)
            - colors: (N, 3) rgb values (uint8)
            - start_idx: Starting global index of this chunk
            - chunk_idx: Chunk number (0-indexed)
        """
        chunk_idx = 0
        start_idx = 0
        
        with open(self.filepath, 'r') as f:
            batch_points = []
            batch_colors = []
            
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    batch_points.append([x, y, z])
                    
                    if len(parts) >= 6:
                        r = min(255, max(0, int(float(parts[3]))))
                        g = min(255, max(0, int(float(parts[4]))))
                        b = min(255, max(0, int(float(parts[5]))))
                        batch_colors.append([r, g, b])
                    else:
                        batch_colors.append([255, 255, 255])
                    
                    # Yield chunk when full
                    if len(batch_points) >= self.chunk_size:
                        points = np.array(batch_points, dtype=np.float32)
                        colors = np.array(batch_colors, dtype=np.uint8)
                        
                        yield points, colors, start_idx, chunk_idx
                        
                        self.processed_points += len(points)
                        start_idx += len(points)
                        chunk_idx += 1
                        batch_points = []
                        batch_colors = []
                        
                except (ValueError, IndexError):
                    continue
        
        # Yield remaining points
        if batch_points:
            points = np.array(batch_points, dtype=np.float32)
            colors = np.array(batch_colors, dtype=np.uint8)
            
            yield points, colors, start_idx, chunk_idx
            
            self.processed_points += len(points)
    
    def get_progress(self) -> float:
        """Get progress as a fraction (0.0 to 1.0)."""
        if self.total_points is None or self.total_points == 0:
            return 0.0
        return self.processed_points / self.total_points


def stream_xyzrgb_txt_chunks(filepath: str, chunk_size: int = 500000) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields chunks of points from a large TXT file.
    
    Memory-efficient: only holds one chunk in memory at a time.
    
    Args:
        filepath: Path to the TXT file
        chunk_size: Number of points per chunk (default: 500K)
    
    Yields:
        Tuple of (points, colors) for each chunk
    """
    reader = StreamingTXTReader(filepath, chunk_size)
    
    for points, colors, start_idx, chunk_idx in reader.stream_chunks():
        yield points, colors


class BinaryNormalWriter:
    """
    Append-mode binary writer for normals.
    
    Writes normals incrementally without storing all results in memory.
    Format per point: x y z nx ny nz (6 float32 = 24 bytes)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize binary writer.
        
        Args:
            filepath: Path to output binary file
        """
        self.filepath = filepath
        self.total_written = 0
        
        # Create output directory
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Open file in append mode
        self.file = open(filepath, 'ab')
        print(f"  Binary writer opened: {filepath}")
    
    def write_chunk(self, points: np.ndarray, normals: np.ndarray) -> int:
        """
        Write a chunk of normals to the binary file.
        
        Args:
            points: (N, 3) xyz coordinates
            normals: (N, 3) normal vectors
        
        Returns:
            Number of points written
        """
        n_points = len(points)
        
        if n_points == 0:
            return 0
        
        # Stack points and normals: (x,y,z,nx,ny,nz) per point
        data = np.hstack([points, normals]).astype(np.float32)
        
        # Write to file
        data.tofile(self.file)
        
        self.total_written += n_points
        return n_points
    
    def close(self):
        """Close the binary file."""
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Binary writer closed. Total points written: {self.total_written:,}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def estimate_normals_streaming(
    input_file: str,
    output_file: str,
    k: int = 10,
    chunk_size: int = 500000,
    device: str = None,
    total_points: int = None
) -> int:
    """
    Streaming normal estimation pipeline.
    
    Processes point clouds in chunks without loading everything into memory.
    Writes results incrementally to a binary file.
    
    Args:
        input_file: Path to input TXT file
        output_file: Path to output binary file
        k: Number of neighbors for normal estimation
        chunk_size: Points per chunk (default: 500K)
        device: 'cuda', 'cpu', or None for auto-detect
        total_points: Pre-counted total points (skip counting if provided)
    
    Returns:
        Total number of points processed
    """
    from .normals import estimate_normals_chunked, get_device
    
    if device is None:
        device = get_device()
    
    print("="*60)
    print("STREAMING NORMAL ESTIMATION")
    print("="*60)
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Chunk size: {chunk_size:,}")
    print(f"  K neighbors: {k}")
    print(f"  Device: {device}")
    
    start_time = time.time()
    
    # Initialize reader and writer
    # Skip counting if total_points already provided
    skip_count = total_points is not None
    reader = StreamingTXTReader(input_file, chunk_size, count_points=not skip_count)
    if total_points is not None:
        reader.total_points = total_points
        print(f"  Total points: {reader.total_points:,} (pre-counted)")
    
    writer = BinaryNormalWriter(output_file)
    
    total_processed = 0
    
    try:
        for points, colors, start_idx, chunk_idx in reader.stream_chunks():
            chunk_time = time.time()
            
            # Compute normals for this chunk
            normals, _ = estimate_normals_chunked(
                points, 
                k=k, 
                chunk_size=chunk_size,
                device=device
            )
            
            # Write to binary file immediately
            writer.write_chunk(points, normals)
            
            total_processed += len(points)
            
            # Progress reporting
            elapsed = time.time() - start_time
            progress = reader.get_progress()
            points_per_sec = total_processed / elapsed if elapsed > 0 else 0
            remaining = (elapsed / progress * (1 - progress)) if progress > 0 else 0
            
            print(f"  Chunk {chunk_idx+1}: {len(points):,} points | "
                  f"Total: {total_processed:,} ({progress*100:.1f}%) | "
                  f"Speed: {points_per_sec:,.0f} pts/sec | "
                  f"ETA: {remaining:.0f}s")
    
    finally:
        writer.close()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"STREAMING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total points: {total_processed:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average speed: {total_processed/total_time:,.0f} pts/sec")
    print(f"  Output: {output_file}")
    
    return total_processed
