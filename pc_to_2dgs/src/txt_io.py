"""
TXT I/O Module

Load and save LiDAR point cloud data from/to TXT files.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np


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
