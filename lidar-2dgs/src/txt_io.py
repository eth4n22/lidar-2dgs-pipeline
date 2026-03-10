"""
TXT I/O Module

Load and save LiDAR point cloud data from/to TXT files.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from .security import validate_file_path
except ImportError:
    # Fallback if security module not available
    def validate_file_path(filepath: str, must_exist: bool = False) -> Path:
        path = Path(filepath)
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return path


def load_xyzrgb_txt(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load point cloud data from a TXT file.

    Expected format: one point per line, "x y z r g b" (space-separated)

    Args:
        filepath: Path to the TXT file

    Returns:
        Dictionary with keys:
            - position: (N, 3) xyz coordinates
            - color: (N, 3) rgb values (0-255)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Stream file line-by-line to handle large files
    positions = []
    colors = []
    line_num = 0

    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Line {line_num}: Expected at least 3 values (x y z), got {len(parts)}")

            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                positions.append([x, y, z])

                # Parse colors if present - use int() for RGB values (0-255)
                if len(parts) >= 6:
                    r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                    # Clamp to valid RGB range
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))
                    colors.append([r, g, b])
                elif len(parts) == 4:
                    # Grayscale intensity - convert to int
                    intensity = int(float(parts[3]))
                    intensity = max(0, min(255, intensity))
                    colors.append([intensity, intensity, intensity])
                else:
                    # No colors - use white (255, 255, 255)
                    colors.append([255, 255, 255])
            except ValueError as e:
                raise ValueError(f"Line {line_num}: Failed to parse values - {e}")

    if len(positions) == 0:
        raise ValueError("No valid points found in file")

    return {
        "position": np.array(positions, dtype=np.float32),
        "color": np.array(colors, dtype=np.uint8)  # RGB values are uint8 (0-255)
    }


def load_xyz_txt(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load point cloud data from a TXT file (xyz only, no colors).

    Args:
        filepath: Path to the TXT file

    Returns:
        Dictionary with 'position' key
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Stream file line-by-line to handle large files
    positions = []
    line_num = 0

    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Line {line_num}: Expected 3 values (x y z), got {len(parts)}")

            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                positions.append([x, y, z])
            except ValueError as e:
                raise ValueError(f"Line {line_num}: Failed to parse - {e}")

    if len(positions) == 0:
        raise ValueError("No valid points found in file")

    return {
        "position": np.array(positions, dtype=np.float32)
    }


def validate_format(filepath: str) -> Tuple[bool, str]:
    """
    Validate that a TXT file has the expected XYZRGB format.

    Args:
        filepath: Path to the TXT file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(filepath)

    if not path.exists():
        return False, f"File not found: {filepath}"

    valid_lines = 0
    line_num = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            if line_num > 100:  # Check first 100 lines
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 3:
                return False, f"Line {line_num}: Expected at least 3 values (x y z)"
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                valid_lines += 1
            except ValueError:
                return False, f"Line {line_num}: Invalid float values"

    if valid_lines == 0:
        return False, "No valid point lines found"

    return True, f"Valid format ({valid_lines} points)"


def save_xyzrgb_txt(filepath: str, points: np.ndarray,
                    colors: Optional[np.ndarray] = None) -> None:
    """
    Save point cloud data to a TXT file.

    Args:
        filepath: Path to output TXT file
        points: (N, 3) xyz coordinates
        colors: Optional (N, 3) rgb values (0-255)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for i in range(points.shape[0]):
            line = f"{points[i, 0]} {points[i, 1]} {points[i, 2]}"
            if colors is not None:
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(line + "\n")


def save_xyz_txt(filepath: str, points: np.ndarray) -> None:
    """
    Save point cloud data to a TXT file (xyz only).

    Args:
        filepath: Path to output TXT file
        points: (N, 3) xyz coordinates
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")


def detect_format(filepath: str) -> str:
    """
    Detect the format of a TXT point cloud file.

    Args:
        filepath: Path to the TXT file

    Returns:
        Format string: "xyz", "xyzrgb", or "unknown"
    """
    path = Path(filepath)

    if not path.exists():
        return "unknown"

    xyz_count = 0
    xyzrgb_count = 0
    line_num = 0

    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            if line_num > 100:  # Check first 100 lines
                break
            
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 6:
                xyzrgb_count += 1
            elif len(parts) >= 3:
                xyz_count += 1

    if xyzrgb_count > xyz_count:
        return "xyzrgb"
    elif xyz_count > 0:
        return "xyz"
    return "unknown"


def load_point_cloud(filepath: str) -> Dict[str, np.ndarray]:
    """
    Smart loader that detects and loads point cloud from TXT.

    Args:
        filepath: Path to the point cloud file

    Returns:
        Dictionary with 'position' and optionally 'color' keys
    """
    fmt = detect_format(filepath)

    if fmt == "xyzrgb":
        return load_xyzrgb_txt(filepath)
    elif fmt == "xyz":
        return load_xyz_txt(filepath)
    else:
        # Try xyzrgb first, then xyz
        try:
            return load_xyzrgb_txt(filepath)
        except ValueError:
            return load_xyz_txt(filepath)
