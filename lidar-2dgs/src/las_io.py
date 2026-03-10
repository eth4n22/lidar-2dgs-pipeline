"""
LAS/LAZ I/O Module

Support for ASPRS LAS format (industry standard for LiDAR).
Uses laspy for pure Python implementation (no C++ dependencies).

Supported:
- Reading: .las, .laz files
- Writing: .las files
- Attributes: position, intensity, classification, return number, etc.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Try importing laspy, provide helpful error if not available
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

try:
    import lazrs
    HAS_LAZRS = True
except ImportError:
    HAS_LAZRS = False


def load_las(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load ASPRS LAS/LAS file.

    Args:
        filepath: Path to .las or .laz file

    Returns:
        Dictionary with:
            - position: (N, 3) xyz coordinates
            - intensity: (N,) intensity values
            - classification: (N,) classification codes
            - return_number: (N,) return numbers
            - num_returns: (N,) number of returns per pulse
            - scan_angle: (N,) scan angle
            - metadata: Dictionary with LAS header info

    Raises:
        ImportError: If laspy is not installed
        FileNotFoundError: If file doesn't exist
    """
    if not HAS_LASPY:
        raise ImportError(
            "laspy not installed. Run: pip install laspy\n"
            "For LAZ support: pip install lazrs"
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check for LAZ support
    if path.suffix.lower() == '.laz' and not HAS_LAZRS:
        raise ImportError(
            "LAZ (compressed LAS) support requires lazrs.\n"
            "Run: pip install lazrs"
        )

    # Read LAS file
    try:
        las = laspy.read(str(path))
    except Exception as e:
        raise ValueError(f"Failed to read LAS file: {e}")

    # Extract points
    points = np.column_stack([
        las.x,
        las.y,
        las.z
    ]).astype(np.float32)

    result = {
        "position": points,
        "metadata": {
            "file_path": str(path),
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "version": las.header.version,
            "point_format": las.header.point_format,
            "point_count": len(las.points),
            "scale": (las.header.scale[0], las.header.scale[1], las.header.scale[2]),
            "offset": (las.header.offset[0], las.header.offset[1], las.header.offset[2]),
            "bounds": (
                (las.header.min[0], las.header.max[0]),
                (las.header.min[1], las.header.max[1]),
                (las.header.min[2], las.header.max[2])
            ),
            "vlr_count": len(las.header.vlrs)
        }
    }

    # Extract available dimensions - use dtype.names for structured array
    dim_names = las.points.array.dtype.names or ()

    if 'intensity' in dim_names:
        result["intensity"] = las.intensity.astype(np.float32)

    if 'classification' in dim_names:
        result["classification"] = las.classification.astype(np.uint8)

    if 'return_number' in dim_names:
        result["return_number"] = las.return_number.astype(np.uint8)

    if 'num_returns' in dim_names:
        result["num_returns"] = las.num_returns.astype(np.uint8)

    if 'scan_angle' in dim_names:
        result["scan_angle"] = las.scan_angle.astype(np.float32)

    if 'red' in dim_names and 'green' in dim_names and 'blue' in dim_names:
        # RGB colors (typically 16-bit)
        red = las.red.astype(np.float32)
        green = las.green.astype(np.float32)
        blue = las.blue.astype(np.float32)

        # Normalize to 0-255 range
        red = np.clip(red / 65535.0 * 255.0, 0, 255)
        green = np.clip(green / 65535.0 * 255.0, 0, 255)
        blue = np.clip(blue / 65535.0 * 255.0, 0, 255)

        result["color"] = np.column_stack([red, green, blue]).astype(np.float32)

    return result


def save_las(filepath: str,
             position: np.ndarray,
             intensity: Optional[np.ndarray] = None,
             classification: Optional[np.ndarray] = None,
             color: Optional[np.ndarray] = None,
             scale: Tuple[float, float, float] = (0.01, 0.01, 0.01)) -> None:
    """
    Save point cloud to ASPRS LAS format.

    Args:
        filepath: Output .las file path
        position: (N, 3) xyz coordinates
        intensity: Optional (N,) intensity values
        classification: Optional (N,) classification codes
        color: Optional (N, 3) RGB values (0-255)
        scale: XYZ scale factors (default: 0.01 = 1cm precision)
    """
    if not HAS_LASPY:
        raise ImportError("laspy not installed. Run: pip install laspy")

    from laspy import Header, PointFormat, LasData

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine point format based on available data
    if color is not None:
        point_format = 2  # RGB
    else:
        point_format = 0  # Basic

    # Create header
    header = Header(version="1.2")
    header.point_format = PointFormat(point_format)
    header.scale = scale

    # Calculate bounds
    header.min = [position[:, 0].min(), position[:, 1].min(), position[:, 2].min()]
    header.max = [position[:, 0].max(), position[:, 1].max(), position[:, 2].max()]

    # Create LAS data
    las = LasData(header)

    # Set positions (handle scale/offset)
    las.x = position[:, 0]
    las.y = position[:, 1]
    las.z = position[:, 2]

    # Set optional dimensions
    if intensity is not None:
        las.intensity = intensity.astype(np.uint16)

    if classification is not None:
        las.classification = classification.astype(np.uint8)

    if color is not None:
        # Convert 0-255 to 0-65535 for LAS format
        las.red = (color[:, 0] / 255.0 * 65535.0).astype(np.uint16)
        las.green = (color[:, 1] / 255.0 * 65535.0).astype(np.uint16)
        las.blue = (color[:, 2] / 255.0 * 65535.0).astype(np.uint16)

    # Write file
    las.write(str(path))
    print(f"  Saved LAS file: {path}")


def detect_format(filepath: str) -> str:
    """
    Detect point cloud format from file extension.

    Args:
        filepath: Path to file

    Returns:
        Format string: "las", "laz", "txt", "ply", or "unknown"
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    format_map = {
        '.las': 'las',
        '.laz': 'laz',
        '.txt': 'txt',
        '.ply': 'ply'
    }

    return format_map.get(ext, 'unknown')


def load_point_cloud(filepath: str) -> Dict[str, np.ndarray]:
    """
    Universal point cloud loader.

    Automatically detects format and loads accordingly.

    Args:
        filepath: Path to point cloud file

    Returns:
        Dictionary with position, optional color/intensity
    """
    fmt = detect_format(filepath)

    if fmt == 'las' or fmt == 'laz':
        return load_las(filepath)
    elif fmt == 'txt':
        from src.txt_io import load_xyzrgb_txt
        data = load_xyzrgb_txt(filepath)
        # Convert to standard format
        return {
            "position": data["position"],
            "color": data.get("color"),
            "metadata": {"format": "txt"}
        }
    elif fmt == 'ply':
        from src.export_ply import read_ply
        return read_ply(filepath)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_las_header(filepath: str) -> Dict:
    """
    Read only LAS header without loading points.

    Useful for previewing large files.

    Args:
        filepath: Path to .las or .laz file

    Returns:
        Dictionary with header information
    """
    if not HAS_LASPY:
        raise ImportError("laspy not installed. Run: pip install laspy")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Quick header read using laspy
    try:
        with laspy.open(str(path)) as las_file:
            header = las_file.header
            return {
                "file_path": str(path),
                "file_name": path.name,
                "version": header.version,
                "point_count": header.point_count,
                "point_format": header.point_format,
                "bounds": (
                    (header.min[0], header.max[0]),
                    (header.min[1], header.max[1]),
                    (header.min[2], header.max[2])
                ),
                "scale": header.scale,
                "offset": header.offset
            }
    except Exception as e:
        raise ValueError(f"Failed to read LAS header: {e}")


def sample_las(filepath: str, n_points: int = 10000) -> Dict[str, np.ndarray]:
    """
    Sample N points from LAS file for quick preview.

    Args:
        filepath: Path to .las or .laz file
        n_points: Number of points to sample

    Returns:
        Dictionary with sampled position, intensity, color
    """
    if not HAS_LASPY:
        raise ImportError("laspy not installed. Run: pip install laspy")

    import random

    path = Path(filepath)
    total_points = laspy.read(str(path)).header.point_count

    if n_points >= total_points:
        return load_las(filepath)

    # Random sampling
    indices = sorted(random.sample(range(total_points), n_points))

    with laspy.open(str(path)) as las_file:
        # Read all points (laspy 2.x doesn't have vlrs parameter)
        las = laspy.read(str(path))
        sampled_points = las.points[indices]

        result = {
            "position": np.column_stack([
                sampled_points.x,
                sampled_points.y,
                sampled_points.z
            ]).astype(np.float32),
            "metadata": {
                "sampled_from": total_points,
                "sample_count": n_points
            }
        }

        if 'intensity' in sampled_points.array.dtype.names:
            result["intensity"] = sampled_points.intensity.astype(np.float32)

        if 'red' in sampled_points.array.dtype.names:
            result["color"] = np.column_stack([
                sampled_points.red,
                sampled_points.green,
                sampled_points.blue
            ]).astype(np.float32)

        return result


def sample_las_streaming(filepath: str, n_points: int = 10000, chunk_size: int = 500000) -> Dict[str, np.ndarray]:
    """
    Sample N points from LAS file using streaming (memory-efficient).

    For very large files, this samples points across the entire file
    rather than loading everything into memory.

    Args:
        filepath: Path to .las or .laz file
        n_points: Number of points to sample
        chunk_size: Chunk size for reading (default 500K)

    Returns:
        Dictionary with sampled position, intensity, color
    """
    if not HAS_LASPY:
        raise ImportError("laspy not installed. Run: pip install laspy")

    import random

    path = Path(filepath)
    with laspy.open(str(path)) as f:
        total_points = f.header.point_count

    if n_points >= total_points:
        return load_las(filepath)

    # Stratified sampling across file chunks
    n_chunks = (total_points + chunk_size - 1) // chunk_size
    points_per_chunk = max(1, n_points // n_chunks)

    sampled_points = []
    sampled_intensity = []
    sampled_color = []

    chunk_indices = list(range(n_chunks))
    random.shuffle(chunk_indices)

    sampled_total = 0

    for chunk_idx in chunk_indices:
        if sampled_total >= n_points:
            break

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_points)
        actual_chunk_size = end - start

        # Sample from this chunk
        chunk_sample_size = min(points_per_chunk, n_points - sampled_total)
        if chunk_sample_size <= 0:
            continue

        chunk_indices_local = sorted(random.sample(range(actual_chunk_size), chunk_sample_size))

        with laspy.open(str(path)) as las_file:
            # Read just this chunk
            las_file.seek(start)
            chunk_points = las_file.read_points(actual_chunk_size)

            sampled_points.append(np.column_stack([
                chunk_points.x,
                chunk_points.y,
                chunk_points.z
            ]).astype(np.float32))

            if 'intensity' in chunk_points.array.dtype.names:
                sampled_intensity.append(chunk_points.intensity.astype(np.float32))

            if 'red' in chunk_points.array.dtype.names:
                if not sampled_color:
                    sampled_color = []
                sampled_color.append(np.column_stack([
                    chunk_points.red,
                    chunk_points.green,
                    chunk_points.blue
                ]).astype(np.float32))

            sampled_total += chunk_sample_size

    result = {
        "position": np.vstack(sampled_points).astype(np.float32),
        "metadata": {
            "sampled_from": total_points,
            "sample_count": sampled_total
        }
    }

    if sampled_intensity:
        result["intensity"] = np.concatenate(sampled_intensity)

    if sampled_color:
        result["color"] = np.vstack(sampled_color).astype(np.float32)

    return result
