"""
PLY Export Module

Export Gaussian surfels to PLY format for 3DGS/2DGS rendering.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import struct


# Standard property order for 2DGS/3DGS PLY files
SURFEL_PROPERTIES = [
    ("x", "float"), ("y", "float"), ("z", "float"),  # position
    ("nx", "float"), ("ny", "float"), ("nz", "float"),  # normal
    ("tx", "float"), ("ty", "float"), ("tz", "float"),  # tangent
    ("bx", "float"), ("by", "float"), ("bz", "float"),  # bitangent
    ("opacity", "float"),  # opacity
    ("sx", "float"), ("sy", "float"), ("sz", "float"),  # scale
    ("rx", "float"), ("ry", "float"), ("rz", "float"), ("rw", "float"),  # rotation (quaternion)
    ("red", "float"), ("green", "float"), ("blue", "float"),  # color (0-1)
]


def write_ply(filepath: str, surfels: Dict[str, np.ndarray], binary: bool = False) -> None:
    """
    Write surfel data to a PLY file.

    Args:
        filepath: Path to output PLY file
        surfels: Dictionary of surfel attributes with keys:
            - position: (N, 3) xyz coordinates
            - normal: (N, 3) surface normals
            - tangent: (N, 3) tangent vectors
            - bitangent: (N, 3) bitangent vectors
            - opacity: (N,) opacity values
            - scale: (N, 3) scale parameters
            - rotation: (N, 4) quaternion [x, y, z, w]
            - color: (N, 3) rgb values (0-1)
        binary: If True, write binary PLY (smaller, ~3x smaller)
    
    Raises:
        ValueError: If surfel dictionary is missing required keys
    """
    # Validate required keys
    required_keys = ["position", "normal", "tangent", "bitangent", 
                     "opacity", "scale", "rotation", "color"]
    for key in required_keys:
        if key not in surfels:
            raise ValueError(f"Missing required surfel key: {key}")
    
    n_surfels = surfels["position"].shape[0]
    
    # Create output directory if needed
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build PLY header
    format_str = "binary_little_endian" if binary else "ascii"
    header_lines = [
        "ply",
        f"format {format_str} 1.0",
        f"element vertex {n_surfels}",
    ]
    
    # Add property definitions
    property_names = [
        "x", "y", "z",           # position
        "nx", "ny", "nz",        # normal
        "tx", "ty", "tz",        # tangent
        "bx", "by", "bz",        # bitangent
        "opacity",               # opacity
        "sx", "sy", "sz",        # scale
        "rx", "ry", "rz", "rw",  # rotation (quaternion xyzw)
        "red", "green", "blue",  # color
    ]
    
    for prop_name in property_names:
        header_lines.append(f"property float {prop_name}")
    
    header_lines.append("end_header")
    
    if binary:
        # Write binary PLY
        with open(path, 'wb') as f:
            # Write header
            header_bytes = "\n".join(header_lines).encode('ascii') + b"\n"
            f.write(header_bytes)
            
            # Write vertex data as binary float32 (little-endian)
            for i in range(n_surfels):
                data = []
                data.extend(surfels["position"][i].tolist())
                data.extend(surfels["normal"][i].tolist())
                data.extend(surfels["tangent"][i].tolist())
                data.extend(surfels["bitangent"][i].tolist())
                data.append(float(surfels["opacity"][i]))
                data.extend(surfels["scale"][i].tolist())
                data.extend(surfels["rotation"][i].tolist())
                data.extend(surfels["color"][i].tolist())
                
                # Pack as float32 little-endian using struct
                f.write(struct.pack(f'<{len(data)}f', *data))
    else:
        # Write ASCII PLY (original behavior)
        with open(path, 'w') as f:
            f.write("\n".join(header_lines) + "\n")
            
            # Write vertex data
            for i in range(n_surfels):
                values = []
                values.extend(surfels["position"][i].tolist())
                values.extend(surfels["normal"][i].tolist())
                values.extend(surfels["tangent"][i].tolist())
                values.extend(surfels["bitangent"][i].tolist())
                values.append(float(surfels["opacity"][i]))
                values.extend(surfels["scale"][i].tolist())
                values.extend(surfels["rotation"][i].tolist())
                values.extend(surfels["color"][i].tolist())
                
                formatted = []
                for v in values:
                    if isinstance(v, float):
                        formatted.append(f"{v:.10f}")
                    else:
                        formatted.append(str(v))
                
                f.write(" ".join(formatted) + "\n")
    
    format_name = "binary" if binary else "ASCII"
    print(f"  Wrote {n_surfels} surfels to {filepath} ({format_name})")


def read_ply(filepath: str, max_memory_gb: float = 4.0, start_idx: int = 0, 
             num_points: int = None) -> Dict[str, np.ndarray]:
    """
    Read surfel data from a PLY file (ASCII or binary).

    Args:
        filepath: Path to input PLY file
        max_memory_gb: Maximum memory to use in GB. Default 4GB.
                       Set to 0 to disable downsampling (may use lots of RAM).
        start_idx: Starting index for reading (for chunked reading)
        num_points: Number of points to read (None = read all or based on max_memory_gb)

    Returns:
        Dictionary of surfel attributes
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If PLY format is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")
    
    file_size = path.stat().st_size
    
    # Read header
    with open(path, 'rb') as f:
        header_raw = f.read(2048)
    
    # Decode header
    try:
        header_str = header_raw.decode('ascii')
    except:
        header_str = header_raw.decode('utf-8', errors='replace')
    
    lines = header_str.split('\n')
    
    # Parse header
    is_binary = False
    n_surfels = 0
    n_props = 0  # Number of properties per vertex
    header_size = 0
    
    for i, line in enumerate(lines):
        if line.startswith("format"):
            is_binary = "binary" in line.lower()
        elif line.startswith("element vertex"):
            n_surfels = int(line.split()[-1])
        elif line.startswith("property"):
            n_props += 1
        elif line.strip() == "end_header":
            # Find actual header size in bytes by searching for "end_header\n"
            header_size = 0
            with open(path, 'rb') as f:
                marker = b"end_header"
                marker_len = len(marker)
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    idx = chunk.find(marker)
                    if idx != -1:
                        # header_size is where data starts (after end_header\n)
                        header_size += idx + marker_len + 1  # +1 for newline
                        break
                    header_size += len(chunk)
            break
    
    if n_surfels == 0:
        raise ValueError("No vertices found in PLY file")
    
    if n_props == 0:
        raise ValueError("No properties found in PLY file")
    
    print(f"  Detected {n_surfels} vertices, {n_props} properties, binary={is_binary}")
    
    # Check if file is too large for available memory
    # Estimate memory needed: vertex_data = n_surfels * n_props * 4 bytes
    estimated_memory = n_surfels * n_props * 4
    
    # If file > max_memory_gb or estimated memory > limit, use downsampling for viewer
    MAX_MEMORY = max_memory_gb * 1024**3 if max_memory_gb > 0 else float('inf')
    use_downsample = estimated_memory > MAX_MEMORY
    
    # Handle chunked reading (start_idx and num_points)
    # If these are specified, we always read in chunked mode
    use_chunked_read = (start_idx > 0 or num_points is not None) and is_binary
    
    if use_chunked_read:
        # Chunked reading mode - read specific range
        if num_points is None or num_points < 0:
            num_points = n_surfels - start_idx
        if start_idx >= n_surfels:
            return None  # Beyond end of file
        end_idx = min(start_idx + num_points, n_surfels)
        actual_read = end_idx - start_idx
        
        bytes_per_point = n_props * 4
        data_size = actual_read * bytes_per_point
        offset_bytes = header_size + start_idx * bytes_per_point
        
        print(f"    Reading chunk: {start_idx} to {end_idx} ({actual_read:,} points)")
        
        with open(path, 'rb') as f:
            f.seek(offset_bytes)
            vertex_data = np.frombuffer(
                f.read(data_size), 
                dtype=np.float32
            ).reshape(-1, n_props)
        
        print(f"    Read chunk: {vertex_data.shape}")
        
    elif use_downsample and max_memory_gb > 0:
        # Calculate downsampled count to fit in memory
        max_points = int(MAX_MEMORY // (n_props * 4))
        sample_count = min(n_surfels, max_points)
        step = max(1, n_surfels // sample_count)  # uniform sampling stride, ensure at least 1
        print(f"    File too large ({estimated_memory/1024**3:.1f}GB), sampling ~{sample_count:,} points (every {step}th)")
        
        # Read sampled points - memory efficient: read chunk, sample, repeat
        if is_binary:
            bytes_per_point = n_props * 4
            chunk_size = 1_000_000  # Read 1M points at a time
            
            vertex_data = []
            sampled_count = 0
            with open(path, 'rb') as f:
                f.seek(header_size)
                offset = 0
                while sampled_count < sample_count and offset < n_surfels:
                    points_to_read = min(chunk_size, n_surfels - offset)
                    chunk_data = np.frombuffer(
                        f.read(points_to_read * bytes_per_point), 
                        dtype=np.float32
                    ).reshape(-1, n_props)
                    
                    # Sample every step-th point from this chunk
                    sampled = chunk_data[::step]
                    vertex_data.append(sampled)
                    sampled_count += len(sampled)
                    offset += points_to_read
                    print(f"    Sampled progress: {sampled_count:,} points")
            
            vertex_data = np.vstack(vertex_data)[:sample_count]
            print(f"    Final sampled: {len(vertex_data):,} points")
        else:
            # For ASCII, read and sample
            vertex_data = []
            sampled_count = 0
            with open(path, 'r') as f:
                header_end = 0
                for i, line in enumerate(f):
                    if line.strip() == "end_header":
                        header_end = i + 1
                        break
                
                offset = 0
                while sampled_count < sample_count and offset < n_surfels:
                    chunk_lines = []
                    for j in range(min(100000, n_surfels - offset)):
                        line = f.readline()
                        if line:
                            chunk_lines.append([float(x) for x in line.strip().split()])
                    
                    if chunk_lines:
                        chunk_data = np.array(chunk_lines, dtype=np.float32)
                        sampled = chunk_data[::step]
                        vertex_data.append(sampled)
                        sampled_count += len(sampled)
                    offset += len(chunk_lines)
            
            if vertex_data:
                vertex_data = np.vstack(vertex_data)[:sample_count]
            else:
                vertex_data = np.zeros((0, n_props), dtype=np.float32)
    elif is_binary:
        # Standard reading for smaller files
        data_size = file_size - header_size
        needed_floats = n_surfels * n_props
        bytes_needed = needed_floats * 4  # 4 bytes per float32
        
        with open(path, 'rb') as f:
            f.seek(header_size)
            vertex_data = np.frombuffer(
                f.read(bytes_needed), 
                dtype=np.float32
            ).reshape(-1, n_props)
    else:
        # ASCII format
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Skip header, read vertex data
        vertex_data = []
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = i + 1
                break
        
        for i in range(header_end, header_end + n_surfels):
            if i < len(lines):
                values = [float(x) for x in lines[i].strip().split()]
                vertex_data.append(values)
        
        vertex_data = np.array(vertex_data, dtype=np.float64)
    
    # Print debug info
    print(f"  Read {n_surfels} surfels, data shape: {vertex_data.shape}")
    
    # Map properties to surfel dictionary
    surfels = {}
    
    # Expected property order (21 properties total)
    prop_map = {
        0: ("x", "position", 3),
        3: ("nx", "normal", 3),
        6: ("tx", "tangent", 3),
        9: ("bx", "bitangent", 3),
        12: ("opacity", "opacity", 1),
        13: ("sx", "scale", 3),
        16: ("rx", "rotation", 4),
        20: ("red", "color", 3),
    }
    
    for start_idx, (prop_name, surfel_key, count) in prop_map.items():
        end_idx = start_idx + count
        if end_idx <= vertex_data.shape[1]:
            data = vertex_data[:, start_idx:end_idx]
            if count == 1:
                data = data.flatten()
            surfels[surfel_key] = data
    
    format_type = "binary" if is_binary else "ASCII"
    print(f"  Read {n_surfels} surfels from {filepath} ({format_type})")
    
    return surfels


def ply_header_from_surfels(n_surfels: int, binary: bool = False) -> str:
    """
    Generate PLY header for surfel data.

    Args:
        n_surfels: Number of surfels in the file
        binary: If True, generate header for binary PLY format

    Returns:
        PLY header string
    """
    format_str = "binary_little_endian 1.0" if binary else "ascii 1.0"
    
    header = [
        "ply",
        f"format {format_str}",
        f"element vertex {n_surfels}",
    ]
    
    # Add properties in standard order
    property_names = [
        "x", "y", "z",           # position
        "nx", "ny", "nz",        # normal
        "tx", "ty", "tz",        # tangent
        "bx", "by", "bz",        # bitangent
        "opacity",               # opacity
        "sx", "sy", "sz",        # scale
        "rx", "ry", "rz", "rw",  # rotation
        "red", "green", "blue",  # color
    ]
    
    for prop_name in property_names:
        header.append(f"property float {prop_name}")
    
    header.append("end_header")
    
    return "\n".join(header) + "\n"
