"""
PLY Export Module

Export Gaussian surfels to PLY format for 3DGS/2DGS rendering.
Enhanced with support for both rotation/scale and covariance formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

# Properties for covariance format (alternative to rotation/scale)
COVARIANCE_PROPERTIES = [
    ("x", "float"), ("y", "float"), ("z", "float"),  # position
    ("nx", "float"), ("ny", "float"), ("nz", "float"),  # normal
    ("c_xx", "float"), ("c_xy", "float"), ("c_xz", "float"),  # covariance row 1
    ("c_yx", "float"), ("c_yy", "float"), ("c_yz", "float"),  # covariance row 2
    ("c_zx", "float"), ("c_zy", "float"), ("c_zz", "float"),  # covariance row 3
    ("opacity", "float"),  # opacity
    ("red", "float"), ("green", "float"), ("blue", "float"),  # color (0-1)
]


def validate_surfels(surfels: Dict[str, np.ndarray], format: str = "standard") -> None:
    """
    Validate surfel dictionary has all required keys.

    Args:
        surfels: Dictionary of surfel attributes
        format: "standard" for rotation/scale, "covariance" for cov matrices

    Raises:
        ValueError: If required keys are missing
    """
    if format == "standard":
        required = ["position", "normal", "tangent", "bitangent",
                    "opacity", "scale", "rotation", "color"]
    else:
        required = ["position", "normal", "covariance", "opacity", "color"]

    for key in required:
        if key not in surfels:
            raise ValueError(f"Missing required surfel key: {key}")


def write_ply(filepath: str, surfels: Dict[str, np.ndarray],
              binary: bool = True, verbose: bool = True) -> None:
    """
    Write surfel data to a PLY file.

    Args:
        filepath: Path to output PLY file
        surfels: Dictionary of surfel attributes:
            - position: (N, 3) xyz coordinates
            - normal: (N, 3) surface normals
            - tangent: (N, 3) tangent vectors
            - bitangent: (N, 3) bitangent vectors
            - opacity: (N,) or (N, 1) opacity values
            - scale: (N, 3) scale parameters
            - rotation: (N, 4) quaternion [x, y, z, w]
            - color: (N, 3) rgb values (0-1)
        binary: If True, write binary PLY (smaller, faster)
        verbose: Print progress messages
    """
    validate_surfels(surfels, "standard")

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

    if verbose:
        format_name = "binary" if binary else "ASCII"
        print(f"  Writing {n_surfels} surfels to {filepath} ({format_name})...")

    if binary:
        # Write binary PLY - vectorized for performance
        with open(path, 'wb') as f:
            # Write header
            header_bytes = "\n".join(header_lines).encode('ascii') + b"\n"
            f.write(header_bytes)

            # Pre-allocate single array for all vertex data (23 floats per vertex)
            # Layout: position(3) + normal(3) + tangent(3) + bitangent(3) + opacity(1) + scale(3) + rotation(4) + color(3) = 23
            vertex_data = np.empty((n_surfels, 23), dtype=np.float32)

            # Helper function to normalize array shapes
            def normalize_1d(arr, name):
                """Normalize 1D arrays - flatten any 2D to 1D."""
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    return arr
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    return arr.ravel()
                elif arr.ndim == 2 and arr.shape[1] in (3, 4):
                    # Multi-channel data - flatten for single-column assignment
                    return arr.ravel()
                else:
                    raise ValueError(f"{name} has unexpected shape {arr.shape}, expected (N,) or (N,1) or (N,3)")

            def normalize_3d(arr, name):
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
                elif arr.ndim == 1 and arr.shape[0] == 3:
                    return arr.reshape(1, 3)
                else:
                    raise ValueError(f"{name} has unexpected shape {arr.shape}, expected (N, 3)")

            def normalize_4d(arr, name):
                """Handle quaternions and other 4-component arrays."""
                arr = np.asarray(arr)
                if arr.ndim == 2 and arr.shape[1] == 4:
                    return arr
                else:
                    raise ValueError(f"{name} has unexpected shape {arr.shape}, expected (N, 4)")

            # Fill in all columns at once (vectorized)
            vertex_data[:, 0:3] = normalize_3d(surfels["position"], "position")
            vertex_data[:, 3:6] = normalize_3d(surfels["normal"], "normal")
            vertex_data[:, 6:9] = normalize_3d(surfels["tangent"], "tangent")
            vertex_data[:, 9:12] = normalize_3d(surfels["bitangent"], "bitangent")
            vertex_data[:, 12] = normalize_1d(surfels["opacity"], "opacity")
            vertex_data[:, 13:16] = normalize_3d(surfels["scale"], "scale")
            vertex_data[:, 16:20] = normalize_4d(surfels["rotation"], "rotation")
            vertex_data[:, 20:23] = normalize_3d(surfels["color"], "color")

            # Write as contiguous float32 array (little-endian is native on x86)
            vertex_data.tofile(f)
    else:
        # Write ASCII PLY
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

    if verbose:
        print(f"  Successfully wrote {n_surfels} surfels")


def write_ply_covariance(filepath: str, surfels: Dict[str, np.ndarray],
                         binary: bool = True, verbose: bool = True) -> None:
    """
    Write surfel data to PLY using covariance matrices instead of rotation/scale.

    Args:
        filepath: Path to output PLY file
        surfels: Dictionary of surfel attributes:
            - position: (N, 3) xyz coordinates
            - normal: (N, 3) surface normals
            - covariance: (N, 3, 3) covariance matrices
            - opacity: (N,) opacity values
            - color: (N, 3) rgb values (0-1)
        binary: If True, write binary PLY
        verbose: Print progress messages
    """
    validate_surfels(surfels, "covariance")

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
        "c_xx", "c_xy", "c_xz",  # covariance row 1
        "c_yx", "c_yy", "c_yz",  # covariance row 2
        "c_zx", "c_zy", "c_zz",  # covariance row 3
        "opacity",               # opacity
        "red", "green", "blue",  # color
    ]

    for prop_name in property_names:
        header_lines.append(f"property float {prop_name}")

    header_lines.append("end_header")

    if verbose:
        format_name = "binary" if binary else "ASCII"
        print(f"  Writing {n_surfels} surfels (covariance format) to {filepath} ({format_name})...")

    if binary:
        with open(path, 'wb') as f:
            header_bytes = "\n".join(header_lines).encode('ascii') + b"\n"
            f.write(header_bytes)

            # Pre-allocate array for all vertex data (16 floats per vertex)
            vertex_data = np.empty((n_surfels, 16), dtype=np.float32)

            # Fill in all columns at once (vectorized)
            vertex_data[:, 0:3] = surfels["position"]           # x, y, z
            vertex_data[:, 3:6] = surfels["normal"]             # nx, ny, nz
            # Flatten covariance matrices row-major (9 floats per vertex)
            vertex_data[:, 6:15] = surfels["covariance"].reshape(n_surfels, 9)
            # Handle opacity - ensure 1D array of correct length
            opacity = np.asarray(surfels["opacity"]).ravel()
            vertex_data[:, 15] = opacity
            vertex_data[:, 16:19] = surfels["color"]           # red, green, blue

            # Write as contiguous float32 array
            vertex_data.tofile(f)
    else:
        with open(path, 'w') as f:
            f.write("\n".join(header_lines) + "\n")

            for i in range(n_surfels):
                values = []
                values.extend(surfels["position"][i].tolist())
                values.extend(surfels["normal"][i].tolist())
                cov = surfels["covariance"][i]
                values.extend([cov[0, 0], cov[0, 1], cov[0, 2]])
                values.extend([cov[1, 0], cov[1, 1], cov[1, 2]])
                values.extend([cov[2, 0], cov[2, 1], cov[2, 2]])
                values.append(float(surfels["opacity"][i]))
                values.extend(surfels["color"][i].tolist())

                formatted = [f"{v:.10f}" for v in values]
                f.write(" ".join(formatted) + "\n")

    if verbose:
        print(f"  Successfully wrote {n_surfels} surfels")


def read_ply(filepath: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Read surfel data from a PLY file.

    Args:
        filepath: Path to input PLY file
        verbose: Print progress messages

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

    try:
        header_str = header_raw.decode('ascii')
    except:
        header_str = header_raw.decode('utf-8', errors='replace')

    lines = header_str.split('\n')

    # Parse header
    is_binary = False
    n_surfels = 0
    n_props = 0
    header_size = 0

    for i, line in enumerate(lines):
        if line.startswith("format"):
            is_binary = "binary" in line.lower()
        elif line.startswith("element vertex"):
            n_surfels = int(line.split()[-1])
        elif line.startswith("property"):
            n_props += 1
        elif line.strip() == "end_header":
            # Find actual header size
            with open(path, 'rb') as f:
                marker = b"end_header"
                marker_len = len(marker)
                header_size = 0
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    idx = chunk.find(marker)
                    if idx != -1:
                        header_size += idx + marker_len + 1
                        break
                    header_size += len(chunk)
            break

    if n_surfels == 0:
        raise ValueError("No vertices found in PLY file")

    if n_props == 0:
        raise ValueError("No properties found in PLY file")

    if verbose:
        print(f"  Detected {n_surfels} vertices, {n_props} properties, binary={is_binary}")

    # Read vertex data
    if is_binary:
        data_size = file_size - header_size

        with open(path, 'rb') as f:
            f.seek(header_size)
            raw_data = f.read(data_size)

        n_floats = len(raw_data) // 4
        n_complete = n_floats // n_props

        if n_complete < n_surfels:
            raise ValueError(f"Binary data incomplete: expected {n_surfels * n_props} floats, got {n_floats}")

        needed_floats = n_surfels * n_props
        floats = struct.unpack(f'<{needed_floats}f', raw_data[:needed_floats * 4])
        vertex_data = np.array(floats, dtype=np.float32).reshape(-1, n_props)
    else:
        with open(path, 'r') as f:
            lines = f.readlines()

        vertex_data = []
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = i + 1
                break

        for i in range(header_end, min(header_end + n_surfels, len(lines))):
            values = [float(x) for x in lines[i].strip().split()]
            if len(values) == n_props:
                vertex_data.append(values)

        vertex_data = np.array(vertex_data, dtype=np.float64)
        n_surfels = len(vertex_data)

    if verbose:
        print(f"  Read {n_surfels} surfels, data shape: {vertex_data.shape}")

    # Detect format and map properties
    surfels = {}

    # Check if this is covariance format (13 props) or standard format (21 props)
    if n_props == 13:
        # Covariance format
        prop_map = {
            0: ("x", "position", 3),
            3: ("nx", "normal", 3),
            6: ("c_xx", "covariance", 9),  # 3x3 matrix flattened
            15: ("opacity", "opacity", 1),
            16: ("red", "color", 3),
        }

        for start_idx, (prop_name, surfel_key, count) in prop_map.items():
            end_idx = start_idx + count
            if end_idx <= vertex_data.shape[1]:
                data = vertex_data[:, start_idx:end_idx]
                if count == 1:
                    data = data.flatten()
                if count == 9:
                    # Reshape to 3x3 matrices
                    data = data.reshape(-1, 3, 3)
                surfels[surfel_key] = data

    else:
        # Standard format (21 props)
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
    if verbose:
        print(f"  Successfully read {n_surfels} surfels from {filepath} ({format_type})")

    return surfels


# =============================================================================
# INCREMENTAL PLY WRITING (for billion+ point clouds)
# =============================================================================


class IncrementalPlyWriter:
    """
    Write PLY files incrementally, chunk by chunk.
    
    Essential for billion+ point clouds where accumulating everything in memory
    would exceed available RAM.
    
    Usage:
        writer = IncrementalPlyWriter("output.ply")
        writer.write_header(total_expected=10000000)  # Optional: set expected count
        for chunk in large_dataset:
            writer.write_chunk(chunk)
        writer.finalize()  # Updates vertex count if needed
    """
    
    def __init__(self, filepath: str, binary: bool = True, verbose: bool = True):
        """
        Initialize incremental PLY writer.
        
        Args:
            filepath: Output PLY file path
            binary: Write binary PLY (smaller, faster)
            verbose: Print progress messages
        """
        self.filepath = Path(filepath)
        self.binary = binary
        self.verbose = verbose
        
        self.file = None
        self.header_size = 0
        self.vertex_count = 0
        self.property_size = 21 * 4  # 21 floats per vertex, 4 bytes each
        self.data_start = 0
        self.is_finalized = False
        
    def _get_property_names(self) -> List[str]:
        """Get PLY property names in order."""
        return [
            "x", "y", "z",           # position
            "nx", "ny", "nz",        # normal
            "tx", "ty", "tz",        # tangent
            "bx", "by", "bz",        # bitangent
            "opacity",                # opacity
            "sx", "sy", "sz",       # scale
            "rx", "ry", "rz", "rw",  # rotation
            "red", "green", "blue",  # color
        ]
    
    def write_header(self, total_expected: Optional[int] = None) -> None:
        """
        Write PLY header.
        
        Args:
            total_expected: Expected total vertices (for header element count).
                           If None, writes 0 with fixed-width padding (10 digits).
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        format_str = "binary_little_endian" if self.binary else "ascii"
        
        # Use fixed-width padding to keep header size constant
        # 10 digits supports up to 9,999,999,999 vertices
        if total_expected is not None:
            vertex_count_str = f"{total_expected:010d}"
        else:
            vertex_count_str = "0000000000"  # Placeholder, will be updated
        
        header_lines = [
            "ply",
            f"format {format_str} 1.0",
            f"element vertex {vertex_count_str}",
        ]
        
        for prop_name in self._get_property_names():
            header_lines.append(f"property float {prop_name}")
        
        header_lines.append("end_header")
        
        header_str = "\n".join(header_lines) + "\n"
        
        self.file = open(self.filepath, 'wb' if self.binary else 'w')
        
        if self.binary:
            self.file.write(header_str.encode('ascii'))
        else:
            self.file.write(header_str)
        
        self.header_size = len(header_str)
        self.data_start = self.header_size
        
        if self.verbose:
            mode = "binary" if self.binary else "ASCII"
            print(f"  Opened PLY file for incremental writing ({mode})")
    
    def write_chunk(self, surfels: Dict[str, np.ndarray]) -> None:
        """
        Write a chunk of surfels to the file.
        
        Args:
            surfels: Dictionary of surfel attributes for this chunk
        """
        if self.file is None:
            raise RuntimeError("Must call write_header() before writing chunks")
        
        if self.is_finalized:
            raise RuntimeError("Cannot write to finalized file")
        
        n_surfels = len(surfels["position"])
        self.vertex_count += n_surfels
        
        if self.binary:
            # Write binary chunk
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
                
                self.file.write(struct.pack(f'<{len(data)}f', *data))
        else:
            # Write ASCII chunk
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
                
                formatted = [f"{v:.10f}" if isinstance(v, float) else str(v) for v in values]
                self.file.write(" ".join(formatted) + "\n")
    
    def finalize(self) -> None:
        """
        Finalize the PLY file.
        
        Updates the vertex count in the header if it was unknown (0).
        Uses fixed-width format to keep header size constant.
        """
        if self.file is None:
            raise RuntimeError("No file open")
        
        if self.is_finalized:
            return
        
        self.file.close()
        self.is_finalized = True
        
        # Use fixed-width padding to keep header size constant
        old_line = b"element vertex 0000000000"
        new_line = f"element vertex {self.vertex_count:010d}".encode('ascii')
        
        if self.binary:
            # Read the current file
            with open(self.filepath, 'rb') as f:
                content = f.read()
            
            if old_line in content:
                content = content.replace(old_line, new_line, 1)  # Only first occurrence
                
                # Write back
                with open(self.filepath, 'wb') as f:
                    f.write(content)
        else:
            # ASCII mode: simpler string replacement
            with open(self.filepath, 'r') as f:
                content = f.read()
            
            old_line_ascii = "element vertex 0000000000"
            new_line_ascii = f"element vertex {self.vertex_count:010d}"
            
            if old_line_ascii in content:
                content = content.replace(old_line_ascii, new_line_ascii, 1)
                
                with open(self.filepath, 'w') as f:
                    f.write(content)
        
        if self.verbose:
            print(f"  Finalized: {self.vertex_count:,} total surfels")
    
    def __enter__(self) -> 'IncrementalPlyWriter':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures file is finalized."""
        if exc_type is None:
            self.finalize()
        else:
            # Error occurred - try to close file
            if self.file is not None:
                self.file.close()


def write_ply_incremental(filepath: str, surfel_chunks: List[Dict[str, np.ndarray]],
                          binary: bool = True, verbose: bool = True) -> int:
    """
    Write multiple surfel chunks to a PLY file incrementally.
    
    Args:
        filepath: Output PLY file path
        surfel_chunks: List of surfel dictionaries (each chunk)
        binary: Write binary PLY
        verbose: Print progress
        
    Returns:
        Total number of surfels written
    """
    total_surfels = sum(len(chunk["position"]) for chunk in surfel_chunks)
    
    with IncrementalPlyWriter(filepath, binary=binary, verbose=verbose) as writer:
        writer.write_header(total_expected=total_surfels)
        for i, chunk in enumerate(surfel_chunks):
            if verbose and (i + 1) % 10 == 0:
                print(f"    Writing chunk {i + 1}/{len(surfel_chunks)}...")
            writer.write_chunk(chunk)
    
    return total_surfels


def write_ply_streaming(filepath: str, chunk_iterator,
                        binary: bool = True, verbose: bool = True) -> int:
    """
    Write surfels from a streaming iterator to a PLY file.
    
    This function supports billion+ point clouds by:
    1. Not accumulating chunks in memory
    2. Writing each chunk as it's generated
    3. Updating header at the end
    
    Args:
        filepath: Output PLY file path
        chunk_iterator: Iterator yielding surfel dictionaries
        binary: Write binary PLY
        verbose: Print progress
        
    Returns:
        Total number of surfels written
    
    Example:
        def generate_chunks():
            for chunk in large_dataset:
                yield process_chunk(chunk)
        
        total = write_ply_streaming("output.ply", generate_chunks())
    """
    with IncrementalPlyWriter(filepath, binary=binary, verbose=verbose) as writer:
        writer.write_header()
        
        chunk_count = 0
        for chunk in chunk_iterator:
            writer.write_chunk(chunk)
            chunk_count += 1
            
            if verbose and chunk_count % 10 == 0:
                print(f"    Processed {writer.vertex_count:,} surfels...")
    
    return writer.vertex_count
