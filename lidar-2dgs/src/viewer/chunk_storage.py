"""
Chunk Storage for Streaming Octree Viewer

Handles efficient storage and retrieval of surfel chunks for streaming.
Implements binary format for fast I/O.
"""

import struct
import os
from pathlib import Path
from typing import Dict, Optional, BinaryIO, Tuple, Any
import numpy as np


# Chunk file header format (little-endian)
# magic: 4 bytes ("2DGS")
# version: 4 bytes (uint32)
# num_surfels: 4 bytes (uint32)
# has_colors: 1 byte (bool)
# has_opacity: 1 byte (bool)
# reserved: 2 bytes
CHUNK_HEADER_FORMAT = struct.Struct('<4s I I B B 2x')
CHUNK_HEADER_SIZE = CHUNK_HEADER_FORMAT.size

# Surfel data formats
# position: 3 * float32 = 12 bytes
# normal: 3 * float32 = 12 bytes
# scale: 3 * float32 = 12 bytes
# rotation: 4 * float32 = 16 bytes
# color: 3 * uint8 = 3 bytes (padded to 4 for alignment)
# opacity: 1 * float32 = 4 bytes (padded)
SURFEL_SIZE = 12 + 12 + 12 + 16 + 4 + 4  # 60 bytes per surfel
SURFEL_SIZE_NO_COLOR = 12 + 12 + 12 + 16 + 4  # 56 bytes per surfel (no color)
SURFEL_SIZE_NO_OPACITY = 12 + 12 + 12 + 16 + 3  # 55 bytes per surfel (no color/opacity)


class ChunkStorage:
    """
    Storage manager for surfel chunks.
    
    Provides efficient binary I/O for streaming octree nodes.
    """
    
    CHUNK_MAGIC = b'2DGS'
    CHUNK_VERSION = 1
    
    def __init__(self, base_path: str, mode: str = 'r'):
        """
        Initialize chunk storage.
        
        Args:
            base_path: Base directory for chunk files
            mode: 'r' for read-only, 'w' for write
        """
        self.base_path = Path(base_path)
        self.mode = mode
        self._write_handles: Dict[str, BinaryIO] = {}
        
        if mode == 'w':
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_chunk_path(self, node_id: str, lod: int = 0) -> Path:
        """Get path for a chunk file."""
        return self.base_path / f"chunk_{node_id}_lod{lod}.bin"
    
    def write_chunk(self, node_id: str, surfels: Dict[str, np.ndarray],
                   lod: int = 0, chunk_path: Optional[Path] = None) -> Tuple[str, int, int]:
        """
        Write a chunk to storage.
        
        Args:
            node_id: Node identifier
            surfels: Dictionary of surfel arrays
            lod: LOD level
            chunk_path: Optional explicit path
            
        Returns:
            Tuple of (chunk_file, chunk_offset, chunk_size)
        """
        if chunk_path is None:
            chunk_path = self.get_chunk_path(node_id, lod)
        
        # Ensure directory exists
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_surfels = len(surfels['position'])
        has_colors = 'color' in surfels and surfels['color'] is not None
        has_opacity = 'opacity' in surfels and surfels['opacity'] is not None
        
        # Calculate data size
        if has_colors and has_opacity:
            data_size = num_surfels * SURFEL_SIZE
        elif has_colors:
            data_size = num_surfels * (SURFEL_SIZE - 1)  # No opacity
        else:
            data_size = num_surfels * (SURFEL_SIZE - 7)  # No color/opacity
        
        # Write header and data
        with open(chunk_path, 'wb') as f:
            # Write header
            header = CHUNK_HEADER_FORMAT.pack(
                self.CHUNK_MAGIC,
                self.CHUNK_VERSION,
                num_surfels,
                1 if has_colors else 0,
                1 if has_opacity else 0
            )
            f.write(header)
            
            # Write surfel data - vectorized for performance
            # Pre-allocate combined array for all float32 data
            num_floats = num_surfels * 13  # 3 pos + 3 norm + 3 scale + 4 rot = 13 floats
            float_data = np.empty(num_floats, dtype=np.float32)
            
            # Fill float data - use concatenate to handle sliced arrays
            # Note: sliced arrays (e.g., surfels['position'][indices]) are still 2D
            idx = 0
            float_data[idx:idx + num_surfels * 3] = np.ascontiguousarray(surfels['position']).ravel()
            idx += num_surfels * 3
            float_data[idx:idx + num_surfels * 3] = np.ascontiguousarray(surfels['normal']).ravel()
            idx += num_surfels * 3
            float_data[idx:idx + num_surfels * 3] = np.ascontiguousarray(surfels['scale']).ravel()
            idx += num_surfels * 3
            float_data[idx:idx + num_surfels * 4] = np.ascontiguousarray(surfels['rotation']).ravel()
            
            # Write float data as contiguous bytes
            f.write(float_data.tobytes())
            
            # Write color data if present
            if has_colors:
                # Convert colors to uint8 array
                color_data = np.clip(surfels['color'], 0, 255).astype(np.uint8)
                # Add alpha padding column
                if color_data.shape[1] == 3:
                    color_data = np.column_stack([color_data, np.full(num_surfels, 255, dtype=np.uint8)])
                f.write(color_data.tobytes())
            
            # Write opacity data if present
            if has_opacity:
                opacity_data = np.ascontiguousarray(surfels['opacity']).astype(np.float32)
                f.write(opacity_data.tobytes())
        
        return str(chunk_path), CHUNK_HEADER_SIZE, data_size
    
    def read_chunk(self, chunk_path: str, lod: int = 0) -> Dict[str, np.ndarray]:
        """
        Read a chunk from storage.
        
        Args:
            chunk_path: Path to chunk file
            lod: LOD level (for validation)
            
        Returns:
            Dictionary of surfel arrays
        """
        with open(chunk_path, 'rb') as f:
            # Read header
            header = f.read(CHUNK_HEADER_SIZE)
            magic, version, num_surfels, has_colors, has_opacity = \
                CHUNK_HEADER_FORMAT.unpack(header)
            
            if magic != self.CHUNK_MAGIC:
                raise ValueError(f"Invalid chunk magic: {magic}")
            if version != self.CHUNK_VERSION:
                raise ValueError(f"Unsupported chunk version: {version}")
            
            # Pre-allocate arrays
            surfels = {
                'position': np.zeros((num_surfels, 3), dtype=np.float32),
                'normal': np.zeros((num_surfels, 3), dtype=np.float32),
                'scale': np.zeros((num_surfels, 3), dtype=np.float32),
                'rotation': np.zeros((num_surfels, 4), dtype=np.float32),
            }
            
            if has_colors:
                surfels['color'] = np.zeros((num_surfels, 3), dtype=np.uint8)
            if has_opacity:
                surfels['opacity'] = np.zeros(num_surfels, dtype=np.float32)
            
            # Read surfel data - vectorized for performance
            # Read all float data at once (position + normal + scale + rotation = 13 floats per surfel)
            float_bytes = f.read(num_surfels * 13 * 4)  # 4 bytes per float32
            float_data = np.frombuffer(float_bytes, dtype=np.float32)
            
            # Reshape and assign
            idx = 0
            surfels['position'] = float_data[idx:idx + num_surfels * 3].reshape(num_surfels, 3)
            idx += num_surfels * 3
            surfels['normal'] = float_data[idx:idx + num_surfels * 3].reshape(num_surfels, 3)
            idx += num_surfels * 3
            surfels['scale'] = float_data[idx:idx + num_surfels * 3].reshape(num_surfels, 3)
            idx += num_surfels * 3
            surfels['rotation'] = float_data[idx:idx + num_surfels * 4].reshape(num_surfels, 4)
            
            # Read color data if present
            if has_colors:
                color_bytes = f.read(num_surfels * 4)  # RGB + padding
                surfels['color'] = np.frombuffer(color_bytes, dtype=np.uint8).reshape(num_surfels, 4)[:, :3]
            
            # Read opacity data if present
            if has_opacity:
                opacity_bytes = f.read(num_surfels * 4)
                surfels['opacity'] = np.frombuffer(opacity_bytes, dtype=np.float32)
        
        return surfels
    
    def chunk_exists(self, node_id: str, lod: int = 0) -> bool:
        """Check if a chunk exists."""
        return self.get_chunk_path(node_id, lod).exists()
    
    def delete_chunk(self, node_id: str, lod: int = 0) -> bool:
        """Delete a chunk from storage."""
        path = self.get_chunk_path(node_id, lod)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def get_chunk_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get information about a chunk file without loading full data.
        
        Args:
            filepath: Path to chunk file
            
        Returns:
            Dictionary with chunk metadata
        """
        return get_chunk_info(filepath)
    
    def close(self):
        """Close all write handles."""
        for handle in self._write_handles.values():
            if handle and not handle.closed:
                handle.close()
        self._write_handles.clear()
    
    def __del__(self):
        """Destructor - ensure file handles are closed."""
        if hasattr(self, '_write_handles'):
            self.close()


def load_chunk_from_file(filepath: str, lod: int = 0) -> Dict[str, np.ndarray]:
    """
    Convenience function to load a chunk file.
    
    Args:
        filepath: Path to chunk file
        lod: LOD level (for validation)
        
    Returns:
        Dictionary of surfel arrays
    """
    storage = ChunkStorage(os.path.dirname(filepath), mode='r')
    return storage.read_chunk(filepath, lod)


def save_surfels_to_chunks(surfels: Dict[str, np.ndarray],
                          output_dir: str,
                          node_id: str = "root",
                          chunk_size: int = 10000,
                          lod_levels: int = 1) -> Dict[int, str]:
    """
    Save surfels to multiple chunks for LOD.
    
    Args:
        surfels: Dictionary of surfel arrays
        output_dir: Output directory
        node_id: Base node ID
        chunk_size: Surfels per chunk
        lod_levels: Number of LOD levels to generate
        
    Returns:
        Dictionary of LOD level -> chunk path
    """
    storage = ChunkStorage(output_dir, mode='w')
    
    num_surfels = len(surfels['position'])
    chunk_paths = {}
    
    # Full resolution (LOD 0)
    if num_surfels <= chunk_size:
        chunk_path, offset, size = storage.write_chunk(
            f"{node_id}_lod0", surfels, lod=0
        )
        chunk_paths[0] = chunk_path
    else:
        # Split into multiple chunks
        for i in range(0, num_surfels, chunk_size):
            chunk_surfels = {
                'position': surfels['position'][i:i+chunk_size],
                'normal': surfels['normal'][i:i+chunk_size],
                'scale': surfels['scale'][i:i+chunk_size],
                'rotation': surfels['rotation'][i:i+chunk_size],
            }
            if 'color' in surfels:
                chunk_surfels['color'] = surfels['color'][i:i+chunk_size]
            if 'opacity' in surfels:
                chunk_surfels['opacity'] = surfels['opacity'][i:i+chunk_size]
            
            chunk_id = f"{node_id}_chunk{i//chunk_size}_lod0"
            chunk_path, offset, size = storage.write_chunk(chunk_id, chunk_surfels, lod=0)
            chunk_paths[0] = chunk_path
    
    storage.close()
    return chunk_paths


def get_chunk_info(filepath: str) -> Dict[str, any]:
    """
    Get information about a chunk file without loading full data.
    
    Args:
        filepath: Path to chunk file
        
    Returns:
        Dictionary with chunk metadata
    """
    with open(filepath, 'rb') as f:
        header = f.read(CHUNK_HEADER_SIZE)
        magic, version, num_surfels, has_colors, has_opacity = \
            CHUNK_HEADER_FORMAT.unpack(header)
        
        file_size = os.path.getsize(filepath)
        data_size = file_size - CHUNK_HEADER_SIZE
        
        if has_colors and has_opacity:
            bytes_per_surfel = SURFEL_SIZE
        elif has_colors:
            bytes_per_surfel = SURFEL_SIZE - 1
        else:
            bytes_per_surfel = SURFEL_SIZE - 7
        
        return {
            'magic': magic,
            'version': version,
            'num_surfels': num_surfels,
            'has_colors': bool(has_colors),
            'has_opacity': bool(has_opacity),
            'file_size': file_size,
            'data_size': data_size,
            'bytes_per_surfel': bytes_per_surfel,
            'estimated_memory': num_surfels * 64  # ~64 bytes per surfel in RAM
        }
