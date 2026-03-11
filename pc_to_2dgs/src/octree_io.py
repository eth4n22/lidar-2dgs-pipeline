"""
2DGS Octree Chunk Reader

Load and stream .2dgs_octree chunks for large point cloud viewing.
This enables viewing files larger than available RAM.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import struct


class OctreeMetadata:
    """Metadata for the octree."""
    def __init__(self):
        self.bounding_box = None
        self.num_surfels = 0
        self.chunk_size = 0
        
    @classmethod
    def load(cls, path: str) -> 'OctreeMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        meta = cls()
        meta.bounding_box = data.get('bounding_box')
        meta.num_surfels = data.get('num_surfels', 0)
        meta.chunk_size = data.get('chunk_size', 0)
        return meta


class OctreeNode:
    """Represents a node in the octree."""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.chunk_file: Optional[str] = None
        self.num_surfels: int = 0
        self.bounds = None  # (min_x, min_y, min_z, max_x, max_y, max_z)
        self.children: List['OctreeNode'] = []
        
    @classmethod
    def from_dict(cls, data: dict) -> 'OctreeNode':
        """Create node from dictionary."""
        node = cls(data.get('node_id', ''))
        node.chunk_file = data.get('chunk_file')
        node.num_surfels = data.get('num_surfels', 0)
        
        bounds = data.get('bounds', {})
        if bounds:
            node.bounds = (
                bounds.get('min_x', 0), bounds.get('min_y', 0), bounds.get('min_z', 0),
                bounds.get('max_x', 0), bounds.get('max_y', 0), bounds.get('max_z', 0)
            )
        
        children = data.get('children', [])
        node.children = [cls.from_dict(c) for c in children]
        return node


def load_chunk(chunk_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a single chunk from a .bin file.
    
    Returns surfel dictionary or None if failed.
    """
    if not os.path.exists(chunk_path):
        return None
    
    try:
        # Read binary chunk file
        with open(chunk_path, 'rb') as f:
            data = f.read()
        
        if len(data) < 4:
            return None
        
        # First 4 bytes: number of surfels
        num_surfels = struct.unpack('<I', data[:4])[0]
        
        if num_surfels == 0:
            return None
        
        # Each surfel is 23 floats (92 bytes)
        # Format: x, y, z, nx, ny, nz, tx, ty, tz, bx, by, bz, opacity, sx, sy, sz, rx, ry, rz, rw, r, g, b
        FLOATS_PER_SURFEL = 23
        EXPECTED_SIZE = 4 + num_surfels * FLOATS_PER_SURFEL * 4
        
        # Handle truncated or inconsistent chunk files gracefully
        actual_data_size = len(data) - 4
        actual_floats = actual_data_size // 4
        
        if actual_floats < num_surfels * FLOATS_PER_SURFEL:
            print(f"Warning: Chunk file {chunk_path} is truncated or inconsistent")
            # Calculate how many complete surfels we can read
            num_surfels = actual_floats // FLOATS_PER_SURFEL
            if num_surfels == 0:
                return None
        
        # Unpack only the complete surfels
        floats_data_size = num_surfels * FLOATS_PER_SURFEL * 4
        floats = struct.unpack(f'<{num_surfels * FLOATS_PER_SURFEL}f', data[4:4 + floats_data_size])
        floats = np.array(floats, dtype=np.float32).reshape(num_surfels, FLOATS_PER_SURFEL)
        
        # Extract components
        surfels = {
            'position': floats[:, 0:3],
            'normal': floats[:, 3:6],
            'tangent': floats[:, 6:9],
            'bitangent': floats[:, 9:12],
            'opacity': floats[:, 12],
            'scale': floats[:, 13:16],
            'rotation': floats[:, 16:20],  # quaternion
            'color': floats[:, 20:23],
        }
        
        return surfels
        
    except Exception as e:
        print(f"Error loading chunk {chunk_path}: {e}")
        return None


def load_chunk_lightweight(chunk_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load only position and color from a chunk file (memory efficient for viewing).
    
    Only loads 6 floats per surfel (position + color) instead of 23.
    This reduces memory usage by ~75%.
    """
    if not os.path.exists(chunk_path):
        return None
    
    try:
        with open(chunk_path, 'rb') as f:
            data = f.read()
        
        if len(data) < 4:
            return None
        
        num_surfels = struct.unpack('<I', data[:4])[0]
        if num_surfels == 0:
            return None
        
        FLOATS_PER_SURFEL = 23
        BYTES_PER_SURFEL = FLOATS_PER_SURFEL * 4
        
        # Handle truncated/inconsistent chunks - calculate actual surfels
        actual_data_size = len(data) - 4
        actual_floats = actual_data_size // 4
        actual_surfels = actual_floats // FLOATS_PER_SURFEL
        
        if actual_surfels == 0:
            return None
        
        if actual_surfels < num_surfels:
            print(f"Warning: Chunk {chunk_path} truncated, {actual_surfels} vs {num_surfels} surfels")
            num_surfels = actual_surfels
        
        # Read all surfel data and extract position and color
        # This correctly handles the non-contiguous layout of surfel data:
        # Each surfel is 23 floats (92 bytes): position(3), normal(3), tangent(3), 
        # bitangent(3), opacity(1), scale(3), rotation(4), color(3)
        all_data = np.frombuffer(data[4:4 + num_surfels * BYTES_PER_SURFEL], 
                                 dtype=np.float32).reshape(num_surfels, FLOATS_PER_SURFEL)
        
        positions = all_data[:, 0:3]  # Position: floats 0,1,2
        colors = all_data[:, 20:23]   # Color: floats 20,21,22
        
        return {
            'position': positions,
            'color': colors,
        }
        
    except Exception as e:
        print(f"Error loading chunk {chunk_path}: {e}")
        return None


class OctreeViewer:
    """
    Streaming viewer for .2dgs_octree format.
    
    Loads chunks on-demand based on camera position.
    """
    
    def __init__(self, octree_dir: str, cache_size: int = 50):
        """
        Initialize octree viewer.
        
        Args:
            octree_dir: Path to .2dgs_octree directory
            cache_size: Number of chunks to keep in memory
        """
        self.octree_dir = Path(octree_dir)
        self.cache_size = cache_size
        self.metadata: Optional[OctreeMetadata] = None
        self.root_node: Optional[OctreeNode] = None
        self._chunk_cache: Dict[str, Dict] = {}
        
        # Load metadata and hierarchy
        self._load_octree()
    
    def _load_octree(self):
        """Load octree metadata and hierarchy."""
        metadata_path = self.octree_dir / "metadata.json"
        hierarchy_path = self.octree_dir / "hierarchy.json"
        
        # Load metadata if exists
        if metadata_path.exists():
            self.metadata = OctreeMetadata.load(str(metadata_path))
            print(f"Loaded metadata: {self.metadata.num_surfels} surfels")
        else:
            # Infer from chunk files
            self.metadata = OctreeMetadata()
            self.metadata.num_surfels = 0
            chunk_files = list(Path(self.octree_dir).glob("chunk_*.bin"))
            print(f"No metadata.json - inferring from {len(chunk_files)} chunk files")
        
        # Load hierarchy if exists, otherwise create from chunks
        if hierarchy_path.exists():
            with open(hierarchy_path, 'r') as f:
                data = json.load(f)
            self.root_node = OctreeNode.from_dict(data)
            print(f"Loaded octree hierarchy")
        else:
            # Create minimal hierarchy from chunk files
            self.root_node = OctreeNode("root")
            chunk_files = list(Path(self.octree_dir).glob("chunk_*.bin"))
            
            # Estimate total points from file sizes
            total_points = 0
            for chunk_file in chunk_files:
                size = chunk_file.stat().st_size
                # Subtract 4 byte header, divide by 23 floats * 4 bytes
                n_points = max(0, (size - 4) // (23 * 4))
                total_points += n_points
                
                # Create node for chunk
                name = chunk_file.stem.replace("chunk_", "").replace("_lod0", "")
                node = OctreeNode(name)
                node.chunk_file = chunk_file.name
                node.num_surfels = n_points
                self.root_node.children.append(node)
            
            if self.metadata:
                self.metadata.num_surfels = total_points
            print(f"Created hierarchy from {len(chunk_files)} chunks, ~{total_points:,} total points")
    
    def get_all_chunk_nodes(self) -> List[str]:
        """Get list of all node IDs that have chunks."""
        nodes = []
        
        # If we have hierarchy, use it
        if self.root_node and self.root_node.children:
            def traverse(node: OctreeNode):
                if node.chunk_file and node.num_surfels > 0:
                    nodes.append(node.node_id)
                for child in node.children:
                    traverse(child)
            traverse(self.root_node)
        else:
            # Fallback: scan chunk files directly
            for chunk_file in self.octree_dir.glob("chunk_*.bin"):
                # Extract node_id from filename like chunk_0_1_2_lod0.bin
                name = chunk_file.stem  # chunk_0_1_2_lod0
                if name.startswith("chunk_"):
                    node_id = name.replace("chunk_", "").replace("_lod0", "")
                    nodes.append(node_id)
        
        return nodes
    
    def load_chunks(self, node_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Load multiple chunks and combine them.
        
        Returns combined surfel dictionary.
        """
        all_surfels = {
            'position': [],
            'normal': [],
            'tangent': [],
            'bitangent': [],
            'opacity': [],
            'scale': [],
            'rotation': [],
            'color': [],
        }
        
        for node_id in node_ids:
            surfels = self.get_chunk(node_id)
            if surfels is not None:
                for key in all_surfels:
                    all_surfels[key].append(surfels[key])
        
        # Combine
        if all_surfels['position']:
            for key in all_surfels:
                all_surfels[key] = np.concatenate(all_surfels[key], axis=0)
        else:
            return None
        
        return all_surfels
    
    def load_chunks_lightweight(self, node_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Load multiple chunks with only position and color (memory efficient).
        
        This uses ~75% less memory than load_chunks() since it only loads
        6 floats per surfel instead of 23.
        
        Returns combined surfel dictionary with only 'position' and 'color' keys.
        """
        all_positions = []
        all_colors = []
        
        for node_id in node_ids:
            chunk_file = self._find_chunk_file(node_id)
            if chunk_file:
                surfels = load_chunk_lightweight(chunk_file)
                if surfels is not None:
                    all_positions.append(surfels['position'])
                    all_colors.append(surfels['color'])
        
        # Combine
        if all_positions:
            positions = np.vstack(all_positions)
            colors = np.vstack(all_colors)
            return {
                'position': positions,
                'color': colors,
            }
        else:
            return None
    
    def get_chunk(self, node_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get chunk by node ID, using cache if available.
        """
        # Check cache first
        if node_id in self._chunk_cache:
            return self._chunk_cache[node_id]
        
        # Find chunk file
        chunk_file = self._find_chunk_file(node_id)
        if not chunk_file:
            return None
        
        # Load chunk
        surfels = load_chunk(chunk_file)
        if surfels is not None:
            # Add to cache
            self._chunk_cache[node_id] = surfels
            
            # Evict old chunks if cache is full
            if len(self._chunk_cache) > self.cache_size:
                # Remove oldest (first) item
                oldest = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest]
        
        return surfels
    
    def _find_chunk_file(self, node_id: str) -> Optional[str]:
        """Find chunk file path for a node ID."""
        # First check if file exists directly
        chunk_name = f"chunk_{node_id}_lod0.bin"
        chunk_path = self.octree_dir / chunk_name
        
        if chunk_path.exists():
            return str(chunk_path)
        
        # Try without node_id prefix
        alt_name = f"chunk_{node_id}_lod0.bin"
        alt_path = self.octree_dir / alt_name
        if alt_path.exists():
            return str(alt_path)
        
        # Search all chunk files for matching node
        for chunk_file in self.octree_dir.glob("chunk_*.bin"):
            if f"_{node_id}_" in chunk_file.name or chunk_file.name.endswith(f"_{node_id}_lod0.bin"):
                return str(chunk_file)
        
        return None
    
    def get_bounding_box(self):
        """Get the bounding box of the octree."""
        if self.metadata and self.metadata.bounding_box:
            bb = self.metadata.bounding_box
            return {
                'min': [bb['min_x'], bb['min_y'], bb['min_z']],
                'max': [bb['max_x'], bb['max_y'], bb['max_z']],
            }
        return None


def is_octree_directory(path: str) -> bool:
    """Check if a path is a .2dgs_octree directory."""
    p = Path(path)
    if not p.is_dir():
        return False
    
    # Check for required files
    has_metadata = (p / "metadata.json").exists()
    has_hierarchy = (p / "hierarchy.json").exists()
    
    return has_metadata or has_hierarchy


def convert_ply_to_octree(ply_path: str, output_dir: str = None, chunk_size: int = 100000) -> str:
    """
    Convert a PLY file to .2dgs_octree streaming format.
    
    This enables viewing files larger than available RAM by loading
    only visible chunks on demand.
    
    Uses streaming PLY reader to avoid loading entire file into RAM.
    
    Args:
        ply_path: Path to input PLY file
        output_dir: Output directory (default: <ply_path>.2dgs_octree)
        chunk_size: Points per chunk for spatial distribution (default: 100k)
    
    Returns:
        Path to created .2dgs_octree directory
    """
    import json
    from collections import defaultdict
    from .export_ply import read_ply, stream_ply_chunks
    
    ply_path = Path(ply_path)
    if output_dir is None:
        output_dir = str(ply_path.with_suffix('.2dgs_octree'))
    else:
        output_dir = str(Path(output_dir))
    
    octree_dir = Path(output_dir)
    octree_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {ply_path.name} to .2dgs_octree format...")
    print(f"Output: {octree_dir}")
    
    # Sample to get bounding box
    print("Estimating bounding box (sampling 100k points)...")
    sample = read_ply(str(ply_path), max_memory_gb=1)  # ~100k points with 1GB limit
    
    min_bound = sample['position'].min(axis=0)
    max_bound = sample['position'].max(axis=0)
    
    # Expand slightly to ensure all points are covered
    range_vals = max_bound - min_bound
    range_vals = np.maximum(range_vals, 0.001)  # Avoid division by zero
    
    min_bound = min_bound - range_vals * 0.01
    max_bound = max_bound + range_vals * 0.01
    range_vals = max_bound - min_bound
    
    del sample  # Free memory
    
    grid_size = 8
    chunk_size_world = range_vals / grid_size
    
    print(f"Bounding box: {min_bound} to {max_bound}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size}")
    
    # First get total point count
    with open(ply_path, 'rb') as f:
        header_size = 0
        n_surfels_total = 0
        while True:
            line = f.readline()
            if not line:
                break
            header_size += len(line)
            if line.startswith(b'element vertex'):
                n_surfels_total = int(line.split()[2])
            elif line.strip() == b'end_header':
                header_size += len(line)
                break
    
    print(f"Total points to convert: {n_surfels_total:,}")
    
    # Process in chunks using STREAMING reader - memory efficient!
    print("Reading and distributing points (streaming)...")
    chunks = defaultdict(lambda: {'data': [], 'min': [float('inf')]*3, 'max': [float('-inf')]*3})
    
    # Use streaming reader - processes PLY in chunks without loading entire file
    # Default streaming chunk size is 100k points
    stream_chunk_size = min(chunk_size, 100000)  # Match the function's default
    
    total_processed = 0
    for surfels in stream_ply_chunks(str(ply_path), chunk_size=stream_chunk_size, verbose=False):
        positions = surfels.get('position', [])
        if len(positions) == 0:
            break
        
        n_points = len(positions)
        
        # Compute chunk indices
        chunk_indices = ((positions - min_bound) / chunk_size_world).astype(np.int32)
        chunk_indices = np.clip(chunk_indices, 0, grid_size - 1)
        
        # Build data array in correct order
        data = np.column_stack([
            surfels['position'],
            surfels['normal'],
            surfels['tangent'],
            surfels['bitangent'],
            surfels['opacity'][:, None] if surfels['opacity'].ndim == 1 else surfels['opacity'],
            surfels['scale'],
            surfels['rotation'],
            surfels['color']
        ])
        
        # Group by chunk
        for i in range(n_points):
            idx = tuple(chunk_indices[i])
            chunk_key = f"{idx[0]}_{idx[1]}_{idx[2]}"
            chunks[chunk_key]['data'].append(data[i])
            
            # Update bounds
            pt = positions[i]
            for j in range(3):
                if pt[j] < chunks[chunk_key]['min'][j]:
                    chunks[chunk_key]['min'][j] = pt[j]
                if pt[j] > chunks[chunk_key]['max'][j]:
                    chunks[chunk_key]['max'][j] = pt[j]
        
        total_processed += n_points
        print(f"  Processed {total_processed:,} points...")
        
        del surfels, data, positions, chunk_indices
    
    # Write chunks
    print(f"Writing {len(chunks)} chunks (incremental to avoid memory issues)...")
    
    hierarchy = {
        'node_id': 'root',
        'bounds': {
            'min_x': float(min_bound[0]), 'min_y': float(min_bound[1]), 'min_z': float(min_bound[2]),
            'max_x': float(max_bound[0]), 'max_y': float(max_bound[1]), 'max_z': float(max_bound[2])
        },
        'children': []
    }
    
    total_written = 0
    n_props = 23
    
    # Write each chunk individually to avoid memory issues
    for chunk_key, chunk_info in chunks.items():
        if not chunk_info['data']:
            continue
        
        # Process this chunk in smaller pieces to avoid memory issues
        # Convert list of arrays to a single array piece by piece
        chunk_arrays = chunk_info['data']
        n_total = len(chunk_arrays)  # Each entry is one surfel (23 floats), not the array length
        
        if n_total == 0:
            continue
        
        # Write chunk file directly without concatenating all at once
        chunk_file = f"chunk_{chunk_key}_lod0.bin"
        chunk_path = octree_dir / chunk_file
        
        with open(chunk_path, 'wb') as f:
            # Write number of surfels
            f.write(struct.pack('<I', n_total))
            
            # Write each piece directly to avoid large concatenation
            for arr in chunk_arrays:
                arr_float = np.ascontiguousarray(arr, dtype=np.float32)
                f.write(arr_float.tobytes())
        
        hierarchy['children'].append({
            'node_id': chunk_key,
            'chunk_file': chunk_file,
            'num_surfels': n_total,
            'bounds': {
                'min_x': float(chunk_info['min'][0]), 'min_y': float(chunk_info['min'][1]), 'min_z': float(chunk_info['min'][2]),
                'max_x': float(chunk_info['max'][0]), 'max_y': float(chunk_info['max'][1]), 'max_z': float(chunk_info['max'][2])
            },
            'children': []
        })
        
        total_written += n_total
        # Clear from memory
        chunk_info['data'] = []
    
    print(f"  Wrote {len(chunks)} chunks, {total_written:,} total points")
    
    # Write metadata
    metadata = {
        'num_surfels': total_written,
        'bounding_box': hierarchy['bounds'],
        'chunk_size': chunk_size,
        'format_version': 1
    }
    
    with open(octree_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write hierarchy
    with open(octree_dir / 'hierarchy.json', 'w') as f:
        json.dump(hierarchy, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Total surfels: {total_written:,}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Output: {octree_dir}")
    
    return str(octree_dir)
