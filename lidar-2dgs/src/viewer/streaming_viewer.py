"""
Streaming Octree Viewer - Potree-style viewer for 2DGS surfels

Key features:
- Out-of-core chunk loading (not all into RAM)
- LOD-based progressive rendering
- Frustum culling
- Camera-aware chunk streaming

Advantages over Potree:
- Automatic 100:1 downsampling before conversion (configurable)
- 1 surfel visually represents ~100-1000 points worth of detail
- ~20x memory reduction: 100 points (1200 bytes) → 1 surfel (60 bytes)
- Continuous surfaces instead of discrete points
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict

from .octree_types import OctreeNode, OctreeMetadata, BoundingBox
from .chunk_storage import ChunkStorage, load_chunk_from_file


@dataclass
class Frustum:
    """View frustum for culling."""
    # Defined by 6 planes: (normal, distance)
    planes: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    
    @classmethod
    def from_projection_matrix(cls, proj_matrix: np.ndarray, 
                               view_matrix: np.ndarray) -> 'Frustum':
        """
        Create frustum from projection and view matrices.
        
        Args:
            proj_matrix: 4x4 projection matrix
            view_matrix: 4x4 view matrix
            
        Returns:
            Frustum object
        """
        # Combine matrices
        combined = proj_matrix @ view_matrix
        
        # Extract frustum planes (simplified - extracts 6 planes)
        planes = []
        
        # Left plane
        plane = combined[:, 0] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        # Right plane
        plane = -combined[:, 0] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        # Bottom plane
        plane = combined[:, 1] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        # Top plane
        plane = -combined[:, 1] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        # Near plane
        plane = combined[:, 2] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        # Far plane
        plane = -combined[:, 2] + combined[:, 3]
        norm = np.linalg.norm(plane[:3])
        planes.append((plane[:3] / norm, plane[3] / norm))
        
        return cls(planes=planes)
    
    def intersects_aabb(self, bbox: BoundingBox) -> bool:
        """Check if frustum intersects axis-aligned bounding box."""
        # Get box corners
        corners = np.array([
            [bbox.min_x, bbox.min_y, bbox.min_z],
            [bbox.max_x, bbox.min_y, bbox.min_z],
            [bbox.min_x, bbox.max_y, bbox.min_z],
            [bbox.max_x, bbox.max_y, bbox.min_z],
            [bbox.min_x, bbox.min_y, bbox.max_z],
            [bbox.max_x, bbox.min_y, bbox.max_z],
            [bbox.min_x, bbox.max_y, bbox.max_z],
            [bbox.max_x, bbox.max_y, bbox.max_z],
        ])
        
        for normal, dist in self.planes:
            # Check if all corners are behind the plane
            if np.all(np.dot(corners, normal) + dist < 0):
                return False
        
        return True


class LRUCache:
    """LRU cache with thread safety."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def put(self, key: str, value: Dict) -> None:
        """Add item to cache."""
        with self._lock:
            if key in self._cache:
                # Update and move to end
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new item
                self._cache[key] = value
                
                # Evict oldest if over limit
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
    
    @property
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            return len(self._cache)


class StreamingOctreeViewer:
    """
    Potree-style streaming viewer for 2DGS surfels.
    
    Loads octree nodes on-demand based on camera position and view frustum.
    Does NOT load entire dataset into RAM.
    """
    
    def __init__(self, ply_file_path: str, 
                 cache_size: int = 100,
                 max_memory_mb: float = 500.0,
                 preload_metadata: bool = True):
        """
        Initialize streaming viewer.
        
        Args:
            ply_file_path: Path to PLY file or .2dgs_octree directory
            cache_size: Maximum number of chunks to keep in memory
            max_memory_mb: Maximum memory usage in MB
            preload_metadata: Whether to load octree metadata at init
        """
        self.ply_file_path = Path(ply_file_path)
        self.cache_size = cache_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Determine if input is directory or file
        if self.ply_file_path.is_dir():
            self.octree_dir = self.ply_file_path
            self.metadata_path = self.octree_dir / "metadata.json"
        else:
            self.octree_dir = self.ply_file_path.with_suffix('.2dgs_octree')
            self.metadata_path = self.octree_dir / "metadata.json"
        
        # State
        self.metadata: Optional[OctreeMetadata] = None
        self.root_node: Optional[OctreeNode] = None
        self._chunk_storage: Optional[ChunkStorage] = None
        
        # Cache
        self._chunk_cache = LRUCache(max_size=cache_size)
        self._loading_threads: Set[str] = set()
        self._load_lock = threading.Lock()
        
        # Camera state
        self.camera_position = np.array([0.0, 0.0, 0.0])
        self.camera_orientation = np.eye(3)
        self._visible_nodes: Set[str] = set()
        
        # Statistics
        self._stats = {
            'total_chunks_loaded': 0,
            'total_bytes_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_frame_load_time': 0.0
        }
        
        if preload_metadata:
            self.load_metadata()
    
    @property
    def chunk_storage(self) -> ChunkStorage:
        """Get or create chunk storage."""
        if self._chunk_storage is None:
            self._chunk_storage = ChunkStorage(str(self.octree_dir), mode='r')
        return self._chunk_storage
    
    def get_all_chunk_nodes(self) -> List[str]:
        """
        Get all node IDs that have chunk files.
        
        Returns:
            List of node IDs with chunks
        """
        nodes_with_chunks = []
        
        def traverse(node: OctreeNode) -> None:
            if node.chunk_file is not None and node.num_surfels > 0:
                nodes_with_chunks.append(node.node_id)
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    traverse(child)
        
        if self.root_node is not None:
            traverse(self.root_node)
        
        return nodes_with_chunks
    
    @property
    def chunk_storage(self) -> ChunkStorage:
        """Get or create chunk storage."""
        if self._chunk_storage is None:
            self._chunk_storage = ChunkStorage(str(self.octree_dir), mode='r')
        return self._chunk_storage
    
    def load_metadata(self) -> None:
        """Load octree metadata (small, fast)."""
        if not self.metadata_path.exists():
            # Need to build octree from source file
            self._build_octree()
        else:
            self.metadata = OctreeMetadata.load(str(self.metadata_path))
            
            # Load octree hierarchy
            hierarchy_path = self.octree_dir / "hierarchy.json"
            if hierarchy_path.exists():
                import json
                with open(hierarchy_path, 'r') as f:
                    data = json.load(f)
                self.root_node = OctreeNode.from_dict(data)
    
    def _build_octree(self) -> None:
        """Build octree from source PLY file."""
        from src.export_ply import read_ply
        from src.surfels import build_surfels
        from scipy.spatial import cKDTree
        
        print(f"Building octree from {self.ply_file_path}...")
        start_time = time.time()
        
        # Ensure octree directory exists
        self.octree_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chunk storage in write mode for building
        self._chunk_storage = ChunkStorage(str(self.octree_dir), mode='w')
        
        # Load surfels from PLY
        surfels = read_ply(str(self.ply_file_path))
        
        # Create metadata
        self.metadata = OctreeMetadata.from_surfels(
            surfels,
            max_depth=8,
            chunk_size=10000,
            source_file=str(self.ply_file_path)
        )
        
        # Build octree structure
        self.root_node = OctreeNode(
            node_id="root",
            depth=0,
            bounds=self.metadata.bounding_box
        )
        
        # Distribute surfels to octree nodes
        self._distribute_surfels(self.root_node, surfels)
        
        # Save metadata and hierarchy
        self.metadata.save(str(self.metadata_path))
        
        import json
        hierarchy_path = self.octree_dir / "hierarchy.json"
        with open(hierarchy_path, 'w') as f:
            json.dump(self.root_node.to_dict(), f, indent=2)
        
        # Switch to read mode for viewing
        self._chunk_storage = ChunkStorage(str(self.octree_dir), mode='r')
        
        print(f"Octree built in {time.time() - start_time:.2f}s")
    
    def _distribute_surfels(self, node: OctreeNode, surfels: Dict[str, np.ndarray]) -> None:
        """Distribute surfels to octree nodes."""
        num_surfels = len(surfels['position'])
        
        if num_surfels <= self.metadata.chunk_size or node.depth >= self.metadata.max_depth:
            # This node gets the surfels directly
            self._save_node_chunk(node, surfels)
            return
        
        # Find which child each surfel belongs to
        centers = node.bounds.center
        half = node.bounds.half_size
        
        # Classify surfels by octant
        surfel_indices = {i: [] for i in range(8)}
        
        for i in range(num_surfels):
            pos = surfels['position'][i]
            dx = 1 if pos[0] >= centers[0] else 0
            dy = 1 if pos[1] >= centers[1] else 0
            dz = 1 if pos[2] >= centers[2] else 0
            octant = dx + 2*dy + 4*dz
            surfel_indices[octant].append(i)
        
        # Create children and distribute
        children = node.subdivide(str(self.octree_dir))
        
        for i, child in enumerate(children):
            indices = surfel_indices[i]
            if len(indices) == 0:
                continue
            
            child_surfels = {
                'position': surfels['position'][indices],
                'normal': surfels['normal'][indices],
                'scale': surfels['scale'][indices],
                'rotation': surfels['rotation'][indices],
            }
            if 'color' in surfels:
                child_surfels['color'] = surfels['color'][indices]
            if 'opacity' in surfels:
                child_surfels['opacity'] = surfels['opacity'][indices]
            
            self._distribute_surfels(child, child_surfels)
    
    def _save_node_chunk(self, node: OctreeNode, surfels: Dict[str, np.ndarray]) -> None:
        """Save surfels for a node to chunk file."""
        chunk_path, offset, size = self.chunk_storage.write_chunk(
            node.node_id, surfels, lod=0
        )
        node.chunk_file = chunk_path
        node.chunk_offset = offset
        node.chunk_size = size
        node.num_surfels = len(surfels['position'])
    
    def update_visible_chunks(self, camera_position: np.ndarray,
                              frustum: Frustum,
                              max_depth: Optional[int] = None) -> List[str]:
        """
        Update which chunks are visible based on camera and frustum.
        
        Args:
            camera_position: Camera position in world space
            frustum: View frustum for culling
            max_depth: Maximum depth to traverse (None for auto)
            
        Returns:
            List of node IDs that should be loaded
        """
        if max_depth is None:
            max_depth = self.metadata.max_depth
        
        self.camera_position = camera_position
        self._visible_nodes.clear()
        
        # Find visible nodes
        self._find_visible_nodes(self.root_node, frustum, camera_position, max_depth)
        
        # Request loading of visible chunks
        nodes_to_load = []
        for node_id in self._visible_nodes:
            if self._chunk_cache.get(node_id) is None:
                nodes_to_load.append(node_id)
        
        return nodes_to_load
    
    def _find_visible_nodes(self, node: OctreeNode, frustum: Frustum,
                           camera_position: np.ndarray, max_depth: int) -> None:
        """Recursively find visible nodes."""
        # Check if node is in frustum
        if not frustum.intersects_aabb(node.bounds):
            return
        
        # Calculate distance for LOD selection
        distance = np.linalg.norm(camera_position - node.bounds.center)
        
        # Determine if we should descend or load this node
        node_size = np.linalg.norm(node.bounds.size)
        
        if node_size / distance < 0.01 or node.depth >= max_depth:
            # Load this node
            self._visible_nodes.add(node.node_id)
        else:
            # Check children
            if node.is_leaf:
                self._visible_nodes.add(node.node_id)
            else:
                for child in node.children:
                    self._find_visible_nodes(child, frustum, camera_position, max_depth)
    
    def load_chunk(self, node_id: str, lod: int = 0, 
                   block: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Load a chunk from disk.
        
        Args:
            node_id: Node identifier
            lod: LOD level
            block: Whether to block until loaded
            
        Returns:
            Dictionary of surfel arrays, or None if not available
        """
        # Check cache first
        cached = self._chunk_cache.get(node_id)
        if cached is not None:
            self._stats['cache_hits'] += 1
            return cached
        
        self._stats['cache_misses'] += 1
        
        # Debug: Print chunk ID if cache miss
        if self._stats['cache_misses'] < 5:
            print(f"  CACHE MISS: {node_id}")
        
        # Check if already being loaded
        with self._load_lock:
            if node_id in self._loading_threads:
                if block:
                    # Wait for loading to complete
                    while node_id in self._loading_threads:
                        time.sleep(0.001)
                    return self._chunk_cache.get(node_id)
                else:
                    return None
        
        # Find chunk path
        chunk_path = self.chunk_storage.get_chunk_path(node_id, lod)
        if not chunk_path.exists():
            # Try default LOD
            if lod != 0:
                chunk_path = self.chunk_storage.get_chunk_path(node_id, 0)
                if not chunk_path.exists():
                    return None
            else:
                return None
        
        # Load chunk
        surfels = self.chunk_storage.read_chunk(str(chunk_path), lod)
        
        # Update stats
        self._stats['total_chunks_loaded'] += 1
        self._stats['total_bytes_loaded'] += chunk_path.stat().st_size
        
        # Add to cache
        self._chunk_cache.put(node_id, surfels)
        
        return surfels
    
    def load_chunks_async(self, node_ids: List[str], lod: int = 0) -> None:
        """
        Asynchronously load multiple chunks.
        
        Args:
            node_ids: List of node identifiers
            lod: LOD level
        """
        for node_id in node_ids:
            if self._chunk_cache.get(node_id) is not None:
                continue
            
            with self._load_lock:
                if node_id in self._loading_threads:
                    continue
            
            def _load():
                try:
                    self.load_chunk(node_id, lod, block=True)
                finally:
                    with self._load_lock:
                        self._loading_threads.discard(node_id)
            
            thread = threading.Thread(target=_load, daemon=True)
            thread.start()
    
    def get_visible_surfels(self) -> Dict[str, np.ndarray]:
        """
        Get all visible surfels for rendering.
        
        Returns:
            Dictionary of concatenated surfel arrays
        """
        all_surfels = {}
        
        for node_id in self._visible_nodes:
            surfels = self._chunk_cache.get(node_id)
            if surfels is None:
                # Try to load synchronously (will block)
                surfels = self.load_chunk(node_id, block=True)
            
            if surfels is None:
                continue
            
            # Concatenate surfels
            for key, value in surfels.items():
                if key not in all_surfels:
                    all_surfels[key] = []
                all_surfels[key].append(value)
        
        # Convert lists to arrays
        for key in all_surfels:
            arrays = [np.asarray(a) for a in all_surfels[key]]
            
            # Handle 1D vs 2D arrays
            if arrays[0].ndim == 1:
                # 1D array (opacity) - use concatenate
                all_surfels[key] = np.concatenate(arrays)
            else:
                # 2D array - use vstack
                all_surfels[key] = np.vstack(arrays)
        
        return all_surfels
    
    def unload_distant_chunks(self, camera_position: np.ndarray, 
                              max_distance: float = 1000.0) -> None:
        """
        Unload chunks that are too far from camera.
        
        Args:
            camera_position: Current camera position
            max_distance: Maximum distance to keep chunks
        """
        to_remove = []
        
        for node_id in self._chunk_cache._cache.keys():
            # Find node bounds (need to traverse octree)
            node = self._find_node(node_id)
            if node is None:
                continue
            
            distance = np.linalg.norm(camera_position - node.bounds.center)
            if distance > max_distance:
                to_remove.append(node_id)
        
        for node_id in to_remove:
            self._chunk_cache.remove(node_id)
    
    def _find_node(self, node_id: str) -> Optional[OctreeNode]:
        """Find a node by ID in the octree."""
        if self.root_node is None:
            return None

        if self.root_node.node_id == node_id:
            return self.root_node

        def search(node: OctreeNode) -> Optional[OctreeNode]:
            """Recursively search for node by ID."""
            if node.node_id == node_id:
                return node

            # Only search children if they exist
            if node.children is not None:
                for child in node.children:
                    found = search(child)
                    if found is not None:
                        return found

            return None

        return search(self.root_node)
    
    def clear_cache(self) -> None:
        """Clear the chunk cache."""
        self._chunk_cache.clear()
    
    @property
    def stats(self) -> Dict:
        """Get viewer statistics."""
        cache_size = self._chunk_cache.size
        return {
            'chunks_cached': cache_size,
            'total_chunks_loaded': self._stats['total_chunks_loaded'],
            'total_bytes_loaded': self._stats['total_bytes_loaded'] / (1024 * 1024),
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': (
                self._stats['cache_hits'] / 
                (self._stats['cache_hits'] + self._stats['cache_misses']) * 100
                if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0 else 0
            ),
            'visible_nodes': len(self._visible_nodes)
        }
    
    def get_bounding_box(self) -> Optional[BoundingBox]:
        """Get the overall bounding box."""
        return self.metadata.bounding_box if self.metadata else None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._chunk_storage:
            self._chunk_storage.close()
        self.clear_cache()


# Convenience function
def create_streaming_viewer(ply_file_path: str, **kwargs) -> StreamingOctreeViewer:
    """
    Create a streaming viewer for a PLY file.

    Args:
        ply_file_path: Path to PLY file
        **kwargs: Additional arguments to StreamingOctreeViewer

    Returns:
        Configured StreamingOctreeViewer instance
    """
    return StreamingOctreeViewer(ply_file_path, **kwargs)


def main():
    """Main entrypoint for the streaming viewer."""
    parser = argparse.ArgumentParser(
        description='Streaming Octree Viewer for 2DGS Surfels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m viewer input.ply
  python -m viewer input.ply --point-mode
  python -m viewer input.ply --cache-size 200
  python -m viewer input.ply --cache-size 100 --point-mode
        """
    )

    parser.add_argument('input', type=str,
                        help='Path to input PLY file or .2dgs_octree directory')
    parser.add_argument('--point-mode', action='store_true',
                        help='Render as points instead of Gaussian splats')
    parser.add_argument('--cache-size', type=int, default=100,
                        help='Maximum number of chunks to keep in cache (default: 100)')
    parser.add_argument('--max-memory', type=float, default=500.0,
                        help='Maximum memory usage in MB (default: 500.0)')

    args = parser.parse_args()

    # Verify input file exists
    import os
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create and launch the viewer
    try:
        viewer = StreamingOctreeViewer(
            ply_file_path=args.input,
            cache_size=args.cache_size,
            max_memory_mb=args.max_memory,
            preload_metadata=True
        )

        # Get bounding box info
        bbox = viewer.get_bounding_box()
        if bbox:
            print(f"Loaded: {args.input}")
            print(f"Bounding box: {bbox.min_x:.2f},{bbox.min_y:.2f},{bbox.min_z:.2f} to "
                  f"{bbox.max_x:.2f},{bbox.max_y:.2f},{bbox.max_z:.2f}")
        else:
            print(f"Warning: Could not load metadata for {args.input}")

        print(f"\nViewer initialized successfully!")
        print(f"Use WASD + mouse to navigate.")
        print(f"Press SPACEBAR to toggle point/splat mode.")
        print(f"Press ESC to exit.\n")

        # Import and run the OpenGL renderer
        try:
            import glfw
            from .opengl_renderer import GaussianSplatRenderer
        except ImportError as e:
            print(f"Error: Missing OpenGL/GLFW dependency - {e}", file=sys.stderr)
            print("Install with: pip install PyOpenGL glfw", file=sys.stderr)
            sys.exit(1)

        renderer = GaussianSplatRenderer(
            width=1280,
            height=720,
            title=f"2DGS Viewer: {args.input}"
        )

        # Upload visible surfels to GPU
        visible = viewer.get_visible_surfels()
        if visible and len(visible.get('position', [])) > 0:
            renderer.upload_surfels(visible)
            print(f"Uploaded {len(visible['position'])} surfels to GPU")
        else:
            print("Warning: No surfels to upload!")

        # Main render loop
        render_mode = 'points' if args.point_mode else 'splats'
        renderer._render_mode = render_mode

        print(f"Starting render loop (mode: {render_mode})...")

        while not renderer._use_opengl or not glfw.window_should_close(renderer.window):
            # Update visible chunks based on camera
            view = renderer._get_view_matrix()
            proj = renderer._get_projection_matrix()
            frustum = Frustum.from_projection_matrix(proj, view)

            visible_nodes = viewer.update_visible_chunks(
                renderer.camera_pos,
                frustum,
                max_depth=4
            )

            # Async load visible chunks
            viewer.load_chunks_async(visible_nodes)

            # Get all visible surfels
            surfels = viewer.get_visible_surfels()

            # Upload to GPU if changed
            if surfels and len(surfels.get('position', [])) > 0:
                renderer.upload_surfels(surfels)

            # Render
            renderer.render()

            # Print stats periodically
            stats = viewer.stats
            if renderer._frame_count % 60 == 0:
                print(f"FPS: {renderer._stats['fps']:.1f}, "
                      f"Cache: {stats['chunks_cached']}, "
                      f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")

        viewer.close()
        glfw.terminate()

    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Install required packages with: pip install PyOpenGL glfw", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
