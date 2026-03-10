"""
Streaming Viewer Package

Potree-style streaming viewer for 2DGS surfels with:
- Out-of-core chunk loading (not all into RAM)
- LOD-based rendering
- Frustum culling
- Progressive loading based on camera position
"""

from .streaming_viewer import StreamingOctreeViewer
from .chunk_storage import ChunkStorage, load_chunk_from_file, save_surfels_to_chunks
from .octree_types import OctreeNode, OctreeMetadata, BoundingBox

__all__ = [
    'StreamingOctreeViewer',
    'ChunkStorage',
    'load_chunk_from_file',
    'save_surfels_to_chunks',
    'OctreeNode',
    'OctreeMetadata',
    'BoundingBox'
]
