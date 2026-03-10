#!/usr/bin/env python3
"""
Main entry point for the 2DGS Streaming Viewer.

Integrates StreamingOctreeViewer with GaussianSplatRenderer for
out-of-core rendering of billion+ point clouds.

Usage:
    python -m tools.streaming_viewer_main <input.ply>
    python -m tools.streaming_viewer_main <input.ply> --cache-size 200
    python -m tools.streaming_viewer_main <input.ply> --point-mode
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path so 'src' module can be found
_parent_dir = Path(__file__).parent.parent.resolve()
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

import numpy as np

from src.viewer.streaming_viewer import StreamingOctreeViewer, Frustum
from src.viewer.opengl_renderer import GaussianSplatRenderer


def run_viewer(ply_path: str, cache_size: int = 100, 
               point_mode: bool = False, width: int = 1280, 
               height: int = 720) -> None:
    """
    Run the streaming viewer.
    
    Args:
        ply_path: Path to input PLY file
        cache_size: Number of chunks to keep in cache
        point_mode: Use point rendering instead of splats
        width: Window width
        height: Window height
    """
    print(f"Loading {ply_path}...")
    print(f"Cache size: {cache_size} chunks")
    print("-" * 50)
    
    # Initialize viewer (creates octree if needed)
    viewer = StreamingOctreeViewer(
        ply_path,
        cache_size=cache_size,
        max_memory_mb=500.0,
        preload_metadata=True
    )
    
    # Debug: Check if root node was loaded
    if viewer.root_node is None:
        print("ERROR: root_node is None! Octree not loaded.")
        print(f"  metadata exists: {viewer.metadata_path.exists()}")
        print(f"  hierarchy exists: {viewer.octree_dir / 'hierarchy.json'}")
        print(f"  metadata: {viewer.metadata}")
        print("\nTrying to build octree from PLY file...")
        # Try building octree from source PLY
        ply_file = str(viewer.ply_file_path)
        if Path(ply_file).exists():
            viewer.load_metadata()  # This should trigger _build_octree
            if viewer.root_node:
                print(f"  Successfully built octree!")
    else:
        print(f"  root_node loaded: {viewer.root_node.node_id}")
        print(f"  root bounds: {viewer.root_node.bounds}")
    
    # Get bounding box
    bbox = viewer.get_bounding_box()
    if bbox:
        center = bbox.center
        print(f"Bounding box: {bbox}")
        print(f"Center: {center}")
    
    # Initialize renderer
    renderer = GaussianSplatRenderer(width=width, height=height)
    
    # Set initial camera position based on bounding box
    if bbox:
        # Calculate bounding box size for proper camera distance
        bbox_size = np.array([
            bbox.max_x - bbox.min_x,
            bbox.max_y - bbox.min_y,
            bbox.max_z - bbox.min_z
        ])
        
        # Calculate bounding sphere radius (half the diagonal)
        bbox_center = np.array([
            (bbox.min_x + bbox.max_x) / 2,
            (bbox.min_y + bbox.max_y) / 2,
            (bbox.min_z + bbox.max_z) / 2
        ])
        bbox_diagonal = np.sqrt(np.sum(bbox_size ** 2))
        sphere_radius = bbox_diagonal / 2.0
        
        # Position camera at 1-2x the bounding sphere radius (closer for small data)
        # This ensures the entire model is visible
        camera_distance = sphere_radius * 1.5
        if camera_distance < 1.0:  # Minimum distance for very small data
            camera_distance = 1.5
        
        # Position camera offset in X-Y plane to get a better viewing angle
        # Position at an angle rather than directly above
        initial_pos = np.array([
            bbox_center[0],
            bbox_center[1] + camera_distance * 0.5,
            bbox_center[2] + camera_distance * 0.8
        ], dtype=np.float32)
        
        # Set camera to look at the center of the data
        renderer.camera_position = initial_pos
        renderer.camera_target = np.array(bbox_center, dtype=np.float32)
        
        print(f"Bounding sphere radius: {sphere_radius:.2f}")
        print(f"Initial camera pos: {initial_pos}")
        print(f"Camera looking at: {bbox_center}")
    
    # Set render mode if point mode requested
    if point_mode:
        renderer.render_mode = 'points'
    
    # Statistics
    last_time = time.time()
    frame_count = 0
    fps = 0.0
    initial_load_complete = False  # Separate flag for chunk loading
    
    print("\nControls:")
    print("  W/S: Move forward/backward")
    print("  A/D: Move left/right")
    print("  Q/E: Move down/up")
    print("  SPACE: Toggle point/splat mode")
    print("  ESC: Exit")
    print("-" * 50)
    
    try:
        while not renderer.should_close():
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Poll events and handle input (updates renderer.camera_position internally)
            renderer.poll_events()
            renderer.handle_input()
            
            # Get camera matrices from renderer (uses renderer.camera_position internally)
            view_matrix = renderer._get_view_matrix()
            proj_matrix = renderer._get_projection_matrix()
            
            # Create frustum for culling
            frustum = Frustum.from_projection_matrix(proj_matrix, view_matrix)
            
            # Update visible chunks based on renderer's camera position
            nodes_to_load = viewer.update_visible_chunks(
                renderer.camera_position, frustum, max_depth=6
            )
            
            # If no visible nodes found (frustum culling too aggressive), load fallback chunks
            if len(viewer._visible_nodes) == 0:
                print("  WARNING: No visible nodes from frustum culling, loading fallback chunks...")
                # Try to load some chunks regardless of visibility
                all_chunks = viewer.get_all_chunk_nodes()
                if all_chunks:
                    # Load first batch of chunks
                    fallback_chunks = all_chunks[:50]
                    viewer._visible_nodes.update(fallback_chunks)
                    nodes_to_load = fallback_chunks
                    print(f"  Loading {len(fallback_chunks)} fallback chunks")
                elif viewer.root_node:
                    # Try root node directly if it has a chunk
                    if hasattr(viewer.root_node, 'chunk_file') and viewer.root_node.chunk_file:
                        viewer._visible_nodes.add(viewer.root_node.node_id)
                        nodes_to_load = [viewer.root_node.node_id]
                        print(f"  Loading root node chunk directly")
                    elif hasattr(viewer.root_node, 'children') and viewer.root_node.children:
                        # Fallback: use root children
                        for child in viewer.root_node.children:
                            viewer._visible_nodes.add(child.node_id)
                        nodes_to_load = [child.node_id for child in viewer.root_node.children]
                        print(f"  Loading {len(nodes_to_load)} root children")
            
            # Initial load on first frame (keep as backup)
            if not initial_load_complete:
                print("  Loading initial chunks...")
                all_chunks = viewer.get_all_chunk_nodes()
                print(f"  Found {len(all_chunks)} nodes with chunks")
                if all_chunks:
                    # Load more chunks initially for large datasets
                    # Use 50 chunks as default, or fewer if not available
                    num_initial = min(50, len(all_chunks))
                    initial_chunks = all_chunks[:num_initial]
                    viewer._visible_nodes.update(initial_chunks)
                    nodes_to_load = initial_chunks
                    print(f"  Loading first {len(nodes_to_load)} chunks")
                    initial_load_complete = True
                else:
                    # Fallback: try root children
                    print("  No chunks found, trying root children...")
                    if viewer.root_node and hasattr(viewer.root_node, 'children') and viewer.root_node.children:
                        for child in viewer.root_node.children:
                            viewer._visible_nodes.add(child.node_id)
                        nodes_to_load = [child.node_id for child in viewer.root_node.children]
                        print(f"  Loading {len(nodes_to_load)} root children")
                        initial_load_complete = True
                    else:
                        # Last resort: load root node directly
                        print("  ERROR: Could not find any chunks to load!")
                        print(f"    root_node: {viewer.root_node}")
                        if viewer.root_node and viewer.root_node.chunk_file:
                            viewer._visible_nodes.add(viewer.root_node.node_id)
                            nodes_to_load = [viewer.root_node.node_id]
                            initial_load_complete = True
            
            # Async load visible chunks
            if nodes_to_load:
                viewer.load_chunks_async(nodes_to_load)
            
            # Get visible surfels for rendering
            surfels = viewer.get_visible_surfels()
            
            # Debug: Print detailed info on first few frames (reduced output)
            if frame_count < 1:
                num_surfels = len(surfels.get('position', [])) if surfels else 0
                if num_surfels > 0:
                    print(f"  Loaded {num_surfels} surfels, {len(viewer._visible_nodes)} visible nodes")
                else:
                    print(f"  WARNING: No surfels loaded, visible nodes: {list(viewer._visible_nodes)[:5]}...")
            
            # Upload surfels to GPU (only if changed)
            if surfels and len(surfels.get('position', [])) > 0:
                renderer.upload_surfels(surfels)
                if frame_count == 0:
                    print(f"  Uploaded {len(surfels['position'])} surfels to GPU")
            elif frame_count == 0:
                print("  WARNING: No surfels to upload!")
            
            # Render frame
            renderer.render()
            
            # Print stats periodically
            frame_count += 1
            if frame_count % 60 == 0:
                viewer_stats = viewer.stats
                renderer_stats = renderer.stats
                fps = renderer_stats.get('fps', 0.0)
                print(f"FPS: {fps:.1f} | Chunks: {viewer_stats['chunks_cached']} | "
                      f"Visible: {viewer_stats['visible_nodes']} | "
                      f"Cache hit: {viewer_stats['cache_hit_rate']:.1f}%")
                frame_count = 0
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        viewer.close()
        renderer.close()
        print("Viewer closed")


def main():
    parser = argparse.ArgumentParser(
        description="2DGS Streaming Viewer - View billion+ point clouds"
    )
    parser.add_argument('input', help='Input PLY file')
    parser.add_argument('--cache-size', type=int, default=100,
                        help='Number of chunks to cache (default: 100)')
    parser.add_argument('--point-mode', action='store_true',
                        help='Use point rendering instead of Gaussian splats')
    parser.add_argument('--width', type=int, default=1280,
                        help='Window width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Window height (default: 720)')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    run_viewer(
        args.input,
        cache_size=args.cache_size,
        point_mode=args.point_mode,
        width=args.width,
        height=args.height
    )


if __name__ == '__main__':
    main()
