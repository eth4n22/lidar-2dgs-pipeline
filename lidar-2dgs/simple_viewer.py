#!/usr/bin/env python3
"""
Simple PLY Viewer - Loads and displays PLY files directly.

For large files (>500MB), automatically uses streaming mode if .2dgs_octree directory exists.

Usage:
    python simple_viewer.py <input.ply>
    python simple_viewer.py <input.2dgs_octree/>
"""
import sys
import time
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, '.')

from src.viewer.opengl_renderer import GaussianSplatRenderer
from src.export_ply import read_ply


def get_file_size_mb(filepath):
    """Get file size in MB."""
    return filepath.stat().st_size / (1024 * 1024)


def run_viewer(ply_path: str, width: int = 1280, height: int = 720) -> None:
    """Run simple PLY viewer with automatic format detection."""
    input_path = Path(ply_path)
    
    # Check if input is a .2dgs_octree directory (streaming format)
    if input_path.is_dir() and input_path.suffix == '.2dgs_octree':
        print(f"Detected .2dgs_octree directory - using streaming viewer...")
        # Import streaming viewer
        from src.viewer.streaming_viewer import StreamingOctreeViewer
        from src.viewer.octree_types import Frustum
        
        viewer = StreamingOctreeViewer(
            str(input_path),
            cache_size=100,
            max_memory_mb=2000.0,
            preload_metadata=True
        )
        
        renderer = GaussianSplatRenderer(width=width, height=height, 
                                         title=f"Streaming Viewer: {input_path.name}")
        
        # Get bounding box
        bbox = viewer.get_bounding_box()
        if bbox:
            center = np.array([
                (bbox.min_x + bbox.max_x) / 2,
                (bbox.min_y + bbox.max_y) / 2,
                (bbox.min_z + bbox.max_z) / 2
            ], dtype=np.float32)
            size = np.array([
                bbox.max_x - bbox.min_x,
                bbox.max_y - bbox.min_y,
                bbox.max_z - bbox.min_z
            ])
            diagonal = np.sqrt(np.sum(size ** 2))
            camera_dist = max(diagonal * 1.5, 2.0)
            camera_pos = np.array([
                center[0],
                center[1] + camera_dist * 0.3,
                center[2] + camera_dist
            ], dtype=np.float32)
            renderer.camera_position = camera_pos
            renderer.camera_target = center
        
        print("\nControls: W/S/A/D/Q/E, SPACE=mode, ESC=exit")
        
        while not renderer.should_close():
            renderer.poll_events()
            renderer.handle_input()
            
            # Get camera matrices
            view = renderer._get_view_matrix()
            proj = renderer._get_projection_matrix()
            frustum = Frustum.from_projection_matrix(proj, view)
            
            # Update visible chunks
            viewer.update_visible_chunks(renderer.camera_position, frustum, max_depth=6)
            
            # Load visible chunks
            if viewer._visible_nodes:
                viewer.load_chunks_async(list(viewer._visible_nodes))
            else:
                # Load initial chunks if none visible
                all_chunks = viewer.get_all_chunk_nodes()[:20]
                viewer._visible_nodes.update(all_chunks)
                viewer.load_chunks_async(all_chunks)
            
            surfels = viewer.get_visible_surfels()
            if surfels and len(surfels.get('position', [])) > 0:
                renderer.upload_surfels(surfels)
            
            renderer.render()
            time.sleep(0.001)
        
        viewer.close()
        renderer.close()
        print("Viewer closed.")
        return
    
    # Regular PLY file loading
    print(f"Loading {ply_path}...")
    file_size_mb = get_file_size_mb(input_path)
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Check if there's a corresponding .2dgs_octree directory
    octree_dir = input_path.with_suffix('.2dgs_octree')
    if octree_dir.exists():
        print(f"Found .2dgs_octree directory! Using streaming mode for better performance...")
        # Re-call this function with the octree directory
        run_viewer(str(octree_dir), width, height)
        return
    
    # Load PLY file directly
    if file_size_mb > 500:
        print("WARNING: Large file! This may take a while or run out of memory.")
        print("Consider converting to .2dgs_octree format first for better performance.")
    
    surfels = read_ply(ply_path)
    num_surfels = len(surfels['position'])
    print(f"Loaded {num_surfels} surfels")
    
    # Get bounding box
    positions = surfels['position']
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    
    print(f"Bounding box: min={bbox_min}, max={bbox_max}")
    print(f"Center: {bbox_center}")
    
    # Calculate camera position (above and back from center)
    bbox_size = bbox_max - bbox_min
    bbox_diagonal = np.sqrt(np.sum(bbox_size ** 2))
    camera_distance = bbox_diagonal * 1.5
    
    camera_pos = np.array([
        bbox_center[0],
        bbox_center[1] + camera_distance * 0.3,  # angled view
        bbox_center[2] + camera_distance
    ], dtype=np.float32)
    
    print(f"Initial camera pos: {camera_pos}")
    print(f"Looking at: {bbox_center}")
    
    # Initialize renderer
    renderer = GaussianSplatRenderer(width=width, height=height, title=f"PLY Viewer: {ply_path}")
    
    # Upload surfels to GPU
    print(f"Uploading {num_surfels} surfels to GPU...")
    renderer.upload_surfels(surfels)
    
    # Set camera
    renderer.camera_position = camera_pos
    renderer.camera_target = np.array(bbox_center, dtype=np.float32)
    
    # Main loop
    print("\nControls:")
    print("  W/S: Move forward/backward")
    print("  A/D: Move left/right")
    print("  Q/E: Move down/up")
    print("  SPACE: Toggle point/splat mode")
    print("  +/-: Adjust point size")
    print("  ESC: Exit")
    print("-" * 50)
    
    last_time = time.time()
    frame_count = 0
    
    while not renderer.should_close():
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        
        # Handle input
        renderer.poll_events()
        renderer.handle_input()
        
        # Render
        renderer.render()
        
        # Stats every second
        frame_count += 1
        if frame_count % 60 == 0:
            stats = renderer.stats
            fps = stats.get('fps', 0.0)
            print(f"FPS: {fps:.1f} | Surfels: {num_surfels}")
            frame_count = 0
    
    renderer.close()
    print("Viewer closed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_viewer.py <input.ply>")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    run_viewer(ply_path)
