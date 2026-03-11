#!/usr/bin/env python3
"""
Quick 2DGS Viewer - Run with just: python viewer.py

Supports:
- Regular PLY files (small to medium)
- .2dgs_octree directories (large files - streaming mode)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.export_ply import read_ply
from src.txt_io import load_xyzrgb_txt
from src.octree_io import OctreeViewer, is_octree_directory, load_chunk_lightweight

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D not installed.")
    print("Run: pip install open3d")
    sys.exit(1)


class SurfelViewer:
    def __init__(self, input_path: str, original_txt_path: str = None):
        self.input_path = input_path
        self.original_txt_path = original_txt_path
        self.show_axis = True
        self.show_surfels = True
        self.coord = None
        self.surfel_pcd = None
        self.original_pcd = None
        self.current_pcd = None
        self.octree_viewer = None
        self.is_streaming = False
        
        # Check if input is a .2dgs_octree directory
        if is_octree_directory(input_path):
            print(f"Detected .2dgs_octree directory - using streaming mode")
            self._init_streaming_viewer(input_path)
        else:
            # Regular PLY file
            self._init_ply_viewer(input_path, original_txt_path)
    
    def _init_streaming_viewer(self, octree_dir: str):
        """Initialize streaming viewer for .2dgs_octree format with camera-based chunk loading."""
        import numpy as np
        from pathlib import Path
        
        self.is_streaming = True
        
        # Use OctreeViewer for proper streaming
        self.octree_viewer = OctreeViewer(octree_dir, cache_size=100)
        
        # Get all chunk node IDs
        all_nodes = self.octree_viewer.get_all_chunk_nodes()
        total_chunks = len(all_nodes)
        total_surfels = self.octree_viewer.metadata.num_surfels
        
        import sys
        print(f"Octree streaming ready:", file=sys.stderr)
        print(f"  Total chunks: {total_chunks}", file=sys.stderr)
        print(f"  Total surfels: {total_surfels:,}", file=sys.stderr)
        print(f"  Cache size: {self.octree_viewer.cache_size}", file=sys.stderr)
        sys.stderr.flush()
        
        # Track loaded chunks for logging
        self._loaded_chunk_ids = set()
        self._total_visible_surfels = 0
        
        # Initial load - get root-level chunks (top of octree)
        # These are the largest chunks covering the whole scene
        root_chunks = []
        for node in self.octree_viewer.root_node.children:
            if node.chunk_file:
                root_chunks.append(node.node_id)
        
        print(f"  Root chunks: {len(root_chunks)}", file=sys.stderr)
        sys.stderr.flush()
        
        # Load initial set of chunks
        initial_chunks = root_chunks[:min(50, len(root_chunks))]
        if initial_chunks:
            self._load_chunks_streaming(initial_chunks)
        
        # Get bounding box for camera
        bbox = self.octree_viewer.get_bounding_box()
        if bbox:
            center = np.array([
                (bbox['min'][0] + bbox['max'][0]) / 2,
                (bbox['min'][1] + bbox['max'][1]) / 2,
                (bbox['min'][2] + bbox['max'][2]) / 2
            ])
        else:
            center = np.array([0, 0, 0])
        
        self._octree_center = center
        self._all_chunk_nodes = all_nodes
        self._all_chunk_ids = all_nodes  # For status logging
        self._chunk_bounds = {}
        
        # Pre-compute chunk bounds from hierarchy
        self._compute_chunk_bounds()
        
        # Create coordinate frame
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="2DGS Streaming Viewer", width=1280, height=720)
        
        # Add geometries
        self.current_pcd = self.surfel_pcd
        self.vis.add_geometry(self.current_pcd)
        self.vis.add_geometry(self.coord)
        
        # Settings
        render = self.vis.get_render_option()
        render.background_color = [0.1, 0.1, 0.1]
        render.point_size = 2
        
        # Camera - zoom out to see entire scene
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat([0, 0, 0])
        self.vis.get_view_control().set_zoom(0.3)
        
        # Register key callbacks
        self.vis.register_key_callback(ord('A'), self.toggle_axis)
        self.vis.register_key_callback(ord('a'), self.toggle_axis)
        self.vis.register_key_callback(ord('R'), self.reload_chunks)
        self.vis.register_key_callback(ord('r'), self.reload_chunks)
        self.vis.register_key_callback(ord('L'), self.log_chunk_status)
        self.vis.register_key_callback(ord('l'), self.log_chunk_status)
        self.vis.register_key_callback(ord('Q'), self.quit)
        self.vis.register_key_callback(ord('q'), self.quit)
    
    def log_chunk_status(self, vis):
        """Log current chunk streaming status."""
        import sys
        if not self.is_streaming:
            print("Not in streaming mode", file=sys.stderr)
            sys.stderr.flush()
            return
        
        print("\n" + "="*50, file=sys.stderr)
        print("CHUNK STREAMING STATUS", file=sys.stderr)
        print("="*50, file=sys.stderr)
        print(f"Total chunks in octree: {len(self._all_chunk_ids)}", file=sys.stderr)
        print(f"Currently loaded: {len(self._loaded_chunk_ids)}", file=sys.stderr)
        print(f"Total visible surfels: {self._total_visible_surfels:,}", file=sys.stderr)
        print(f"Cache size: {self.octree_viewer.cache_size}", file=sys.stderr)
        
        # Get camera info
        cam = self.vis.get_view_control()
        lookat = cam.get_lookat()
        front = cam.get_front()
        print(f"Camera lookat: {lookat}", file=sys.stderr)
        print(f"Camera front: {front}", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)
        sys.stderr.flush()
    
    def reload_chunks(self, vis):
        """Reload chunks based on current camera position."""
        if not self.is_streaming or not hasattr(self, 'octree_viewer'):
            return
        
        # Update visible chunks based on camera
        self._update_visible_chunks()
    
    def _compute_chunk_bounds(self):
        """Pre-compute bounding boxes for each chunk."""
        import numpy as np
        import sys
        
        # Use hierarchy bounds if available
        if hasattr(self.octree_viewer, 'root_node') and self.octree_viewer.root_node:
            def traverse(node, depth=0):
                if node.chunk_file and node.bounds:
                    self._chunk_bounds[node.node_id] = node.bounds
                for child in node.children:
                    traverse(child, depth+1)
            traverse(self.octree_viewer.root_node)
        
        # If no bounds from hierarchy, estimate from chunk data
        if not self._chunk_bounds:
            print("  Warning: No chunk bounds in hierarchy, loading sample chunks...", file=sys.stderr)
            sample_nodes = self._all_chunk_nodes[:10]
            for node_id in sample_nodes:
                surfels = self.octree_viewer.get_chunk(node_id)
                if surfels is not None and len(surfels.get('position', [])) > 0:
                    pos = surfels['position']
                    self._chunk_bounds[node_id] = (
                        pos[:,0].min(), pos[:,1].min(), pos[:,2].min(),
                        pos[:,0].max(), pos[:,1].max(), pos[:,2].max()
                    )
            sys.stderr.flush()
    
    def _load_chunks_streaming(self, node_ids: list):
        """Load chunks and create point cloud geometry."""
        import numpy as np
        import sys
        
        # Load chunks with lightweight mode
        surfels = self.octree_viewer.load_chunks_lightweight(node_ids)
        if surfels is None:
            return
        
        positions = surfels['position']
        colors = surfels['color']
        
        # Translate to center
        positions = positions - self._octree_center
        
        # Create point cloud
        self.surfel_pcd = o3d.geometry.PointCloud()
        self.surfel_pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float32))
        self.surfel_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        
        # Track loaded chunks
        self._loaded_chunk_ids = set(node_ids)
        self._total_visible_surfels = len(positions)
        
        print(f"  Loaded {len(node_ids)} chunks, {len(positions):,} surfels", file=sys.stderr)
        sys.stderr.flush()
    
    def _update_visible_chunks(self):
        """Update which chunks are loaded based on camera position."""
        import numpy as np
        import sys
        
        cam = self.vis.get_view_control()
        lookat = cam.get_lookat()
        camera_pos = np.array(lookat)  # This is actually the lookat point
        
        # Get camera position from the camera object
        # Open3D's get_view_control doesn't directly give camera position
        # We use lookat + a offset based on front vector
        front = np.array(cam.get_front())
        # Estimate camera is behind the lookat point
        # For now, use lookat as approximate camera position
        
        view_dist = 10.0  # Assume ~10 units view distance
        camera_pos = np.array(lookat) - front * view_dist
        
        print(f"\n=== Chunk Selection Update ===", file=sys.stderr)
        print(f"Camera position: {camera_pos}", file=sys.stderr)
        print(f"Lookat: {lookat}", file=sys.stderr)
        
        # Find chunks within view distance
        max_view_dist = 20.0  # Load chunks within 20 units
        visible_nodes = []
        
        for node_id in self._all_chunk_nodes:
            bounds = self._chunk_bounds.get(node_id)
            if bounds is None:
                # No bounds, include it
                visible_nodes.append(node_id)
                continue
            
            # Check if chunk center is within view distance
            center_x = (bounds[0] + bounds[3]) / 2
            center_y = (bounds[1] + bounds[4]) / 2
            center_z = (bounds[2] + bounds[5]) / 2
            chunk_center = np.array([center_x, center_y, center_z])
            
            dist = np.linalg.norm(chunk_center - camera_pos)
            if dist < max_view_dist:
                visible_nodes.append(node_id)
        
        # Limit to cache size
        cache_size = self.octree_viewer.cache_size
        if len(visible_nodes) > cache_size:
            # Sort by distance and take closest
            visible_nodes.sort(key=lambda nid: self._get_chunk_distance(nid, camera_pos))
            visible_nodes = visible_nodes[:cache_size]
        
        # Determine what to load/unload
        new_chunks = set(visible_nodes) - self._loaded_chunk_ids
        old_chunks = self._loaded_chunk_ids - set(visible_nodes)
        
        print(f"Currently loaded: {len(self._loaded_chunk_ids)} chunks", file=sys.stderr)
        print(f"Visible: {len(visible_nodes)} chunks", file=sys.stderr)
        print(f"Chunks to LOAD: {len(new_chunks)}", file=sys.stderr)
        print(f"Chunks to UNLOAD: {len(old_chunks)}", file=sys.stderr)
        
        if new_chunks or old_chunks:
            # Reload visible chunks
            self._load_chunks_streaming(visible_nodes)
        
        print(f"Total visible surfels: {self._total_visible_surfels:,}", file=sys.stderr)
        print(f"================================\n", file=sys.stderr)
        sys.stderr.flush()
    
    def _get_chunk_distance(self, node_id: str, camera_pos: np.ndarray) -> float:
        """Get distance from camera to chunk center."""
        import numpy as np
        bounds = self._chunk_bounds.get(node_id)
        if bounds is None:
            return float('inf')
        
        center = np.array([
            (bounds[0] + bounds[3]) / 2,
            (bounds[1] + bounds[4]) / 2,
            (bounds[2] + bounds[5]) / 2
        ])
        return np.linalg.norm(center - camera_pos)
    
    def _init_ply_viewer(self, ply_path: str, original_txt_path: str = None):
        """Initialize regular PLY viewer."""
        # Load surfels
        print(f"Loading: {ply_path}")
        surfels = read_ply(ply_path, max_memory_gb=0)  # Load all points, no downsampling
        n_points = len(surfels["position"])
        print(f"Loaded {n_points} surfels")
        
        # Create surfel point cloud
        import numpy as np
        self.surfel_pcd = o3d.geometry.PointCloud()
        self.surfel_pcd.points = o3d.utility.Vector3dVector(surfels["position"].astype(np.float32))
        self.surfel_pcd.colors = o3d.utility.Vector3dVector(surfels["color"].astype(np.float32))
        self.surfel_pcd = self.surfel_pcd.translate(-self.surfel_pcd.get_center())
        
        # Load original point cloud if available
        if original_txt_path and os.path.exists(original_txt_path):
            print(f"Loading original: {original_txt_path}")
            points, colors = load_xyzrgb_txt(original_txt_path)
            self.original_pcd = o3d.geometry.PointCloud()
            self.original_pcd.points = o3d.utility.Vector3dVector(points)
            self.original_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            self.original_pcd = self.original_pcd.translate(-self.original_pcd.get_center())
            print(f"Original point cloud: {len(self.original_pcd.points)} points")
        else:
            print(f"No original TXT found for: {ply_path}")
            if original_txt_path:
                print(f"  Path checked: {original_txt_path}")
        
        # Create coordinate frame
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="2DGS Surfel Viewer", width=1280, height=720)
        
        # Start with surfels
        self.current_pcd = self.surfel_pcd
        self.vis.add_geometry(self.current_pcd)
        self.vis.add_geometry(self.coord)
        
        # Settings
        render = self.vis.get_render_option()
        render.background_color = [0.1, 0.1, 0.1]
        render.point_size = 3
        
        # Camera
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat([0, 0, 0])
        self.vis.get_view_control().set_zoom(0.5)
        
        # Register key callbacks
        self.vis.register_key_callback(ord('A'), self.toggle_axis)
        self.vis.register_key_callback(ord('a'), self.toggle_axis)
        self.vis.register_key_callback(ord('P'), self.toggle_points)
        self.vis.register_key_callback(ord('p'), self.toggle_points)
        self.vis.register_key_callback(ord('Q'), self.quit)
        self.vis.register_key_callback(ord('q'), self.quit)
        
    def toggle_axis(self, vis):
        """Toggle axis visibility."""
        self.show_axis = not self.show_axis
        if self.show_axis:
            self.vis.add_geometry(self.coord)
            print("Axis: ON (press A)")
        else:
            self.vis.remove_geometry(self.coord)
            print("Axis: OFF (press A)")
        
    def toggle_points(self, vis):
        """Toggle between surfels and original point cloud."""
        if self.original_pcd is None:
            print("No original point cloud loaded (press P)")
            return
            
        self.show_surfels = not self.show_surfels
        
        # Determine which point cloud to show
        new_pcd = self.surfel_pcd if self.show_surfels else self.original_pcd
        
        # Remove old, add new
        self.vis.remove_geometry(self.current_pcd)
        self.vis.add_geometry(new_pcd)
        self.current_pcd = new_pcd
        
        # Adjust point size
        render = self.vis.get_render_option()
        render.point_size = 3 if self.show_surfels else 1
        
        if self.show_surfels:
            print("View: 2DGS Surfels (large points) - press P")
        else:
            print("View: Original Point Cloud (small points) - press P")
        
    def quit(self, vis):
        """Quit the viewer."""
        print("Quitting...")
        self.vis.close()
        
    def run(self):
        """Run the viewer."""
        if self.is_streaming:
            print("\nControls (Streaming Mode):")
            print("  Left-drag: Rotate | Scroll: Zoom")
            print("  A: Toggle axis | R: Reload chunks | L: Log status | Q: Quit")
        else:
            print("\nControls:")
            print("  Left-drag: Rotate | Scroll: Zoom")
            print("  A: Toggle axis | P: Toggle 2DGS/Original | Q: Quit")
        
        self.vis.run()
        self.vis.destroy_window()


def main():
    script_dir = Path(__file__).parent
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Use provided path
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found.")
            sys.exit(1)
    else:
        # Default: check for .2dgs_octree first, then PLY
        octree_candidates = list((script_dir / "data" / "output").glob("*.2dgs_octree"))
        if octree_candidates:
            input_path = str(octree_candidates[0])
            print(f"Using octree: {input_path}")
        else:
            input_path = str(script_dir / "data" / "output" / "output.ply")
    
    # Determine if input is a directory or file
    input_p = Path(input_path)
    
    # Find matching txt file for PLY (not for octree)
    original_txt = None
    if input_p.suffix == '.ply':
        ply_name = input_p.stem
        name_variants = [ply_name]
        if ply_name.endswith("_binary"):
            name_variants.append(ply_name.replace("_binary", ""))
        elif "_" in ply_name:
            base_name = ply_name.rsplit("_", 1)[0]
            if base_name:
                name_variants.append(base_name)
        
        txt_candidates = []
        for variant in name_variants:
            txt_candidates.extend((script_dir / "data" / "input").glob(f"{variant}*.txt"))
            txt_candidates.extend((script_dir / "data" / "input").glob(f"{variant}.txt"))
        
        original_txt = txt_candidates[0] if txt_candidates else None
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found.")
        print("Usage: python viewer.py [input.ply|input.2dgs_octree]")
        sys.exit(1)
    
    viewer = SurfelViewer(input_path, str(original_txt) if original_txt else None)
    viewer.run()


if __name__ == "__main__":
    main()
