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
        """Initialize streaming viewer for .2dgs_octree format - loads ALL chunks without truncation."""
        import numpy as np
        from pathlib import Path
        
        self.is_streaming = True
        
        # Get all chunk files directly
        octree_path = Path(octree_dir)
        chunk_files = sorted(octree_path.glob("chunk_*.bin"))
        
        if not chunk_files:
            print("ERROR: No chunk files found!")
            return
        
        total_chunks = len(chunk_files)
        print(f"Found {total_chunks} chunks in octree")
        
        # Load ALL chunks using lightweight mode (only position + color)
        # This uses ~75% less memory than loading full surfel data
        BATCH_SIZE = 50
        
        all_positions = []
        all_colors = []
        
        print(f"Loading ALL {total_chunks} chunks with lightweight mode...")
        
        for batch_start in range(0, total_chunks, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_chunks)
            batch_files = chunk_files[batch_start:batch_end]
            
            for chunk_file in batch_files:
                surfels = load_chunk_lightweight(str(chunk_file))
                if surfels is not None and len(surfels.get('position', [])) > 0:
                    all_positions.append(surfels['position'])
                    all_colors.append(surfels['color'])
            
            loaded = batch_end
            print(f"  Loaded {loaded}/{total_chunks} chunks ({loaded * 100 // total_chunks}%)")
        
        if not all_positions:
            print("ERROR: No surfels loaded!")
            return
        
        # Combine all batches
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)
        
        del all_positions, all_colors
        
        n_points = len(positions)
        print(f"Total: {n_points:,} surfels loaded from all {total_chunks} chunks")
        
        # Create point cloud
        import numpy as np
        self.surfel_pcd = o3d.geometry.PointCloud()
        self.surfel_pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float32))
        self.surfel_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        
        # Center the data
        center = self.surfel_pcd.get_center()
        self.surfel_pcd = self.surfel_pcd.translate(-center)
        
        # Get bounding box
        bbox = self.surfel_pcd.get_axis_aligned_bounding_box()
        print(f"Bounding box: {bbox.get_print_info()}")
        
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
        self.vis.register_key_callback(ord('Q'), self.quit)
        self.vis.register_key_callback(ord('q'), self.quit)
    
    def reload_chunks(self, vis):
        """Reload chunks (for streaming mode)."""
        if not self.is_streaming:
            return
        
        # All chunks are already loaded at initialization
        print("All chunks already loaded - no need to reload")
    
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
            print("  A: Toggle axis | R: Reload chunks | Q: Quit")
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
