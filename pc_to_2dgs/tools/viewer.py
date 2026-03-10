#!/usr/bin/env python3
"""
2DGS Surfel Viewer

A simple viewer for 2DGS PLY surfel files using Open3D.
Supports both ASCII and binary PLY formats.
Automatically converts large files to .2dgs_octree streaming format.

Usage:
    python tools/viewer.py                    # View default output.ply
    python tools/viewer.py path/to/file.ply   # View specific file
    python tools/viewer.py --help             # Show help
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D not installed.")
    print("Install with: pip install open3d")
    sys.exit(1)

from src.export_ply import read_ply
from src.txt_io import load_xyzrgb_txt
from src.octree_io import is_octree_directory, convert_ply_to_octree, OctreeViewer, load_chunk_lightweight


class SurfelViewer:
    def __init__(self, filepath: str, original_txt_path: str = None, force_octree: bool = False):
        self.filepath = filepath
        self.original_txt_path = original_txt_path
        self.show_axis = True
        self.show_surfels = True
        self.coord = None
        self.surfel_pcd = None
        self.original_pcd = None
        self.current_pcd = None
        self.use_octree = False
        self.octree_viewer = None
        
        # Check if input is already a .2dgs_octree directory
        input_path = Path(filepath)
        
        # Check for .2dgs_octree directory
        octree_dir = input_path.with_suffix('.2dgs_octree')
        
        if input_path.is_dir() and input_path.suffix == '.2dgs_octree':
            # Already an octree directory
            print(f"Loading .2dgs_octree: {filepath}")
            self.use_octree = True
            self._init_octree_viewer(filepath)
            return
        
        # Check if .2dgs_octree exists (even partial)
        if octree_dir.exists():
            chunk_files = list(octree_dir.glob("chunk_*.bin"))
            if chunk_files:
                print(f"Found .2dgs_octree directory with {len(chunk_files)} chunks! Using streaming mode...")
                self.use_octree = True
                self._init_octree_viewer(str(octree_dir))
                return
        
        # Force octree mode or auto-convert on memory error
        if force_octree:
            print("Converting to .2dgs_octree format (forced)...")
            octree_path = convert_ply_to_octree(filepath)
            self.use_octree = True
            self._init_octree_viewer(octree_path)
            return
        
        # Check file size to decide loading strategy
        file_size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        is_large_file = file_size_mb > 500  # > 500MB is considered large
        
        if is_large_file:
            print(f"Large file detected ({file_size_mb:.0f}MB)")
            print("Converting to .2dgs_octree format first (this may take a while)...")
            octree_path = convert_ply_to_octree(filepath)
            self.use_octree = True
            self._init_octree_viewer(octree_path)
            return
        
        # Try to load PLY directly for smaller files
        print(f"Loading: {filepath}")
        
        try:
            surfels = read_ply(filepath, max_memory_gb=0)  # Load all points, no downsampling
            n_points = len(surfels["position"])
            print(f"Loaded {n_points} surfels")
            
        except (MemoryError, RuntimeError) as e:
            print(f"\nMemory error: {e}")
            print("Converting to .2dgs_octree format...")
            octree_path = convert_ply_to_octree(filepath)
            self.use_octree = True
            self._init_octree_viewer(octree_path)
            return
            print(f"\nMemory error loading file: {e}")
            print("Automatically converting to .2dgs_octree format...")
            print("This may take a few minutes for large files...\n")
            
            # Auto-convert to octree
            octree_path = convert_ply_to_octree(filepath)
            
            self.use_octree = True
            self._init_octree_viewer(octree_path)
            return
        
        except Exception as e:
            print(f"Error loading PLY: {e}")
            print("Trying .2dgs_octree conversion...")
            octree_path = convert_ply_to_octree(filepath)
            self.use_octree = True
            self._init_octree_viewer(octree_path)
            return
        
        # Continue with normal PLY initialization (not octree)
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
            print(f"No original TXT found for: {filepath}")
            if original_txt_path:
                print(f"  Path checked: {original_txt_path}")
        
        # Create coordinate frame
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="2DGS Surfel Viewer", width=1280, height=720)
        
        # Start with surfels
        self.current_pcd = self.surfel_pcd
        self.vis.add_geometry(self.current_pcd)
        self.vis.add_geometry(self.coord)
        
        # Get render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 3
        
        # Setup camera
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat(self.surfel_pcd.get_center())
        self.vis.get_view_control().set_zoom(0.5)
        
        # Register key callbacks
        self.vis.register_key_callback(ord('A'), self.toggle_axis)
        self.vis.register_key_callback(ord('a'), self.toggle_axis)
        self.vis.register_key_callback(ord('P'), self.toggle_points)
        self.vis.register_key_callback(ord('p'), self.toggle_points)
        self.vis.register_key_callback(ord('Q'), self.quit)
        self.vis.register_key_callback(ord('q'), self.quit)
    
    def _init_octree_viewer(self, octree_dir: str):
        """Initialize octree streaming viewer - loads ALL chunks without truncation."""
        import numpy as np
        
        # Don't create OctreeViewer cache for full loading - we'll use lightweight loading directly
        # Get all chunk node IDs by scanning directory
        octree_path = Path(octree_dir)
        chunk_files = sorted(octree_path.glob("chunk_*.bin"))
        
        if not chunk_files:
            print("Error: No chunks found in octree")
            return
        
        total_chunks = len(chunk_files)
        print(f"Octree has {total_chunks} chunks")
        
        # Load ALL chunks using lightweight loading (only position + color)
        # This uses ~75% less memory than loading full surfel data
        BATCH_SIZE = 50
        
        all_positions = []
        all_colors = []
        
        print(f"Loading ALL {total_chunks} chunks with lightweight mode (this may take a moment)...")
        print(f"  Using memory-efficient mode: only loading position + color (6 floats/surfel)")
        
        for batch_start in range(0, total_chunks, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_chunks)
            batch_files = chunk_files[batch_start:batch_end]
            
            for chunk_file in batch_files:
                surfels = load_chunk_lightweight(str(chunk_file))
                if surfels is not None and len(surfels.get('position', [])) > 0:
                    all_positions.append(surfels['position'])
                    all_colors.append(surfels['color'])
            
            # Progress update
            loaded = batch_end
            print(f"  Loaded {loaded}/{total_chunks} chunks ({loaded * 100 // total_chunks}%)")
        
        if not all_positions:
            print("Error: No surfels loaded from octree")
            return
        
        # Combine all batches into single arrays
        # Memory: 6 floats * 4 bytes * N points
        # For 100M points: ~2.4GB (much better than ~9GB with full surfel data)
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)
        
        # Clear intermediate lists to free some memory
        del all_positions, all_colors
        
        n_points = len(positions)
        print(f"Total: {n_points:,} surfels loaded from all {total_chunks} chunks")
        
        # Create surfel point cloud
        self.surfel_pcd = o3d.geometry.PointCloud()
        self.surfel_pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float32))
        self.surfel_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        self.surfel_pcd = self.surfel_pcd.translate(-self.surfel_pcd.get_center())
        
        # Load original point cloud if available
        if self.original_txt_path and os.path.exists(self.original_txt_path):
            print(f"Loading original: {self.original_txt_path}")
            points, colors = load_xyzrgb_txt(self.original_txt_path)
            self.original_pcd = o3d.geometry.PointCloud()
            self.original_pcd.points = o3d.utility.Vector3dVector(points)
            self.original_pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            self.original_pcd = self.original_pcd.translate(-self.original_pcd.get_center())
            print(f"Original point cloud: {len(self.original_pcd.points)} points")
        else:
            print(f"No original TXT found for: {self.filepath}")
            if self.original_txt_path:
                print(f"  Path checked: {self.original_txt_path}")
        
        # Create coordinate frame
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="2DGS Surfel Viewer", width=1280, height=720)
        
        # Start with surfels
        self.current_pcd = self.surfel_pcd
        self.vis.add_geometry(self.current_pcd)
        self.vis.add_geometry(self.coord)
        
        # Get render options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 3
        
        # Setup camera
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat(self.surfel_pcd.get_center())
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
        print("\nControls:")
        print("  Left-drag: Rotate | Scroll: Zoom")
        print("  A: Toggle axis | P: Toggle 2DGS/Original | Q: Quit")
        
        self.vis.run()
        self.vis.destroy_window()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="View 2DGS PLY files")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to PLY file (default: output.ply)")
    parser.add_argument("--txt", dest="txt_file", default=None,
                        help="Path to original TXT point cloud for comparison")
    parser.add_argument("--help-all", action="store_true", help="Show full help")
    
    args = parser.parse_args()
    
    # Determine file path
    script_dir = Path(__file__).parent.parent
    
    if args.help_all:
        print(__doc__)
        return
        
    if args.file:
        filepath = Path(args.file)
        if not filepath.is_absolute():
            filepath = script_dir / args.file
    else:
        # Look for any .ply file in output directory
        output_dir = script_dir / "data" / "output"
        ply_files = sorted(output_dir.glob("*.ply"))
        if ply_files:
            filepath = ply_files[-1]  # Use the most recent
        else:
            filepath = script_dir / "data" / "output" / "output.ply"
    
    # Determine original txt file
    original_txt = None
    if args.txt_file:
        original_txt = Path(args.txt_file)
        if not original_txt.is_absolute():
            original_txt = script_dir / args.txt_file
    else:
        # Look for matching txt file based on PLY filename
        ply_name = filepath.stem
        # Try different name variations
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
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        print("\nUsage: python tools/viewer.py [path/to/file.ply]")
        print("Example: python tools/viewer.py data/output/myfile.ply")
        sys.exit(1)
    
    viewer = SurfelViewer(str(filepath), str(original_txt) if original_txt else None)
    viewer.run()


if __name__ == "__main__":
    main()
