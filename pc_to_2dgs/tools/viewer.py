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

# Get project root (parent of pc_to_2dgs)
_tool_dir = Path(__file__).parent
_pc2dgs_dir = _tool_dir.parent
_project_root = _pc2dgs_dir.parent

# Add both pc2dgs and project root to path so 'src' module can be found
if str(_pc2dgs_dir) not in sys.path:
    sys.path.insert(0, str(_pc2dgs_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
        """Initialize octree streaming viewer with camera-based chunk loading."""
        import numpy as np
        import sys
        
        # Use OctreeViewer for proper streaming
        from src.octree_io import OctreeViewer
        self.octree_viewer = OctreeViewer(octree_dir, cache_size=100)
        
        # Get all chunk node IDs
        all_nodes = self.octree_viewer.get_all_chunk_nodes()
        total_chunks = len(all_nodes)
        total_surfels = self.octree_viewer.metadata.num_surfels
        
        print(f"Octree streaming ready:", file=sys.stderr)
        print(f"  Total chunks: {total_chunks}", file=sys.stderr)
        print(f"  Total surfels: {total_surfels:,}", file=sys.stderr)
        print(f"  Cache size: {self.octree_viewer.cache_size}", file=sys.stderr)
        sys.stderr.flush()
        
        # Track loaded chunks for logging
        self._loaded_chunk_ids = set()
        self._total_visible_surfels = 0
        self._all_chunk_ids = all_nodes
        
        # Get bounding box for camera FIRST (needed for chunk loading)
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
        self._chunk_bounds = {}
        
        # Initial load - get root-level chunks
        # If octree has all chunks as children, just use all_nodes
        root_chunks = []
        if self.octree_viewer.root_node and self.octree_viewer.root_node.children:
            for node in self.octree_viewer.root_node.children:
                if node.chunk_file:
                    root_chunks.append(node.node_id)
        
        # If no hierarchy, use all nodes as root
        if not root_chunks:
            root_chunks = all_nodes[:]
        
        print(f"  Initial chunks to load: {len(root_chunks)}", file=sys.stderr)
        sys.stderr.flush()
        
        # Load initial chunks (limited by cache size)
        initial_chunks = root_chunks[:min(self.octree_viewer.cache_size, len(root_chunks))]
        if initial_chunks:
            self._load_chunks_streaming(initial_chunks)
        
        # Pre-compute chunk bounds
        self._compute_chunk_bounds()
        
        # Create coordinate frame
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
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
        
        # Camera
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
    
    def _compute_chunk_bounds(self):
        """Pre-compute bounding boxes for each chunk."""
        import numpy as np
        import sys
        
        if hasattr(self.octree_viewer, 'root_node') and self.octree_viewer.root_node:
            def traverse(node, depth=0):
                if node.chunk_file and node.bounds:
                    self._chunk_bounds[node.node_id] = node.bounds
                for child in node.children:
                    traverse(child, depth+1)
            traverse(self.octree_viewer.root_node)
        
        if not self._chunk_bounds:
            print("  Warning: No chunk bounds, loading samples...", file=sys.stderr)
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
    
    def reload_chunks(self, vis):
        """Reload chunks based on camera position."""
        if not hasattr(self, 'octree_viewer'):
            return
        self._update_visible_chunks()
    
    def _update_visible_chunks(self):
        """Update chunks based on camera."""
        import numpy as np
        import sys
        
        # Use stored camera position instead of trying to get from Open3D
        # Open3D's ViewControl doesn't have get_lookat()
        camera_pos = getattr(self, '_camera_position', np.array([0, 0, 10]))
        
        print(f"\n=== Chunk Selection ===", file=sys.stderr)
        print(f"Camera: {camera_pos}", file=sys.stderr)
        
        max_dist = 20.0
        visible = []
        for nid in self._all_chunk_nodes:
            b = self._chunk_bounds.get(nid)
            if b is None:
                visible.append(nid)
                continue
            center = np.array([(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2])
            if np.linalg.norm(center - camera_pos) < max_dist:
                visible.append(nid)
        
        cache = self.octree_viewer.cache_size
        if len(visible) > cache:
            visible.sort(key=lambda x: self._get_dist(x, camera_pos))
            visible = visible[:cache]
        
        new_c = set(visible) - self._loaded_chunk_ids
        old_c = self._loaded_chunk_ids - set(visible)
        
        print(f"Loaded: {len(self._loaded_chunk_ids)} | Visible: {len(visible)} | +{len(new_c)} -{len(old_c)}", file=sys.stderr)
        print(f"Surfels: {self._total_visible_surfels:,}", file=sys.stderr)
        sys.stderr.flush()
        
        if new_c or old_c:
            self._load_chunks_streaming(visible)
    
    def _get_dist(self, node_id, cam_pos):
        import numpy as np
        b = self._chunk_bounds.get(node_id)
        if b is None: return float('inf')
        center = np.array([(b[0]+b[3])/2, (b[1]+b[4])/2, (b[2]+b[5])/2])
        return np.linalg.norm(center - cam_pos)
    
    def log_chunk_status(self, vis):
        """Log current streaming status."""
        import sys
        if not hasattr(self, 'octree_viewer'):
            return
        print("\n" + "="*40, file=sys.stderr)
        print("STREAMING STATUS", file=sys.stderr)
        print("="*40, file=sys.stderr)
        print(f"Total chunks: {len(self._all_chunk_ids)}", file=sys.stderr)
        print(f"Loaded: {len(self._loaded_chunk_ids)}", file=sys.stderr)
        print(f"Surfels: {self._total_visible_surfels:,}", file=sys.stderr)
        print("="*40 + "\n", file=sys.stderr)
        sys.stderr.flush()
        
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
        if hasattr(self, 'octree_viewer'):
            print("\nControls (Streaming):")
            print("  Left-drag: Rotate | Scroll: Zoom")
            print("  A: Toggle axis | R: Reload chunks | L: Status | Q: Quit")
        else:
            print("\nControls:")
            print("  Left-drag: Rotate | Scroll: Zoom")
            print("  A: Toggle axis | P: Toggle 2DGS/Original | Q: Quit")
        
        self.vis.run()
        self.vis.destroy_window()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="View 2DGS PLY files")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to PLY file or .2dgs_octree directory (default: output.ply)")
    parser.add_argument("--txt", dest="txt_file", default=None,
                        help="Path to original TXT point cloud for comparison")
    parser.add_argument("--force-ply", action="store_true",
                        help="Force loading PLY directly instead of using octree")
    parser.add_argument("--help-all", action="store_true", help="Show full help")
    
    args = parser.parse_args()
    
    # Determine file path - use global project root
    script_dir = _pc2dgs_dir
    
    # Check if path is to a .2dgs_octree directory
    input_path = Path(args.file) if args.file else None
    if input_path and input_path.is_dir() and str(input_path).endswith('.2dgs_octree'):
        # Already an octree directory - use it directly
        filepath = str(input_path)
    elif input_path and input_path.suffix == '.ply':
        # User selected a PLY file - check if octree version exists
        if not args.force_ply:
            octree_path = input_path.with_suffix('.2dgs_octree')
            if octree_path.exists():
                chunk_files = list(octree_path.glob("chunk_*.bin"))
                if chunk_files:
                    print(f"Found .2dgs_octree version! Using streaming mode...")
                    viewer = SurfelViewer(str(octree_path), args.txt_file)
                    viewer.run()
                    return
        # Load PLY directly
        filepath = str(input_path)
    else:
        # Default to output.ply
        filepath = str(script_dir / "data" / "output" / "output.ply")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        # Try .2dgs_octree version as fallback
        fallback = Path(filepath).with_suffix('.2dgs_octree')
        if fallback.exists():
            print(f"Trying .2dgs_octree version instead...")
            viewer = SurfelViewer(str(fallback), args.txt_file)
            viewer.run()
            return
        sys.exit(1)
    
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
