#!/usr/bin/env python3
"""
TXT to 2DGS Converter CLI

Convert LiDAR point cloud TXT files to 2D Gaussian surfel PLY files.
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LiDAR point cloud (TXT) to 2DGS surfel (PLY) format"
    )
    
    # Required arguments
    parser.add_argument("-i", "--input", required=True, 
                        help="Input TXT file path")
    parser.add_argument("-o", "--output", required=True,
                        help="Output PLY file path")
    
    # Preprocessing options
    parser.add_argument("--voxel", type=float, default=None,
                        help="Voxel size for downsampling (meters, 0 to disable)")
    parser.add_argument("--outlier_threshold", type=float, default=None,
                        help="Z-score threshold for outlier removal")
    
    # Normal estimation options
    parser.add_argument("--knn_normals", action="store_true",
                        help="Use KNN-based normal estimation")
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="Number of neighbors for KNN (default: 20)")
    
    # Output options
    parser.add_argument("--binary", action="store_true",
                        help="Write binary PLY instead of ASCII")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    import numpy as np
    args = parse_args()

    # Get script directory and add src to path
    script_dir = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(script_dir))

    print("=" * 60)
    print("TXT to 2DGS Conversion")
    print("=" * 60)

    # 1. Load TXT file
    print(f"\n[1/5] Loading {args.input}...")
    from src.txt_io import load_xyzrgb_txt
    points, colors = load_xyzrgb_txt(args.input)
    print(f"      Loaded {len(points):,} points")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")

    # Voxel downsampling if requested
    if args.voxel is not None and args.voxel > 0:
        print(f"      Voxel downsampling (size={args.voxel})...")
        from src.preprocess import voxel_downsample
        points, unique_indices = voxel_downsample(points, colors, voxel_size=args.voxel)
        colors = colors[unique_indices]
        print(f"      After voxel: {len(points):,} points")

    # Outlier removal if requested
    if args.outlier_threshold is not None:
        print(f"      Outlier removal (threshold={args.outlier_threshold})...")
        from src.preprocess import remove_outliers
        points, mask = remove_outliers(points, colors, std_ratio=args.outlier_threshold)
        colors = colors[mask]
        print(f"      After outlier removal: {len(points):,} points")

    # 3. Estimate normals
    print(f"\n[3/5] Estimating normals (K={args.k_neighbors})...")
    from src.normals import estimate_normals_knn
    normals, curvatures = estimate_normals_knn(points, k=args.k_neighbors)
    print(f"      Computed {len(normals):,} normals")

    # 4. Build surfels
    print("\n[4/5] Building Gaussian surfels...")
    from src.surfels import build_surfels
    surfels = build_surfels(points, normals, colors=colors)
    print(f"      Created {len(surfels['position']):,} surfels")

    # 5. Export PLY
    print(f"\n[5/5] Exporting to {args.output}...")
    from src.export_ply import write_ply
    write_ply(args.output, surfels, binary=args.binary, verbose=True)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
