#!/usr/bin/env python3
"""
TXT to 2DGS Converter CLI - Survey Grade

Supports:
- TXT, LAS, LAZ input formats
- GPU/CPU auto-detection
- Survey-grade normal estimation with uncertainty
- Validation against original LiDAR points
- Optimized for large datasets
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import yaml
from typing import Optional

import numpy as np

from src.txt_io import load_xyzrgb_txt
from src.security import validate_file_path
from src.las_io import load_las, detect_format, load_point_cloud
from src.preprocess import (
    preprocess_point_cloud,
    remove_outliers_statistical,
    voxel_downsample
)
from src.normals import (
    estimate_normals_knn,
    estimate_normals_with_uncertainty,
    validate_normals_against_points,
    filter_normals_by_uncertainty,
    get_device
)
from src.surfels import build_surfels
from src.export_ply import write_ply, write_ply_covariance
from src.metrics import prune_surfels, quality_report
from src.io_parallel import ParallelIO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert LiDAR point cloud to 2DGS surfel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Formats:
  TXT: x y z [r g b]
  LAS: ASPRS LAS format (industry standard)
  LAZ: Compressed LAS format

Examples:
  %(prog)s input.txt output.ply
  %(prog)s survey.las output.ply
  %(prog)s data.laz -o output.ply --uncertainty --report
        """
    )

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Input file (TXT, LAS, or LAZ)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output PLY file path")

    # Device options
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (deterministic, recommended for survey)")

    # Configuration
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file (default: config/default.yaml)")

    # Preprocessing
    parser.add_argument("--voxel", type=float, default=None,
                        help="Voxel size for downsampling (meters). Overrides auto-downsampling.")
    parser.add_argument("--downsample", action="store_true",
                        help="Enable automatic downsampling (reduces memory usage)")
    parser.add_argument("--downsample-ratio", type=float, default=100.0,
                        help="Target downsampling ratio (default: 100.0 = 100:1). Higher = more aggressive.")
    parser.add_argument("--outlier_threshold", type=float, default=2.0,
                        help="Outlier removal threshold (default: 2.0)")
    parser.add_argument("--outlier_k", type=int, default=20,
                        help="Neighbors for outlier detection (default: 20)")

    # Normal estimation
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="Neighbors for normal estimation (default: 20)")
    parser.add_argument("--up_vector", type=str, default="0,0,1",
                        help="Up vector (default: 0,0,1)")

    # Survey-grade options
    parser.add_argument("--uncertainty", action="store_true",
                        help="Enable uncertainty quantification")
    parser.add_argument("--max_uncertainty", type=float, default=0.5,
                        help="Max uncertainty threshold (default: 0.5)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate normals against points")
    parser.add_argument("--report", action="store_true",
                        help="Generate quality report")

    # Surfel options
    parser.add_argument("--sigma_tangent", type=float, default=0.05,
                        help="Sigma along tangent (meters, default: 0.05)")
    parser.add_argument("--sigma_normal", type=float, default=0.002,
                        help="Sigma along normal (meters, default: 0.002)")
    parser.add_argument("--opacity", type=float, default=0.8,
                        help="Default opacity (default: 0.8)")

    # Pruning
    parser.add_argument("--prune", action="store_true",
                        help="Enable pruning")
    parser.add_argument("--min_opacity", type=float, default=0.01,
                        help="Min opacity (default: 0.01)")
    parser.add_argument("--min_planarity", type=float, default=0.4,
                        help="Min planarity (default: 0.4)")

    # Output
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Binary PLY (default: True)")
    parser.add_argument("--covariance", action="store_true",
                        help="Use covariance format")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_stat(label: str, value) -> None:
    print(f"  {label:.<40} {value}")


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def main() -> int:
    args = parse_args()
    start_time = time.time()
    
    # Load configuration file
    config = load_config(getattr(args, 'config', None))
    
    # Validate and sanitize input/output paths
    try:
        input_path = validate_file_path(args.input, must_exist=True)
        output_path = validate_file_path(args.output, must_exist=False)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: Invalid file path - {e}")
        return 1

    # Determine device (config overrides CLI args)
    if config.get('normals', {}).get('use_gpu', False) or args.gpu:
        device = 'cuda'
    elif args.cpu:
        device = 'cpu'
    else:
        device = get_device()

    # Get up_vector from config or args
    up_vector_config = config.get('normals', {}).get('up_vector', None)
    if up_vector_config:
        up_vector = tuple(up_vector_config)
    else:
        up_vector = tuple(float(x) for x in args.up_vector.split(','))

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_section("STEP 1: Loading Data")

    fmt = detect_format(str(input_path))

    if args.verbose:
        print(f"  Input file: {args.input}")
        print(f"  Format: {fmt.upper()}")
        print(f"  Device: {device.upper()}")

    try:
        data = load_point_cloud(str(input_path))
        points = data["position"]
        colors = data.get("color")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    original_count = points.shape[0]
    print_stat("Points loaded", f"{original_count:,}")
    if colors is not None:
        print_stat("Colors loaded", colors.shape[0])

    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print_section("STEP 2: Preprocessing")

    # Outlier removal
    if args.outlier_threshold is not None:
        outlier_result = remove_outliers_statistical(
            points, k=args.outlier_k, std_multiplier=args.outlier_threshold
        )
        points = outlier_result["position"]
        if colors is not None:
            colors = colors[outlier_result["mask"]]
        print_stat("Outliers removed", outlier_result["removed_count"])

    # Downsampling (optional, for memory efficiency)
    # WARNING: Downsampling reduces quality by removing points
    # - 100:1 ratio = 99% of points removed (aggressive, ~20x memory savings)
    # - 10:1 ratio = 90% removed (moderate, ~2x memory savings)
    # - No downsampling = full quality, 1:1 conversion
    if args.voxel and args.voxel > 0:
        # User-specified voxel size
        voxel_result = voxel_downsample(points, voxel_size=args.voxel, colors=colors)
        points = voxel_result["position"]
        colors = voxel_result.get("color")
        reduction = 100 * (1 - len(points) / original_count)
        print_stat("Voxel downsampling", f"{reduction:.1f}% reduction, {len(points):,} points")
        print_stat("Voxel size", f"{args.voxel:.4f}m")
    elif args.downsample:
        # Automatic downsampling with configurable ratio
        from src.preprocess import calculate_voxel_size_for_ratio
        auto_voxel_size = calculate_voxel_size_for_ratio(points, target_ratio=args.downsample_ratio)
        voxel_result = voxel_downsample(points, voxel_size=auto_voxel_size, colors=colors)
        points_before = len(points)
        points = voxel_result["position"]
        colors = voxel_result.get("color")
        reduction = 100 * (1 - len(points) / points_before)
        print_stat("Auto downsampling", f"{args.downsample_ratio:.0f}:1 ratio, {reduction:.1f}% reduction")
        print_stat("Points after", f"{len(points):,}")
        print_stat("Voxel size", f"{auto_voxel_size:.4f}m")
        print_stat("Quality tradeoff", f"~{reduction:.0f}% points removed (fine details may be lost)")
    else:
        print_stat("Downsampling", "DISABLED (full quality)")

    print_stat("Final points", f"{points.shape[0]:,}")

    # =========================================================================
    # STEP 3: Normal Estimation
    # =========================================================================
    print_section("STEP 3: Normal Estimation")

    if args.uncertainty:
        normal_data = estimate_normals_with_uncertainty(
            points, k_neighbors=args.k_neighbors, up_vector=up_vector, device=device
        )
        normals = normal_data["normals"]
        uncertainty = normal_data["uncertainty"]

        print_stat("Normals computed", normals.shape[0])
        print_stat("Mean uncertainty", f"{uncertainty.mean():.4f}")

        if args.max_uncertainty < 1.0:
            filtered = filter_normals_by_uncertainty(normals, uncertainty, args.max_uncertainty)
            points = points[filtered["mask"]]
            normals = filtered["normals"]
            if colors is not None:
                colors = colors[filtered["mask"]]
            print_stat("Filtered", filtered["removed_count"])
    else:
        normals = estimate_normals_knn(
            points, k_neighbors=args.k_neighbors, up_vector=up_vector, device=device
        )
        print_stat("Normals computed", normals.shape[0])

    # =========================================================================
    # STEP 4: Validation
    # =========================================================================
    if args.validate:
        print_section("STEP 4: Validation")
        validation = validate_normals_against_points(normals, points)
        print_stat("Mean consistency", f"{validation['consistency'].mean():.4f}")
        print_stat("Low consistency", f"{np.sum(validation['consistency'] < 0.8)}")

    # =========================================================================
    # STEP 5: Surfel Construction
    # =========================================================================
    print_section("STEP 5: Surfel Construction")

    surfels = build_surfels(
        points, normals, colors=colors,
        sigma_tangent=args.sigma_tangent,
        sigma_normal=args.sigma_normal,
        opacity=args.opacity
    )
    print_stat("Surfels created", surfels["position"].shape[0])

    # =========================================================================
    # STEP 6: Quality Report
    # =========================================================================
    if args.report:
        print_section("STEP 6: Quality Report")
        print(quality_report(surfels, points))

    # =========================================================================
    # STEP 7: Pruning
    # =========================================================================
    if args.prune:
        print_section("STEP 7: Pruning")
        surfels, stats = prune_surfels(
            surfels, points, min_opacity=args.min_opacity, min_planarity=args.min_planarity
        )
        print_stat("After pruning", surfels["position"].shape[0])

    # =========================================================================
    # STEP 8: Export
    # =========================================================================
    print_section("STEP 8: Export")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.covariance:
        write_ply_covariance(str(output_path), surfels, binary=args.binary)
    else:
        write_ply(str(output_path), surfels, binary=args.binary)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time

    print_section("SUMMARY")
    print_stat("Input points", f"{original_count:,}")
    print_stat("Output surfels", f"{surfels['position'].shape[0]:,}")
    print_stat("Device", device.upper())
    print_stat("Time", f"{elapsed:.2f}s")
    print_stat("Rate", f"{original_count/elapsed/1000:.1f}K pts/sec")

    print("\n" + "=" * 60)
    print("  Complete!")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
