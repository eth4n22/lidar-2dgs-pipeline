#!/usr/bin/env python3
"""
AR Export CLI

Convert LiDAR point clouds to AR-ready formats:
- GLB (web, cross-platform)
- USDZ (iPhone AR Quick Look)
- OBJ (universal 3D format)
- QR code support for AR links

Usage:
    python tools/export_ar.py input.las --output ./ar_output --usdz
    python tools/export_ar.py input.las --output ./ar_output --ar_url https://mysite.com/model.usdz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.las_io import load_point_cloud, detect_format
from src.txt_io import load_xyzrgb_txt
from src.normals import estimate_normals_knn
from src.surfels import build_surfels
from src.ar_export import ARExporter, surfels_to_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LiDAR data to AR formats (GLB, USDZ, OBJ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.las --output ./ar_output
  %(prog)s input.laz --output ./ar_output --ar_url https://example.com/scan.usdz
  %(prog)s input.txt --output ./ar_output --qr --name "Building Scan"
        """
    )

    parser.add_argument("input", help="Input file (LAS, LAZ, TXT, PLY)")

    # Output
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory for AR files")

    # Model naming
    parser.add_argument("-n", "--name", default="model",
                        help="Name for output files (default: model)")

    # AR options
    parser.add_argument("--gltf", action="store_true",
                        help="Export GLB format")
    parser.add_argument("--usdz", action="store_true",
                        help="Export USDZ format (iPhone AR)")
    parser.add_argument("--obj", action="store_true",
                        help="Export OBJ format")
    parser.add_argument("--all", action="store_true",
                        help="Export all formats")

    # QR code
    parser.add_argument("--qr", action="store_true",
                        help="Generate QR code for AR link")
    parser.add_argument("--ar_url", default=None,
                        help="URL for AR model (required for QR)")

    # Processing options
    parser.add_argument("--voxel", type=float, default=None,
                        help="Voxel size for downsampling (meters)")
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="Neighbors for normal estimation (default: 20)")
    parser.add_argument("--resolution", type=int, default=4,
                        help="Mesh resolution (higher = smoother, default: 4)")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> int:
    args = parse_args()
    start_time = time.time()

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_section("STEP 1: Loading Data")

    fmt = detect_format(args.input)
    if args.verbose:
        print(f"  Input: {args.input}")
        print(f"  Format: {fmt.upper()}")

    try:
        data = load_point_cloud(args.input)
        points = data["position"]
        colors = data.get("color")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    print(f"  Points: {len(points):,}")
    if colors is not None:
        print(f"  Colors: {len(colors):,}")

    # =========================================================================
    # STEP 2: Downsample (if requested)
    # =========================================================================
    if args.voxel and args.voxel > 0:
        print_section("STEP 2: Downsampling")
        from src.preprocess import voxel_downsample

        voxel_result = voxel_downsample(points, voxel_size=args.voxel, colors=colors)
        points = voxel_result["position"]
        colors = voxel_result.get("color")

        reduction = 100 * (1 - len(points) / len(data["position"]))
        print(f"  Reduced: {reduction:.1f}% ({len(points):,} points)")

    # =========================================================================
    # STEP 3: Estimate Normals
    # =========================================================================
    print_section("STEP 3: Normal Estimation")

    normals = estimate_normals_knn(points, k_neighbors=args.k_neighbors)
    print(f"  Normals: {len(normals):,}")

    # =========================================================================
    # STEP 4: Build Surfels
    # =========================================================================
    print_section("STEP 4: Building Surfels")

    surfels = build_surfels(
        points, normals, colors=colors,
        sigma_tangent=0.05,
        sigma_normal=0.002
    )
    print(f"  Surfels: {len(surfels['position']):,}")

    # =========================================================================
    # STEP 5: Export to AR Formats
    # =========================================================================
    print_section("STEP 5: AR Export")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = ARExporter()

    # Determine formats to export
    export_all = args.all or not (args.gltf or args.usdz or args.obj)
    formats = []
    if export_all or args.gltf:
        formats.append("GLB")
    if export_all or args.usdz:
        formats.append("USDZ")
    if export_all or args.obj:
        formats.append("OBJ")

    print(f"  Exporting: {', '.join(formats)}")

    result = exporter.export(
        surfels=surfels,
        output_dir=str(output_dir),
        model_name=args.name,
        ar_url=args.ar_url
    )

    # Print results
    for key, path in result.items():
        if key in ["glb", "obj", "usdz"]:
            print(f"  ✓ {key.upper()}: {Path(path).name}")

    if "usdz_note" in result:
        print(f"\n  Note: {result['usdz_note']}")

    # =========================================================================
    # STEP 6: QR Code (if requested)
    # =========================================================================
    if args.qr and args.ar_url:
        print_section("STEP 6: QR Code")

        # Generate QR code info
        qr_info = exporter.generate_qr_info(
            ar_url=args.ar_url,
            output_dir=str(output_dir),
            model_name=args.name
        )

        print(f"  AR URL: {args.ar_url}")
        print(f"  Metadata: {Path(qr_info['metadata']).name}")
        print("\n  QR Code Generation:")
        print("  - Use any QR code generator with the AR URL above")
        print("  - Recommended: qr-code-generator.com, Canva, or Adobe tools")
        print("  - Test on both iOS (Camera) and Android (Google Lens)")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time

    print_section("SUMMARY")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Time: {elapsed:.1f}s")

    if "glb" in result:
        size_mb = Path(result["glb"]).stat().st_size / (1024 * 1024)
        print(f"  GLB Size: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("  AR Export Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
