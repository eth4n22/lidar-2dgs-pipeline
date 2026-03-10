#!/usr/bin/env python3
"""Convert large LAS file to 2DGS - NO DOWNSAMPLING."""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.las_io import sample_las
from src.normals_large import estimate_normals_faiss
from src.surfels import build_surfels
from src.export_ply import write_ply

def main():
    input_file = "C:/Users/65889/Desktop/HDB/Code/lidar-2dgs_DISABLED/data/input/project2025-09-25-10-09-19_1.las"
    output_file = "data/output/project2025_2dgs.ply"
    
    # Sample 10M points with GPU acceleration
    SAMPLE_SIZE = 10000000  # 10 million points
    
    print("=" * 60)
    print("LAS to 2DGS Conversion - 10M points + GPU")
    print("=" * 60)
    
    # Step 1: Sample points
    print(f"\n[1/4] Sampling {SAMPLE_SIZE:,} points from LAS file...")
    data = sample_las(input_file, n_points=SAMPLE_SIZE)
    points = data["position"]
    print(f"      Sampled {len(points):,} points (NO DOWNSAMPLING)")
    
    # Get colors - use intensity if available
    colors = None
    if "color" in data and data["color"] is not None:
        colors = data["color"]
        print(f"      Using RGB colors from LAS file")
    elif "intensity" in data and data["intensity"] is not None:
        intensity = data["intensity"]
        if intensity.max() > intensity.min():
            intensity_norm = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
        else:
            intensity_norm = np.full_like(intensity, 128, dtype=np.uint8)
        colors = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
        print(f"      Using intensity as grayscale color")
    else:
        z = points[:, 2]
        z_norm = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
        colors = np.column_stack([z_norm, z_norm, z_norm])
        print(f"      Using height-based grayscale color")
    
    # Step 2: Estimate normals using FAISS with GPU
    print("\n[2/4] Estimating normals with FAISS GPU...")
    normals = estimate_normals_faiss(points, k_neighbors=20, use_gpu=True)
    print(f"      Computed {len(normals):,} normals")
    
    # Step 3: Build surfels - NO voxel downsampling
    print("\n[3/4] Building Gaussian surfels (every point becomes a surfel)...")
    surfels = build_surfels(points, normals, colors=colors)
    print(f"      Created {len(surfels['position']):,} surfels")
    
    # Step 4: Export
    print("\n[4/4] Exporting to PLY...")
    write_ply(output_file, surfels, binary=True, verbose=True)
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
