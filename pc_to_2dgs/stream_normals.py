#!/usr/bin/env python3
"""
Normal Estimation Pipeline - Hybrid Mode

Supports multiple processing modes:
- fast: GPU-accelerated processing (best for <20M points)
- streaming: Memory-efficient sequential processing (for 100M+ points)
- spatial_streaming: Voxel-based spatial partitioning for improved locality (for 50M+ points)

Usage:
    python stream_normals.py input.txt output.bin
    python stream_normals.py input.txt output.bin --mode fast
    python stream_normals.py input.txt output.bin --mode streaming
    python stream_normals.py input.txt output.bin --mode spatial_streaming
    python stream_normals.py input.txt output.bin --mode auto
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.txt_io import StreamingTXTReader, estimate_normals_streaming, BinaryNormalWriter
from src.normals import estimate_normals_chunked, estimate_normals_knn, get_device
from src.spatial_partition import (
    streaming_voxel_partition,
    load_spatial_chunk,
    cleanup_spatial_chunks
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Normal estimation with automatic mode selection"
    )
    
    parser.add_argument("input", help="Input TXT file path")
    parser.add_argument("output", help="Output binary file path")
    parser.add_argument("--mode", choices=["auto", "fast", "streaming", "spatial_streaming"], 
                       default="auto",
                       help="Processing mode: auto (default), fast, streaming, or spatial_streaming")
    parser.add_argument("--chunk-size", type=int, default=500000,
                       help="Points per chunk (default: 500,000)")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of neighbors (default: 10)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--cleanup", action="store_true", default=True,
                       help="Clean up temporary spatial chunk files (default: True)")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                       help="Don't clean up temporary spatial chunk files")
    
    return parser.parse_args()


def count_points(filepath: str) -> int:
    """Count total points in a TXT file."""
    reader = StreamingTXTReader(filepath, chunk_size=1000000)
    return reader.total_points


def estimate_normals_fast(
    input_file: str,
    output_file: str,
    k: int = 10,
    device: str = None
) -> int:
    """
    Fast GPU-accelerated normal estimation.
    
    Loads all points into memory using full load (not streaming).
    Uses optimized GPU processing. Best for datasets under ~20M points.
    """
    import numpy as np
    import time
    
    print("="*60)
    print("FAST GPU NORMAL ESTIMATION")
    print("="*60)
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  K neighbors: {k}")
    print(f"  Device: {device}")
    
    # Load all points using full numpy load (fast mode)
    print("\n[Timing] Load: ...")
    load_start = time.time()
    import os
    print(f"  Loading {os.path.basename(input_file)} (full load)...")
    data = np.loadtxt(input_file, dtype=np.float32, comments='#', delimiter=None)
    points = data[:, 0:3]
    n_points = len(points)
    load_time = time.time() - load_start
    print(f"[Timing] Load: {load_time:.2f}s ({n_points:,} points)")
    
    # Estimate normals using fast GPU method (no chunking, no halo)
    print("\n[Timing] Normals: ...")
    norm_start = time.time()
    
    normals, _ = estimate_normals_knn(
        points,
        k=k,
        use_gpu=True
    )
    
    norm_time = time.time() - norm_start
    print(f"[Timing] Normals: {norm_time:.2f}s ({n_points/norm_time:,.0f} pts/sec)")
    
    # Skip surfels for now (fast mode)
    surfels_time = 0.0
    
    # Write to binary
    print("\n[Timing] Export: ...")
    write_start = time.time()
    
    out_data = np.hstack([points, normals]).astype(np.float32)
    
    with open(output_file, 'wb') as f:
        out_data.tofile(f)
    
    export_time = time.time() - write_start
    print(f"[Timing] Export: {export_time:.2f}s")
    
    total_time = load_time + norm_time + surfels_time + export_time
    
    print(f"\n{'='*60}")
    print(f"FAST MODE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total points: {n_points:,}")
    print(f"  Load: {load_time:.2f}s ({n_points/load_time:,.0f} pts/sec)")
    print(f"  Normals: {norm_time:.2f}s ({n_points/norm_time:,.0f} pts/sec)")
    print(f"  Surfels: {surfels_time:.2f}s")
    print(f"  Export: {export_time:.2f}s")
    print(f"  TOTAL: {total_time:.2f}s ({n_points/total_time:,.0f} pts/sec)")
    
    return n_points


def estimate_normals_spatial_streaming(
    input_file: str,
    output_file: str,
    k: int = 10,
    chunk_size: int = 500000,
    device: str = None,
    cleanup: bool = True
) -> int:
    """
    Spatial streaming mode: partition points into voxels, then process each spatial chunk.
    
    This mode:
    1. Partitions input into spatial chunks (voxels)
    2. Processes each chunk independently
    3. Writes results to a single output file
    
    Benefits:
    - Better KNN locality (points in same voxel are spatially close)
    - Improved cache utilization
    - Better scaling for very large datasets
    """
    import numpy as np
    
    if device is None:
        device = get_device()
    
    print("="*60)
    print("SPATIAL STREAMING NORMAL ESTIMATION")
    print("="*60)
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  K neighbors: {k}")
    print(f"  Device: {device}")
    
    # Create temporary directory for spatial chunks
    temp_dir = Path(input_file).parent / f"{Path(input_file).stem}_spatial_chunks"
    
    try:
        # Step 1: Spatial partitioning
        print("\n[SPATIAL] Partitioning points into voxels...")
        partition_start = time.time()
        
        chunk_files = streaming_voxel_partition(
            str(input_file),
            str(temp_dir),
            target_points_per_chunk=chunk_size
        )
        
        partition_time = time.time() - partition_start
        n_chunks = len(chunk_files)
        
        if n_chunks == 0:
            print("[ERROR] No spatial chunks created")
            return 0
        
        print(f"[SPATIAL] Partitioning completed in {partition_time:.2f}s")
        
        # Step 2: Process each spatial chunk
        print("\n[SPATIAL] Processing spatial chunks...")
        process_start = time.time()
        
        # Open binary writer for output
        writer = BinaryNormalWriter(output_file)
        
        total_points = 0
        for chunk_idx, chunk_path in enumerate(chunk_files):
            # Load spatial chunk
            chunk_points = load_spatial_chunk(chunk_path)
            chunk_n = len(chunk_points)
            
            if chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1:
                progress = (chunk_idx + 1) / n_chunks * 100
                print(f"  Chunk {chunk_idx + 1}/{n_chunks}: {chunk_n:,} points ({progress:.1f}%)")
            
            # Estimate normals for this chunk
            chunk_normals, _ = estimate_normals_chunked(
                chunk_points,
                k=k,
                device=device
            )
            
            # Write to output
            writer.write_chunk(chunk_points, chunk_normals)
            total_points += chunk_n
        
        writer.close()
        process_time = time.time() - process_start
        
        # Calculate stats
        avg_points = total_points / n_chunks if n_chunks > 0 else 0
        avg_speed = total_points / process_time if process_time > 0 else 0
        
        total_time = partition_time + process_time
        
        print(f"\n{'='*60}")
        print(f"SPATIAL STREAMING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total chunks: {n_chunks}")
        print(f"  Total points: {total_points:,}")
        print(f"  Avg points/chunk: {avg_points:,.0f}")
        print(f"  Partition time: {partition_time:.2f}s")
        print(f"  Processing time: {process_time:.2f}s ({avg_speed:,.0f} pts/sec)")
        print(f"  TOTAL time: {total_time:.2f}s")
        
        return total_points
        
    finally:
        # Cleanup temporary files
        if cleanup and temp_dir.exists():
            cleanup_spatial_chunks(str(temp_dir))


def estimate_normals_hybrid(
    input_file: str,
    output_file: str,
    mode: str = "auto",
    k: int = 10,
    chunk_size: int = 500000,
    device: str = None,
    cleanup: bool = True
) -> int:
    """
    Hybrid normal estimation with automatic mode selection.
    
    Args:
        input_file: Input TXT file
        output_file: Output binary file
        mode: 'auto', 'fast', 'streaming', or 'spatial_streaming'
        k: Number of neighbors
        chunk_size: Chunk size for streaming mode
        device: Device to use
        cleanup: Whether to clean up temporary files
    
    Returns:
        Total points processed
    """
    # Get device
    if device is None:
        device = get_device()
    
    print("="*60)
    print("NORMAL ESTIMATION PIPELINE")
    print("="*60)
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  K neighbors: {k}")
    print(f"  Device: {device}")
    
    # Count points first
    print("\n  Counting points in input file...")
    n_points = count_points(input_file)
    
    # Determine mode - use explicit mode, no auto threshold switching
    selected_mode = mode
    if mode == "auto":
        selected_mode = "fast"  # Default to fast if auto
    
    print(f"\n[MODE] Using: {selected_mode}")
    print(f"[MODE] Dataset size: {n_points:,} points")
    
    if selected_mode == "fast":
        return estimate_normals_fast(
            input_file=input_file,
            output_file=output_file,
            k=k,
            device=device
        )
    elif selected_mode == "streaming":
        return estimate_normals_streaming(
            input_file=input_file,
            output_file=output_file,
            k=k,
            chunk_size=chunk_size,
            device=device,
            total_points=n_points
        )
    else:  # spatial_streaming
        return estimate_normals_spatial_streaming(
            input_file=input_file,
            output_file=output_file,
            k=k,
            chunk_size=chunk_size,
            device=device,
            cleanup=cleanup
        )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert to absolute paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Run hybrid estimation
    total_points = estimate_normals_hybrid(
        input_file=str(input_path),
        output_file=str(output_path),
        mode=args.mode,
        k=args.k,
        chunk_size=args.chunk_size,
        device=args.device,
        cleanup=args.cleanup
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
