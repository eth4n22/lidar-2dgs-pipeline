#!/usr/bin/env python3
"""
Normal Estimation Pipeline - Hybrid Mode

Supports both fast and streaming modes:
- fast: GPU-accelerated chunked processing (best for <20M points)
- streaming: Memory-efficient processing (for 100M+ points)

Usage:
    python stream_normals.py input.txt output.bin
    python stream_normals.py input.txt output.bin --mode fast
    python stream_normals.py input.txt output.bin --mode streaming
    python stream_normals.py input.txt output.bin --mode auto
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.txt_io import StreamingTXTReader, estimate_normals_streaming
from src.normals import estimate_normals_chunked, estimate_normals_knn, get_device


# Threshold for switching to streaming mode (20M points)
STREAMING_THRESHOLD = 20_000_000


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Normal estimation with automatic mode selection"
    )
    
    parser.add_argument("input", help="Input TXT file path")
    parser.add_argument("output", help="Output binary file path")
    parser.add_argument("--mode", choices=["auto", "fast", "streaming"], default="auto",
                       help="Processing mode: auto (default), fast, or streaming")
    parser.add_argument("--chunk-size", type=int, default=500000,
                       help="Points per chunk (default: 500,000)")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of neighbors (default: 10)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                       help="Device to use (default: auto-detect)")
    
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
    
    Loads all points into memory but uses optimized GPU processing.
    Best for datasets under ~20M points.
    """
    from src.txt_io import load_xyzrgb_txt
    
    import time
    
    print("="*60)
    print("FAST GPU NORMAL ESTIMATION")
    print("="*60)
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  K neighbors: {k}")
    print(f"  Device: {device}")
    
    # Load all points
    print("\n[Timing] Loading points...")
    load_start = time.time()
    points, colors = load_xyzrgb_txt(input_file)
    load_time = time.time() - load_start
    n_points = len(points)
    print(f"[Timing] Load time: {load_time:.2f}s ({n_points:,} points)")
    
    # Estimate normals using fast GPU method
    print("\n[Timing] Computing normals...")
    norm_start = time.time()
    
    normals, _ = estimate_normals_knn(
        points,
        k=k,
        use_gpu=True,
        use_chunked=True
    )
    
    norm_time = time.time() - norm_start
    print(f"[Timing] Normal estimation: {norm_time:.2f}s ({n_points/norm_time:,.0f} pts/sec)")
    
    # Write to binary
    print("\n[Timing] Writing to binary...")
    write_start = time.time()
    
    import numpy as np
    data = np.hstack([points, normals]).astype(np.float32)
    
    with open(output_file, 'wb') as f:
        data.tofile(f)
    
    write_time = time.time() - write_start
    print(f"[Timing] Write time: {write_time:.2f}s")
    
    total_time = load_time + norm_time + write_time
    
    print(f"\n{'='*60}")
    print(f"FAST MODE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total points: {n_points:,}")
    print(f"  Load time: {load_time:.2f}s ({n_points/load_time:,.0f} pts/sec)")
    print(f"  Normal time: {norm_time:.2f}s ({n_points/norm_time:,.0f} pts/sec)")
    print(f"  Write time: {write_time:.2f}s")
    print(f"  TOTAL time: {total_time:.2f}s ({n_points/total_time:,.0f} pts/sec)")
    
    return n_points


def estimate_normals_hybrid(
    input_file: str,
    output_file: str,
    mode: str = "auto",
    k: int = 10,
    chunk_size: int = 500000,
    device: str = None
) -> int:
    """
    Hybrid normal estimation with automatic mode selection.
    
    Args:
        input_file: Input TXT file
        output_file: Output binary file
        mode: 'auto', 'fast', or 'streaming'
        k: Number of neighbors
        chunk_size: Chunk size for streaming mode
        device: Device to use
    
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
    
    # Determine mode
    if mode == "auto":
        if n_points >= STREAMING_THRESHOLD:
            selected_mode = "streaming"
        else:
            selected_mode = "fast"
    else:
        selected_mode = mode
    
    print(f"\n[MODE] Using: {selected_mode}")
    print(f"[MODE] Dataset size: {n_points:,} points")
    
    if selected_mode == "fast":
        return estimate_normals_fast(
            input_file=input_file,
            output_file=output_file,
            k=k,
            device=device
        )
    else:  # streaming - pass pre-counted n_points to avoid double counting
        return estimate_normals_streaming(
            input_file=input_file,
            output_file=output_file,
            k=k,
            chunk_size=chunk_size,
            device=device,
            total_points=n_points  # Pass pre-counted to skip counting
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
        device=args.device
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
