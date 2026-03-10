#!/usr/bin/env python3
"""
High-Performance TXT to 2DGS Converter for Large Datasets

Optimized for billions/millions of points using:
- FAISS for ultra-fast neighbor search (~100x faster than KD-tree)
- Parallel processing (multiprocessing)
- Memory-mapped I/O
- Chunked processing

Usage:
    %(prog)s input.txt output.ply --stream
    %(prog)s input.txt output.ply --view
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import numpy as np

# Rich for modern CLI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style
from rich.theme import Theme
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.color import Color
from rich.style import StyleType

# Custom theme
custom_theme = Theme({
    "info": "dim cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "accent": "cyan",
    "dim": "dim",
    "header": "bold cyan on black",
})

console = Console(theme=custom_theme, force_terminal=True)


# ═══════════════════════════════════════════════════════════════════════════
# ELABORATE ASCII ART BANNER
# ═══════════════════════════════════════════════════════════════════════════

BANNER = """
[bold on black]
[/bold on black]
[bold on black][/bold on black]
[bold on black]    ╔═══════════════════════════════════════════════════════════════════════════════════╗[/bold on black]
[bold on black]    ║[/bold on black]                                                                               [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ███████╗████████╗ █████╗ ██████╗ ████████╗ █████╗  ██████╗██╗  ██╗         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██║  ██║         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ███████║   ██║   ███████║██████╔╝   ██║   ███████║╚█████╗ ███████║         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ██╔══██║   ██║   ██╔══██║██╔═══╝    ██║   ██╔══██║ ╚═══██╗██╔══██║         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ██║  ██║   ██║   ██║  ██║██║        ██║   ██║  ██║██████╔╝██║  ██║         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]      ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝         [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]                                                                               [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]         ██████╗ ██████╗ ██████╗ ████████╗ ██████╗ ███████╗                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██║     ██║   ██║███████║   ██║   ██║   ██║██████╔╝                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██║     ██║   ██║██╔══██║   ██║   ██║   ██║██╔══██╗                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╔╝██║  ██║                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]         ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝                 [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]                                                                               [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██████╗ █████╗ ██╗      ██╗██╗███╗   ██╗ ██████╗                     [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██╔══██╗██╔══██╗██║      ██║██║████╗  ██║██╔════╝                     [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██████╔╝███████║██║      ██║██║██╔██╗ ██║██║  ███╗                    [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██╔═══╝ ██╔══██║██║      ██║██║██║╚██╗██║██║   ██║                    [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ██║     ██║  ██║███████╗██║██║██║ ╚████║╚██████╔╝                    [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]        ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝                     [bold on black]║[/bold on black]
[bold on black]    ║[/bold on black]                                                                               [bold on black]║[/bold on black]
[bold on black]    ╚═══════════════════════════════════════════════════════════════════════════════════╝[/bold on black]
"""


# ═══════════════════════════════════════════════════════════════════════════
# LIVE STATS PANEL
# ═══════════════════════════════════════════════════════════════════════════

class LiveStats:
    """Real-time updating stats panel during processing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.chunks_processed = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.points_processed = 0
        self.surfels_generated = 0
        self.current_stage = "Initializing..."
        self.memory_usage = 0
        self.rate = 0.0
        self.eta = "N/A"
        
    def update(self, chunks_processed=None, total_chunks=None, current_chunk=None,
               points_processed=None, surfels_generated=None, current_stage=None):
        if chunks_processed is not None:
            self.chunks_processed = chunks_processed
        if total_chunks is not None:
            self.total_chunks = total_chunks
        if current_chunk is not None:
            self.current_chunk = current_chunk
        if points_processed is not None:
            self.points_processed = points_processed
        if surfels_generated is not None:
            self.surfels_generated = surfels_generated
        if current_stage is not None:
            self.current_stage = current_stage
        
        # Calculate rate and ETA
        elapsed = time.time() - self.start_time
        if elapsed > 0 and self.chunks_processed > 0:
            self.rate = self.chunks_processed / elapsed
            remaining = self.total_chunks - self.chunks_processed
            if remaining > 0 and self.rate > 0:
                eta_seconds = remaining / self.rate
                self.eta = f"{eta_seconds:.1f}s"
        
        # Update memory
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_usage = process.memory_info().rss / (1024**2)  # MB
        except:
            self.memory_usage = 0
    
    def get_panel(self) -> Panel:
        """Get the current stats panel."""
        elapsed = time.time() - self.start_time
        
        # Create status grid
        grid = Table(box=box.SIMPLE, pad_edge=False, show_header=False)
        grid.add_column(style="cyan", width=20)
        grid.add_column(style="green", justify="right", width=15)
        
        # Row 1: Stage and progress
        progress = f"[{self.chunks_processed}/{self.total_chunks}]"
        grid.add_row(f"📍 Stage", f"[bold cyan]{self.current_stage}[/bold cyan]")
        grid.add_row(f"📊 Progress", f"{progress}")
        
        # Row 2: Stats
        grid.add_row(f"⏱️  Elapsed", f"[yellow]{elapsed:.1f}s[/yellow]")
        grid.add_row(f"⏳ ETA", f"[yellow]{self.eta}[/yellow]")
        
        # Row 3: Data stats
        grid.add_row(f"📦 Points", f"[white]{self.points_processed:,}[/white]")
        grid.add_row(f"✨ Surfels", f"[white]{self.surfels_generated:,}[/white]")
        
        # Row 4: System
        grid.add_row(f"💾 RAM", f"[magenta]{self.memory_usage:.1f} MB[/magenta]")
        grid.add_row(f"🚀 Rate", f"[green]{self.rate:.2f} chunks/s[/green]")
        
        return Panel(
            grid,
            title="[bold cyan]📈 Live Statistics[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )


def print_banner():
    """Print elaborate ASCII art banner with animation effect."""
    console.clear()
    console.print(BANNER)
    console.print()


def create_progress_bar(total: int, description: str = "Processing"):
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(spinner="dots"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•", style="dim"),
        TextColumn("[cyan]{task.completed}/{task.total}", style="cyan"),
        TimeElapsedColumn(),
        TextColumn("•", style="dim"),
        TimeRemainingColumn(),
        console=console,
        expand=True
    )


def print_section(title: str, emoji: str = "🚀") -> None:
    """Print a modern section header with transition effect."""
    console.print()
    console.print(f"[bold cyan]│[/bold cyan]" * 3 + f" {emoji} [bold cyan]{title}[/bold cyan] " + f"[bold cyan]│[/bold cyan]" * 3, justify="center")
    console.print()


def print_stat(label: str, value, color: str = "green") -> None:
    """Print a statistic with color."""
    console.print(f"  [dim]├─ {label}[/dim]   [bold {color}]{value}[/bold {color}]")


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_summary_table(stats: Dict):
    """Print a beautiful summary table."""
    table = Table(
        title="[bold cyan]📋 Conversion Summary[/bold cyan]",
        box=box.ROUNDED,
        style="cyan",
        title_style="bold cyan",
        header_style="bold cyan",
        row_styles=["", "dim"]
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    
    for key, value in stats.items():
        table.add_row(key, str(value))
    
    console.print()
    console.print(table)


def load_with_spinner(message: str):
    """Show animated spinner while loading."""
    return console.status(f"[bold cyan]{message}...[/bold cyan]", spinner="bouncingBall")


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


from src.txt_io import load_xyzrgb_txt
from src.preprocess import voxel_downsample, calculate_voxel_size_for_ratio
from src.normals_large import (
    estimate_normals_large,
    LargePointCloudLoader,
    get_recommended_method,
    HAS_FAISS
)
from src.surfels import build_surfels
from src.export_ply import write_ply, IncrementalPlyWriter, write_ply_streaming


def process_chunked_pipeline_streaming(
    loader: LargePointCloudLoader,
    output_path: str,
    chunk_size: int = 50000,
    k_neighbors: int = 20,
    up_vector: tuple = (0.0, 0.0, 1.0),
    sigma_tangent: float = 0.05,
    sigma_normal: float = 0.002,
    method: str = 'auto',
    voxel_size: Optional[float] = None,
    binary: bool = True,
    verbose: bool = False
) -> Dict:
    """TRUE streaming pipeline with live stats panel."""
    total_points = loader.total_points
    n_chunks = (total_points + chunk_size - 1) // chunk_size
    
    print_section("Processing in TRUE Streaming Mode", "⚡")
    print_stat("Total points", format_number(total_points))
    print_stat("Chunk size", format_number(chunk_size))
    print_stat("Number of chunks", n_chunks)
    print_stat("Method", method)
    print_stat("Output", output_path)
    
    start_time = time.time()
    total_surfels_written = 0
    
    # Calculate automatic voxel size for downsampling if not specified
    # Note: This function doesn't have access to args, so we use default 100:1
    # For custom ratios, use --voxel flag instead
    if voxel_size is None or voxel_size <= 0:
        # Sample first chunk to estimate bounding box
        sample_chunk = loader.load_chunk(0, min(chunk_size, total_points))
        if len(sample_chunk) > 0:
            # Estimate full bbox from sample (rough approximation)
            auto_voxel_size = calculate_voxel_size_for_ratio(sample_chunk, target_ratio=100.0)
            if verbose:
                console.print(f"[cyan]Auto downsampling: {auto_voxel_size:.4f}m voxel size (100:1 ratio)[/cyan]")
                console.print(f"[yellow]Note: Downsampling is now OFF by default. Use --downsample to enable.[/yellow]")
        else:
            auto_voxel_size = 0.01
    else:
        auto_voxel_size = voxel_size
    
    # Initialize live stats
    stats = LiveStats()
    stats.total_chunks = n_chunks
    stats.current_stage = "Processing"
    
    # Create live display with stats panel
    with Live(stats.get_panel(), console=console, refresh_per_second=4) as live:
        # Use IncrementalPlyWriter for true streaming
        with IncrementalPlyWriter(output_path, binary=binary, verbose=verbose) as writer:
            # Write header with unknown count
            writer.write_header()
            
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total_points)
                
                # Update live stats
                stats.current_chunk = chunk_idx + 1
                stats.chunks_processed = chunk_idx
                stats.update()
                live.refresh()
                
                if verbose:
                    console.print(f"  Processing chunk {chunk_idx + 1}/{n_chunks}")
                
                # Load chunk from disk
                chunk_points = loader.load_chunk(start, end)
                
                if len(chunk_points) == 0:
                    continue
                
                # Automatic downsampling for memory efficiency (if enabled)
                if auto_voxel_size > 0:
                    chunk_result = voxel_downsample(chunk_points, voxel_size=auto_voxel_size)
                    chunk_points = chunk_result["position"]
                    chunk_colors = chunk_result.get("color")
                else:
                    chunk_colors = None
                
                # Estimate normals
                normals = estimate_normals_large(
                    chunk_points,
                    k_neighbors=min(k_neighbors, len(chunk_points) - 1),
                    up_vector=up_vector,
                    method=method
                )
                
                # Build surfels
                surfels = build_surfels(
                    chunk_points,
                    normals,
                    colors=chunk_colors,
                    sigma_tangent=sigma_tangent,
                    sigma_normal=sigma_normal
                )
                
                # Write chunk to disk
                writer.write_chunk(surfels)
                
                chunk_surfels = len(surfels["position"])
                total_surfels_written += chunk_surfels
                
                # Update live stats
                stats.points_processed = end
                stats.surfels_generated = total_surfels_written
                stats.chunks_processed = chunk_idx + 1
                live.refresh()
    
    elapsed = time.time() - start_time
    console.print()
    print_stat("Processing time", f"{elapsed:.2f}s")
    print_stat("Total surfels written", format_number(total_surfels_written))
    print_stat("Rate", f"{total_points / elapsed / 1000:.1f}K pts/sec")
    
    return {"total_surfels": total_surfels_written, "elapsed": elapsed}


def main() -> int:
    """Main entry point."""
    # Print elaborate banner
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="High-performance LiDAR to 2DGS for large datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True,
                        help="Input TXT file path")
    parser.add_argument("-o", "--output", required=True,
                        help="Output PLY file path")
    
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode (memory-mapped I/O for large files)")
    parser.add_argument("--auto", action="store_true",
                        help="Automatically choose best mode based on file size (recommended)")
    parser.add_argument("--chunk_size", type=int, default=50000,
                        help="Points per chunk for streaming (default: 50000)")
    
    parser.add_argument("--method", type=str, default='auto',
                        choices=['auto', 'faiss', 'gpu', 'parallel'],
                        help="Normal estimation method (default: auto)")
    
    parser.add_argument("--voxel", type=float, default=None,
                        help="Voxel size for downsampling (meters). Overrides auto-downsampling.")
    parser.add_argument("--downsample", action="store_true",
                        help="Enable automatic downsampling (reduces memory usage)")
    parser.add_argument("--downsample-ratio", type=float, default=100.0,
                        help="Target downsampling ratio (default: 100.0 = 100:1). Higher = more aggressive.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N points (for testing)")
    
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="Number of neighbors for normal estimation (default: 20)")
    parser.add_argument("--up_vector", type=str, default="0,0,1",
                        help="Up vector for normal orientation (default: 0,0,1)")
    
    parser.add_argument("--sigma_tangent", type=float, default=0.05,
                        help="Sigma along tangent plane (default: 0.05)")
    parser.add_argument("--sigma_normal", type=float, default=0.002,
                        help="Sigma along normal (default: 0.002)")
    
    parser.add_argument("--binary", action="store_true", default=True,
                        help="Write binary PLY (default: True)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--view", action="store_true",
                        help="Launch viewer after conversion")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════
    print_section("Loading Data", "📂")
    
    up_vector = tuple(float(x) for x in args.up_vector.split(','))
    
    file_size = Path(args.input).stat().st_size
    print_stat("File size", f"{file_size / (1024**3):.2f} GB")
    
    original_count = 0
    
    # Auto mode: detect file size and choose best processing method
    USE_STREAMING = args.stream
    if args.auto:
        print_section("Auto-Detecting Best Mode", "🔍")
        # Quick count of points without loading full file
        from src.txt_io import detect_format
        # Estimate points from file size (rough: ~30 bytes per point for xyzrgb)
        estimated_points = file_size / 30
        print_stat("Estimated points", format_number(int(estimated_points)))
        
        # Thresholds:
        # - < 10M points: in-memory mode (fast, full quality)
        # - 10M-50M points: streaming with moderate downsampling
        # - 50M+ points: streaming with aggressive downsampling
        if estimated_points < 10_000_000:
            USE_STREAMING = False
            print_stat("Mode", "In-memory (fast, full quality)")
        elif estimated_points < 50_000_000:
            USE_STREAMING = True
            args.downsample = True
            args.downsample_ratio = 50.0
            print_stat("Mode", "Streaming + moderate downsampling (50:1)")
        else:
            USE_STREAMING = True
            args.downsample = True
            args.downsample_ratio = 100.0
            print_stat("Mode", "Streaming + aggressive downsampling (100:1)")
    elif args.stream:
        print_stat("Mode", "Streaming (manual)")
    else:
        print_stat("Mode", "In-memory (manual)")
    
    if USE_STREAMING:
        with load_with_spinner("Loading point cloud...") as status:
            loader = LargePointCloudLoader(args.input)
            original_count = loader.total_points
            print_stat("Points loaded (streaming)", format_number(original_count))
            
            if args.sample and args.sample < original_count:
                print_warning("Sampling with streaming not supported, loading full file")
                USE_STREAMING = False
    else:
        with load_with_spinner("Loading point cloud...") as status:
            try:
                data = load_xyzrgb_txt(args.input)
                original_count = len(data["position"])
                print_stat("Points loaded", format_number(original_count))
            except Exception as e:
                print_error(f"Error loading file: {e}")
                return 1
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Sampling
    # ═══════════════════════════════════════════════════════════════════
    if args.sample and args.sample < original_count and not USE_STREAMING:
        print_section("Sampling", "🎲")
        with load_with_spinner("Sampling..."):
            indices = np.random.choice(original_count, args.sample, replace=False)
            data["position"] = data["position"][indices]
            if data.get("color") is not None:
                data["color"] = data["color"][indices]
            print_stat("Sampled points", format_number(args.sample))
            original_count = args.sample
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2.5: Downsampling (optional, for memory efficiency)
    # ═══════════════════════════════════════════════════════════════════
    if not USE_STREAMING:
        if not args.downsample:
            print_section("Downsampling", "📉")
            print_success("DISABLED - Full quality preserved")
        elif args.voxel and args.voxel > 0:
            # User-specified voxel size
            print_section("Voxel Downsampling", "📉")
            voxel_result = voxel_downsample(data["position"], voxel_size=args.voxel, colors=data.get("color"))
            points_before = len(data["position"])
            data["position"] = voxel_result["position"]
            if "color" in data and voxel_result.get("color") is not None:
                data["color"] = voxel_result["color"]
            reduction = 100 * (1 - len(data["position"]) / points_before)
            print_success(f"Downsampled: {format_number(len(data['position']))} points ({reduction:.1f}% reduction)")
            print_stat("Voxel size", f"{args.voxel:.4f}m")
            print_warning(f"Quality: ~{reduction:.0f}% of points removed (fine details may be lost)")
        else:
            # Automatic downsampling with configurable ratio
            print_section("Auto Downsampling", "📉")
            auto_voxel_size = calculate_voxel_size_for_ratio(data["position"], target_ratio=args.downsample_ratio)
            points_before = len(data["position"])
            voxel_result = voxel_downsample(data["position"], voxel_size=auto_voxel_size, colors=data.get("color"))
            data["position"] = voxel_result["position"]
            if "color" in data and voxel_result.get("color") is not None:
                data["color"] = voxel_result["color"]
            reduction = 100 * (1 - len(data["position"]) / points_before)
            print_success(f"Downsampled: {format_number(len(data['position']))} points")
            print_stat("Ratio", f"{args.downsample_ratio:.0f}:1")
            print_stat("Reduction", f"{reduction:.1f}%")
            print_stat("Voxel size", f"{auto_voxel_size:.4f}m")
            print_warning(f"Quality tradeoff: ~{reduction:.0f}% points removed (fine details may be lost)")
            if args.downsample_ratio >= 100:
                print_stat("Memory savings", "~20x after Gaussian conversion")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Normal Estimation
    # ═══════════════════════════════════════════════════════════════════
    print_section("Estimating Normals", "📐")
    
    if not USE_STREAMING:
        method = args.method if args.method != 'auto' else get_recommended_method(original_count)
        print_stat("Method", method)
        
        with load_with_spinner("Computing normals..."):
            normals = estimate_normals_large(
                data["position"],
                k_neighbors=min(args.k_neighbors, len(data["position"]) - 1),
                up_vector=up_vector,
                method=method
            )
        print_success(f"Normals computed: {format_number(len(normals))}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Surfel Construction
    # ═══════════════════════════════════════════════════════════════════
    print_section("Building Surfels", "✨")
    
    if not USE_STREAMING:
        with load_with_spinner("Creating Gaussian surfels..."):
            surfels = build_surfels(
                data["position"],
                normals,
                colors=data.get("color"),
                sigma_tangent=args.sigma_tangent,
                sigma_normal=args.sigma_normal
            )
        print_success(f"Surfels created: {format_number(len(surfels['position']))}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Export
    # ═══════════════════════════════════════════════════════════════════
    print_section("Exporting", "💾")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if USE_STREAMING:
        # For streaming mode, pass voxel_size (None = auto, 0 = disabled)
        if args.voxel:
            stream_voxel_size = args.voxel
        elif args.downsample:
            stream_voxel_size = None  # auto
        else:
            stream_voxel_size = 0.0  # disabled
        with load_with_spinner("Exporting with streaming..."):
            stream_result = process_chunked_pipeline_streaming(
                loader=LargePointCloudLoader(args.input),
                output_path=str(output_path),
                chunk_size=args.chunk_size,
                k_neighbors=args.k_neighbors,
                up_vector=up_vector,
                sigma_tangent=args.sigma_tangent,
                sigma_normal=args.sigma_normal,
                method=args.method,
                voxel_size=stream_voxel_size,
                binary=args.binary,
                verbose=args.verbose
            )
            surfels = {"position": np.zeros((stream_result["total_surfels"], 3), dtype=np.float32)}
    else:
        export_start = time.time()
        write_ply(str(output_path), surfels, binary=args.binary, verbose=args.verbose)
        print_stat("Export time", f"{time.time() - export_start:.2f}s")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    
    stats = {
        "Input points": format_number(original_count),
        "Output surfels": format_number(len(surfels["position"])),
        "Mode": "TRUE streaming (no RAM accumulation)" if USE_STREAMING else "standard",
        "Total time": f"{total_time:.2f}s",
        "Throughput": f"{original_count / total_time / 1000:.1f}K pts/sec",
        "Output file": str(output_path.resolve())
    }
    print_summary_table(stats)
    
    console.print()
    print_success("Conversion complete!")
    
    # Launch viewer if requested
    if args.view:
        console.print()
        print_section("Launching Viewer", "🖥️")
        from tools.streaming_viewer_main import run_viewer
        run_viewer(str(output_path), cache_size=100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
