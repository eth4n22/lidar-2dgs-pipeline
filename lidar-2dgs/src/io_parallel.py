"""
Multi-Threaded I/O Module

Parallel file reading for large datasets.
Uses ThreadPoolExecutor for concurrent I/O operations.

Benefits:
- Faster loading of multiple files
- Overlap I/O with computation
- Efficient CPU utilization
"""

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class ParallelIO:
    """
    Multi-threaded file I/O handler.

    Enables parallel reading/writing of multiple files
    and concurrent processing of large point clouds.
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize parallel I/O handler.

        Args:
            max_workers: Maximum concurrent threads (default: CPU count)
        """
        self.max_workers = max_workers or os.cpu_count() or 4
        self._lock = threading.Lock()
        self._stats = {
            "files_read": 0,
            "files_written": 0,
            "bytes_read": 0,
            "bytes_written": 0
        }

    def read_files_parallel(self,
                           files: List[str],
                           loader_func: Callable[[str], Dict]) -> Dict[str, Dict]:
        """
        Read multiple files in parallel.

        Args:
            files: List of file paths
            loader_func: Function to load each file (returns dict with 'position', etc.)

        Returns:
            Dictionary mapping filename to loaded data
        """
        results = {}
        errors = []

        if len(files) == 1:
            # Single file - no parallelization needed
            try:
                results[files[0]] = loader_func(files[0])
            except Exception as e:
                errors.append((files[0], str(e)))
            return results, errors

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as executor:
            futures = {executor.submit(loader_func, f): f for f in files}

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    data = future.result()
                    results[filename] = data
                    with self._lock:
                        self._stats["files_read"] += 1
                except Exception as e:
                    errors.append((filename, str(e)))

        return results, errors

    def write_files_parallel(self,
                             files: List[str],
                             data_list: List[Dict],
                             writer_func: Callable[[str, Dict], None]) -> Dict[str, Any]:
        """
        Write multiple files in parallel.

        Args:
            files: List of output file paths
            data_list: List of data dictionaries to write
            writer_func: Function to write each file

        Returns:
            Dictionary with success/failure counts
        """
        if len(files) != len(data_list):
            raise ValueError("files and data_list must have same length")

        success = []
        errors = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as executor:
            futures = {
                executor.submit(writer_func, f, d): f
                for f, d in zip(files, data_list)
            }

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                    success.append(filename)
                    with self._lock:
                        self._stats["files_written"] += 1
                except Exception as e:
                    errors.append((filename, str(e)))

        return {"success": success, "errors": errors}

    def chunk_and_process(self,
                          data: np.ndarray,
                          process_func: Callable[[np.ndarray], np.ndarray],
                          chunk_size: int = 50000) -> np.ndarray:
        """
        Process large array in chunks using multiple threads.

        Args:
            data: Large numpy array
            process_func: Function to apply to each chunk
            chunk_size: Number of elements per chunk

        Returns:
            Processed array
        """
        n = len(data)
        n_chunks = (n + chunk_size - 1) // chunk_size

        # Create chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunks.append(data[start:end])

        results = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, n_chunks)) as executor:
            futures = {executor.submit(process_func, chunk): i for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                results.append((future.result(), futures[future]))

        # Reorder results
        results.sort(key=lambda x: x[1])
        return np.vstack([r[0] for r in results])

    def get_stats(self) -> Dict:
        """Get I/O statistics."""
        with self._lock:
            return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset I/O statistics."""
        with self._lock:
            self._stats = {
                "files_read": 0,
                "files_written": 0,
                "bytes_read": 0,
                "bytes_written": 0
            }


def estimate_chunk_size(n_points: int,
                       available_memory: int = 8 * 1024**3,
                       overhead_factor: float = 2.0) -> int:
    """
    Estimate optimal chunk size based on available memory.

    Args:
        n_points: Total number of points
        available_memory: Available RAM in bytes (default: 8GB)
        overhead_factor: Memory overhead multiplier

    Returns:
        Optimal chunk size
    """
    # Assume 32 bytes per point (3 floats + extras)
    bytes_per_point = 32 * overhead_factor

    max_chunk = available_memory // bytes_per_point

    # Return minimum of max_chunk and total points
    return min(max_chunk, n_points)


class ProgressTracker:
    """
    Thread-safe progress tracker for parallel operations.
    """

    def __init__(self, total: int, description: str = "Processing"):
        self._total = total
        self._completed = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._description = description
        self._last_report = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        with self._lock:
            self._completed += n
            elapsed = time.time() - self._start_time

            # Report every 5% or 1 second
            if self._completed - self._last_report >= max(self._total // 20, 1):
                percent = 100.0 * self._completed / self._total
                rate = self._completed / elapsed if elapsed > 0 else 0
                eta = (self._total - self._completed) / rate if rate > 0 else 0

                print(f"\r{self._description}: {self._completed}/{self._total} "
                      f"({percent:.1f}%) - {rate:.1f} items/s - "
                      f"ETA: {eta:.1f}s", end="", flush=True)

                self._last_report = self._completed

    def finish(self) -> None:
        """Mark as complete."""
        elapsed = time.time() - self._start_time
        print(f"\r{self._description}: {self._completed}/{self._total} "
              f"(100.0%) - Total time: {elapsed:.1f}s")

    @property
    def completed(self) -> int:
        """Get completed count."""
        with self._lock:
            return self._completed

    @property
    def total(self) -> int:
        """Get total count."""
        with self._lock:
            return self._total


def parallel_normal_estimation(points: np.ndarray,
                               k_neighbors: int = 20,
                               n_workers: int = None) -> np.ndarray:
    """
    Estimate normals using parallel processing.

    Divides point cloud into chunks and processes in parallel.

    Args:
        points: (N, 3) point cloud
        k_neighbors: Number of neighbors
        n_workers: Number of parallel workers

    Returns:
        (N, 3) normals
    """
    from scipy.spatial import cKDTree
    from src.normals import orient_normals_consistently

    n_workers = n_workers or os.cpu_count() or 4
    n = len(points)

    # Build KD-tree once
    tree = cKDTree(points)

    # Estimate optimal chunk size
    chunk_size = max(10000, n // n_workers)

    def process_chunk(args):
        start, end = args
        chunk = points[start:end]
        _, indices = tree.query(chunk, k=k_neighbors + 1)
        # Use int32 indices to save memory
        indices = indices.astype(np.int32) if indices.dtype != np.int32 else indices

        normals = np.zeros_like(chunk)

        for i in range(len(chunk)):
            global_idx = start + i
            neighbors = points[indices[i, 1:]]

            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            cov = np.dot(centered.T, centered) / k_neighbors

            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            normals[i] = normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return normals / norms

    # Create chunks
    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    # Process in parallel
    with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    normals = np.vstack(results)

    # Orient consistently
    normals = orient_normals_consistently(normals)

    return normals


def batch_process_files(file_list: List[str],
                        process_func: Callable[[str], None],
                        max_workers: int = None) -> Dict:
    """
    Process multiple files in batch.

    Args:
        file_list: List of file paths
        process_func: Function to process each file
        max_workers: Maximum parallel workers

    Returns:
        Dictionary with success/failure info
    """
    max_workers = max_workers or os.cpu_count() or 4
    results = {"success": [], "failed": []}
    tracker = ProgressTracker(len(file_list), "Processing files")

    def process_one(filepath):
        try:
            process_func(filepath)
            return filepath, True, None
        except Exception as e:
            return filepath, False, str(e)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(file_list))) as executor:
        futures = {executor.submit(process_one, f): f for f in file_list}

        for future in as_completed(futures):
            filepath, success, error = future.result()
            if success:
                results["success"].append(filepath)
            else:
                results["failed"].append((filepath, error))
            tracker.update()

    tracker.finish()
    return results
