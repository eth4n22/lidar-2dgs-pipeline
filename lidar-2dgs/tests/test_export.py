"""Tests for export_ply module."""

import tempfile
import os
import numpy as np
import pytest
from typing import Dict

from src.export_ply import write_ply, IncrementalPlyWriter, write_ply_incremental, write_ply_streaming


class TestWritePly:
    """Tests for write_ply function."""

    def _make_full_surfels(self, count: int = 2) -> Dict:
        """Create surfels with all required keys.

        Note: Colors should be in 0-1 range for PLY export.
        If you have 0-255 RGB values, divide by 255.0 first.
        """
        return {
            "position": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)[:count],
            "normal": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)[:count],
            "tangent": np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)[:count],
            "bitangent": np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)[:count],
            "scale": np.array([[0.05, 0.05, 0.002], [0.05, 0.05, 0.002]], dtype=np.float32)[:count],
            "rotation": np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)[:count],
            # Colors in 0-1 range (PLy standard)
            "color": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)[:count],
            "opacity": np.array([0.8, 0.8], dtype=np.float32)[:count],
        }
    
    def test_write_ply_basic(self):
        """Test basic PLY file writing."""
        surfels = self._make_full_surfels(2)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            write_ply(filepath, surfels, binary=True)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_ply_ascii(self):
        """Test PLY file writing in ASCII format."""
        surfels = self._make_full_surfels(2)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            write_ply(filepath, surfels, binary=False)
            
            assert os.path.exists(filepath)
            # ASCII file should have "end_header" line
            with open(filepath, 'r') as f:
                content = f.read()
            assert "end_header" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_ply_binary(self):
        """Test PLY file writing in binary format."""
        surfels = self._make_full_surfels(2)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            write_ply(filepath, surfels, binary=True)
            
            assert os.path.exists(filepath)
            # Binary file should have "binary_little_endian" header
            with open(filepath, 'r') as f:
                header = f.read(100)
            assert "binary_little_endian" in header
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_ply_empty_surfels(self):
        """Test PLY writing with empty surfels."""
        surfels = {
            "position": np.zeros((0, 3), dtype=np.float32),
            "normal": np.zeros((0, 3), dtype=np.float32),
            "tangent": np.zeros((0, 3), dtype=np.float32),
            "bitangent": np.zeros((0, 3), dtype=np.float32),
            "scale": np.zeros((0, 3), dtype=np.float32),
            "rotation": np.zeros((0, 4), dtype=np.float32),
            "color": np.zeros((0, 3), dtype=np.float32),
            "opacity": np.zeros((0,), dtype=np.float32),
        }
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            write_ply(filepath, surfels, binary=True)
            
            assert os.path.exists(filepath)
            # File should exist but be minimal
            with open(filepath, 'r') as f:
                header = f.read()
            assert "element vertex 0" in header
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_ply_verbose(self):
        """Test verbose PLY writing."""
        surfels = self._make_full_surfels(1)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            # Should not raise any errors
            write_ply(filepath, surfels, binary=True, verbose=True)
            
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_write_ply_with_opacity(self):
        """Test PLY writing with opacity values."""
        surfels = {
            "position": np.array([[0, 0, 0]], dtype=np.float32),
            "normal": np.array([[0, 0, 1]], dtype=np.float32),
            "tangent": np.array([[1, 0, 0]], dtype=np.float32),
            "bitangent": np.array([[0, 1, 0]], dtype=np.float32),
            "scale": np.array([[0.05, 0.05, 0.002]], dtype=np.float32),
            "rotation": np.array([[1, 0, 0, 0]], dtype=np.float32),
            "opacity": np.array([0.5], dtype=np.float32),
            # Colors in 0-1 range (PLy standard)
            "color": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        }
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            write_ply(filepath, surfels, binary=True)
            
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestIncrementalPlyWriter:
    """Tests for IncrementalPlyWriter - streaming PLY writing for billion+ points."""
    
    def _make_surfel_chunk(self, start_idx: int, count: int) -> Dict:
        """Create a test surfel chunk."""
        positions = np.array([[i, 0, 0] for i in range(start_idx, start_idx + count)], dtype=np.float32)
        return {
            "position": positions,
            "normal": np.array([[0, 0, 1]] * count, dtype=np.float32),
            "tangent": np.array([[1, 0, 0]] * count, dtype=np.float32),
            "bitangent": np.array([[0, 1, 0]] * count, dtype=np.float32),
            "opacity": np.array([0.9] * count, dtype=np.float32),
            "scale": np.array([[0.05, 0.05, 0.002]] * count, dtype=np.float32),
            "rotation": np.array([[1, 0, 0, 0]] * count, dtype=np.float32),
            # Colors in 0-1 range (PLy standard)
            "color": np.array([[1.0, 0.0, 0.0]] * count, dtype=np.float32),
        }
    
    def test_incremental_writer_basic(self):
        """Test basic incremental PLY writing."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            writer = IncrementalPlyWriter(filepath, binary=True, verbose=False)
            writer.write_header(total_expected=1000)
            
            # Write multiple chunks
            for i in range(10):
                chunk = self._make_surfel_chunk(i * 100, 100)
                writer.write_chunk(chunk)
            
            writer.finalize()
            
            assert os.path.exists(filepath)
            file_size = os.path.getsize(filepath)
            # Each vertex has 21 floats = 84 bytes + header overhead
            assert file_size > 1000 * 84  
            
            # Read back and verify
            from src.export_ply import read_ply
            surfels = read_ply(filepath, verbose=False)
            assert len(surfels["position"]) == 1000
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_incremental_writer_context_manager(self):
        """Test incremental writer with context manager."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            with IncrementalPlyWriter(filepath, binary=True, verbose=False) as writer:
                writer.write_header(total_expected=500)
                for i in range(5):
                    chunk = self._make_surfel_chunk(i * 100, 100)
                    writer.write_chunk(chunk)
            
            from src.export_ply import read_ply
            surfels = read_ply(filepath, verbose=False)
            assert len(surfels["position"]) == 500
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_incremental_writer_unknown_count(self):
        """Test incremental writing without known vertex count."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            # Don't specify total_expected - writes 0 with fixed padding
            with IncrementalPlyWriter(filepath, binary=True, verbose=False) as writer:
                writer.write_header()  # Will write element vertex 0000000000
                
                for i in range(3):
                    chunk = self._make_surfel_chunk(i * 50, 50)
                    writer.write_chunk(chunk)
            
            # Should have correct count after finalize (fixed-width: 10 digits)
            with open(filepath, 'rb') as f:
                header_bytes = f.read(500)
            header_str = header_bytes.decode('ascii', errors='replace')
            assert "element vertex 0000000150" in header_str
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_write_ply_incremental(self):
        """Test write_ply_incremental helper function."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            chunks = [
                self._make_surfel_chunk(0, 100),
                self._make_surfel_chunk(100, 100),
                self._make_surfel_chunk(200, 100),
            ]
            
            total = write_ply_incremental(filepath, chunks, binary=True, verbose=False)
            
            assert total == 300
            
            from src.export_ply import read_ply
            surfels = read_ply(filepath, verbose=False)
            assert len(surfels["position"]) == 300
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_write_ply_streaming(self):
        """Test write_ply_streaming with generator."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            def chunk_generator():
                for i in range(5):
                    yield self._make_surfel_chunk(i * 100, 100)
            
            total = write_ply_streaming(filepath, chunk_generator(), binary=True, verbose=False)
            
            assert total == 500
            
            from src.export_ply import read_ply
            surfels = read_ply(filepath, verbose=False)
            assert len(surfels["position"]) == 500
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_incremental_writer_ascii(self):
        """Test incremental writer in ASCII mode."""
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            with IncrementalPlyWriter(filepath, binary=False, verbose=False) as writer:
                writer.write_header(total_expected=200)
                
                for i in range(2):
                    chunk = self._make_surfel_chunk(i * 100, 100)
                    writer.write_chunk(chunk)
            
            # Check header (fixed-width: 10 digits)
            with open(filepath, 'r') as f:
                content = f.read()
            assert "ascii" in content
            assert "element vertex 0000000200" in content
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_memory_efficiency(self):
        """Test that streaming writes don't accumulate in memory."""
        import sys
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            filepath = f.name
        
        try:
            with IncrementalPlyWriter(filepath, binary=True, verbose=False) as writer:
                writer.write_header(total_expected=10000)
                
                chunk_size = 1000
                memory_before = sys.getsizeof(writer.file)
                
                for i in range(10):
                    chunk = self._make_surfel_chunk(i * chunk_size, chunk_size)
                    writer.write_chunk(chunk)
                    
                    # Writer memory should not grow with accumulated data
                    memory_during = sys.getsizeof(writer.file)
                    # Memory should stay roughly constant
                    assert memory_during <= memory_before * 2
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
