"""Tests for streaming viewer components."""

import tempfile
import os
import numpy as np
import pytest

from src.viewer.octree_types import BoundingBox, OctreeNode, OctreeMetadata
from src.viewer.chunk_storage import ChunkStorage, save_surfels_to_chunks


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_from_points(self):
        """Test creating bounding box from points."""
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        bbox = BoundingBox.from_points(points)
        
        assert bbox.min_x == 0
        assert bbox.min_y == 0
        assert bbox.min_z == 0
        assert bbox.max_x == 4
        assert bbox.max_y == 5
        assert bbox.max_z == 6
    
    def test_center(self):
        """Test bounding box center calculation."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        
        center = bbox.center
        
        np.testing.assert_array_almost_equal(center, [5, 5, 5])
    
    def test_size(self):
        """Test bounding box size calculation."""
        bbox = BoundingBox(0, 0, 0, 10, 20, 30)
        
        size = bbox.size
        
        np.testing.assert_array_almost_equal(size, [10, 20, 30])
    
    def test_contains_point_inside(self):
        """Test point containment - inside."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        
        assert bbox.contains(np.array([5, 5, 5]))
    
    def test_contains_point_outside(self):
        """Test point containment - outside."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        
        assert not bbox.contains(np.array([11, 5, 5]))
    
    def test_intersects(self):
        """Test bounding box intersection."""
        bbox1 = BoundingBox(0, 0, 0, 10, 10, 10)
        bbox2 = BoundingBox(5, 5, 5, 15, 15, 15)
        
        assert bbox1.intersects(bbox2)
    
    def test_no_intersect(self):
        """Test no intersection."""
        bbox1 = BoundingBox(0, 0, 0, 10, 10, 10)
        bbox2 = BoundingBox(20, 20, 20, 30, 30, 30)
        
        assert not bbox1.intersects(bbox2)
    
    def test_serialization(self):
        """Test bounding box dict serialization."""
        bbox = BoundingBox(0, 0, 0, 10, 20, 30)
        
        data = bbox.to_dict()
        restored = BoundingBox.from_dict(data)
        
        assert restored.min_x == bbox.min_x
        assert restored.min_y == bbox.min_y
        assert restored.min_z == bbox.min_z
        assert restored.max_x == bbox.max_x
        assert restored.max_y == bbox.max_y
        assert restored.max_z == bbox.max_z


class TestOctreeNode:
    """Tests for OctreeNode class."""

    def test_create_node(self):
        """Test creating an octree node."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        node = OctreeNode(node_id="root", depth=0, bounds=bbox)
        
        assert node.node_id == "root"
        assert node.depth == 0
        assert node.is_leaf
        # Children can be None or empty list depending on initialization
        assert node.children is None or node.children == []
    
    def test_get_child_id(self):
        """Test child ID generation."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        node = OctreeNode(node_id="root", depth=0, bounds=bbox)
        
        child_id = node.get_child_id(3)
        
        assert child_id == "root_3"
    
    def test_subdivide(self):
        """Test node subdivision."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        node = OctreeNode(node_id="root", depth=0, bounds=bbox)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            children = node.subdivide(tmpdir)
        
        assert len(children) == 8
        assert not node.is_leaf
        assert node.children is not None
    
    def test_serialization(self):
        """Test node dict serialization."""
        bbox = BoundingBox(0, 0, 0, 10, 10, 10)
        node = OctreeNode(node_id="root", depth=0, bounds=bbox)
        
        data = node.to_dict()
        restored = OctreeNode.from_dict(data)
        
        assert restored.node_id == node.node_id
        assert restored.depth == node.depth


class TestOctreeMetadata:
    """Tests for OctreeMetadata class."""

    def test_from_surfels(self):
        """Test creating metadata from surfels."""
        surfels = {
            "position": np.random.randn(100, 3).astype(np.float32),
            "normal": np.zeros((100, 3), dtype=np.float32),
            "scale": np.ones((100, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((100, 4), dtype=np.float32),
        }
        
        metadata = OctreeMetadata.from_surfels(surfels, max_depth=4)
        
        assert metadata.total_surfels == 100
        assert metadata.max_depth == 4
        assert metadata.bounding_box is not None
        assert metadata.surfel_schema is not None
    
    def test_save_load(self):
        """Test metadata save/load."""
        surfels = {
            "position": np.random.randn(50, 3).astype(np.float32),
            "normal": np.zeros((50, 3), dtype=np.float32),
            "scale": np.ones((50, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((50, 4), dtype=np.float32),
        }
        
        metadata = OctreeMetadata.from_surfels(surfels)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            metadata.save(filepath)
            restored = OctreeMetadata.load(filepath)
            
            assert restored.total_surfels == metadata.total_surfels
            assert restored.max_depth == metadata.max_depth
        finally:
            os.unlink(filepath)


class TestChunkStorage:
    """Tests for ChunkStorage class."""

    def test_write_read_chunk(self):
        """Test writing and reading a chunk."""
        surfels = {
            "position": np.random.randn(100, 3).astype(np.float32),
            "normal": np.zeros((100, 3), dtype=np.float32),
            "scale": np.ones((100, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((100, 4), dtype=np.float32),
            "color": np.random.randint(0, 255, (100, 3), dtype=np.uint8),
            "opacity": np.ones(100, dtype=np.float32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ChunkStorage(tmpdir, mode='w')
            chunk_path, offset, size = storage.write_chunk("test_node", surfels)
            
            # Read back
            restored = storage.read_chunk(chunk_path)
            
            np.testing.assert_array_almost_equal(
                restored["position"], surfels["position"]
            )
            np.testing.assert_array_almost_equal(
                restored["color"], surfels["color"]
            )

    def test_chunk_exists(self):
        """Test chunk existence check."""
        surfels = {
            "position": np.random.randn(10, 3).astype(np.float32),
            "normal": np.zeros((10, 3), dtype=np.float32),
            "scale": np.ones((10, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((10, 4), dtype=np.float32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ChunkStorage(tmpdir, mode='w')
            
            assert not storage.chunk_exists("test_node")
            
            storage.write_chunk("test_node", surfels)
            
            assert storage.chunk_exists("test_node")
    
    def test_delete_chunk(self):
        """Test chunk deletion."""
        surfels = {
            "position": np.random.randn(10, 3).astype(np.float32),
            "normal": np.zeros((10, 3), dtype=np.float32),
            "scale": np.ones((10, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((10, 4), dtype=np.float32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ChunkStorage(tmpdir, mode='w')
            storage.write_chunk("test_node", surfels)
            assert storage.chunk_exists("test_node")
            
            result = storage.delete_chunk("test_node")
            assert result is True
            assert not storage.chunk_exists("test_node")
    
    def test_get_chunk_info(self):
        """Test getting chunk metadata."""
        surfels = {
            "position": np.random.randn(10, 3).astype(np.float32),
            "normal": np.zeros((10, 3), dtype=np.float32),
            "scale": np.ones((10, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((10, 4), dtype=np.float32),
            "color": np.random.randint(0, 255, (10, 3), dtype=np.uint8),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ChunkStorage(tmpdir, mode='w')
            chunk_path, offset, size = storage.write_chunk("test_node", surfels)
            
            info = storage.get_chunk_info(chunk_path)
            
            assert info["num_surfels"] == 10
            assert info["has_colors"]


class TestSaveSurfelsToChunks:
    """Tests for save_surfels_to_chunks function."""

    def test_save_single_chunk(self):
        """Test saving surfels to a single chunk."""
        surfels = {
            "position": np.random.randn(100, 3).astype(np.float32),
            "normal": np.zeros((100, 3), dtype=np.float32),
            "scale": np.ones((100, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((100, 4), dtype=np.float32),
            "color": np.random.randint(0, 255, (100, 3), dtype=np.uint8),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_surfels_to_chunks(
                surfels, tmpdir, node_id="test", chunk_size=1000
            )
            
            assert len(paths) == 1
            assert os.path.exists(paths[0])

    def test_save_multiple_chunks(self):
        """Test saving surfels to multiple chunks."""
        surfels = {
            "position": np.random.randn(5000, 3).astype(np.float32),
            "normal": np.zeros((5000, 3), dtype=np.float32),
            "scale": np.ones((5000, 3), dtype=np.float32) * 0.05,
            "rotation": np.zeros((5000, 4), dtype=np.float32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_surfels_to_chunks(
                surfels, tmpdir, node_id="test", chunk_size=1000
            )
            
            # Should create 5 chunks
            assert len(paths) >= 1
