"""
Octree Types and Data Structures

Core types for the streaming octree viewer.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
import json
from pathlib import Path


class LodLevel(Enum):
    """LOD level for progressive rendering."""
    LODOUNT = 0  # Ultra low detail
    LOD_LOW = 1
    LOD_MEDIUM = 2
    LOD_HIGH = 3
    LOD_FULL = 4  # Full detail


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    
    @property
    def center(self) -> np.ndarray:
        """Get center of bounding box."""
        return np.array([
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2
        ])
    
    @property
    def size(self) -> np.ndarray:
        """Get size of bounding box."""
        return np.array([
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z
        ])
    
    @property
    def half_size(self) -> np.ndarray:
        """Get half-size (extent from center)."""
        return self.size / 2
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.min_x <= point[0] <= self.max_x and
            self.min_y <= point[1] <= self.max_y and
            self.min_z <= point[2] <= self.max_z
        )
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects another."""
        return not (
            self.max_x < other.min_x or
            self.min_x > other.max_x or
            self.max_y < other.min_y or
            self.min_y > other.max_y or
            self.max_z < other.min_z or
            self.min_z > other.max_z
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'min_x': float(self.min_x),
            'min_y': float(self.min_y),
            'min_z': float(self.min_z),
            'max_x': float(self.max_x),
            'max_y': float(self.max_y),
            'max_z': float(self.max_z)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        """Create from dictionary."""
        return cls(
            min_x=data['min_x'],
            min_y=data['min_y'],
            min_z=data['min_z'],
            max_x=data['max_x'],
            max_y=data['max_y'],
            max_z=data['max_z']
        )
    
    @classmethod
    def from_points(cls, points: np.ndarray) -> 'BoundingBox':
        """Create bounding box from array of points."""
        return cls(
            min_x=float(points[:, 0].min()),
            min_y=float(points[:, 1].min()),
            min_z=float(points[:, 2].min()),
            max_x=float(points[:, 0].max()),
            max_y=float(points[:, 1].max()),
            max_z=float(points[:, 2].max())
        )
    
    @classmethod
    def from_center_size(cls, center: np.ndarray, size: np.ndarray) -> 'BoundingBox':
        """Create bounding box from center and size."""
        half = size / 2
        return cls(
            min_x=float(center[0] - half[0]),
            min_y=float(center[1] - half[1]),
            min_z=float(center[2] - half[2]),
            max_x=float(center[0] + half[0]),
            max_y=float(center[1] + half[1]),
            max_z=float(center[2] + half[2])
        )


@dataclass
class OctreeNode:
    """Node in the streaming octree."""
    node_id: str
    depth: int
    bounds: BoundingBox
    chunk_file: Optional[str] = None
    chunk_offset: int = 0
    chunk_size: int = 0
    num_surfels: int = 0
    children: Optional[List['OctreeNode']] = None
    is_leaf: bool = True
    lod_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # LOD -> data
    
    def __post_init__(self):
        # Keep children as None if not provided, only initialize empty list if explicitly set
        pass
    
    def get_child_id(self, child_index: int) -> str:
        """Get ID for child node."""
        return f"{self.node_id}_{child_index}"
    
    def subdivide(self, chunk_dir: str) -> List['OctreeNode']:
        """
        Subdivide this node into 8 children.
        
        Args:
            chunk_dir: Directory to store chunk files
            
        Returns:
            List of 8 child nodes
        """
        center = self.bounds.center
        half = self.bounds.half_size
        
        # 8 children in XYZ order
        offsets = [
            (-1, -1, -1),  # 0: -x, -y, -z
            (1, -1, -1),   # 1: +x, -y, -z
            (-1, 1, -1),   # 2: -x, +y, -z
            (1, 1, -1),    # 3: +x, +y, -z
            (-1, -1, 1),   # 4: -x, -y, +z
            (1, -1, 1),    # 5: +x, -y, +z
            (-1, 1, 1),    # 6: -x, +y, +z
            (1, 1, 1),     # 7: +x, +y, +z
        ]
        
        children = []
        for i, (dx, dy, dz) in enumerate(offsets):
            # Calculate child bounds
            child_center = center + np.array([dx * half[0]/2, dy * half[1]/2, dz * half[2]/2])
            child_size = half * 2  # Full child size
            child_bounds = BoundingBox.from_center_size(child_center, child_size)
            
            child = OctreeNode(
                node_id=self.get_child_id(i),
                depth=self.depth + 1,
                bounds=child_bounds,
                chunk_file=None,
                is_leaf=True
            )
            children.append(child)
        
        self.children = children
        self.is_leaf = False
        
        return children
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'bounds': self.bounds.to_dict(),
            'chunk_file': self.chunk_file,
            'chunk_offset': self.chunk_offset,
            'chunk_size': self.chunk_size,
            'num_surfels': self.num_surfels,
            'is_leaf': self.is_leaf,
            'children': [c.to_dict() for c in self.children] if self.children else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OctreeNode':
        """Create from dictionary."""
        node = cls(
            node_id=data['node_id'],
            depth=data['depth'],
            bounds=BoundingBox.from_dict(data['bounds']),
            chunk_file=data.get('chunk_file'),
            chunk_offset=data.get('chunk_offset', 0),
            chunk_size=data.get('chunk_size', 0),
            num_surfels=data.get('num_surfels', 0),
            is_leaf=data.get('is_leaf', True)
        )
        if data.get('children'):
            node.children = [cls.from_dict(c) for c in data['children']]
        return node


@dataclass
class OctreeMetadata:
    """Metadata for the entire octree."""
    version: str = "1.0.0"
    total_surfels: int = 0
    bounding_box: Optional[BoundingBox] = None
    max_depth: int = 8
    chunk_size: int = 10000
    surfel_schema: Dict[str, str] = field(default_factory=dict)
    created_from: Optional[str] = None
    creation_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'total_surfels': self.total_surfels,
            'bounding_box': self.bounding_box.to_dict() if self.bounding_box else None,
            'max_depth': self.max_depth,
            'chunk_size': self.chunk_size,
            'surfel_schema': self.surfel_schema,
            'created_from': self.created_from,
            'creation_time': self.creation_time
        }
    
    def save(self, filepath: str) -> None:
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'OctreeMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            version=data.get('version', '1.0.0'),
            total_surfels=data.get('total_surfels', 0),
            bounding_box=BoundingBox.from_dict(data['bounding_box']) if data.get('bounding_box') else None,
            max_depth=data.get('max_depth', 8),
            chunk_size=data.get('chunk_size', 10000),
            surfel_schema=data.get('surfel_schema', {}),
            created_from=data.get('created_from'),
            creation_time=data.get('creation_time')
        )
    
    @classmethod
    def from_surfels(cls, surfels: Dict[str, np.ndarray], 
                     max_depth: int = 8,
                     chunk_size: int = 10000,
                     source_file: Optional[str] = None) -> 'OctreeMetadata':
        """
        Create metadata from surfel data.
        
        Args:
            surfels: Dictionary of surfel arrays
            max_depth: Maximum octree depth
            chunk_size: Target surfels per chunk
            source_file: Source file path
        """
        import datetime
        
        positions = surfels['position']
        bbox = BoundingBox.from_points(positions)
        
        return cls(
            version="1.0.0",
            total_surfels=len(positions),
            bounding_box=bbox,
            max_depth=max_depth,
            chunk_size=chunk_size,
            surfel_schema={
                'position': 'float32[3]',
                'normal': 'float32[3]',
                'scale': 'float32[3]',
                'rotation': 'float32[4]',
                'color': 'uint8[3]',
                'opacity': 'float32'
            },
            created_from=source_file,
            creation_time=datetime.datetime.now().isoformat()
        )
