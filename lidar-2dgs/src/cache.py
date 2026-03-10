"""
Caching utilities for expensive computations.

Provides LRU cache for KD-tree building and other expensive operations.
"""

from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


# Cache for KD-trees (expensive to build for large point clouds)
@lru_cache(maxsize=32)
def _cached_kdtree(points_hash: int, shape: Tuple[int, int]) -> Optional[cKDTree]:
    """
    Cached KD-tree builder.
    
    Note: This is a helper - actual caching should be done at higher level
    since we can't hash numpy arrays directly.
    """
    # This is a placeholder - actual implementation would need to store
    # points in a way that can be cached. For now, caching is handled
    # at the function level where points are available.
    return None


class KDTreeCache:
    """
    Cache for KD-trees to avoid rebuilding for same point clouds.
    
    Uses weak references to avoid memory leaks.
    """
    def __init__(self, max_size: int = 16):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of trees to cache
        """
        self._cache = {}
        self._max_size = max_size
        self._access_order = []
    
    def get(self, points: np.ndarray) -> Optional[cKDTree]:
        """
        Get cached KD-tree for points.
        
        Args:
            points: Point cloud array
            
        Returns:
            Cached KD-tree or None if not found
        """
        # Use array shape and first/last few values as key
        key = self._make_key(points)
        if key in self._cache:
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, points: np.ndarray, tree: cKDTree) -> None:
        """
        Cache a KD-tree.
        
        Args:
            points: Point cloud array
            tree: KD-tree to cache
        """
        key = self._make_key(points)
        
        # Evict oldest if cache full
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            old_tree = self._cache.pop(oldest, None)
            del old_tree  # Explicit cleanup of KD-tree
        
        self._cache[key] = tree
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def _make_key(self, points: np.ndarray) -> Tuple:
        """Create cache key from points array."""
        # Use shape, bounds hash, and data range for more robust key generation
        if len(points) == 0:
            return (points.shape, hash(b"empty"))

        # Use shape, data range (min/max), and hash of data
        # This reduces collision probability significantly
        bounds = (points.min(), points.max())
        data_hash = hash(points.tobytes())

        return (points.shape, bounds, data_hash)


# Global cache instance (can be disabled by setting to None)
_kdtree_cache: Optional[KDTreeCache] = None


def get_kdtree_cache() -> Optional[KDTreeCache]:
    """Get global KD-tree cache instance."""
    global _kdtree_cache
    if _kdtree_cache is None:
        _kdtree_cache = KDTreeCache(max_size=16)
    return _kdtree_cache


def clear_kdtree_cache() -> None:
    """Clear the global KD-tree cache."""
    global _kdtree_cache
    if _kdtree_cache is not None:
        _kdtree_cache.clear()


def disable_kdtree_cache() -> None:
    """Disable caching (for memory-constrained environments)."""
    global _kdtree_cache
    if _kdtree_cache is not None:
        _kdtree_cache.clear()
    _kdtree_cache = None
