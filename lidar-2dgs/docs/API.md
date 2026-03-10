# LiDAR 2DGS Library Documentation

Complete reference for the lidar-2dgs Python library functions and modules.

**Note:** This is a Python library (not a web API). Functions are called directly from Python code.

## Core Modules

### `src.normals`

Surface normal estimation from point clouds.

#### `estimate_normals_knn(points, k_neighbors=20, up_vector=(0.0, 0.0, 1.0), device=None)`

Estimate surface normals using K-Nearest Neighbors and PCA.

**Parameters:**
- `points` (np.ndarray): (N, 3) xyz coordinates, float32
- `k_neighbors` (int): Number of neighbors for local neighborhood (minimum 3)
- `up_vector` (Tuple[float, float, float]): Reference vector for consistent normal orientation
- `device` (Optional[str]): 'cuda', 'mps', or 'cpu'. Auto-detects if None

**Returns:**
- `np.ndarray`: (N, 3) surface normals (unit length, consistently oriented)

**Raises:**
- `ValueError`: If inputs are invalid
- `RuntimeError`: If GPU is requested but unavailable

**Example:**
```python
from src.normals import estimate_normals_knn
import numpy as np

points = np.random.randn(1000, 3).astype(np.float32)
normals = estimate_normals_knn(points, k_neighbors=20)
assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)  # Unit normals
```

#### `estimate_normals_with_uncertainty(points, k_neighbors=20, up_vector=(0.0, 0.0, 1.0), device=None)`

Estimate normals with uncertainty quantification.

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with:
  - `normals`: (N, 3) unit normals
  - `uncertainty`: (N,) uncertainty scores [0, 1]
  - `planarity`: (N,) planarity scores

**Example:**
```python
from src.normals import estimate_normals_with_uncertainty

result = estimate_normals_with_uncertainty(points, k_neighbors=20)
normals = result["normals"]
uncertainty = result["uncertainty"]  # Low = reliable, High = ambiguous
```

---

### `src.preprocess`

Point cloud preprocessing operations.

#### `voxel_downsample(points, voxel_size=0.01, colors=None)`

Downsample point cloud using voxel grid filtering.

**Parameters:**
- `points` (np.ndarray): (N, 3) xyz coordinates
- `voxel_size` (float): Size of voxels in meters
- `colors` (Optional[np.ndarray]): (N, 3) rgb values (0-255)

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with 'position' and optionally 'color' keys

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
from src.preprocess import voxel_downsample

result = voxel_downsample(points, voxel_size=0.05, colors=colors)
downsampled_points = result["position"]
downsampled_colors = result.get("color")
```

#### `remove_outliers_statistical(points, k=20, std_multiplier=2.0)`

Remove outliers using statistical outlier removal.

**Parameters:**
- `points` (np.ndarray): (N, 3) xyz coordinates
- `k` (int): Number of neighbors to analyze
- `std_multiplier` (float): Threshold multiplier for global std

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with 'position', 'mask', and 'removed_count' keys

**Example:**
```python
from src.preprocess import remove_outliers_statistical

result = remove_outliers_statistical(points, k=20, std_multiplier=2.0)
clean_points = result["position"]
print(f"Removed {result['removed_count']} outliers")
```

---

### `src.surfels`

2D Gaussian surfel construction.

#### `build_surfels(points, normals, colors=None, sigma_tangent=0.05, sigma_normal=0.002, opacity=0.8)`

Build 2D Gaussian surfels from points, normals, and colors.

**Parameters:**
- `points` (np.ndarray): (N, 3) xyz coordinates
- `normals` (np.ndarray): (N, 3) surface normals
- `colors` (Optional[np.ndarray]): (N, 3) rgb values (0-255)
- `sigma_tangent` (float): Spread along tangent plane (meters)
- `sigma_normal` (float): Thickness along normal (meters)
- `opacity` (float): Default opacity for all surfels [0, 1]

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with surfel attributes:
  - `position`: (N, 3) xyz
  - `normal`: (N, 3)
  - `tangent`: (N, 3)
  - `bitangent`: (N, 3)
  - `opacity`: (N,)
  - `scale`: (N, 3)
  - `rotation`: (N, 4) quaternion [x, y, z, w]
  - `color`: (N, 3) rgb (0-1)

**Raises:**
- `ValueError`: If inputs are invalid

**Example:**
```python
from src.surfels import build_surfels

surfels = build_surfels(points, normals, colors=colors)
# Export to PLY for rendering
from src.export_ply import write_ply
write_ply("output.ply", surfels, binary=True)
```

---

### `src.export_ply`

PLY file export for 2DGS-compatible formats.

#### `write_ply(filepath, surfels, binary=True, verbose=True)`

Write surfels to PLY file format.

**Parameters:**
- `filepath` (str): Output file path
- `surfels` (Dict[str, np.ndarray]): Surfel dictionary
- `binary` (bool): Use binary format (faster, smaller)
- `verbose` (bool): Print progress information

**Raises:**
- `ValueError`: If surfel dictionary is invalid

---

## Complete Pipeline Example

```python
import numpy as np
from src.txt_io import load_xyzrgb_txt
from src.preprocess import remove_outliers_statistical, voxel_downsample
from src.normals import estimate_normals_knn
from src.surfels import build_surfels
from src.export_ply import write_ply

# Load point cloud
data = load_xyzrgb_txt("input.txt")
points = data["position"]
colors = data.get("color")

# Preprocessing
outlier_result = remove_outliers_statistical(points, k=20, std_multiplier=2.0)
points_clean = outlier_result["position"]
colors_clean = colors[outlier_result["mask"]] if colors is not None else None

# Optional: Voxel downsampling for large datasets
voxel_result = voxel_downsample(points_clean, voxel_size=0.05, colors=colors_clean)
points_down = voxel_result["position"]
colors_down = voxel_result.get("color")

# Normal estimation
normals = estimate_normals_knn(points_down, k_neighbors=20)

# Surfel construction
surfels = build_surfels(points_down, normals, colors=colors_down)

# Export
write_ply("output.ply", surfels, binary=True)
```

---

## Error Handling

All functions validate inputs and raise descriptive errors:

- `ValueError`: Invalid input parameters or data
- `RuntimeError`: Hardware/device unavailable
- `ImportError`: Missing optional dependencies

**Example error handling:**
```python
try:
    normals = estimate_normals_knn(points, device='cuda')
except RuntimeError as e:
    print(f"GPU unavailable: {e}")
    normals = estimate_normals_knn(points, device='cpu')
```

---

## Performance Tips

1. **Use FAISS for large datasets** (>100K points): Automatically used when available
2. **Voxel downsampling**: Reduces processing time for very large point clouds
3. **GPU acceleration**: Automatically used if CUDA available and dataset >10K points
4. **Streaming mode**: For datasets exceeding RAM, use chunked processing

---

## Type Hints

All functions include complete type hints for IDE support and static analysis.

```python
from typing import Dict, Optional, Tuple
import numpy as np

def estimate_normals_knn(
    points: np.ndarray,
    k_neighbors: int = 20,
    up_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    device: Optional[str] = None
) -> np.ndarray:
    ...
```
