# pc_to_2dgs Technical Report

## 1. OVERVIEW

### Purpose
`pc_to_2dgs` is a Python pipeline that converts LiDAR point cloud data (XYZRGB TXT format) into 2D Gaussian Splatting (2DGS) compatible surfel representations. The output is used for rendering with 2DGS techniques.

### Input/Output Formats

**Input Format (TXT):**
```
x y z r g b
0.0 0.0 0.0 255 0 0
1.0 0.5 0.2 0 255 128
-0.5 1.2 0.8 128 0 255
```
- One point per line, space-separated
- 6 values per point: x, y, z, r, g, b
- All coordinates are floats (meters assumed)
- RGB values are typically 0-255 or 0.0-1.0

**Output Format (PLY):**
- 23 float properties per vertex:
  - Position: `x, y, z` (3 floats)
  - Normal: `nx, ny, nz` (3 floats)
  - Tangent: `tx, ty, tz` (3 floats)
  - Bitangent: `bx, by, bz` (3 floats)
  - Opacity: `opacity` (1 float)
  - Scale: `sx, sy, sz` (3 floats)
  - Rotation (quaternion): `rx, ry, rz, rw` (4 floats)
  - Color: `red, green, blue` (3 floats)

### Where It Fits in the Project
This is one of two conversion pipelines:
- `lidar-2dgs`: Survey-grade precision pipeline (complex normals, uncertainty estimation)
- `pc_to_2dgs`: Memory-safe streaming pipeline for large point clouds (100M+ points)

---

## 2. END-TO-END PIPELINE FLOW

### File Structure
```
pc_to_2dgs/
├── gui.py                    # GUI interface
├── stream_normals.py         # CLI streaming interface  
├── tools/
│   └── txt_to_2dgs.py        # Basic conversion tool
├── src/
│   ├── normals.py            # Normal estimation (CORE)
│   ├── surfels.py            # Surfel generation
│   ├── txt_io.py             # Streaming I/O
│   └── spatial_partition.py  # Voxel partitioning
└── data/
    ├── input/                # TXT point clouds
    └── output/               # PLY surfel files
```

### Step-by-Step Pipeline

#### Step 1: File Loading
**Files:** `src/txt_io.py`

**Two loading strategies:**

1. **Full Load (`np.loadtxt`)** - Used by FAST mode
   ```python
   data = np.loadtxt(input_file, dtype=np.float32, comments='#', delimiter=None)
   points = data[:, 0:3]
   colors = data[:, 3:6] if data.shape[1] >= 6 else None
   ```
   - Loads entire file into memory
   - Fast but requires RAM ≥ file size × 2

2. **Streaming Load (`StreamingTXTReader`)** - Used by STREAMING/SPATIAL modes
   ```python
   class StreamingTXTReader:
       def __init__(self, filepath, chunk_size=500000, count_points=True):
           # First pass: count lines (if count_points=True)
           # Memory-efficient chunk-by-chunk reading
       
       def stream_chunks(self):
           # Yields (points, colors, start_idx, chunk_idx)
           # Does NOT store all points in memory
   ```
   - Iterates through file in chunks
   - Memory footprint: O(chunk_size)

#### Step 2: Preprocessing
**Files:** `src/surfels.py` (voxelization)

Optional preprocessing:
- **Statistical Outlier Removal**: Remove isolated points
- **Voxel Downsample**: Grid-based averaging

#### Step 3: Normal Estimation
**File:** `src/normals.py` (DETAILED BELOW)

Three distinct modes:
1. FAST mode: `estimate_normals_knn()` - global KNN, NO chunking
2. STREAMING mode: `estimate_normals_streaming()` - chunk-by-chunk
3. SPATIAL mode: `estimate_normals_spatial_streaming()` - voxel-based with halo

#### Step 4: Surfel Generation
**File:** `src/surfels.py` - `build_surfels()`

Converts (points, normals) → Gaussian surfels:
```python
def build_surfels(points, normals, colors=None, 
                  avg_spacing=None, k_spatial=16):
    """
    Builds 2DGS surfels from point cloud.
    
    For each point:
    1. Find k_spatial nearest neighbors
    2. Compute local covariance to get tangent/bitangent
    3. Set scale based on avg_spacing
    4. Compute quaternion from normal/tangent/bitangent
    """
```

#### Step 5: Export
**Files:** `src/txt_io.py` - `BinaryNormalWriter`, PLY writer

Two output formats:
- **ASCII PLY**: Human-readable, large file size
- **Binary PLY**: Compact, faster I/O

---

## 3. NORMAL ESTIMATION PIPELINE (DETAILED)

### Core Algorithm: KNN + PCA

**Mathematical Foundation:**

Given a point cloud P = {p₁, p₂, ..., pₙ} where each pᵢ ∈ ℝ³:

1. **Find k nearest neighbors** for each point using cKDTree
2. **Compute local covariance matrix** for each point's neighborhood:
   ```
   Cᵢ = (1/k) Σⱼ₌₁ᵏ (pⱼ - μᵢ)(pⱼ - μᵢ)ᵀ
   
   where μᵢ = (1/k) Σⱼ₌₁ᵏ pⱼ is the centroid
   ```

3. **Eigendecomposition** of Cᵢ:
   ```
   Cᵢ v = λ v
   
   Result: λ₀ ≤ λ₁ ≤ λ₂ (sorted eigenvalues)
           v₀, v₁, v₂ (corresponding eigenvectors)
   ```

4. **Normal vector**: Smallest eigenvector v₀ points in normal direction

5. **Curvature**: Ratio of smallest to total eigenvalues:
   ```
   curvatureᵢ = λ₀ / (λ₀ + λ₁ + λ₂)
   ```

### cKDTree KNN Approach

**File:** `src/normals.py`

scipy's cKDTree provides O(N log N) KNN search:
```python
tree = cKDTree(points)  # Build once
distances, indices = tree.query(points, k=k_effective)
# indices[:, 0] is self (distance=0), skip it
neighbor_indices = indices[:, 1:]  # k neighbors
```

### GPU Batch Processing

**File:** `src/normals.py` - `estimate_normals_gpu()`

Key insight: GPU excels at batched linear algebra (PCA/eigendecomposition).

**Batch Processing Flow:**
```python
# 1. Build KDTree ONCE (CPU-bound, O(N log N))
tree = cKDTree(points)

# 2. Move all points to GPU (one-time transfer)
points_gpu = torch.from_numpy(points).cuda()

# 3. Batch loop
n_batches = ceil(n_points / batch_size)
for batch_idx in range(n_batches):
    start = batch_idx * batch_size
    end = min(start + batch_size, n_points)
    
    # CPU: Query KNN for this batch
    indices = tree.query(points[start:end], k=k_effective)[1]
    
    # GPU: Transfer indices
    indices_gpu = torch.from_numpy(indices[:, 1:]).cuda()
    
    # GPU: Gather neighbors
    neighbors = points_gpu[indices_gpu]  # (batch_n, k, 3)
    
    # GPU: Vectorized PCA
    centroids = neighbors.mean(dim=1, keepdim=True)  # (batch_n, 1, 3)
    centered = neighbors - centroids  # (batch_n, k, 3)
    cov = torch.matmul(centered.transpose(1,2), centered) / (k-1)  # (batch_n, 3, 3)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # Batched eigendecomp
    
    # GPU: Extract normals (smallest eigenvector)
    batch_normals = eigenvectors[:, :, 0]  # (batch_n, 3)
    
    # CPU: Transfer results once per batch
    normals[start:end] = batch_normals.cpu().numpy()
```

**Memory Footprint per Batch:**
- Points: `batch_size × 3 × 4 bytes` (float32)
- Neighbors: `batch_size × k × 3 × 4 bytes`
- Covariance: `batch_size × 3 × 3 × 4 bytes`

For 800K points, k=10:
- ~10 MB for points
- ~96 MB for neighbors
- ~29 MB for covariance
- Total GPU: ~135 MB per batch

### Adaptive Batch Sizing

**Calibrated batch sizes to prevent OOM:**
```python
if n_points >= 5_000_000:
    gpu_batch_size = 800_000
elif n_points >= 2_000_000:
    gpu_batch_size = 600_000
else:
    gpu_batch_size = min(n_points, 1_000_000)
```

### OOM Retry Logic

```python
retry_count = 0
max_retries = 3
current_batch = gpu_batch_size

while retry_count < max_retries:
    try:
        return estimate_normals_gpu(points, k=k, batch_size=current_batch, device=device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            retry_count += 1
            old_batch = current_batch
            current_batch = max(MIN_GPU_BATCH, current_batch // 2)
            torch.cuda.empty_cache()
            if current_batch <= MIN_GPU_BATCH:
                # Fallback to CPU
                break
```

### CPU Vectorized Fallback

**File:** `src/normals.py` - `estimate_normals_vectorized_cpu()`

Uses numpy einsum for batch covariance:
```python
# Vectorized: (batch_n, k, 3) - (batch_n, 1, 3)
centered = neighbors - centroids[:, np.newaxis, :]

# einsum for batch covariance: cov[i] = centered[i].T @ centered[i] / (k-1)
cov = np.einsum('bij,bik->bjk', centered, centered) / (k - 1)
```

### Normal Orientation Correction

**Problem:** PCA gives normals in ±normal direction (sign ambiguity)

**Solution:** Flip normals to point away from origin:
```python
dot_products = np.sum(normals * points, axis=1)  # dot(normal, position)
flip_mask = dot_products < 0
normals[flip_mask] *= -1
```

This works because LiDAR scans are typically outward from sensor origin.

---

## 4. MODES OF OPERATION

### FAST Mode

**Entry Points:**
- CLI: `python stream_normals.py input.txt output.bin --mode fast`
- GUI: Uncheck "Halo" toggle

**Characteristics:**
- Loads entire point cloud into memory
- Builds ONE global cKDTree
- GPU batched PCA
- NO spatial chunking
- NO halo computation

**When to Use:**
- Dataset < 20M points
- Sufficient RAM (> 2× dataset size)
- Need maximum speed

**Pros:**
- Fastest for small/medium datasets
- Single KDTree = efficient KNN
- GPU batched = maximum throughput

**Cons:**
- Memory limited
- CUDA OOM on large datasets
- No halo = inaccurate normals at chunk boundaries

**Memory Behavior:**
```
Peak RAM: ~2-3× point cloud size
  - Input array (float32)
  - KDTree (~4N × 3 × 8 bytes for k-d tree)
  - Output normals
  
Peak VRAM: ~batch_size × 150 MB
```

---

### STREAMING Mode

**Entry Points:**
- CLI: `python stream_normals.py input.txt output.bin --mode streaming`

**Characteristics:**
- Streaming TXT reader (never loads full file)
- Chunk-by-chunk processing
- Writes output incrementally
- Each chunk processed independently
- NO halo (boundaries between chunks are approximate)

**When to Use:**
- Dataset > 100M points
- Limited RAM
- Streaming from network source

**Pros:**
- Constant memory footprint
- Can process arbitrarily large files
- Progress reporting per chunk

**Cons:**
- No global spatial awareness
- Boundary normals are approximate (no halo)
- Slower than FAST for same dataset size

**Memory Behavior:**
```
Peak RAM: O(chunk_size)
  - Single chunk in memory
  - KDTree for single chunk
  - Output written incrementally
  
Peak VRAM: O(chunk_size × halo_factor)
```

---

### SPATIAL_STREAMING Mode (Recommended for Large Data)

**Entry Points:**
- CLI: `python stream_normals.py input.txt output.bin --mode spatial_streaming`
- GUI: Check "Halo" toggle

**Characteristics:**
- Voxel-based spatial partitioning
- **Halo expansion** for accurate boundary normals
- Processes each voxel independently
- Writes output incrementally

**When to Use:**
- Dataset > 10M points
- Need accurate normals everywhere
- Large-scale survey data

**Pros:**
- Accurate normals at ALL boundaries (halo)
- Memory efficient (chunk-based)
- Handles 100M+ points

**Cons:**
- Slower than FAST (halo computation overhead)
- More complex logic
- Potential halo explosion on sparse data

**Memory Behavior:**
```
Peak RAM: O(halo_size) << full_dataset
Peak VRAM: O(halo_size × PCA_factor)
```

---

## 5. OPTIMIZATIONS IMPLEMENTED

### 1. GPU Transfer Optimization

**Issue:** Early implementation transferred indices to GPU, then queried GPU-based KNN (slow).

**Fix:** Keep cKDTree on CPU, GPU only for PCA:
```python
# OLD (slow): GPU-based KNN
tree_gpu = faiss.IndexFlatL2(points_gpu)  # Too slow on Windows

# NEW (fast): CPU KNN + GPU PCA
tree = cKDTree(points)  # CPU, fast
indices = tree.query(batch_points, k=k_effective)  # CPU query
indices_gpu = torch.from_numpy(indices[:, 1:]).cuda()  # Transfer indices
neighbors = points_gpu[indices_gpu]  # GPU gather
# PCA on GPU...
```

### 2. Batch KNN Query

**Issue:** Querying KDTree for each point individually is slow.

**Fix:** Batch query entire dataset once, process in chunks:
```python
# OLD (slow): Query in loop
for point in points:
    neighbors = tree.query(point, k=k)  # N queries
    
# NEW (fast): Single query, batched processing
indices = tree.query(points, k=k_effective)  # 1 query, N×k results
# Process in batches...
```

### 3. Calibrated Batch Sizing

**Issue:** Fixed batch size caused OOM on large datasets.

**Fix:** Adaptive batch sizing based on dataset size:
```python
if n_points >= 5_000_000:
    batch_size = 800_000  # Stable for RTX 3060
elif n_points >= 2_000_000:
    batch_size = 600_000
else:
    batch_size = min(n_points, 1_000_000)
```

### 4. OOM Retry Logic

**Issue:** Single OOM failure = complete failure.

**Fix:** Graceful degradation with batch size reduction:
```python
current_batch = initial_batch_size
for attempt in range(max_retries):
    try:
        return process_gpu(batch_size=current_batch)
    except OOM:
        current_batch //= 2  # Halve batch
        torch.cuda.empty_cache()
    if current_batch < MIN_BATCH:
        return process_cpu()  # Fallback
```

### 5. CPU-GPU Transfer Minimization

**Issue:** Multiple CPU↔GPU transfers per batch.

**Fix:** Single transfer per batch:
```python
# Transfer indices to GPU (small)
indices_gpu = torch.from_numpy(indices).cuda()

# All computation on GPU
neighbors = points_gpu[indices_gpu]  # GPU gather
cov = torch.matmul(...)  # GPU PCA
normals = eigenvectors[:, :, 0]  # GPU

# Single transfer back
normals_np = normals.cpu().numpy()  # 1 transfer
```

### 6. Vectorized CPU Fallback

**Issue:** Loop-based CPU PCA = very slow.

**Fix:** numpy einsum for batch covariance:
```python
# OLD (slow): Loop
for i in range(n_points):
    cov_i = centered[i].T @ centered[i]
    eigenvalues, eigenvectors = eigh(cov_i)
    normals[i] = eigenvectors[:, 0]

# NEW (fast): Vectorized
cov = np.einsum('bij,bik->bjk', centered, centered) / (k - 1)
# Still need loop for eigendecomp (no batched CPU eigensolver in scipy)
```

### 7. Spatial Partitioning with Halo

**Issue:** Chunk boundaries have inaccurate normals (no neighbors across chunks).

**Fix:** Halo expansion:
```python
halo_expand = avg_spacing * 3  # 3× average point spacing

# Find all points in expanded bounding box
halo_mask = (
    (points[:, 0] >= chunk_min[0] - halo_expand) & 
    (points[:, 0] <= chunk_max[0] + halo_expand) &
    ...
)
halo_indices = np.where(halo_mask)[0]

# Compute normals for ALL halo points
# Extract only core normals using mapping
core_normals = halo_normals[core_positions]
```

### 8. avg_spacing Computation (Once Outside Loop)

**Issue:** Recomputing avg_spacing for each chunk wasted time.

**Fix:** Compute once from sample before chunk loop:
```python
# Outside loop (one-time)
sample_size = min(5000, n_points)
sample_idx = np.random.choice(n_points, sample_size, replace=False)
sample_tree = cKDTree(points[sample_idx])
sample_dist, _ = sample_tree.query(points[sample_idx], k=2)
avg_spacing = np.mean(sample_dist[:, 1])

# Inside loop (reuse)
halo_expand = avg_spacing * 3
```

### 9. GUI/CLI Mode Consistency

**Issue:** GUI and CLI had different mode selection logic.

**Fix:** Unified mode routing:
```python
# GUI
use_halo = self.halo_var.get()
mode = "spatial_streaming" if use_halo else "fast"

# CLI
selected_mode = mode  # Already "fast" or "spatial_streaming"
if selected_mode == "fast":
    return estimate_normals_fast(...)
```

### 10. Removal of Auto Mode Switching

**Issue:** Dataset size auto-triggered chunking in FAST mode.

**Fix:** Explicit mode selection only:
```python
# OLD (confusing): Auto-switch based on size
if n_points > THRESHOLD:
    return estimate_normals_chunked(...)  # Halo enabled unexpectedly!

# NEW (clear): Mode determines behavior
if mode == "fast":
    return estimate_normals_knn(...)  # NO halo
elif mode == "spatial_streaming":
    return estimate_normals_chunked(...)  # WITH halo
```

---

## 6. PERFORMANCE BENCHMARKS

### Test Environment
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- CPU: AMD Ryzen 7 5800X
- RAM: 32GB DDR4
- Storage: NVMe SSD

### Results

| Dataset Size | Mode | Runtime | Throughput | GPU VRAM |
|-------------|------|---------|------------|----------|
| 300K | FAST | ~2s | 150K pts/s | ~150 MB |
| 1M | FAST | ~8s | 125K pts/s | ~200 MB |
| 9M | FAST | ~75s | 120K pts/s | ~400 MB |
| 9M | SPATIAL | ~120s | 75K pts/s | ~300 MB |
| 100M (projected) | STREAMING | ~15min | ~110K pts/s | ~300 MB |

### Observations

1. **FAST mode scales well** up to ~10M points on RTX 3060
2. **SPATIAL mode has overhead** from halo computation but produces better normals
3. **Throughput is relatively constant** for FAST mode (KDTree dominates)
4. **SPATIAL mode throughput lower** due to per-chunk KDTree rebuilds

---

## 7. MEMORY MANAGEMENT STRATEGY

### Why FAST Mode Fails at Scale

FAST mode requires:
```
RAM:  ~3 × N × 4 bytes (float32) = 12 bytes per point
VRAM: batch_size × 135 MB

For 100M points:
  RAM needed: 1.2 GB
  VRAM needed: 400-800 MB per batch
  
Problem: cKDTree O(N) memory overhead
  KDTree ~ 4N × 3 × 8 bytes = 9.6 GB for 100M points
```

At 100M points, cKDTree alone exceeds typical GPU memory.

### Why Streaming Is Needed

Streaming processes data in chunks:
```
Memory = O(chunk_size) << O(N)

For 100M points, chunk_size=500K:
  RAM: ~6 MB per chunk
  KDTree: ~18 MB per chunk
```

### GPU vs CPU Memory Usage

**GPU (FAST mode):**
```
Per batch (800K points):
  points_gpu:     800K × 3 × 4  = 9.6 MB
  neighbors:      800K × 10 × 3 × 4 = 96 MB
  cov_matrix:     800K × 3 × 3 × 4 = 28.8 MB
  -----------------------------------------
  Total per batch:                   ~135 MB
  
  Plus indices transfer: ~3 MB
```

**CPU (STREAMING mode):**
```
Per chunk (500K points):
  chunk_points:   500K × 3 × 4  = 6 MB
  kd_tree:        ~4 × 500K × 3 × 8 ≈ 48 MB
  -----------------------------------------
  Total per chunk:                  ~54 MB
```

### Batch Sizing Logic

```python
# From estimate_normals_knn() lines 712-718
if n_points >= 5_000_000:
    gpu_batch_size = 800_000  # Safe for RTX 3060
elif n_points >= 2_000_000:
    gpu_batch_size = 600_000  # Safety margin
else:
    gpu_batch_size = min(n_points, 1_000_000)  # Small dataset
```

Calibration based on RTX 3060 (12GB) with system also using VRAM.

---

## 8. LIMITATIONS

### CPU KNN Bottleneck

**Issue:** cKDTree query is CPU-bound even in GPU mode.

**Impact:**
- KNN query for 10M points takes ~10s (on CPU)
- GPU PCA for same takes ~5s
- CPU bottleneck limits overall throughput

**Workaround:** Batched KNN (already implemented) reduces overhead.

### GPU Underutilization

**Issue:** GPU only used for PCA, not KNN.

**Impact:**
- FAISS GPU KNN attempted but too slow on Windows
- CPU KNN + GPU PCA = unbalanced workload

**Ideal:** GPU-native KNN (cuVS when stable on Windows).

### Halo Explosion Problem

**Issue:** Sparse regions cause halo to expand dramatically.

**Example:**
```
Dense region (1mm spacing):
  halo_expand = 3mm
  halo contains ~27× core points
  
Sparse region (1m spacing):
  halo_expand = 3m  
  halo contains ~10,000× core points!
```

**Impact:** Memory spikes on sparse datasets.

**Current Mitigation:** Chunk size limits absolute halo size.

### Memory Limits at 100M+

**Issue:** Even streaming mode has limits.

**Breakdown for 100M points:**
```
RAM:  ~1-2 GB (streaming chunks)
VRAM: ~400 MB (per batch)
Disk: Output PLY = ~2.3 GB (23 floats × 100M)
```

**Hard limit:** Output file size > available disk space.

### Lack of True Hierarchical LOD

**Issue:** Single-resolution output.

**Impact:**
- Viewer must LOD at display time
- No multi-resolution octree
- Loading 100M surfels in viewer is slow

**Contrast with `lidar-2dgs`:** Has proper octree-based LOD output.

### TXT Input Bottleneck

**Issue:** np.loadtxt is slow for large files.

**Example:** 100M line file ≈ 6GB TXT
```
Load time: ~5-10 minutes
Processing: ~15 minutes
Total:     ~20-25 minutes
```

**Solution:** Binary input formats (LAS/LAZ, PLY) would speed loading 10×.

---

## 9. FUTURE IMPROVEMENTS

### 1. GPU-Native KNN (cuVS)

**Current State:** CPU cKDTree + GPU PCA = unbalanced

**Future:** Use NVIDIA cuVS for GPU-accelerated KNN
```python
# Target implementation
index = cuvs.nn.cagra.Index(params)
index.build(points_gpu)
distances, indices = index.search(points_gpu, k=k)
```

**Expected Impact:**
- 10-50× faster KNN
- Fully GPU pipeline
- True real-time processing

### 2. Hierarchical LOD

**Current State:** Single resolution output

**Future:** Multi-resolution octree
```
Level 0: 1M surfels (full detail, loaded on zoom)
Level 1: 250K surfels (medium detail)
Level 2: 62K surfels (overview)
```

**Implementation:** Modify spatial partitioning to build octree hierarchy.

### 3. Binary Input Formats

**Current State:** TXT only (slow parse)

**Future:** Support LAS/LAZ (LiDAR standard)
```
pip install laspy
import laspy
las = laspy.read("input.las")
points = np.vstack([las.x, las.y, las.z]).T
```

**Expected Impact:** 5-10× faster file loading.

### 4. Streaming Octree Output

**Current State:** Flat PLY output

**Future:** Proper octree-based 2DGS format
```
chunk_0_0_0.bin   # Root octant
chunk_0_0_0_0.bin # Child octant
...
hierarchy.json    # LOD structure
```

**Implementation:** Already has chunked output structure, needs hierarchy metadata.

### 5. Multi-GPU Scaling

**Current State:** Single GPU

**Future:** Distribute chunks across multiple GPUs
```python
# Target
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
chunk = chunks[rank % n_gpus]
normals = process_gpu(chunk, gpu_id=rank % n_gpus)
```

**Expected Impact:** Linear scaling with GPU count.

### 6. Normal Refinement

**Current State:** Single-pass PCA

**Future:** Iterative refinement
```
Pass 1: Coarse normals via PCA
Pass 2: Graph-based smoothing
Pass 3: Viewpoint-consistent reorientation
```

**Implementation:** Leverage existing `refine_normals()` function.

### 7. Viewer Improvements

**Current State:** Basic PLY viewer

**Future:** WebGL-based streaming viewer
```
- Load LOD levels progressively
- GPU-based point rendering
- Cloud-ready for large datasets
```

---

## APPENDIX: Function Reference

### Core Normal Estimation Functions

| Function | File | Purpose |
|----------|------|---------|
| `estimate_normals_knn()` | normals.py | FAST mode wrapper |
| `estimate_normals_gpu()` | normals.py | GPU batched PCA |
| `estimate_normals_vectorized_cpu()` | normals.py | CPU fallback |
| `estimate_normals_chunked()` | normals.py | SPATIAL mode (halo) |
| `estimate_normals_chunked_parallel()` | normals.py | CPU parallel chunks |
| `partition_point_cloud_spatial()` | normals.py | Voxel partitioning |

### I/O Functions

| Function | File | Purpose |
|----------|------|---------|
| `StreamingTXTReader` | txt_io.py | Streaming file reader |
| `BinaryNormalWriter` | txt_io.py | Incremental binary output |
| `estimate_normals_streaming()` | txt_io.py | STREAMING mode |

### Surfel Functions

| Function | File | Purpose |
|----------|------|---------|
| `build_surfels()` | surfels.py | Point → Surfel conversion |
| `compute_tangent_bitangent()` | surfels.py | Local frame from normals |
| `refine_normals()` | surfels.py | Graph-based refinement |

### Entry Points

| File | Interface | Modes |
|------|-----------|-------|
| `gui.py` | Tkinter GUI | FAST, SPATIAL |
| `stream_normals.py` | CLI | FAST, STREAMING, SPATIAL |
| `tools/txt_to_2dgs.py` | CLI | FAST only |