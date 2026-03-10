### ❌ What Doesn't Work
- **Main viewer** (`tools/streaming_viewer_main.py`) - Window opens but shows nothing
- **Octree-based rendering** - Data isn't being passed from octree to renderer

## Quick Test

Run the minimal test to verify OpenGL works:
```bash
python test_minimal_viewer.py
```
**Expected:** You should see 30 colored points in the window.

## Known Issues

### 1. Octree Not Loading Data
The streaming viewer builds an octree but may not be loading chunks correctly.

**Check:**
- `src/viewer/streaming_viewer.py` - `get_visible_surfels()` method
- `src/viewer/streaming_viewer.py` - `update_visible_chunks()` method
- Console output should show "DEBUG: Found X visible surfels"

### 2. Camera/Frustum Culling Too Aggressive
The frustum culling might be excluding all nodes.

**Check:**
- `src/viewer/streaming_viewer.py` - `_find_visible_nodes()` method
- `src/viewer/streaming_viewer.py` - `Frustum.intersects_aabb()` method
- Try disabling frustum culling temporarily

### 3. Instance VAO Setup
Point rendering uses instanced attributes which may not be configured correctly.

**Check:**
- `src/viewer/opengl_renderer.py` - `_render_points()` method (line ~557)
- `src/viewer/opengl_renderer.py` - `upload_surfels()` method (line ~361)
- Verify position (location 0) and color (location 4) are bound correctly

## Debug Steps

### Step 1: Verify Data Flow
Add debug prints in `tools/streaming_viewer_main.py` around line 130:
```python
surfels = viewer.get_visible_surfels()
print(f"DEBUG: Got {len(surfels.get('position', [])) if surfels else 0} surfels")
```

### Step 2: Check Octree Loading
In `src/viewer/streaming_viewer.py`, check if root node is being loaded:
```python
# In get_visible_surfels(), add:
print(f"DEBUG: Visible nodes: {self._visible_nodes}")
print(f"DEBUG: Cache size: {len(self._chunk_cache)}")
```

### Step 3: Force Load Root Node
Temporarily bypass frustum culling:
```python
# In tools/streaming_viewer_main.py, after line 121:
if len(viewer._visible_nodes) == 0:
    viewer._visible_nodes.add("root")
    viewer.load_chunks_async(["root"])
```

### Step 4: Check Renderer State
In `src/viewer/opengl_renderer.py`, `render()` method:
```python
if self._num_instances == 0:
    print("WARNING: No instances to render!")
    return
```

## Files to Check

1. **`tools/streaming_viewer_main.py`** (line 86-145)
   - Main render loop
   - Data flow from viewer to renderer

2. **`src/viewer/streaming_viewer.py`** (line 485-520)
   - `get_visible_surfels()` - Returns data for rendering
   - `update_visible_chunks()` - Determines what to load

3. **`src/viewer/opengl_renderer.py`** (line 557-620)
   - `_render_points()` - Point rendering code
   - `upload_surfels()` - GPU upload code

4. **`src/viewer/streaming_viewer.py`** (line 380-402)
   - `_find_visible_nodes()` - Frustum culling logic

## Working Reference

Compare with `test_minimal_viewer.py` which works:
- Simple VBO setup (no instancing)
- Direct attribute binding
- Simple shader

The main viewer uses instanced rendering which is more complex.

## Quick Fixes to Try

1. **Disable frustum culling:**
   ```python
   # In _find_visible_nodes(), always return True
   if True:  # Always visible
       self._visible_nodes.add(node.node_id)
   ```

2. **Force load all data:**
   ```python
   # In get_visible_surfels(), load root directly
   root_surfels = self.load_chunk("root", block=True)
   return root_surfels
   ```

3. **Increase point size:**
   ```python
   renderer._point_size = 20.0  # Make points bigger
   ```

4. **Check camera position:**
   ```python
   print(f"Camera: {renderer.camera_position}, Target: {renderer.camera_target}")
   ```

## Expected Console Output

When working, you should see:
```
Loading data/output/sample_output.ply...
Building octree from data/output/sample_output.ply...
Octree built in 0.01s
DEBUG Frame 0: 30 visible surfels
  Uploaded 30 surfels to GPU
```

If you see "0 visible surfels", the octree isn't loading data.

## Next Steps

1. Add debug prints to trace data flow
2. Check if `get_visible_surfels()` returns empty dict
3. Verify octree chunks are being created (check `data/output/sample_output.2dgs_octree/` directory)