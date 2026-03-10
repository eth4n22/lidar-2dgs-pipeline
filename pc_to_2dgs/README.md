# pc_to_2dgs

A Python tool to convert LiDAR point clouds (XYZRGB TXT format) into 2D Gaussian surfel (2DGS-style) PLY files with both CLI and GUI interfaces.

## Overview

This project converts LiDAR point cloud data stored as plain text files into a format suitable for 2D Gaussian Splatting (2DGS) rendering. The output is a PLY file containing Gaussian surfel attributes (position, normal, tangent, bitangent, opacity, scale, rotation, color).

## Input Format

**TXT Point Cloud File:**

- One point per line
- Format: `x y z r g b` (space-separated)
- All values are floats

Example:

```
0.0 0.0 0.0 255 0 0
1.0 0.5 0.2 0 255 128
-0.5 1.2 0.8 128 0 255
```

## Output Format

**PLY Surfel File:**

- PLY format (ASCII or binary)
- 23 float properties per vertex:
  - Position: `x, y, z`
  - Normal: `nx, ny, nz`
  - Tangent: `tx, ty, tz`
  - Bitangent: `bx, by, bz`
  - Opacity: `opacity`
  - Scale: `sx, sy, sz`
  - Rotation (quaternion): `rx, ry, rz, rw`
  - Color: `red, green, blue`

## Pipeline Stages

1. **Parse** - Load TXT file, validate format
2. **Preprocess** - Remove outliers, voxel downsampling
3. **Normals** - Estimate surface normals via KNN
4. **Surfels** - Build Gaussian surfels from points and normals
5. **Export** - Write PLY file (ASCII or binary)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI

Launch the graphical interface for easy conversion:

```bash
python gui.py
```

**GUI Controls:**

- Load a TXT point cloud file
- Adjust preprocessing parameters (outlier removal, voxel size)
- Set KNN parameters for normal estimation
- Choose output location and format (ASCII/binary)
- Click "Start Processing"

### CLI

```bash
# Basic conversion
python tools/txt_to_2dgs.py --input data/input/points.txt --output data/output/surfels.ply

# With voxel downsampling (voxel_size in meters)
python tools/txt_to_2dgs.py -i data/input/points.txt -o out.ply --voxel 0.05

# With KNN normal estimation
python tools/txt_to_2dgs.py -i in.txt -o out.ply --knn_normals --k_neighbors 20

# Binary PLY output (smaller file)
python tools/txt_to_2dgs.py -i in.txt -o out.ply --binary
```

### Help

```bash
python tools/txt_to_2dgs.py --help
```

## Viewer

Use the built-in viewer to inspect the output:

```bash
python viewer.py
```

**Viewer Controls:**

- **Left-drag**: Rotate view
- **Scroll**: Zoom in/out
- **A**: Toggle XYZ axis visibility
- **P**: Toggle between 2DGS surfels and original point cloud
- **Q**: Quit

You can also specify a PLY file and optional TXT file for comparison:

```bash
python tools/viewer.py data/output/myfile.ply --txt data/input/myfile.txt
```

## Project Structure

```
pc_to_2dgs/
├── README.md
├── requirements.txt
├── .gitignore
├── gui.py                    # GUI application
├── main.py                   # Simple CLI entrypoint
├── viewer.py                 # Quick viewer
├── HDB4.png                 # GUI logo
├── data/
│   ├── input/
│   │   ├── auditorium_1.txt  # Example data
│   │   ├── hallway_1.txt
│   │   └── lounge_1.txt
│   └── output/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── txt_io.py            # TXT file I/O
│   ├── preprocess.py         # Outlier removal, voxel downsampling
│   ├── normals.py           # Normal estimation
│   ├── surfels.py           # Surfel construction
│   └── export_ply.py        # PLY export
├── tools/
│   ├── txt_to_2dgs.py       # CLI entrypoint
│   └── viewer.py            # Advanced viewer
└── test_*.py               # Test suite
```

## Requirements

- Python 3.10+
- numpy
- scipy
- open3d (for viewer)
- tkinter (for GUI, included with Python)

## What is 2DGS?

2D Gaussian Splatting represents 3D surfaces as 2D Gaussians (ellipses) instead of raw points. This provides:

- **Smooth blending** - Gaussians overlap and blend smoothly
- **Differentiable rendering** - Can optimize parameters to fit observations
- **Oriented surfaces** - Tangent/bitangent define local surface orientation
- **Neural rendering** - Used in 3DGS/NeRF systems

The output PLY contains all the Gaussian attributes (normals, tangents, scales, rotations) needed for proper 2DGS rendering. Basic viewers like Open3D can only display the position/color components.

## License

MIT
