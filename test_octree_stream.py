#!/usr/bin/env python3
"""Test streaming octree conversion."""
import sys
sys.path.insert(0, 'pc_to_2dgs')

from src.octree_io import convert_ply_to_octree

# Test streaming octree conversion with a small PLY file
print('Testing streaming octree conversion...', file=sys.stderr)
try:
    output = convert_ply_to_octree('pc_to_2dgs/data/output/hallway_4_2dgs.ply', output_dir='pc_to_2dgs/data/output/test_stream_octree')
    print(f'Output: {output}', file=sys.stderr)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
