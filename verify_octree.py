#!/usr/bin/env python3
"""Verify streamed octree output matches original."""
import sys
sys.path.insert(0, 'pc_to_2dgs')

from src.octree_io import load_chunk
import numpy as np

# Load first chunk from both original and streamed octree
print('Loading chunks...', file=sys.stderr)
orig = load_chunk('pc_to_2dgs/data/output/hallway_4_2dgs.2dgs_octree/chunk_3_1_0_lod0.bin')
stream = load_chunk('pc_to_2dgs/data/output/test_stream_octree/chunk_3_1_0_lod0.bin')

print('Original octree:', file=sys.stderr)
print(f'  Points: {len(orig["position"])}', file=sys.stderr)
print(f'  Color[0]: {orig["color"][0]}', file=sys.stderr)

print('Streamed octree:', file=sys.stderr)
print(f'  Points: {len(stream["position"])}', file=sys.stderr)  
print(f'  Color[0]: {stream["color"][0]}', file=sys.stderr)

# Check if positions match
if len(orig['position']) > 0 and len(stream['position']) > 0:
    diff = np.abs(orig['position'][:5] - stream['position'][:5])
    print(f'Position diff (first 5): {diff.max():.6f}', file=sys.stderr)
