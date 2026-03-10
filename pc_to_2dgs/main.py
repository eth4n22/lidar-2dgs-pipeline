#!/usr/bin/env python3
"""
PC to 2DGS Converter - Main Entry Point

Usage:
    python main.py                    # Convert first .txt file in data/input/ to output.ply
    python main.py myfile.txt         # Convert myfile.txt to output.ply
    python main.py myfile.txt out.ply # Custom input and output
    python main.py --help             # Show all options
"""

import sys
from pathlib import Path

# Get script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Ensure we can import from src
sys.path.insert(0, str(SCRIPT_DIR))

# Import and run the CLI
from tools.txt_to_2dgs import main

def find_first_txt():
    """Find the first .txt file in data/input/"""
    input_dir = SCRIPT_DIR / 'data' / 'input'
    for f in input_dir.glob('*.txt'):
        return str(f)
    return None

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Use first .txt file found
        txt_file = find_first_txt()
        if txt_file:
            sys.argv = ['main.py', '-i', txt_file, 
                        '-o', str(SCRIPT_DIR / 'data' / 'output')]
        else:
            print("Error: No .txt files found in data/input/")
            sys.exit(1)
    elif len(sys.argv) == 2:
        # Assume it's a filename in data/input/
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            # Try looking in data/input/
            input_path = SCRIPT_DIR / 'data' / 'input' / sys.argv[1]
        sys.argv = ['main.py', '-i', str(input_path), 
                    '-o', str(SCRIPT_DIR / 'data' / 'output')]
    elif len(sys.argv) == 3:
        sys.argv = ['main.py', '-i', sys.argv[1], '-o', sys.argv[2]]
    
    sys.exit(main())
