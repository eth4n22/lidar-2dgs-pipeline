#!/usr/bin/env python3
"""Wrapper script to run txt_to_2dgs_large with correct PYTHONPATH."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run the main function
from tools.txt_to_2dgs_large import main

if __name__ == '__main__':
    sys.exit(main())
