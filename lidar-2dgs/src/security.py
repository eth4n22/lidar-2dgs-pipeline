"""
Security utilities for file path validation and sanitization.
"""

import os
from pathlib import Path
from typing import Optional


def sanitize_file_path(filepath: str, base_dir: Optional[str] = None) -> Path:
    """
    Sanitize and validate file paths to prevent directory traversal attacks.
    
    Args:
        filepath: Input file path
        base_dir: Optional base directory to restrict paths to
    
    Returns:
        Path: Sanitized Path object
    
    Raises:
        ValueError: If path is invalid or contains dangerous patterns
    """
    path = Path(filepath).resolve()
    
    # Check for dangerous patterns
    path_str = str(path)
    if '..' in path_str or path_str.startswith('~'):
        raise ValueError(f"Invalid path: contains '..' or '~' (potential directory traversal)")
    
    # Restrict to base directory if provided
    if base_dir:
        base = Path(base_dir).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            raise ValueError(f"Path {path} is outside allowed base directory {base}")
    
    return path


def validate_file_path(filepath: str, must_exist: bool = False) -> Path:
    """
    Validate file path and check existence if required.
    
    Args:
        filepath: File path to validate
        must_exist: If True, file must exist
    
    Returns:
        Path: Validated Path object
    
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("File path must be a non-empty string")
    
    # Remove any null bytes
    filepath = filepath.replace('\x00', '')
    
    path = Path(filepath)
    
    # Check for absolute path limits (prevent accessing system directories)
    if path.is_absolute():
        # On Windows, prevent access to system directories
        if os.name == 'nt':
            system_dirs = ['C:\\Windows', 'C:\\System32', 'C:\\Program Files']
            if any(str(path).startswith(d) for d in system_dirs):
                raise ValueError(f"Access to system directory denied: {path}")
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return path


def safe_file_write(filepath: str, content: bytes, base_dir: Optional[str] = None) -> Path:
    """
    Safely write to a file with path validation.
    
    Args:
        filepath: Output file path
        content: Content to write (bytes)
        base_dir: Optional base directory restriction
    
    Returns:
        Path: Path to written file
    """
    path = sanitize_file_path(filepath, base_dir)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    path.write_bytes(content)
    
    return path
