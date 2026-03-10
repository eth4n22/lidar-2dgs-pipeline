"""Tests for txt_io module."""

import tempfile
import os
import numpy as np
import pytest

from src.txt_io import (
    load_xyzrgb_txt,
    load_xyz_txt,
    save_xyzrgb_txt,
    save_xyz_txt,
    detect_format,
    validate_format
)


class TestLoadXyzrgbTxt:
    """Tests for load_xyzrgb_txt function."""

    def test_load_valid_xyzrgb_file(self):
        """Test loading a valid XYZRGB file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0 255 0 0\n")
            f.write("4.0 5.0 6.0 0 255 0\n")
            f.write("7.0 8.0 9.0 0 0 255\n")
            filepath = f.name

        try:
            result = load_xyzrgb_txt(filepath)

            assert "position" in result
            assert "color" in result
            assert result["position"].shape == (3, 3)
            assert result["color"].shape == (3, 3)
            np.testing.assert_array_almost_equal(
                result["position"][0], [1.0, 2.0, 3.0]
            )
            np.testing.assert_array_almost_equal(
                result["color"][0], [255.0, 0.0, 0.0]
            )
        finally:
            os.unlink(filepath)

    def test_load_file_with_comments(self):
        """Test loading file with # comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("1.0 2.0 3.0 255 0 0\n")
            f.write("# Another comment\n")
            f.write("4.0 5.0 6.0 0 255 0\n")
            filepath = f.name

        try:
            result = load_xyzrgb_txt(filepath)
            assert result["position"].shape == (2, 3)
        finally:
            os.unlink(filepath)

    def test_load_file_with_empty_lines(self):
        """Test loading file with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("\n")
            f.write("1.0 2.0 3.0 255 0 0\n")
            f.write("\n")
            f.write("4.0 5.0 6.0 0 255 0\n")
            f.write("\n")
            filepath = f.name

        try:
            result = load_xyzrgb_txt(filepath)
            assert result["position"].shape == (2, 3)
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_xyzrgb_txt("/nonexistent/path/file.txt")

    def test_load_invalid_format(self):
        """Test that ValueError is raised for invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0\n")  # Only 2 values, need at least 3
            filepath = f.name

        try:
            with pytest.raises(ValueError):
                load_xyzrgb_txt(filepath)
        finally:
            os.unlink(filepath)

    def test_load_grayscale_format(self):
        """Test loading file with grayscale intensity."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0 128\n")
            filepath = f.name

        try:
            result = load_xyzrgb_txt(filepath)
            np.testing.assert_array_almost_equal(
                result["color"][0], [128.0, 128.0, 128.0]
            )
        finally:
            os.unlink(filepath)

    def test_load_no_colors(self):
        """Test loading file without colors - defaults to white."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            filepath = f.name

        try:
            result = load_xyzrgb_txt(filepath)
            np.testing.assert_array_almost_equal(
                result["color"][0], [255.0, 255.0, 255.0]
            )
        finally:
            os.unlink(filepath)


class TestLoadXyzTxt:
    """Tests for load_xyz_txt function."""

    def test_load_xyz_only(self):
        """Test loading XYZ-only file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            filepath = f.name

        try:
            result = load_xyz_txt(filepath)
            assert "position" in result
            assert "color" not in result
            assert result["position"].shape == (2, 3)
        finally:
            os.unlink(filepath)


class TestSaveXyzrgbTxt:
    """Tests for save_xyzrgb_txt function."""

    def test_save_and_load_roundtrip(self):
        """Test that saving and loading preserves data."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filepath = f.name

        try:
            save_xyzrgb_txt(filepath, points, colors)
            result = load_xyzrgb_txt(filepath)

            np.testing.assert_array_almost_equal(result["position"], points)
            np.testing.assert_array_almost_equal(result["color"], colors)
        finally:
            os.unlink(filepath)


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_detect_xyzrgb_format(self):
        """Test detection of XYZRGB format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0 255 0 0\n")
            f.write("4.0 5.0 6.0 0 255 0\n")
            filepath = f.name

        try:
            fmt = detect_format(filepath)
            assert fmt == "xyzrgb"
        finally:
            os.unlink(filepath)

    def test_detect_xyz_format(self):
        """Test detection of XYZ-only format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            filepath = f.name

        try:
            fmt = detect_format(filepath)
            assert fmt == "xyz"
        finally:
            os.unlink(filepath)


class TestValidateFormat:
    """Tests for validate_format function."""

    def test_validate_valid_format(self):
        """Test validation of valid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0 255 0 0\n")
            f.write("4.0 5.0 6.0 0 255 0\n")
            filepath = f.name

        try:
            is_valid, msg = validate_format(filepath)
            assert is_valid is True
        finally:
            os.unlink(filepath)

    def test_validate_invalid_format(self):
        """Test validation of invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0\n")  # Only 2 values
            filepath = f.name

        try:
            is_valid, msg = validate_format(filepath)
            assert is_valid is False
        finally:
            os.unlink(filepath)
