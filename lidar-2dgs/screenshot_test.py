#!/usr/bin/env python3
"""
Simple test to save a screenshot of the rendered scene.
This helps verify if rendering is working even if the window isn't displaying properly.
"""
import glfw
import OpenGL.GL as gl
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.export_ply import read_ply
from src.viewer.opengl_renderer import GaussianSplatRenderer

def save_screenshot(window, filename="screenshot.png"):
    """Save the current OpenGL render to a PNG file."""
    width, height = glfw.get_window_size(window)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename)
    print(f"Screenshot saved to {filename}")
    return image

def main():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Request OpenGL 3.3 Core Profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)  # 4x MSAA

    # Create window
    width, height = 1280, 720
    window = glfw.create_window(width, height, "Screenshot Test", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)

    print(f"OpenGL Renderer: {gl.glGetString(gl.GL_RENDERER)}")
    print(f"OpenGL Version: {gl.glGetString(gl.GL_VERSION)}")

    # Load and render
    ply_path = "data/output/office_1_2dgs.ply"
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found")
        glfw.terminate()
        return

    print(f"Loading {ply_path}...")
    surfels = read_ply(ply_path)
    num_surfels = len(surfels['position'])
    print(f"Loaded {num_surfels} surfels")

    renderer = GaussianSplatRenderer(width=width, height=height, title="Screenshot Test")
    renderer.upload_surfels(surfels)

    # Set initial camera view
    positions = surfels['position']
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2

    renderer.camera_position = np.array([bbox_center[0], bbox_center[1] + 20, bbox_center[2] + 20], dtype=np.float32)
    renderer.camera_target = np.array(bbox_center, dtype=np.float32)

    print("Rendering frame...")
    renderer.render()

    print("Saving screenshot...")
    save_screenshot(window, "viewer_screenshot.png")

    # Render a few more frames to show animation
    for i in range(10):
        glfw.poll_events()
        if glfw.window_should_close(window):
            break
        renderer.render()
        glfw.swap_buffers(window)

    print("Done! Check viewer_screenshot.png")

    # Also save a raw numpy array for debugging
    width, height = glfw.get_window_size(window)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    print(f"Image data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Min value: {data.min()}, Max value: {data.max()}")

    glfw.terminate()

if __name__ == "__main__":
    main()
