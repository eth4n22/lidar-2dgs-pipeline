#!/usr/bin/env python3
"""
Test Point Rendering - Draw random colored points to test instanced rendering.
"""
import sys
import time
import ctypes
import numpy as np

try:
    import glfw
    import OpenGL.GL as gl
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)


def main():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to init GLFW")
        return
    
    window = glfw.create_window(800, 600, "Point Test", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    print(f"Renderer: {gl.glGetString(gl.GL_RENDERER)}")
    print(f"Version: {gl.glGetString(gl.GL_VERSION)}")
    
    # Resize callback
    def window_size_callback(window, width, height):
        gl.glViewport(0, 0, width, height)
    
    glfw.set_window_size_callback(window, window_size_callback)
    
    # Set initial viewport
    width, height = glfw.get_window_size(window)
    gl.glViewport(0, 0, width, height)
    
    # Point vertex shader
    vertex_src = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    
    out vec3 v_color;
    
    void main() {
        gl_Position = vec4(position, 1.0);
        gl_PointSize = 10.0;
        v_color = color;
    }
    """
    
    # Fragment shader
    fragment_src = """
    #version 330 core
    in vec3 v_color;
    out vec4 color;
    
    void main() {
        color = vec4(v_color, 1.0);
    }
    """
    
    # Compile shaders
    def compile_shader(type, src):
        shader = gl.glCreateShader(type)
        gl.glShaderSource(shader, src)
        gl.glCompileShader(shader)
        status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not status:
            print(f"Shader error: {gl.glGetShaderInfoLog(shader)}")
        return shader
    
    vs = compile_shader(gl.GL_VERTEX_SHADER, vertex_src)
    fs = compile_shader(gl.GL_FRAGMENT_SHADER, fragment_src)
    
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    
    # Create point data - 100 random colored points in a 3D box
    n_points = 100
    np.random.seed(42)  # For reproducibility
    positions = np.random.uniform(-0.8, 0.8, (n_points, 3)).astype(np.float32)
    
    # Make first point red and at center, rest are random colors
    positions[0] = [0.0, 0.0, 0.0]  # Red at center
    colors = np.random.uniform(0, 1, (n_points, 3)).astype(np.float32)
    colors[0] = [1.0, 0.0, 0.0]  # Red
    
    print(f"Position range: {positions.min():.3f} to {positions.max():.3f}")
    print(f"Color range: {colors.min():.3f} to {colors.max():.3f}")
    
    # Create VAO/VBO
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    
    gl.glBindVertexArray(vao)
    
    # Position VBO
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, positions, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    
    # Color VBO
    color_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, color_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    
    gl.glBindVertexArray(0)
    
    print(f"Drawing {n_points} points")
    print("You should see 100 colored dots, with a RED dot at the center")
    
    # Enable point size
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    
    # Set initial viewport
    width, height = glfw.get_window_size(window)
    gl.glViewport(0, 0, width, height)
    
    while not glfw.window_should_close(window):
        # Handle resize
        w, h = glfw.get_window_size(window)
        gl.glViewport(0, 0, w, h)
        
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        # Clear to dark blue
        gl.glClearColor(0.1, 0.1, 0.3, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        gl.glUseProgram(program)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, n_points)
        gl.glBindVertexArray(0)
        
        glfw.swap_buffers(window)
        time.sleep(0.016)
    
    glfw.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
