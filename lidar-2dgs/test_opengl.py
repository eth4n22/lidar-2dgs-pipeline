#!/usr/bin/env python3
"""
Simple OpenGL Test - Draws a colored triangle and text on screen.
"""
import sys
import time

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
    
    # Create window
    window = glfw.create_window(800, 600, "OpenGL Test", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    print(f"Renderer: {gl.glGetString(gl.GL_RENDERER)}")
    print(f"Version: {gl.glGetString(gl.GL_VERSION)}")
    
    # Simple vertex shader
    vs = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    out vec3 v_color;
    void main() {
        gl_Position = vec4(position, 1.0);
        v_color = color;
    }
    """
    
    # Simple fragment shader
    fs = """
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
    
    vs_id = compile_shader(gl.GL_VERTEX_SHADER, vs)
    fs_id = compile_shader(gl.GL_FRAGMENT_SHADER, fs)
    
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs_id)
    gl.glAttachShader(program, fs_id)
    gl.glLinkProgram(program)
    
    # Triangle vertices (position, color)
    vertices = [
        # position      # color
        -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,  # red bottom-left
         0.5, -0.5, 0.0,  0.0, 1.0, 0.0,  # green bottom-right
         0.0,  0.5, 0.0,  0.0, 0.0, 1.0,  # blue top
    ]
    vertices = (gl.GLfloat * len(vertices))(*vertices)
    
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    
    # Position attribute
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * 4, None)
    
    # Color attribute
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    
    print("\nYou should see a colorful triangle (red, green, blue).")
    print("If you see this, your OpenGL is working!")
    print("Press ESC to exit.\n")
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        gl.glClearColor(0.1, 0.1, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(program)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        
        glfw.swap_buffers(window)
        time.sleep(0.016)
    
    glfw.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
