#!/usr/bin/env python3
"""
Definitive GPU Draw Test - Tests if OpenGL is actually producing fragments.

This test:
1. Prints GL_VENDOR, GL_RENDERER, GL_VERSION at startup
2. Checks framebuffer status
3. Draws a full-screen magenta triangle
4. Reads back the center pixel to verify drawing worked
"""
import sys
import time
import ctypes

# Add src to path
sys.path.insert(0, '.')

from OpenGL.GL import *
from OpenGL.GL import shaders
from src.viewer.opengl_renderer import create_gl_context

# Vertex shader - hardcoded positions, no attributes needed
VERTEX_SHADER_SRC = """#version 330 core
void main() {
    // Full-screen triangle with hardcoded positions in NDC
    // Triangle covers entire screen
    if (gl_VertexID == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
    } else if (gl_VertexID == 1) {
        gl_Position = vec4( 3.0, -1.0, 0.0, 1.0);
    } else {
        gl_Position = vec4(-1.0,  3.0, 0.0, 1.0);
    }
}
"""

# Fragment shader - solid magenta
FRAGMENT_SHADER_SRC = """#version 330 core
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 1.0, 1.0);  // Magenta: R=1, G=0, B=1, A=1
}
"""


def print_gl_info():
    """Print OpenGL context information."""
    print("=" * 60)
    print("OPENGL INFO")
    print("=" * 60)
    print(f"GL_VENDOR:   {glGetString(GL_VENDOR).decode()}")
    print(f"GL_RENDERER: {glGetString(GL_RENDERER).decode()}")
    print(f"GL_VERSION:  {glGetString(GL_VERSION).decode()}")
    print(f"GL_SHADING_LANGUAGE_VERSION: {glGetString(GL_SHADING_LANGUAGE_VERSION).decode()}")
    
    # Check extensions
    num_exts = glGetIntegerv(GL_NUM_EXTENSIONS)
    print(f"GL_NUM_EXTENSIONS: {num_exts}")
    
    # Framebuffer status
    fbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    print(f"GL_FRAMEBUFFER_STATUS: {fbo_status}")
    if fbo_status == GL_FRAMEBUFFER_COMPLETE:
        print("  -> Framebuffer is COMPLETE")
    else:
        print(f"  -> Framebuffer INCOMPLETE: 0x{fbo_status:x}")
    
    print("=" * 60)


def run_gpu_test(width: int = 640, height: int = 480) -> bool:
    """Run the GPU draw test."""
    print(f"\nCreating OpenGL context ({width}x{height})...")
    
    # Create context using the renderer's method
    window, gl_context = create_gl_context(width=width, height=height, title="GPU Draw Test")
    
    if not window:
        print("ERROR: Failed to create window/context")
        return False
    
    print("Context created successfully!")
    
    # Print GL info
    print_gl_info()
    
    # Compile shaders
    print("\nCompiling shaders...")
    try:
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, VERTEX_SHADER_SRC)
        glCompileShader(vertex_shader)
        
        # Check vertex shader compilation
        status = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(vertex_shader)
            print(f"Vertex shader error: {log}")
            return False
        print("Vertex shader compiled successfully")
        
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, FRAGMENT_SHADER_SRC)
        glCompileShader(fragment_shader)
        
        # Check fragment shader compilation
        status = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(fragment_shader)
            print(f"Fragment shader error: {log}")
            return False
        print("Fragment shader compiled successfully")
        
        # Link program
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        
        # Check program linking
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(program)
            print(f"Program link error: {log}")
            return False
        print("Shader program linked successfully\n")
        
    except Exception as e:
        print(f"Shader compilation error: {e}")
        return False
    
    # Create VAO (required for OpenGL 3.3+)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    # Main render loop
    print("Starting render loop...")
    print("If you see MAGENTA triangle on screen, GPU drawing WORKS.")
    print("If you see BLACK screen, GPU drawing FAILS.")
    print("Press ESC to exit.\n")
    
    frame_count = 0
    test_passed = False
    
    while not gl_context.should_close:
        # Poll events
        gl_context.poll_events()
        
        # Check for ESC key
        if gl_context.get_key('escape'):
            print("ESC pressed, exiting...")
            break
        
        # Clear to black
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Disable depth test, culling, blending for this test
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        
        # Use our shader program
        glUseProgram(program)
        
        # Draw full-screen triangle (3 vertices, no VAO attributes needed)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        
        # On frame 5, read back center pixel
        if frame_count == 5:
            print("\n" + "=" * 60)
            print("READING CENTER PIXEL (glReadPixels)")
            print("=" * 60)
            
            x = width // 2
            y = height // 2
            
            # Read RGBA pixels
            pixel_data = glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
            
            # pixel_data is a bytes object, convert to tuple
            if isinstance(pixel_data, bytes):
                r, g, b, a = tuple(pixel_data)
            else:
                r, g, b, a = tuple(pixel_data)
            
            print(f"glReadPixels at ({x}, {y}):")
            print(f"  R={r}, G={g}, B={b}, A={a}")
            print(f"  Hex: 0x{r:02x}{g:02x}{b:02x}{a:02x}")
            
            # Check if magenta (R=255, G=0, B=255, A=255)
            if r > 200 and g < 50 and b > 200 and a > 200:
                print("\n*** TEST PASSED: Got expected MAGENTA pixel! ***")
                print("GPU IS PRODUCING FRAGMENTS CORRECTLY!")
                test_passed = True
            else:
                print(f"\n*** TEST FAILED: Expected magenta (255,0,255,255) ***")
                print(f"Got ({r}, {g}, {b}, {a}) instead.")
                if r < 50 and g < 50 and b < 50 and a > 200:
                    print("This looks like BLACK - GPU is NOT producing fragments!")
                elif r == g == b:
                    print("This looks like GRAYSCALE - possible color buffer issue!")
            
            print("=" * 60 + "\n")
        
        # Swap buffers
        gl_context.swap_buffers()
        
        frame_count += 1
        
        # Exit after 3 seconds if test passed
        if test_passed and frame_count > 180:
            print("Test passed, exiting after 3 seconds...")
            break
    
    # Cleanup
    glDeleteProgram(program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glDeleteVertexArrays(1, [vao])
    
    gl_context.close()
    
    print("\n" + "=" * 60)
    if test_passed:
        print("GPU DRAW TEST: PASSED")
        print("Your GPU is working correctly!")
        print("The issue is in the 2DGS pipeline, not the GPU.")
    else:
        print("GPU DRAW TEST: FAILED")
        print("Your GPU is NOT producing fragments properly!")
        print("This is an environmental/driver issue.")
    print("=" * 60)
    
    return test_passed


if __name__ == "__main__":
    print("=" * 60)
    print("DEFINITIVE GPU DRAW TEST")
    print("=" * 60)
    print("This test will draw a solid magenta full-screen triangle")
    print("and verify if fragments are actually being produced.")
    print()
    
    success = run_gpu_test(width=640, height=480)
    
    sys.exit(0 if success else 1)
