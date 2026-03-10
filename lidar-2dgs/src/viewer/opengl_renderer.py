"""
OpenGL Renderer for 2DGS Surfels - True Gaussian Splatting

Implements efficient OpenGL rendering with:
- True Gaussian splat shader (not just points)
- Instanced rendering for performance
- 2D covariance computation in shader
- Alpha blending for smooth surfaces
"""

import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import ctypes

# OpenGL imports (optional - fall back to software rendering)
try:
    import OpenGL.GL as gl
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


def _check_gl_error():
    """Check for OpenGL errors and print if any."""
    if not HAS_OPENGL:
        return
    while True:
        error = gl.glGetError()
        if error == gl.GL_NO_ERROR:
            break
        error_map = {
            gl.GL_INVALID_ENUM: "GL_INVALID_ENUM",
            gl.GL_INVALID_VALUE: "GL_INVALID_VALUE",
            gl.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            gl.GL_STACK_OVERFLOW: "GL_STACK_OVERFLOW",
            gl.GL_STACK_UNDERFLOW: "GL_STACK_UNDERFLOW",
            gl.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
            gl.GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
        }
        error_name = error_map.get(error, f"Unknown({error})")
        print(f"  GL Error: {error_name}")

try:
    import glfw
    HAS_GLFW = True
except ImportError:
    HAS_GLFW = False


# Gaussian Splat Vertex Shader
# Renders surfels as textured quads (billboards) with Gaussian falloff
GAUSSIAN_SPLAT_VERTEX_SHADER = """
#version 330 core

// Instance attributes
layout(location = 0) in vec3 a_position;      // Surfel center position
layout(location = 1) in vec3 a_normal;         // Surface normal
layout(location = 2) in vec3 a_scale;          // [sigma_tangent, sigma_tangent, sigma_normal]
layout(location = 3) in vec4 a_rotation;       // Quaternion (x, y, z, w)
layout(location = 4) in vec3 a_color;           // RGB color

// Vertex attributes (for the quad)
layout(location = 5) in vec2 a_quad_offset;    // Quad corner offset (-1 to 1)

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_opacity;

// Outputs to fragment shader
out vec3 v_position;       // World position of splat
out vec3 v_normal;         // Surface normal
out vec3 v_color;          // RGB color
out vec2 v_uv;             // UV coordinates (-1 to 1)
out vec2 v_scale;          // Tangent scale for Gaussian
out float v_opacity;        // Per-splat opacity

// Helper: Multiply quaternion by vector
vec3 rotate_vector(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    v_normal = a_normal;
    v_color = a_color;
    v_opacity = u_opacity;
    
    // Compute tangent and bitangent from quaternion
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(a_normal, up)) > 0.99) {
        up = vec3(1.0, 0.0, 0.0);
    }
    vec3 tangent = normalize(cross(a_normal, up));
    vec3 bitangent = normalize(cross(a_normal, tangent));
    
    // Rotate tangent directions by quaternion
    tangent = rotate_vector(a_rotation, tangent);
    bitangent = rotate_vector(a_rotation, bitangent);
    
    // Scale factors for Gaussian
    v_scale = a_scale.xy;  // Tangent plane scale
    
    // Compute corner offset in world space
    vec3 corner_offset = (tangent * a_quad_offset.x * a_scale.x * 2.0 +
                         bitangent * a_quad_offset.y * a_scale.y * 2.0);
    
    // World position of this corner
    vec3 world_pos = a_position + corner_offset;
    v_position = world_pos;
    v_uv = a_quad_offset;
    
    // Project to clip space
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

# Gaussian Splat Fragment Shader
# Computes 2D Gaussian falloff for smooth, continuous surfaces
GAUSSIAN_SPLAT_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in vec2 v_scale;
in float v_opacity;

uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;

out vec4 frag_color;

void main() {
    // Compute 2D Gaussian falloff
    // v_uv is in [-1, 1] range
    float x = v_uv.x;
    float y = v_uv.y;
    
    // Normalize by scale for proper Gaussian
    float ux = x / max(v_scale.x, 0.001);
    float uy = y / max(v_scale.y, 0.001);
    
    // 2D Gaussian: exp(-(x² + y²))
    float gaussian = exp(-0.5 * (ux * ux + uy * uy));
    
    // Discard pixels with very low contribution (optimization)
    if (gaussian < 0.01) {
        discard;
    }
    
    // Lighting calculation
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_dir);
    vec3 view = normalize(u_camera_pos - v_position);
    vec3 half_vec = normalize(light + view);
    
    // Phong shading with Gaussian intensity
    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(normal, half_vec), 0.0), 32.0);
    
    // Combine lighting with Gaussian
    vec3 final_color = v_color * (0.4 + 0.6 * diffuse) + vec3(0.2) * specular;
    
    // Apply Gaussian alpha
    float alpha = v_opacity * gaussian;
    
    // Output premultiplied alpha for proper blending
    frag_color = vec4(final_color * alpha, alpha);
}
"""

# Point Shader (fallback for very large datasets)
POINT_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 4) in vec3 a_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;

out vec3 v_color;

void main() {
    v_color = a_color;
    vec4 clip_pos = u_projection * u_view * vec4(a_position, 1.0);
    gl_Position = clip_pos;
    gl_PointSize = u_point_size * (1.0 / clip_pos.w);
}
"""

POINT_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_color;
uniform float u_opacity;

out vec4 frag_color;

void main() {
    // Circular point
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Gaussian falloff for point
    float alpha = exp(-dist * dist * 8.0) * u_opacity;
    
    frag_color = vec4(v_color * alpha, alpha);
}
"""


class GaussianSplatRenderer:
    """
    True Gaussian Splat Renderer for 2DGS Surfels.
    
    Renders surfels as textured quads with Gaussian falloff,
    producing continuous surfaces instead of point clouds.
    """
    
    def __init__(self, width: int = 1280, height: int = 720,
                 title: str = "2DGS Gaussian Splat Viewer"):
        """
        Initialize Gaussian splat renderer.
        
        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title
        
        if not HAS_GLFW or not HAS_OPENGL:
            print("WARNING: OpenGL/GLFW not available, using software fallback")
            self._use_opengl = False
            return
        
        self._use_opengl = True
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Configure OpenGL context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # MSAA
        
        # Create window
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        
        glfw.make_context_current(self.window)
        
        # Enable features for proper splat rendering
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        
        # Compile shaders
        self._splat_program = self._create_shader_program(
            GAUSSIAN_SPLAT_VERTEX_SHADER, 
            GAUSSIAN_SPLAT_FRAGMENT_SHADER
        )
        self._point_program = self._create_shader_program(
            POINT_VERTEX_SHADER, 
            POINT_FRAGMENT_SHADER
        )
        
        # Create quad geometry for splats (shared by all instances)
        self._quad_vao, self._quad_vbo = self._create_quad()
        
        # Camera state
        self._camera_pos = np.array([0.0, 0.0, 10.0])
        self._camera_target = np.array([0.0, 0.0, 0.0])
        self._camera_up = np.array([0.0, 1.0, 0.0])
        self._fov = 45.0
        
        # Rendering state
        self._opacity = 0.8
        self._point_size = 10.0  # Increased for visibility
        self._render_mode = 'splats'  # 'splats' or 'points'
        self._light_dir = np.array([1.0, 1.0, 1.0])
        
        # Instance data
        self._instance_vao = None
        self._instance_vbo = None
        self._num_instances = 0
        
        # Statistics
        self._stats = {
            'fps': 0.0,
            'frame_time': 0.0,
            'draw_calls': 0
        }
        self._last_frame_time = time.time()
        self._frame_count = 0
    
    def _create_shader_program(self, vertex_src: str, fragment_src: str) -> int:
        """Compile and link shader program."""
        # Vertex shader
        vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader, vertex_src)
        gl.glCompileShader(vertex_shader)
        
        if gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            error_msg = gl.glGetShaderInfoLog(vertex_shader).decode('utf-8', errors='replace')
            print(f"  ERROR: Vertex shader compilation failed: {error_msg}")
            return 0
        
        # Fragment shader
        fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader, fragment_src)
        gl.glCompileShader(fragment_shader)
        
        if gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
            error_msg = gl.glGetShaderInfoLog(fragment_shader).decode('utf-8', errors='replace')
            print(f"  ERROR: Fragment shader compilation failed: {error_msg}")
            return 0
        
        # Link program
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            error_msg = gl.glGetProgramInfoLog(program).decode('utf-8', errors='replace')
            print(f"  ERROR: Program linking failed: {error_msg}")
            return 0
        
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        # Print attribute locations
        print(f"  SHADER: Program {program} created successfully")
        print(f"  ATTR locs: pos={gl.glGetAttribLocation(program, b'a_position')}, "
              f"norm={gl.glGetAttribLocation(program, b'a_normal')}, "
              f"scale={gl.glGetAttribLocation(program, b'a_scale')}, "
              f"rot={gl.glGetAttribLocation(program, b'a_rotation')}, "
              f"color={gl.glGetAttribLocation(program, b'a_color')}")
        
        _check_gl_error()
        return program
    
    def _create_quad(self) -> Tuple[int, int]:
        """Create quad geometry for splat rendering."""
        # Quad corners (-1 to 1)
        quad_vertices = np.array([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            quad_vertices.nbytes,
            quad_vertices.flatten(),
            gl.GL_STATIC_DRAW
        )
        
        # Quad offset (location 5)
        gl.glEnableVertexAttribArray(5)
        gl.glVertexAttribPointer(5, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        gl.glBindVertexArray(0)
        
        return vao, vbo
    
    def upload_surfels(self, surfels: Dict[str, np.ndarray]) -> None:
        """
        Upload surfel data to GPU for instanced rendering.
        
        Args:
            surfels: Dictionary of surfel arrays
        """
        if not self._use_opengl:
            self._cpu_surfels = surfels
            self._num_instances = len(surfels['position'])
            return
        
        self._num_instances = len(surfels['position'])
        
        # Validate required keys
        required_keys = ['position', 'normal', 'scale', 'rotation', 'color']
        for key in required_keys:
            if key not in surfels:
                raise ValueError(f"Missing required key in surfels: {key}")
        
        # Validate data for NaN/Inf
        for key in required_keys:
            data = surfels[key]
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: {key} contains NaN/Inf values (NaN: {nan_count}, Inf: {inf_count})")
        
        # Create interleaved instance data
        # Format: position(3) + normal(3) + scale(3) + rotation(4) + color(3)
        instance_data = np.zeros((self._num_instances, 16), dtype=np.float32)
        
        instance_data[:, 0:3] = surfels['position']  # position
        instance_data[:, 3:6] = surfels['normal']    # normal
        instance_data[:, 6:9] = surfels['scale']     # scale
        instance_data[:, 9:13] = surfels['rotation']  # rotation (quaternion)
        
        # Color: check if already in 0-1 range or 0-255 range
        colors = surfels['color']
        if colors.max() > 1.0:
            # Colors are in 0-255 range, normalize to 0-1
            instance_data[:, 13:16] = colors.astype(np.float32) / 255.0
        else:
            # Colors already in 0-1 range
            instance_data[:, 13:16] = colors.astype(np.float32)
        
        # Validate final data
        nan_count = np.isnan(instance_data).sum()
        if nan_count > 0:
            print(f"  WARNING: instance_data contains {nan_count} NaN values")
        
        _check_gl_error()
        
        # Create VAO and VBO for instances
        self._instance_vao = gl.glGenVertexArrays(1)
        self._instance_vbo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self._instance_vao)
        
        # Bind quad geometry (shared)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._quad_vbo)
        gl.glEnableVertexAttribArray(5)
        gl.glVertexAttribPointer(5, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        # Bind instance data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            instance_data.nbytes,
            instance_data.flatten(),
            gl.GL_STATIC_DRAW
        )
        
        stride = 16 * 4  # 16 floats * 4 bytes
        
        # Set instance attributes
        # Position (location 0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(0, 1)  # Per-instance
        
        # Normal (location 1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glVertexAttribDivisor(1, 1)
        
        # Scale (location 2)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))
        gl.glVertexAttribDivisor(2, 1)
        
        # Rotation (location 3)
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(36))
        gl.glVertexAttribDivisor(3, 1)
        
        # Color (location 4) - at offset 13*4 = 52 bytes
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(52))
        gl.glVertexAttribDivisor(4, 1)
        
        gl.glBindVertexArray(0)
        
        _check_gl_error()
    
    def _get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix."""
        forward = self._camera_target - self._camera_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, self._camera_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, 0] = right[0]
        view[1, 0] = right[1]
        view[2, 0] = right[2]
        view[0, 1] = up[0]
        view[1, 1] = up[1]
        view[2, 1] = up[2]
        view[0, 2] = -forward[0]
        view[1, 2] = -forward[1]
        view[2, 2] = -forward[2]
        view[0, 3] = -np.dot(right, self._camera_pos)
        view[1, 3] = -np.dot(up, self._camera_pos)
        view[2, 3] = np.dot(forward, self._camera_pos)
        
        return view
    
    def _get_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix."""
        aspect = self.width / self.height
        fov_rad = np.radians(self._fov)
        
        f = 1.0 / np.tan(fov_rad / 2)
        
        near = 0.1
        far = 1000.0
        
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -(far + near) / (far - near), -1],
            [0, 0, -(2 * far * near) / (far - near), 0]
        ], dtype=np.float32)
        
        # Debug: Print near/far
        if self._frame_count % 60 == 0:
            print(f"  PROJ: near={near}, far={far}, aspect={aspect:.2f}, fov={self._fov}")
            print(f"  CAM: pos={self._camera_pos}, target={self._camera_target}")
            dist = np.linalg.norm(self._camera_pos - self._camera_target)
            print(f"  CAM dist to target: {dist:.2f}")
        
        return projection
    
    def _render_debug_test_point(self) -> None:
        """Render a single test point at center to validate pipeline."""
        if self._point_program == 0 or self._num_instances == 0:
            return
        
        program = self._point_program
        gl.glUseProgram(program)
        
        # Create a simple test point at origin
        test_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        test_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Red
        
        # Create a simple VBO for test point
        test_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, test_vbo)
        test_data = np.array([test_pos[0], test_pos[1], test_pos[2], 
                              test_color[0], test_color[1], test_color[2]], dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, test_data.nbytes, test_data, gl.GL_STATIC_DRAW)
        
        # Create VAO for test
        test_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(test_vao)
        
        # Position (loc 0)
        pos_loc = gl.glGetAttribLocation(program, b"a_position")
        if pos_loc >= 0:
            gl.glEnableVertexAttribArray(pos_loc)
            gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, None)
        
        # Color (loc 4) - offset 12 bytes
        color_loc = gl.glGetAttribLocation(program, b"a_color")
        if color_loc >= 0:
            gl.glEnableVertexAttribArray(color_loc)
            gl.glVertexAttribPointer(color_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        
        # Set uniforms
        view = self._get_view_matrix()
        projection = self._get_projection_matrix()
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, b"u_view"), 1, gl.GL_FALSE, view.flatten())
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, b"u_projection"), 1, gl.GL_FALSE, projection.flatten())
        gl.glUniform1f(gl.glGetUniformLocation(program, b"u_opacity"), 1.0)
        gl.glUniform1f(gl.glGetUniformLocation(program, b"u_point_size"), 20.0)
        
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        
        # Disable depth test for test point visibility
        gl.glDisable(gl.GL_DEPTH_TEST)
        
        print(f"  TEST: Drawing single red point at origin")
        gl.glDrawArrays(gl.GL_POINTS, 0, 1)
        
        # Check for errors
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            print(f"  TEST: GL error after draw: {err}")
        
        # Restore depth test
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDisable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        
        # Cleanup
        gl.glBindVertexArray(0)
        gl.glDeleteVertexArrays(1, [test_vao])
        gl.glDeleteBuffers(1, [test_vbo])
    
    def _validate_gl_state(self) -> None:
        """Validate OpenGL state before rendering."""
        if not hasattr(self, '_gl_validated'):
            self._gl_validated = True
            
            # Check viewport
            viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            print(f"  GL: Viewport={viewport}")
            print(f"  GL: Window size={self.width}x{self.height}")
            
            # Check depth state
            depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
            print(f"  GL: DEPTH_TEST enabled={bool(depth_test)}")
            
            # Check clear values (skip if function not available)
            try:
                clear_color = gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE)
                clear_depth = gl.glGetFloatv(gl.GL_DEPTH_CLEAR_VALUE)
                print(f"  GL: Clear color={clear_color}, clear depth={clear_depth}")
                
                # Check point size
                point_size_range = gl.glGetFloatv(gl.GL_POINT_SIZE_RANGE)
                print(f"  GL: Point size range={point_size_range}")
            except (AttributeError, TypeError):
                print("  GL: GetFloatv not available")
            
            # Print enabled vertex attribs for instance VAO
            # Check VAO state (skip if function not available)
            if self._instance_vao:
                gl.glBindVertexArray(self._instance_vao)
                print(f"  GL: Instance VAO={self._instance_vao}")
                try:
                    for i in range(5):
                        enabled = gl.glIsEnabledVertexAttribArray(i)
                        print(f"    Attr {i}: enabled={bool(enabled)}")
                except AttributeError:
                    print("  GL: Vertex attrib array check not available")
                gl.glBindVertexArray(0)
            
            # Check for errors
            _check_gl_error()
    
    def _print_enabled_vertex_attribs(self, vao: int) -> None:
        """Print enabled vertex attributes for a VAO."""
        print(f"  VAO {vao} attributes:")
        gl.glBindVertexArray(vao)
        try:
            for i in range(10):  # Check first 10 attributes
                enabled = gl.glIsEnabledVertexAttribArray(i)
                if enabled:
                    print(f"    Attr {i}: ENABLED")
                else:
                    print(f"    Attr {i}: disabled")
        except AttributeError:
            print("  GL: Vertex attrib array check not available")
        gl.glBindVertexArray(0)
    
    def render(self) -> None:
        """Render the current frame."""
        # Validate GL state once
        self._validate_gl_state()
        
        current_time = time.time()
        self._frame_count += 1
        if current_time - self._last_frame_time >= 1.0:
            self._stats['fps'] = self._frame_count / (current_time - self._last_frame_time)
            self._stats['frame_time'] = 1000.0 / max(self._stats['fps'], 0.001)
            self._frame_count = 0
            self._last_frame_time = current_time
        
        if not self._use_opengl:
            self._render_software()
            return
        
        # Set viewport to window dimensions
        gl.glViewport(0, 0, self.width, self.height)
        
        # Debug: Print viewport
        if self._frame_count == 1:
            viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            print(f"  RENDER: glViewport={viewport}, window={self.width}x{self.height}")
        
        # Clear with explicit depth clear
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Debug: Check depth state
        depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        if self._frame_count == 1:
            print(f"  RENDER: DEPTH_TEST={bool(depth_test)}")
        
        # Debug: Check if we have instances
        if self._num_instances == 0:
            if not hasattr(self, '_warned_no_instances'):
                print(f"  WARNING: No instances to render! _num_instances=0")
                self._warned_no_instances = True
            return
        
        # Debug: Check render mode
        if not hasattr(self, '_render_mode_debug'):
            print(f"  DEBUG: Render mode={self._render_mode}, instances={self._num_instances}")
            self._render_mode_debug = True
        
        # Render splats or points
        if self._render_mode == 'splats':
            self._render_splats()
            
            # If splat rendering failed (no VAO), fallback to test point
            if self._instance_vao is None:
                self._render_debug_test_point()
        else:
            self._render_points()
            
            # If point rendering failed (no point VAO), fallback to test point
            if not hasattr(self, '_point_vao') or self._point_vao is None:
                self._render_debug_test_point()
        
        # Check for errors after rendering
        _check_gl_error()
        
        # Swap buffers
        glfw.swap_buffers(self.window)
    
    def _render_splats(self) -> None:
        """Render as true Gaussian splats (textured quads)."""
        if self._instance_vao is None or self._num_instances == 0:
            # Debug: Print warning on first few frames
            if hasattr(self, '_warned_no_vao'):
                return
            self._warned_no_vao = True
            print(f"  WARNING: No VAO or instances. VAO={self._instance_vao}, instances={self._num_instances}")
            return
        
        program = self._splat_program
        gl.glUseProgram(program)
        
        # Set uniforms
        view = self._get_view_matrix()
        projection = self._get_projection_matrix()
        
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, b"u_view"), 
            1, gl.GL_FALSE, view.flatten()
        )
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, b"u_projection"), 
            1, gl.GL_FALSE, projection.flatten()
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(program, b"u_opacity"), 
            self._opacity
        )
        gl.glUniform3fv(
            gl.glGetUniformLocation(program, b"u_camera_pos"), 
            1, self._camera_pos
        )
        gl.glUniform3fv(
            gl.glGetUniformLocation(program, b"u_light_dir"), 
            1, self._light_dir
        )
        
        # Debug: Print rendering info ONCE
        if not hasattr(self, '_render_debug_printed'):
            print(f"  RENDER: VAO={self._instance_vao}, VBO={getattr(self, '_instance_vbo', None)}, "
                  f"instances={self._num_instances}")
            print(f"  RENDER: Drawing GL_TRIANGLE_STRIP with glDrawArraysInstanced")
            print(f"  RENDER: 4 vertices per quad, {self._num_instances} instances")
            # Print enabled attributes
            self._print_enabled_vertex_attribs(self._instance_vao)
            self._render_debug_printed = True
        
        # Draw instanced quads
        gl.glBindVertexArray(self._instance_vao)
        
        glDrawArraysInstanced = getattr(gl, 'glDrawArraysInstanced', None)
        if glDrawArraysInstanced:
            # Only print on first frame
            if not hasattr(self, '_splat_draw_printed'):
                print(f"  SPLAT: Calling glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, {self._num_instances})")
                self._splat_draw_printed = True
            glDrawArraysInstanced(
                gl.GL_TRIANGLE_STRIP,  # 4 vertices per quad
                0, 4,                   # 4 vertices, N instances
                self._num_instances
            )
        else:
            print("  ERROR: glDrawArraysInstanced not available!")
        
        # Check for errors after draw
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            print(f"  SPLAT: GL error after draw: {err}")
        
        gl.glBindVertexArray(0)
        
        _check_gl_error()
        
        self._stats['draw_calls'] = 1
    
    def _render_points(self) -> None:
        """Render as fallback points."""
        if not hasattr(self, '_point_debug_printed'):
            self._point_debug_printed = False
            self._point_shader_locs = {}  # Store actual attribute locations
        
        if self._instance_vao is None or self._num_instances == 0:
            return
        
        program = self._point_program
        gl.glUseProgram(program)
        
        view = self._get_view_matrix()
        projection = self._get_projection_matrix()
        mvp = projection @ view
        
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, b"u_view"), 
            1, gl.GL_FALSE, view.flatten()
        )
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, b"u_projection"), 
            1, gl.GL_FALSE, projection.flatten()
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(program, b"u_opacity"), 
            self._opacity
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(program, b"u_point_size"), 
            self._point_size
        )
        
        # Enable point sprite rendering
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        
        # Get actual attribute locations from point shader
        pos_loc = gl.glGetAttribLocation(program, b"a_position")
        color_loc = gl.glGetAttribLocation(program, b"a_color")
        
        if not hasattr(self, '_point_shader_validated'):
            print(f"  POINT: Validating shader attributes...")
            print(f"    a_position: loc={pos_loc}")
            print(f"    a_color: loc={color_loc}")
            self._point_shader_validated = True
        
        # Create point-only VAO that only binds attributes the point shader has
        if not hasattr(self, '_point_vao') or self._point_vao is None:
            self._point_vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(self._point_vao)
            
            # Bind instance VBO
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_vbo)
            
            # Position: stride=64 bytes (16 floats), offset=0
            if pos_loc >= 0:
                gl.glEnableVertexAttribArray(pos_loc)
                gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 64, ctypes.c_void_p(0))
                gl.glVertexAttribDivisor(pos_loc, 1)
            
            # Color: stride=64 bytes (16 floats), offset=52 bytes (13 floats * 4 bytes)
            if color_loc >= 0:
                gl.glEnableVertexAttribArray(color_loc)
                gl.glVertexAttribPointer(color_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 64, ctypes.c_void_p(52))
                gl.glVertexAttribDivisor(color_loc, 1)
            
            gl.glBindVertexArray(0)
        
        # Draw instanced points
        if not self._point_debug_printed:
            print(f"  POINT: VAO={self._point_vao}, pos_loc={pos_loc}, color_loc={color_loc}, instances={self._num_instances}")
            print(f"  POINT: point_size={self._point_size}, opacity={self._opacity}")
            print(f"  POINT: Drawing GL_POINTS with glDrawArraysInstanced")
            cam_dist = np.linalg.norm(self._camera_pos - self._camera_target)
            print(f"  POINT: cam_dist={cam_dist:.2f}")
            if hasattr(self, '_data_bbox'):
                bbox_center = (self._data_bbox[:3] + self._data_bbox[3:]) / 2
                dist_to_bbox = np.linalg.norm(self._camera_pos - bbox_center)
                print(f"  POINT: bbox_center={bbox_center}, dist_to_bbox={dist_to_bbox:.2f}")
            # Print enabled attributes
            self._print_enabled_vertex_attribs(self._point_vao)
            self._point_debug_printed = True
        
        print(f"  POINT: Calling glDrawArraysInstanced(GL_POINTS, 0, 1, {self._num_instances})")
        
        gl.glBindVertexArray(self._point_vao)
        gl.glDrawArraysInstanced(gl.GL_POINTS, 0, 1, self._num_instances)
        
        # Check for errors after draw
        err = gl.glGetError()
        if err != gl.GL_NO_ERROR:
            print(f"  POINT: GL error after draw: {err}")
        
        gl.glBindVertexArray(0)
        
        _check_gl_error()
        self._stats['draw_calls'] = 1
        
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDisable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        
        self._stats['draw_calls'] = 1
    
    def _render_software(self) -> None:
        """Software fallback."""
        print(f"Frame: {self._num_instances} splats, FPS: {self._stats['fps']:.1f}")
    
    def should_close(self) -> bool:
        """Check if window should close."""
        if not self._use_opengl:
            return True
        return glfw.window_should_close(self.window)
    
    def poll_events(self) -> None:
        """Poll for input events."""
        if not self._use_opengl:
            return
        glfw.poll_events()
    
    def handle_input(self) -> None:
        """Handle keyboard/mouse input."""
        if not self._use_opengl:
            return
        
        # Debug: Check if window has focus
        # (GLFW doesn't provide direct focus check, so we assume if context is current, it works)
        
        # Close on ESC
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            return  # Exit early to avoid processing other keys
        
        # Toggle render mode (only on key press, not hold)
        # We need to track the previous state to detect key press vs hold
        if not hasattr(self, '_space_was_pressed'):
            self._space_was_pressed = False
        
        space_current = glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS
        if space_current and not self._space_was_pressed:
            self._render_mode = 'splats' if self._render_mode == 'points' else 'points'
            print(f"  Switched to {self._render_mode} mode")
        self._space_was_pressed = space_current
        
        # Camera controls - use delta_time for smooth movement
        if not hasattr(self, '_last_input_time'):
            self._last_input_time = time.time()
        
        current_time = time.time()
        delta_time = current_time - self._last_input_time
        self._last_input_time = current_time
        
        # Clamp delta_time to avoid huge jumps
        delta_time = min(delta_time, 0.1)
        
        move_speed = 5.0  # meters per second
        key_speed = move_speed * delta_time
        
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            # Move FORWARD (in the direction we're looking)
            direction = self._camera_target - self._camera_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._camera_pos += direction * key_speed
            self._camera_target += direction * key_speed  # Move target too!
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            # Move BACKWARD
            direction = self._camera_target - self._camera_pos
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self._camera_pos -= direction * key_speed
            self._camera_target -= direction * key_speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            # Move LEFT
            right = np.cross(self._camera_target - self._camera_pos, self._camera_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            self._camera_pos -= right * key_speed
            self._camera_target -= right * key_speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            # Move RIGHT
            right = np.cross(self._camera_target - self._camera_pos, self._camera_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            self._camera_pos += right * key_speed
            self._camera_target += right * key_speed
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            # Move DOWN
            self._camera_pos -= self._camera_up * key_speed
            self._camera_target -= self._camera_up * key_speed
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            # Move UP
            self._camera_pos += self._camera_up * key_speed
            self._camera_target += self._camera_up * key_speed
        
        # Adjust point/splat size (only on key press, not hold)
        if not hasattr(self, '_plus_was_pressed'):
            self._plus_was_pressed = False
        if not hasattr(self, '_minus_was_pressed'):
            self._minus_was_pressed = False
        
        plus_current = glfw.get_key(self.window, glfw.KEY_EQUAL) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_KP_ADD) == glfw.PRESS
        minus_current = glfw.get_key(self.window, glfw.KEY_MINUS) == glfw.PRESS or glfw.get_key(self.window, glfw.KEY_KP_SUBTRACT) == glfw.PRESS
        
        if plus_current and not self._plus_was_pressed:
            self._point_size = min(self._point_size * 1.5, 100.0)
            print(f"  Point size: {self._point_size:.1f}")
        if minus_current and not self._minus_was_pressed:
            self._point_size = max(self._point_size / 1.5, 1.0)
            print(f"  Point size: {self._point_size:.1f}")
        
        self._plus_was_pressed = plus_current
        self._minus_was_pressed = minus_current
    
    def close(self) -> None:
        """Clean up resources."""
        if not self._use_opengl:
            return
        
        if self._instance_vao:
            gl.glDeleteVertexArrays(1, [self._instance_vao])
        if self._instance_vbo:
            gl.glDeleteBuffers(1, [self._instance_vbo])
        if self._quad_vao:
            gl.glDeleteVertexArrays(1, [self._quad_vao])
        if self._splat_program:
            gl.glDeleteProgram(self._splat_program)
        if self._point_program:
            gl.glDeleteProgram(self._point_program)
        
        glfw.terminate()
    
    @property
    def stats(self) -> Dict:
        """Get rendering statistics."""
        return self._stats.copy()
    
    @property
    def camera_position(self) -> np.ndarray:
        """Get camera position."""
        return self._camera_pos.copy()
    
    @camera_position.setter
    def camera_position(self, pos: np.ndarray) -> None:
        """Set camera position."""
        self._camera_pos = np.array(pos, dtype=np.float32)
    
    @property
    def camera_target(self) -> np.ndarray:
        """Get camera target (look-at point)."""
        return self._camera_target.copy()
    
    @camera_target.setter
    def camera_target(self, target: np.ndarray) -> None:
        """Set camera target."""
        self._camera_target = np.array(target, dtype=np.float32)
    
    @property
    def camera_up(self) -> np.ndarray:
        """Get camera up vector."""
        return self._camera_up.copy()
    
    @camera_up.setter
    def camera_up(self, up: np.ndarray) -> None:
        """Set camera up vector."""
        self._camera_up = np.array(up, dtype=np.float32)
    
    @property
    def opacity(self) -> float:
        """Get opacity."""
        return self._opacity
    
    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity."""
        self._opacity = float(np.clip(value, 0.0, 1.0))
    
    @property
    def render_mode(self) -> str:
        """Get render mode."""
        return self._render_mode
    
    @render_mode.setter
    def render_mode(self, mode: str) -> None:
        """Set render mode ('splats' or 'points')."""
        self._render_mode = mode if mode in ['splats', 'points'] else 'splats'


def create_gaussian_splat_renderer(width: int = 1280, height: int = 720,
                                   title: str = "2DGS Gaussian Splat Viewer"
                                   ) -> GaussianSplatRenderer:
    """
    Create a Gaussian splat renderer instance.
    
    Args:
        width: Window width
        height: Window height
        title: Window title
        
    Returns:
        GaussianSplatRenderer instance
    """
    return GaussianSplatRenderer(width, height, title)
