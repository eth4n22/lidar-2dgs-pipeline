"""
AR/USDZ Export Module

Convert Gaussian surfels to mesh formats for AR visualization.
Supports: OBJ, GLTF, and USDZ (via conversion).

Workflow:
    LiDAR → Gaussian Surfels → Mesh (OBJ/GLTF) → USDZ → iPhone AR
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Try importing trimesh for mesh operations
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def surfels_to_mesh(surfels: Dict[str, np.ndarray],
                   resolution: int = 4) -> trimesh.Trimesh:
    """
    Convert Gaussian surfels to a triangular mesh.

    Each Gaussian surfel is approximated as a small quad, then triangulated.

    Args:
        surfels: Surfel dictionary with position, normal, scale, color
        resolution: Number of subdivisions (higher = smoother but slower)

    Returns:
        Trimesh object
    """
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh not installed. Run: pip install trimesh\n"
            "Required for mesh conversion."
        )

    positions = surfels["position"]
    normals = surfels["normal"]
    scales = surfels["scale"]
    colors = surfels.get("color", np.ones((len(positions), 3)))

    # Build vertices and faces
    vertices = []
    faces = []
    vertex_colors = []
    vertex_offset = 0

    for i in range(len(positions)):
        pos = positions[i]
        normal = normals[i]
        scale = scales[i]

        # Create local coordinate system
        up = np.array([0, 0, 1], dtype=np.float32)
        if np.abs(np.dot(normal, up)) > 0.9:
            up = np.array([1, 0, 0], dtype=np.float32)

        tangent = np.cross(up, normal)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        bitangent = np.cross(normal, tangent)

        # Create quad vertices
        sx, sy, sz = scale
        # Tangent direction (wider)
        t = tangent * sx * 2
        # Bitangent direction (wider)
        b = bitangent * sy * 2
        # Normal direction (thin)
        n = normal * sz * 2

        # 4 corners of the quad
        v0 = pos - t - b
        v1 = pos + t - b
        v2 = pos + t + b
        v3 = pos - t + b

        vertices.extend([v0, v1, v2, v3])
        vertex_colors.extend([colors[i]] * 4)

        # Two triangles
        faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
        faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])

        vertex_offset += 4

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    colors_array = np.array(vertex_colors, dtype=np.float64)

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors_array)

    # Basic cleanup
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()

    return mesh


def export_obj(filepath: str, surfels: Dict[str, np.ndarray]) -> None:
    """
    Export surfels to OBJ format.

    Args:
        filepath: Output .obj file path
        surfels: Surfel dictionary
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required. Run: pip install trimesh")

    mesh = surfels_to_mesh(surfels)

    # Export with colors as vertex colors
    mesh.export(filepath, file_type='obj', vertex_color=True)

    print(f"  Exported OBJ: {filepath}")


def export_gltf(filepath: str, surfels: Dict[str, np.ndarray]) -> None:
    """
    Export surfels to GLTF/GLB format.

    GLTF is the modern 3D format that can be:
    - Viewed in browsers (Three.js)
    - Opened in Blender
    - Converted to USDZ using external tools

    Args:
        filepath: Output .gltf or .glb file path
        surfels: Surfel dictionary
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required. Run: pip install trimesh")

    mesh = surfels_to_mesh(surfels)

    # Determine format
    is_binary = Path(filepath).suffix.lower() == '.glb'

    mesh.export(filepath, file_type='glb' if is_binary else 'gltf')

    print(f"  Exported GLTF: {filepath}")


def export_usdz(surfels: Dict[str, np.ndarray],
               output_dir: str,
               model_name: str = "model") -> Dict[str, str]:
    """
    Export surfels to USDZ format for iPhone AR.

    Conversion pipeline:
    1. Export to GLTF/GLB
    2. Use external tool to convert to USDZ

    Note: True USDZ conversion requires macOS or usd-core.

    Args:
        surfels: Surfel dictionary
        output_dir: Output directory
        model_name: Base name for output files

    Returns:
        Dictionary with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Export to GLB first (intermediate format)
    glb_path = output_path / f"{model_name}.glb"
    export_gltf(str(glb_path), surfels)
    results["glb"] = str(glb_path)

    # Try to create USDZ using trimesh (if on macOS or with usd-core)
    usdz_path = output_path / f"{model_name}.usdz"

    try:
        # Try using trimesh's USDZ export (requires usd-core)
        mesh = surfels_to_mesh(surfels)
        mesh.export(str(usdz_path), file_type='usdz')
        results["usdz"] = str(usdz_path)
        print(f"  Exported USDZ: {usdz_path}")
    except Exception as e:
        # Provide alternative instructions
        results["usdz_instructions"] = f"""
USDZ Conversion Options:

1. Using macOS (Terminal):
   # Install Xcode command line tools
   xcode-select --install

   # Convert GLB to USDZ
   /System/Library/PrivateFrameworks/MobileCoreServices.framework/\
Versions/Current/Frameworks/LaunchServices.framework/\
Versions/Current/Support/lsd \
   -i {glb_path} -o {usdz_path}

2. Using usd-core (Python):
   pip install usd-core

3. Online conversion:
   - https://www.vectary.com/ - Free GLTF to USDZ
   - https://www.rapidconvert.org/ - Online converter

4. Blender:
   Import GLB → Export USDZ
"""
        print(f"  USDZ export requires additional setup (see instructions)")

    # Also export OBJ for compatibility
    obj_path = output_path / f"{model_name}.obj"
    export_obj(str(obj_path), surfels)
    results["obj"] = str(obj_path)

    return results


def generate_qr_info(ar_url: str,
                     output_dir: str,
                     model_name: str = "model") -> Dict[str, str]:
    """
    Generate QR code information for AR model.

    Creates metadata and instructions for QR code generation.

    Args:
        ar_url: URL where the USDZ model is hosted
        output_dir: Output directory for files
        model_name: Name of the model

    Returns:
        Dictionary with QR code generation info
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create AR link file (for iOS AR Quick Look)
    # iOS expects: <a href="model.usdz" rel="ar"><img src="preview.png"></a>

    # Simple metadata file
    metadata = {
        "model_name": model_name,
        "ar_url": ar_url,
        "ios_support": True,
        "android_support": True,
        "qr_instructions": """
QR Code Generation for AR:

1. Host USDZ file on a web server (HTTPS required for iOS)

2. Create HTML link for iOS AR Quick Look:
   <html>
   <body>
     <a href="model.usdz" rel="ar">
       <img src="qr-image.png">
     </a>
   </body>
   </html>

3. Generate QR code pointing to your HTML page

4. Tools for QR generation:
   - Python: pip install qrcode
   - Online: qr-code-generator.com
   - Canva, Adobe tools

5. Best practices:
   - Use short URLs (QR codes more readable)
   - Add alt text for accessibility
   - Test on both iOS and Android
"""
    }

    # Write metadata
    metadata_path = output_path / f"{model_name}_ar_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        "metadata": str(metadata_path),
        "ar_url": ar_url
    }


class ARExporter:
    """
    Complete AR export pipeline.

    Usage:
        exporter = ARExporter()
        result = exporter.export(
            surfels=my_surfels,
            output_dir="./ar_output",
            model_name="building_scan"
        )
        # result contains paths to GLB, USDZ, OBJ files
    """

    def __init__(self):
        self.trimesh_available = HAS_TRIMESH

    def export(self,
              surfels: Dict[str, np.ndarray],
              output_dir: str,
              model_name: str = "model",
              ar_url: Optional[str] = None) -> Dict[str, str]:
        """
        Export surfels to all AR formats.

        Args:
            surfels: Surfel dictionary
            output_dir: Output directory
            model_name: Base name for files
            ar_url: Optional URL for AR access

        Returns:
            Dictionary with paths to all exported files
        """
        results = {}

        # Export to GLB (primary format)
        try:
            glb_path = Path(output_dir) / f"{model_name}.glb"
            export_gltf(str(glb_path), surfels)
            results["glb"] = str(glb_path)
        except ImportError as e:
            results["error"] = str(e)
            return results

        # Export to OBJ
        obj_path = Path(output_dir) / f"{model_name}.obj"
        export_obj(str(obj_path), surfels)
        results["obj"] = str(obj_path)

        # Export to USDZ
        usdz_path = Path(output_dir) / f"{model_name}.usdz"
        try:
            mesh = surfels_to_mesh(surfels)
            mesh.export(str(usdz_path), file_type='usdz')
            results["usdz"] = str(usdz_path)
        except Exception as e:
            results["usdz_note"] = str(e)

        # Generate QR code info if URL provided
        if ar_url:
            qr_info = generate_qr_info(ar_url, output_dir, model_name)
            results.update(qr_info)

        return results

    def check_requirements(self) -> Dict[str, bool]:
        """Check which export formats are available."""
        return {
            "trimesh": self.trimesh_available,
            "gltf_export": self.trimesh_available,
            "usdz_export": self.trimesh_available,
            "obj_export": self.trimesh_available
        }
