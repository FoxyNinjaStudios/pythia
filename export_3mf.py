"""
3MF (3D Manufacturing Format) export with color quantization for multi-material 3D printing.

3MF is an XML-based format that natively supports:
- Multiple materials per mesh
- Per-vertex or per-triangle color information
- Proper 3D printing metadata
- Full color preservation for full-color printing services

Color Quantization:
- Reduces vertex color palette to N colors for practical printing
- Groups similar colors together using k-means clustering
- Maps colors to printable materials

Usage:
  mesh_with_colors = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc)
  export_3mf_with_colors(mesh_with_colors, "output.3mf", num_colors=16)
"""

import numpy as np
import trimesh
from typing import Tuple, Optional
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; color quantization will use simple palette reduction")

try:
    from lxml import etree as ET
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as ET


def quantize_colors_kmeans(colors: np.ndarray, num_colors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize vertex colors using k-means clustering.
    
    Args:
        colors: (V, 3) or (V, 4) array of RGB(A) colors in 0-255 range
        num_colors: Target number of colors in the quantized palette
    
    Returns:
        (quantized_colors, palette): Quantized colors and the color palette
    """
    if not SKLEARN_AVAILABLE:
        return quantize_colors_simple(colors, num_colors)
    
    if colors.shape[0] == 0:
        return colors, np.array([[255, 255, 255]], dtype=np.uint8)
    
    # Handle RGBA by clustering on RGB
    rgb_colors = colors[:, :3] if colors.shape[1] >= 3 else colors
    
    # Normalize to 0-1 for clustering
    rgb_norm = rgb_colors.astype(np.float32) / 255.0
    
    # K-means clustering
    num_colors = min(num_colors, len(rgb_colors))
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=3)
    labels = kmeans.fit_predict(rgb_norm)
    
    # Get palette and quantized colors
    palette = (kmeans.cluster_centers_ * 255).astype(np.uint8)
    quantized = palette[labels].astype(np.uint8)
    
    # Preserve alpha channel if present
    if colors.shape[1] == 4:
        quantized = np.column_stack([quantized, colors[:, 3]])
    
    return quantized, palette


def quantize_colors_simple(colors: np.ndarray, num_colors: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple color quantization by reducing bit depth.
    
    Faster than k-means but less effective. Used as fallback when sklearn unavailable.
    """
    if colors.shape[0] == 0:
        return colors, np.array([[255, 255, 255]], dtype=np.uint8)
    
    # Determine bit depth from target color count
    bits_per_channel = max(1, int(np.log2(num_colors) / 3))
    
    # Quantize to reduced palette
    rgb_colors = colors[:, :3] if colors.shape[1] >= 3 else colors
    quantized = ((rgb_colors >> (8 - bits_per_channel)) << (8 - bits_per_channel)).astype(np.uint8)
    
    # Get unique colors as palette
    palette = np.unique(quantized, axis=0)
    
    # Preserve alpha if present
    if colors.shape[1] == 4:
        quantized = np.column_stack([quantized, colors[:, 3]])
    
    return quantized, palette


def export_3mf_with_colors(
    mesh: trimesh.Trimesh,
    filepath: str,
    num_colors: int = 16,
    verbose: bool = False
) -> bool:
    """
    Export a mesh to 3MF format with color information.

    The format supports per-vertex colors, which are preserved and quantized
    for practical multi-material 3D printing.

    Args:
        mesh: trimesh.Trimesh object with optional vertex_colors
        filepath: Output file path (.3mf)
        num_colors: Target palette size (8-256, default 16)
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate and prepare mesh
        if mesh.vertices.shape[0] == 0:
            logger.error("Cannot export empty mesh")
            return False

        # Check for vertex colors (access via mesh.visual.vertex_colors in trimesh)
        has_colors = (hasattr(mesh, 'visual') and
                     hasattr(mesh.visual, 'vertex_colors') and
                     mesh.visual.vertex_colors is not None and
                     len(mesh.visual.vertex_colors) > 0)

        if verbose:
            logger.info(f"Exporting to 3MF: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
            if has_colors:
                logger.info(f"  Vertex colors present, quantizing to {num_colors} colors")

        # Clamp num_colors to valid range
        num_colors = max(2, min(256, num_colors))

        # Quantize colors if present
        if has_colors:
            vertex_colors = mesh.visual.vertex_colors.copy()
            if vertex_colors.shape[1] == 3:
                vertex_colors = np.column_stack([vertex_colors, np.full(len(vertex_colors), 255)])

            quantized_colors, palette = quantize_colors_kmeans(vertex_colors, num_colors)

            if verbose:
                logger.info(f"  Palette: {len(palette)} unique colors")
                logger.info(f"  Color quantization: {np.unique(quantized_colors, axis=0).shape[0]} → {num_colors} colors")
        else:
            quantized_colors = None
            palette = np.array([[200, 200, 200]], dtype=np.uint8)

        # Export to 3MF with proper color support
        if has_colors and quantized_colors is not None:
            _export_3mf_with_vertex_colors(
                mesh, filepath, quantized_colors, palette, verbose=verbose
            )
        else:
            # Fallback: use trimesh's basic export
            mesh.export(filepath, file_type='3mf')

        if verbose:
            logger.info(f"✓ 3MF export complete: {filepath}")

        return True

    except Exception as e:
        logger.error(f"3MF export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_3mf_separated_by_color(
    mesh: trimesh.Trimesh,
    filepath: str,
    num_colors: int = 8,
    min_part_frac: float = 0.02,
    verbose: bool = False,
) -> bool:
    """
    Export a 3MF where **each colour is a separate object**.

    The mesh is clustered by its per-vertex colour (the same clustering used for
    the multi-object GLB, via :func:`part_segmentation.split_mesh_by_color`) and
    every colour group is written as its own ``<object>`` in the 3MF, each
    referencing a solid base material. Slicers such as Bambu Studio then show one
    distinct, filament-assignable part per colour — instead of a single
    flat-coloured mesh — which is what multi-material / AMS printing needs.

    Args:
        mesh: trimesh.Trimesh with per-vertex colours
        filepath: Output ``.3mf`` path
        num_colors: Number of colour groups to split into (2-256)
        min_part_frac: Merge colour groups smaller than this fraction of faces
        verbose: Print progress

    Returns:
        True on success, False otherwise.
    """
    try:
        if mesh.vertices.shape[0] == 0:
            logger.error("Cannot export empty mesh")
            return False

        from part_segmentation import split_mesh_into_parts

        # num_colors <= 0 means auto-detect the number of colours.
        num_colors = 0 if int(num_colors) <= 0 else max(2, min(256, int(num_colors)))
        submeshes, colors = split_mesh_into_parts(
            mesh, n_colors=num_colors, min_part_frac=min_part_frac
        )

        if verbose:
            logger.info(
                f"Exporting per-colour 3MF: {len(submeshes)} colour object(s), "
                f"{mesh.faces.shape[0]} faces total"
            )

        _write_multi_object_3mf(submeshes, colors, filepath)

        if verbose:
            logger.info(f"✓ 3MF export complete: {filepath}")
        return True

    except Exception as e:
        logger.error(f"3MF per-colour export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _write_multi_object_3mf(submeshes, colors, filepath: str) -> None:
    """Write a 3MF package with one ``<object>`` per colour group.

    Each object references a shared ``basematerials`` group (``pid="1"``) via its
    own ``pindex``, giving every colour a solid display colour that slicers read
    as a separate, filament-assignable part.
    """
    CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    MATL = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
    PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"

    import uuid as _uuid

    def _uid():
        return _uuid.uuid4().hex

    out = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append(
        f'<model unit="millimeter" xml:lang="en-US" '
        f'xmlns="{CORE}" xmlns:m="{MATL}" xmlns:p="{PROD}">'
    )
    out.append(" <resources>")

    # A single base-material group holding every colour.
    out.append('  <m:basematerials id="1">')
    for i, col in enumerate(colors):
        r, g, b = int(col[0]), int(col[1]), int(col[2])
        out.append(
            f'   <m:base name="Color {i + 1}" '
            f'displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>'
        )
    out.append("  </m:basematerials>")

    # One object per colour; the object-level pid/pindex paints all its
    # triangles with that colour's base material.
    obj_ids = []
    for i, sub in enumerate(submeshes):
        oid = i + 2  # id 1 is the basematerials group
        obj_ids.append(oid)
        verts = np.asarray(sub.vertices, dtype=np.float64)
        faces = np.asarray(sub.faces, dtype=np.int64)

        out.append(
            f'  <object id="{oid}" p:UUID="{_uid()}" '
            f'type="model" pid="1" pindex="{i}">'
        )
        out.append("   <mesh>")
        out.append("    <vertices>")
        out.extend(
            f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>'
            for x, y, z in verts
        )
        out.append("    </vertices>")
        out.append("    <triangles>")
        out.extend(
            f'     <triangle v1="{int(t0)}" v2="{int(t1)}" v3="{int(t2)}"/>'
            for t0, t1, t2 in faces
        )
        out.append("    </triangles>")
        out.append("   </mesh>")
        out.append("  </object>")

    # Wrap every colour part as a component of ONE assembly object. The parts
    # already share the model's absolute coordinates, so each component uses the
    # identity transform and the pieces reassemble exactly. Building this single
    # object (instead of one build item per colour) stops slicers such as Bambu
    # Studio from auto-arranging the colours as separate objects across the
    # plate — they stay locked in their true relative positions as printable
    # sub-parts of the one model.
    assembly_id = len(submeshes) + 2
    out.append(f'  <object id="{assembly_id}" p:UUID="{_uid()}" type="model">')
    out.append("   <components>")
    out.extend(
        f'    <component objectid="{oid}" p:UUID="{_uid()}" '
        f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>'
        for oid in obj_ids
    )
    out.append("   </components>")
    out.append("  </object>")

    out.append(" </resources>")
    out.append(" <build>")
    out.append(f'  <item objectid="{assembly_id}" p:UUID="{_uid()}" '
               f'transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>')
    out.append(" </build>")
    out.append("</model>")
    model_xml = "\n".join(out).encode("utf-8")

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    ).encode("utf-8")

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships '
        'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    ).encode("utf-8")

    # Bambu Studio / OrcaSlicer read per-part filament (extruder) assignments
    # from Metadata/model_settings.config, NOT from the 3MF base materials.
    # Without it every part defaults to filament 1 even though each part is a
    # distinct object. Map colour i → filament/extruder i+1 (the user re-maps
    # these to physical AMS slots in the slicer). The <part id> matches the
    # component objectid in 3dmodel.model.
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    cfg.append(f'  <object id="{assembly_id}">')
    cfg.append('    <metadata key="name" value="segmented"/>')
    cfg.append('    <metadata key="extruder" value="1"/>')
    for i, oid in enumerate(obj_ids):
        cfg.append(f'    <part id="{oid}" subtype="normal_part">')
        cfg.append(f'      <metadata key="name" value="Color {i + 1}"/>')
        cfg.append(
            '      <metadata key="matrix" '
            'value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>'
        )
        cfg.append(f'      <metadata key="extruder" value="{i + 1}"/>')
        cfg.append("    </part>")
    cfg.append("  </object>")
    cfg.append("</config>")
    model_settings = "\n".join(cfg).encode("utf-8")

    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model_xml)
        z.writestr("Metadata/model_settings.config", model_settings)


def _export_3mf_with_vertex_colors(
    mesh: trimesh.Trimesh,
    filepath: str,
    vertex_colors: np.ndarray,
    palette: np.ndarray,
    verbose: bool = False
) -> None:
    """
    Export mesh with per-vertex colors to 3MF format.

    Creates proper 3MF XML structure with colorgroup elements for color information.
    """
    # First export basic geometry using trimesh
    temp_path = str(Path(filepath).with_stem(Path(filepath).stem + "_temp"))
    mesh.export(temp_path, file_type='3mf')

    # Now enhance the 3MF file with color information
    # 3MF is a ZIP file containing XML and binary data
    try:
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            # Read the model XML
            model_xml_data = zip_ref.read('3D/3dmodel.model')
            
        # Register namespaces to preserve them
        if LXML_AVAILABLE:
            # lxml doesn't allow empty string as prefix, so we handle the default namespace differently
            ET.register_namespace('m', 'http://schemas.microsoft.com/3dmanufacturing/material/2015/02')
            ET.register_namespace('p', 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06')
            ET.register_namespace('b', 'http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02')
            ET.register_namespace('s', 'http://schemas.microsoft.com/3dmanufacturing/slice/2015/07')
            ET.register_namespace('sc', 'http://schemas.microsoft.com/3dmanufacturing/securecontent/2019/04')
        
        # Parse the XML
        root = ET.fromstring(model_xml_data)

        # Find resources element
        resources = root.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}resources')
        if resources is None:
            resources = root.find('resources')

        if resources is not None:
            colorgroup_ns = '{http://schemas.microsoft.com/3dmanufacturing/material/2015/02}'
            ns_core = '{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}'
            
            # Create basematerials group with ID
            basematerials = ET.Element(colorgroup_ns + 'basematerials')
            basematerials.set('id', '1')  # Explicit ID for the material group
            
            # Create a material for each color in palette (1-indexed)
            material_id_map = {}  # map from color index to material ID
            for idx, color in enumerate(palette):
                # Each base material has id starting from 1
                mat_id = idx + 1
                material_id_map[idx + 1] = mat_id
                
                material = ET.SubElement(basematerials, colorgroup_ns + 'base')
                material.set('id', str(mat_id))
                material.set('name', f'Material{mat_id}')
                
                # Store color as RGB hex value
                color_hex = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}FF"
                material.set('displaycolor', color_hex)
            
            resources.insert(0, basematerials)
            
            # Find object and set material property reference
            obj = root.find(f'.//{ns_core}object')
            if obj is not None:
                obj.set('pid', '1')  # Reference to basematerials group id

            # Find mesh element
            mesh_elem = root.find('.//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}mesh')
            if mesh_elem is None:
                mesh_elem = root.find('.//mesh')

            if mesh_elem is not None:
                # Find triangles
                triangles = mesh_elem.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangles')
                if triangles is None:
                    triangles = mesh_elem.find('triangles')

                if triangles is not None:
                    # Map vertices to material indices
                    for tri_elem in triangles:
                        v1_idx = int(tri_elem.get('v1'))
                        v2_idx = int(tri_elem.get('v2'))
                        v3_idx = int(tri_elem.get('v3'))

                        # Get color indices (1-based from _find_color_index)
                        v1_color_idx = _find_color_index(vertex_colors[v1_idx], palette)
                        v2_color_idx = _find_color_index(vertex_colors[v2_idx], palette)
                        v3_color_idx = _find_color_index(vertex_colors[v3_idx], palette)

                        # Set material indices on triangle (1-based)
                        tri_elem.set('p1', str(v1_color_idx))
                        tri_elem.set('p2', str(v2_color_idx))
                        tri_elem.set('p3', str(v3_color_idx))

        # Write back to 3MF file
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            # Copy all files from temp except the model
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                for item in zip_ref.infolist():
                    if item.filename != '3D/3dmodel.model':
                        zip_out.writestr(item, zip_ref.read(item.filename))

            # Write updated model XML
            model_xml_str = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            zip_out.writestr('3D/3dmodel.model', model_xml_str)

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        if verbose:
            logger.info(f"✓ Enhanced 3MF with {len(palette)} color palette and colorgroup metadata")

    except Exception as e:
        logger.warning(f"Could not enhance 3MF with colors: {e}. Using basic export.")
        import traceback
        traceback.print_exc()
        # Fallback: use the basic export
        Path(temp_path).unlink(missing_ok=True)
        mesh.export(filepath, file_type='3mf')


def _find_color_index(vertex_color: np.ndarray, palette: np.ndarray) -> int:
    """Find the closest color in the palette and return its 1-based index."""
    min_dist = float('inf')
    closest_idx = 1
    
    for i, pal_color in enumerate(palette):
        # Only compare RGB, ignore alpha
        dist = np.linalg.norm(vertex_color[:3].astype(float) - pal_color[:3].astype(float))
        if dist < min_dist:
            min_dist = dist
            closest_idx = i + 1
    
    return closest_idx


def estimate_3mf_size(num_vertices: int, num_faces: int, has_colors: bool = True) -> int:
    """
    Estimate 3MF file size in bytes.
    
    3MF is typically 2-5× smaller than STL due to ZIP compression.
    """
    # Base XML + vertices
    base_bytes = 1000 + (num_vertices * 48)  # ~48 bytes per vertex
    
    # Faces
    faces_bytes = num_faces * 36  # ~36 bytes per triangle
    
    # Colors (if present)
    colors_bytes = (num_vertices * 8) if has_colors else 0
    
    # Estimate compressed size (ZIP typically 60-70% of original)
    total = base_bytes + faces_bytes + colors_bytes
    compressed = int(total * 0.65)
    
    return max(1000, compressed)  # At least 1KB
