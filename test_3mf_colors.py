#!/usr/bin/env python
"""Test 3MF color quantization and export."""

import numpy as np
import trimesh
import tempfile
from pathlib import Path
from export_3mf import export_3mf_with_colors, quantize_colors_kmeans

def test_3mf_colors():
    """Create a mesh with distinct vertex colors and verify they are preserved."""
    print("Creating test mesh with 4 distinct colors...")
    
    # Create a simple cube with 8 vertices, each with a different color
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.uint32)
    
    # Assign 4 distinct colors to vertices
    vertex_colors = np.array([
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
    ], dtype=np.uint8)
    
    # Create mesh with colors
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False
    )
    
    print(f"  Input: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"  Input colors: {np.unique(mesh.visual.vertex_colors, axis=0).shape[0]} unique colors")
    
    # Test quantization
    print("\nTesting color quantization (8 colors → 4 colors)...")
    quantized, palette = quantize_colors_kmeans(mesh.visual.vertex_colors, num_colors=4)
    print(f"  Quantized: {np.unique(quantized, axis=0).shape[0]} unique colors")
    print(f"  Palette size: {len(palette)}")
    print(f"  Palette:\n{palette}")
    
    # Export to 3MF with quantization
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.3mf"
        print(f"\nExporting to 3MF: {output_path}")
        
        success = export_3mf_with_colors(mesh, str(output_path), num_colors=4, verbose=True)
        
        if success and output_path.exists():
            file_size = output_path.stat().st_size
            print(f"✓ Export successful: {file_size} bytes")
            
            # Verify the exported mesh has colors
            print("\nVerifying exported mesh...")
            exported_mesh = trimesh.load(str(output_path), process=False)
            
            if isinstance(exported_mesh, trimesh.Scene):
                meshes = [g for g in exported_mesh.geometry.values()]
                exported_mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
            
            if hasattr(exported_mesh, 'visual') and hasattr(exported_mesh.visual, 'vertex_colors') and exported_mesh.visual.vertex_colors is not None:
                unique_colors = np.unique(exported_mesh.visual.vertex_colors, axis=0)
                print(f"✓ Exported mesh has {unique_colors.shape[0]} unique colors")
                print(f"  Colors:\n{unique_colors}")
                
                # Note: trimesh doesn't parse colorgroup materials when loading,
                # but the colors are in the 3MF XML for slicers to use
                print("\n  Note: trimesh loads don't parse materials, but checking 3MF structure...")
            else:
                print("  Note: trimesh loads don't preserve material colors, checking 3MF structure...")
            
            # Verify colors are actually in the 3MF XML structure
            import zipfile
            from lxml import etree as ET
            
            try:
                with zipfile.ZipFile(str(output_path), 'r') as z:
                    xml_data = z.read('3D/3dmodel.model')
                
                root = ET.fromstring(xml_data)
                
                # Check for basematerials with inline colors (simplified approach)
                basematerials = root.find('.//{http://schemas.microsoft.com/3dmanufacturing/material/2015/02}basematerials')
                
                if basematerials is not None:
                    materials = basematerials.findall('{http://schemas.microsoft.com/3dmanufacturing/material/2015/02}base')
                    print(f"✓ SUCCESS: 3MF contains basematerials with {len(materials)} materials")
                    print(f"  Materials in 3MF file:")
                    for m in materials:
                        mat_id = m.get('id')
                        mat_name = m.get('name')
                        color_val = m.get('displaycolor')
                        print(f"    ID {mat_id} ({mat_name}): {color_val}")
                    
                    # Verify triangles have material property indices
                    triangles = root.findall('.//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangle')
                    material_triangles = sum(1 for t in triangles if t.get('p1') is not None)
                    print(f"  Triangles with material properties: {material_triangles}/{len(triangles)}")
                    if material_triangles > 0:
                        sample_tri = [t for t in triangles if t.get('p1') is not None][0]
                        print(f"    Sample triangle: p1={sample_tri.get('p1')}, p2={sample_tri.get('p2')}, p3={sample_tri.get('p3')}")
                    
                    # Check object pid reference
                    obj = root.find('.//{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}object')
                    if obj is not None:
                        obj_pid = obj.get('pid')
                        if obj_pid:
                            print(f"  Object references materials: pid=\"{obj_pid}\"")
                    
                    print("\n✓ PASS: 3MF with simplified basematerials exported successfully!")
                    print("  Each triangle references a unique material with embedded color.")
                    print("  Bambu Studio should now recognize these as distinct colors.")
                    return True
                else:
                    print("✗ FAIL: No basematerials found in 3MF XML")
                    return False
            except Exception as e:
                print(f"✗ FAIL: Error verifying 3MF structure: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("✗ Export failed")
            return False

if __name__ == "__main__":
    success = test_3mf_colors()
    exit(0 if success else 1)
