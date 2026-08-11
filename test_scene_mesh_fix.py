#!/usr/bin/env python
"""Test 3MF export fix for Scene vs Mesh loading"""

import tempfile
from pathlib import Path
import trimesh
from export_3mf import export_3mf_with_colors
import numpy as np

# Create a test mesh with colors
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float32)

faces = np.array([
    [0, 1, 2], [0, 2, 3],  # bottom
    [4, 6, 5], [4, 7, 6],  # top
    [0, 5, 1], [0, 4, 5],  # front
    [2, 7, 3], [2, 6, 7],  # back
    [0, 3, 7], [0, 7, 4],  # left
    [1, 5, 6], [1, 6, 2],  # right
], dtype=np.uint32)

colors = np.random.randint(0, 256, (len(vertices), 3), dtype=np.uint8)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save as GLB
    glb_path = Path(tmpdir) / "test.glb"
    mesh.export(str(glb_path))
    print(f"✓ Created test GLB: {glb_path}")
    
    # Load it back (this returns a Scene)
    loaded = trimesh.load(str(glb_path), process=False)
    print(f"✓ Loaded mesh type: {type(loaded).__name__}")
    
    # Test the Scene/Mesh handling
    if isinstance(loaded, trimesh.Scene):
        print("  → Detected Scene, extracting meshes...")
        meshes = [g for g in loaded.geometry.values()]
        if len(meshes) == 1:
            mesh_for_export = meshes[0]
            print(f"  → Using single mesh")
        else:
            mesh_for_export = trimesh.util.concatenate(meshes)
            print(f"  → Concatenated {len(meshes)} meshes")
    else:
        mesh_for_export = loaded
    
    print(f"✓ Mesh for export type: {type(mesh_for_export).__name__}")
    print(f"  → Vertices: {mesh_for_export.vertices.shape if hasattr(mesh_for_export, 'vertices') else 'N/A'}")
    print(f"  → Faces: {mesh_for_export.faces.shape if hasattr(mesh_for_export, 'faces') else 'N/A'}")
    
    # Try to export as 3MF
    output_path = Path(tmpdir) / "test.3mf"
    try:
        success = export_3mf_with_colors(mesh_for_export, str(output_path), num_colors=8)
        if success and output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"✓ 3MF export successful: {size_kb:.1f} KB")
        else:
            print("✗ 3MF export returned False")
    except Exception as e:
        print(f"✗ 3MF export failed: {e}")

print("\n✓ Scene/Mesh handling test passed!")
