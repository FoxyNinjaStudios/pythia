#!/usr/bin/env python
"""Comprehensive test of 3MF export integration"""

import sys
import tempfile
from pathlib import Path
import json

def test_3mf_module():
    """Test the export_3mf module directly"""
    print("Testing export_3mf module...")
    try:
        from export_3mf import export_3mf_with_colors, quantize_colors_kmeans
        import numpy as np
        import trimesh
        
        # Create a simple colored mesh
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
        
        # Create random colors
        colors = np.random.randint(0, 256, (len(vertices), 3), dtype=np.uint8)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        
        # Test color quantization
        print("  1. Testing color quantization...", end=" ", flush=True)
        quantized = quantize_colors_kmeans(mesh.visual.vertex_colors, num_colors=8)
        if isinstance(quantized, tuple) and len(quantized) == 2:
            quantized_colors, palette = quantized
            if len(quantized_colors) == len(vertices) and len(palette) <= 8:
                print("✓")
            else:
                print("✗")
                return False
        else:
            print("✗")
            return False
        
        # Test 3MF export
        print("  2. Testing 3MF export...", end=" ", flush=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.3mf"
            success = export_3mf_with_colors(mesh, str(output_path), num_colors=8)
            if success and output_path.exists() and output_path.stat().st_size > 0:
                print(f"✓ ({output_path.stat().st_size} bytes)")
            else:
                print("✗")
                return False
        
        print("✓ export_3mf module tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_server_endpoint():
    """Test the server endpoint"""
    print("Testing server 3MF endpoint...")
    try:
        import requests
        
        # Test with invalid job (should get 404 for job not found, not 400 for bad format)
        print("  1. Testing endpoint responds to 3MF format...", end=" ", flush=True)
        resp = requests.get("http://localhost:8005/result/invalid_job", params={"format": "3mf"})
        if resp.status_code == 404:
            print("✓")
        else:
            print(f"✗ (got {resp.status_code})")
            return False
        
        # Test num_colors parameter
        print("  2. Testing num_colors parameter...", end=" ", flush=True)
        resp = requests.get("http://localhost:8005/result/invalid_job", params={"format": "3mf", "num_colors": 32})
        if resp.status_code == 404:
            print("✓")
        else:
            print(f"✗ (got {resp.status_code})")
            return False
        
        print("✓ Server endpoint tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_ui_configuration():
    """Test that UI configuration is correct"""
    print("Testing UI configuration...")
    try:
        ui_path = Path("/Users/tejaswigowda/Downloads/pythia/static/index.html")
        content = ui_path.read_text()
        
        # Check EXPORT_FORMATS includes 3MF
        print("  1. Checking EXPORT_FORMATS...", end=" ", flush=True)
        if '"3mf"' in content and '"3MF (Multi-color)"' in content:
            print("✓")
        else:
            print("✗")
            return False
        
        # Check 3MF case in runExport
        print("  2. Checking runExport handler...", end=" ", flush=True)
        if 'case "3mf":' in content and 'format=3mf' in content:
            print("✓")
        else:
            print("✗")
            return False
        
        # Check color quantization control
        print("  3. Checking color quantization control...", end=" ", flush=True)
        if 'opt-3mf-colors' in content and '3MF color palette' in content:
            print("✓")
        else:
            print("✗")
            return False
        
        # Check onExportOpt updates color value
        print("  4. Checking color slider update...", end=" ", flush=True)
        if 'opt-3mf-colors-val' in content:
            print("✓")
        else:
            print("✗")
            return False
        
        print("✓ UI configuration tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE 3MF EXPORT INTEGRATION TEST")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Test module
    if not test_3mf_module():
        all_passed = False
    
    # Test server
    if not test_server_endpoint():
        all_passed = False
    
    # Test UI
    if not test_ui_configuration():
        all_passed = False
    
    print("="*60)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
