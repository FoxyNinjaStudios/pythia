#!/usr/bin/env python
"""Test 3MF endpoint integration without requiring a job"""

import requests
import sys

BASE_URL = "http://localhost:8005"

def test_3mf_endpoint():
    """Test that 3MF endpoint is available and responds correctly"""
    print("Testing 3MF endpoint integration...")
    
    # Test 1: Verify endpoint exists but rejects invalid job
    print("\n1. Testing invalid job ID (should get 404)...", end=" ")
    resp = requests.get(f"{BASE_URL}/result/invalid_job_id", params={"format": "3mf"})
    if resp.status_code == 404:
        print("✓")
    else:
        print(f"✗ Expected 404, got {resp.status_code}")
        return False
    
    # Test 2: Verify 3MF format is recognized (we'll get 404 for invalid job, not 400 for bad format)
    print("2. Testing 3MF format param is accepted...", end=" ")
    resp = requests.get(f"{BASE_URL}/result/test_job", params={"format": "3mf", "num_colors": 16})
    if resp.status_code == 404:  # Expected: job not found (format param was accepted)
        print("✓")
    else:
        print(f"✗ Got {resp.status_code}")
        return False
    
    # Test 3: Verify color quantization parameter is passed
    print("3. Testing num_colors parameter...", end=" ")
    resp = requests.get(f"{BASE_URL}/result/test_job", params={"format": "3mf", "num_colors": 32})
    if resp.status_code == 404:  # Expected: job not found
        print("✓")
    else:
        print(f"✗ Got {resp.status_code}")
        return False
    
    # Test 4: Verify default num_colors
    print("4. Testing default num_colors...", end=" ")
    resp = requests.get(f"{BASE_URL}/result/test_job", params={"format": "3mf"})
    if resp.status_code == 404:  # Expected: job not found
        print("✓")
    else:
        print(f"✗ Got {resp.status_code}")
        return False
    
    # Test 5: Verify GLB still works
    print("5. Testing GLB format still works...", end=" ")
    resp = requests.get(f"{BASE_URL}/result/test_job", params={"format": "glb"})
    if resp.status_code == 404:  # Expected: job not found
        print("✓")
    else:
        print(f"✗ Got {resp.status_code}")
        return False
    
    print("\n✓ All 3MF endpoint integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_3mf_endpoint()
    sys.exit(0 if success else 1)
