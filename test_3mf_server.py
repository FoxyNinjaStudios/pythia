#!/usr/bin/env python
"""Test 3MF export endpoint with server API"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8005"

def test_3mf_export():
    """Test 3MF export through the server"""
    print("Testing 3MF server export endpoint...")
    
    # We'll use a cached result if available, or create a simple test
    # For now, let's just verify the endpoint exists and handles requests
    
    # Try to list existing jobs
    jobs_resp = requests.get(f"{BASE_URL}/jobs")
    print(f"Jobs endpoint: {jobs_resp.status_code}")
    if jobs_resp.ok:
        jobs = jobs_resp.json()
        print(f"Available jobs: {len(jobs)}")
        
        # If there are jobs, try to export the last one as 3MF
        if jobs:
            job_ids = list(jobs.keys())
            latest_job = job_ids[-1]
            print(f"\nTesting 3MF export on job: {latest_job}")
            
            # Test with different color quantization values
            for num_colors in [8, 16, 32]:
                print(f"  Testing with {num_colors} colors...", end=" ", flush=True)
                try:
                    resp = requests.get(
                        f"{BASE_URL}/result/{latest_job}",
                        params={"format": "3mf", "num_colors": num_colors},
                        timeout=30
                    )
                    
                    if resp.status_code == 200:
                        size_kb = len(resp.content) / 1024
                        print(f"✓ {size_kb:.1f} KB")
                        
                        # Save the first one as example
                        if num_colors == 16:
                            output_path = Path("/tmp/test_3mf_export.3mf")
                            output_path.write_bytes(resp.content)
                            print(f"    Saved example to {output_path}")
                    else:
                        print(f"✗ HTTP {resp.status_code}: {resp.text[:100]}")
                except Exception as e:
                    print(f"✗ Error: {e}")
    else:
        print(f"Could not list jobs: {jobs_resp.status_code}")

if __name__ == "__main__":
    test_3mf_export()
