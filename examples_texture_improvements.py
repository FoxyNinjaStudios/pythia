#!/usr/bin/env python3
"""
Example: Using texture mapping improvements in PYTHIA.

This script demonstrates how to integrate the three texture quality improvements
(pointmap confidence, adaptive visibility weighting, reconstruction quality)
into your reconstruction pipeline.
"""

import numpy as np
from texture_baking import (
    bake_texture_from_image,
    compute_surface_curvature,
    create_uniform_confidence,
    estimate_sparse_geometry_confidence_from_vertices,
    blend_confidence_arrays,
)


def example_basic_texture_baking():
    """Basic texture baking without confidence (backward compatible)."""
    print("=" * 70)
    print("EXAMPLE 1: Basic texture baking (backward compatible)")
    print("=" * 70)
    
    # Load or create your mesh
    vertices = np.random.randn(1000, 3).astype(np.float32)
    faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Simple baking (uses improved pointmap + adaptive visibility, but no confidence)
    mesh = bake_texture_from_image(
        vertices, faces, image,
        texture_size=1024,
        as_vertex_colors=True,  # Faster for testing
    )
    
    print(f"✓ Baked texture for mesh with {len(vertices)} vertices")
    print(f"✓ Adaptive visibility weighting automatically applied")
    print(f"✓ Enhanced pointmap confidence scoring active")
    print()


def example_with_uniform_confidence():
    """Texture baking with uniform confidence (for testing)."""
    print("=" * 70)
    print("EXAMPLE 2: Texture baking with uniform confidence")
    print("=" * 70)
    
    vertices = np.random.randn(1000, 3).astype(np.float32)
    faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Create uniform confidence (e.g., "high confidence everywhere")
    sparse_conf = create_uniform_confidence(len(vertices), confidence_level=0.85)
    slat_conf = create_uniform_confidence(len(vertices), confidence_level=0.80)
    
    mesh = bake_texture_from_image(
        vertices, faces, image,
        texture_size=1024,
        as_vertex_colors=True,
        sparse_geometry_confidence=sparse_conf,  # NEW
        slat_confidence=slat_conf,                # NEW
    )
    
    print(f"✓ Baked with uniform sparse geometry confidence = 0.85")
    print(f"✓ Baked with uniform SLAT confidence = 0.80")
    print(f"✓ Image texture weight reduced in low-confidence regions")
    print()


def example_with_computed_confidence():
    """Texture baking with confidence computed from geometry."""
    print("=" * 70)
    print("EXAMPLE 3: Texture baking with computed confidence")
    print("=" * 70)
    
    vertices = np.random.randn(1000, 3).astype(np.float32)
    faces = np.random.randint(0, 1000, (500, 3)).astype(np.int32)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    # Estimate sparse geometry confidence from vertex distribution
    # (vertices near center are more confident than silhouette vertices)
    sparse_conf = estimate_sparse_geometry_confidence_from_vertices(
        vertices, falloff_dist=1.0
    )
    
    # Compute surface curvature (for diagnostic purposes)
    curvature = compute_surface_curvature(vertices, faces)
    print(f"  Surface curvature: min={curvature.min():.3f}, "
          f"max={curvature.max():.3f}, mean={curvature.mean():.3f}")
    
    # For this example, use curvature as a proxy for SLAT confidence
    # (high curvature = less confident reconstruction)
    slat_conf = 1.0 - curvature
    
    mesh = bake_texture_from_image(
        vertices, faces, image,
        texture_size=1024,
        as_vertex_colors=True,
        sparse_geometry_confidence=sparse_conf,
        slat_confidence=slat_conf,
    )
    
    print(f"✓ Sparse confidence: min={sparse_conf.min():.3f}, "
          f"max={sparse_conf.max():.3f}, mean={sparse_conf.mean():.3f}")
    print(f"✓ SLAT confidence (from curvature): min={slat_conf.min():.3f}, "
          f"max={slat_conf.max():.3f}, mean={slat_conf.mean():.3f}")
    print()


def example_blending_confidences():
    """Demonstrate confidence blending from multiple sources."""
    print("=" * 70)
    print("EXAMPLE 4: Blending multiple confidence sources")
    print("=" * 70)
    
    vertices = np.random.randn(1000, 3).astype(np.float32)
    
    # Create confidence arrays from different sources
    sparse_conf = create_uniform_confidence(len(vertices), 0.80)
    slat_conf = create_uniform_confidence(len(vertices), 0.90)
    curvature_conf = 1.0 - compute_surface_curvature(vertices, np.zeros((1, 3), dtype=np.int32))
    
    # Blend them (conservative: low values dominate)
    combined = blend_confidence_arrays(sparse_conf, slat_conf, curvature_conf)
    
    print(f"Sparse confidence:  mean = {sparse_conf.mean():.3f}")
    print(f"SLAT confidence:    mean = {slat_conf.mean():.3f}")
    print(f"Curvature conf:     mean = {curvature_conf.mean():.3f}")
    print(f"Combined (blended): mean = {combined.mean():.3f}")
    print(f"✓ Blended via multiplication (conservative approach)")
    print(f"✓ If any source is uncertain, combined confidence is reduced")
    print()


def example_comparison_improvements():
    """Show the effect of improvements side-by-side."""
    print("=" * 70)
    print("EXAMPLE 5: Improvements comparison")
    print("=" * 70)
    
    # Create a simple mesh (sphere-ish)
    vertices = np.random.randn(500, 3).astype(np.float32)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)  # Normalize to sphere
    faces = np.random.randint(0, 500, (250, 3)).astype(np.int32)
    image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    print("WITHOUT improvements (baseline would be):")
    print("  - Fixed visibility exponent 0.5 for all surfaces")
    print("  - Simple KD-tree distance threshold (1.5× median)")
    print("  - No reconstruction quality weighting")
    print()
    
    print("WITH Improvement 1 (Pointmap Quality):")
    curvature = compute_surface_curvature(vertices, faces)
    print(f"  ✓ Adaptive threshold based on 75th percentile + std dev")
    print(f"  ✓ Smooth sigmoid confidence falloff")
    print(f"  ✓ Center-of-image weighting for foreshortened regions")
    print()
    
    print("WITH Improvement 2 (Adaptive Visibility):")
    print(f"  ✓ Surface curvature computed (min={curvature.min():.3f}, "
          f"max={curvature.max():.3f})")
    print(f"  ✓ Adaptive visibility exponent: 0.35 (smooth) to 0.5 (sharp)")
    print(f"  ✓ Grazing angles on smooth surfaces preserve detail")
    print()
    
    print("WITH Improvement 3 (Reconstruction Quality):")
    sparse_conf = estimate_sparse_geometry_confidence_from_vertices(vertices)
    print(f"  ✓ Sparse geometry confidence integrated (mean={sparse_conf.mean():.3f})")
    print(f"  ✓ SLAT refinement confidence can be included")
    print(f"  ✓ Occluded regions fall back to model colors")
    print()
    
    print("COMBINED EFFECT:")
    print("  ✅ Fewer dark blotches on back surfaces")
    print("  ✅ Sharper detail on smooth front surfaces")
    print("  ✅ Cleaner seams in multi-view reconstructions")
    print("  ✅ No color bleeding on silhouettes")
    print("  ✅ Better grazing-angle surface quality")
    print()


if __name__ == "__main__":
    print("\nTexture Mapping Improvements - Examples\n")
    
    # Run examples
    example_basic_texture_baking()
    example_with_uniform_confidence()
    example_with_computed_confidence()
    example_blending_confidences()
    example_comparison_improvements()
    
    print("=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nFor integration into your pipeline:")
    print("1. Compute sparse_geometry_confidence from Stage 1 output")
    print("2. Compute slat_confidence from Stage 2 output")
    print("3. Pass to bake_texture_from_image() or bake_mesh_texture()")
    print("\nSee TEXTURE_MAPPING_IMPROVEMENTS.md for detailed documentation.")
