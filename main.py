# SAM-3D: Single-image 3D reconstruction for Apple Silicon
# 
# Usage:
#     uv run python main.py --image images/shutterstock_stylish_kidsroom_1640806567/image.png --mask-dir images/shutterstock_stylish_kidsroom_1640806567 --mask-index 14 --output outputs/voxels.stl

import os
import sys
import time
import argparse
import resource

# Set environment before any imports
os.environ.setdefault('LIDRA_SKIP_INIT', '1')

# Reconstruction runs on the CPU. The upstream native-Metal reconstruction
# kernels were removed, so the 3-D generative pipeline is CPU-only; MoGe depth
# still runs on Metal (MPS).

# Limit CPU cores to prevent system freeze
MAX_CORES = 14
os.environ['OMP_NUM_THREADS'] = str(MAX_CORES)
os.environ['MKL_NUM_THREADS'] = str(MAX_CORES)
os.environ['OPENBLAS_NUM_THREADS'] = str(MAX_CORES)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(MAX_CORES)
os.environ['NUMEXPR_MAX_THREADS'] = str(MAX_CORES)

import torch
torch.set_num_threads(MAX_CORES)
torch.set_num_interop_threads(MAX_CORES // 2)

# Disable MPS detection for the 3-D generative pipeline (must run before any
# pipeline module reads torch.backends.mps). MoGe depth still runs on Metal via
# SAM3D_MOGE_DEVICE.
if torch.backends.mps.is_available():
    os.environ['SAM3D_MOGE_DEVICE'] = 'mps'   # keep MoGe depth on Metal
torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
print("[SAM3D] 3-D reconstruction on CPU; MoGe depth on Metal.")

import numpy as np
from PIL import Image



def get_memory_gb():
    """Get current memory usage in GB (macOS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 ** 3)


def load_image(path: str) -> np.ndarray:
    """Load image from file path."""
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path: str) -> np.ndarray:
    """Load mask from file path."""
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def load_mask_from_file(path: str) -> np.ndarray:
    """Load mask from PNG/JPG file."""
    return load_mask(path)


def load_mask_from_dir(mask_dir: str, index: int) -> np.ndarray:
    """Load mask from directory using index (SAM format)."""
    mask_path = os.path.join(mask_dir, f"{index}.png")
    if not os.path.exists(mask_path):
        # Fallback to .jpg if .png not found
        if os.path.exists(os.path.join(mask_dir, f"{index}.jpg")):
            mask_path = os.path.join(mask_dir, f"{index}.jpg")
        else:
            raise FileNotFoundError(f"Mask path {mask_path} (or .jpg) does not exist")
    return load_mask(mask_path)


def export_voxels_to_stl(coords: torch.Tensor, output_path: str, voxel_size: float = 1.0):
    """Export sparse voxel coordinates as an STL mesh."""
    import trimesh
    
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    
    xyz = coords[:, 1:4].astype(np.float32)  # Skip batch index
    xyz = xyz - xyz.mean(axis=0)  # Center
    xyz = xyz * voxel_size
    
    print(f"[STL] Creating mesh from {len(xyz)} voxels...")
    
    meshes = []
    cube = trimesh.creation.box(extents=[voxel_size * 0.9] * 3)
    
    for i, (x, y, z) in enumerate(xyz):
        voxel_cube = cube.copy()
        voxel_cube.apply_translation([x, y, z])
        meshes.append(voxel_cube)
        
        if (i + 1) % 5000 == 0:
            print(f"[STL] Processed {i + 1}/{len(xyz)} voxels...")
    
    print("[STL] Merging voxels into single mesh...")
    combined = trimesh.util.concatenate(meshes)
    combined.export(output_path, file_type='stl')
    print(f"[STL] Saved to: {output_path}")
    
    return combined



def run_pipeline(
    image_path: str,
    mask_path: str = None,
    mask_dir: str = None,
    mask_index: int = 0,
    output_path: str = "output.glb",
    inference_steps: int = 12,      # stage 2 (SLAT: texture & refinement) — genuine flow matching
    ss_steps: int = 2,              # stage 1 (sparse structure) — shortcut-distilled, shipped default
    seed: int = 42,
    output_mesh: bool = True,
    cache_dir: str = None,
    simplify_ratio: float = None,
    load_slat: str = None,
    texture_bake: bool = False,
    texture_bake_source: str = "gaussian",
    texture_size: int = 2048,
    vertex_color_source: str = "gaussian",
    distill: bool = False,          # stage 2 (SLAT) distillation — SLAT is NOT distilled; keep off
    ss_distill: bool = True,        # stage 1 shortcut distillation — required for shortcut sampling
    refine_mask: bool = False,      # opt-in: clean + anti-alias (soft-alpha) the mask before use
    smooth_iterations: int = 0,     # opt-in: Taubin-smooth the mesh to de-stair the 64^3 voxel grid
    full_res_geometry: bool = True,  # keep large objects at native 64^3 (prune interior voxels); env SAM3D_FULL_RES_GEOMETRY=0 disables
):
    """
    Run SAM-3D pipeline: image + mask -> 3D mesh.

    Args:
        image_path: Path to input image
        mask_path: Path to mask file (PNG/JPG)
        mask_dir: Directory containing SAM masks (alternative to mask_path)
        mask_index: Index of mask in mask_dir
        output_path: Path for output file (GLB for mesh, STL for voxels)
        inference_steps: Number of diffusion steps (higher = better quality, slower)
        seed: Random seed for reproducibility
        output_mesh: If True, run full pipeline for smooth mesh. If False, voxels only.
        cache_dir: Directory to cache intermediate outputs (SLAT). Second run will skip Stages 0-2.
        simplify_ratio: Ratio of triangles to remove (0.0=none, 0.95=heavy). None picks a
            default: 0.9 when baking a texture (the portable rasterizer is slow on a full-res
            mesh), 0.0 otherwise (max geometry quality for vertex color).
        load_slat: Path to a cached SLAT .pt file to load (skips stages 0-2).
        texture_bake: If True, bake a UV texture atlas instead of per-vertex color.
        texture_bake_source: "gaussian" (decode the Gaussian appearance rep) or "vertex".
        texture_size: Baked texture edge length in pixels.
        vertex_color_source: For the default (non-bake) path, where per-vertex color comes
            from: "gaussian" (saturated SH-DC appearance, recommended) or "mesh" (the
            decoder's washed-out vertex head).
        inference_steps: Stage-2 (SLAT texture & refinement) flow-matching steps. Default 12.
        ss_steps: Stage-1 (sparse-structure / geometry) steps. This model is shortcut-
            distilled; 2 is the shipped default. Values above 4 rarely help.
        distill: Distill stage 2 (SLAT). The released SLAT weights are not distilled, so
            leave this off. Default False.
        ss_distill: Use shortcut-distilled sampling for stage 1 (step-size conditioning,
            CFG-free, ~1 eval/step). Required for the low ss_steps to be valid. Default True.
    """
    from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory

    # Resolve the simplification default. Baking rasterizes every face into the UV
    # atlas in pure Python, so a full-res mesh (hundreds of K faces) is very slow;
    # default to 0.9 when baking. Vertex color needs no simplification.
    if simplify_ratio is None:
        simplify_ratio = 0.9 if texture_bake else 0.0
        if texture_bake:
            print(f"[INFO] --bake: defaulting to --simplify {simplify_ratio} "
                  f"(pass --simplify 0 for full-res, much slower)")

    print("=" * 60)
    print("SAM-3D MPS Pipeline")
    print("=" * 60)
    
    # Ensure directories exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check for MPS
    if torch.backends.mps.is_available():
        print("MPS available: True")
    else:
        print("MPS available: False (using CPU - will be slow!)")
        
    print(f"Initial memory: {get_memory_gb():.1f} GB")
    print()
    
    # Load image
    print(f"[INPUT] Loading image: {image_path}")
    image = load_image(image_path)
    print(f"  Image shape: {image.shape}")
    
    # Load mask
    if mask_path:
        print(f"[INPUT] Loading mask: {mask_path}")
        mask = load_mask_from_file(mask_path)
    elif mask_dir:
        print(f"[INPUT] Loading mask from {mask_dir}, index {mask_index}")
        mask = load_mask_from_dir(mask_dir, mask_index)
    else:
        raise ValueError("Must provide either --mask or --mask-dir + --mask-index")
    
    print(f"  Mask shape: {mask.shape}")

    # Optional: clean + anti-alias the mask (soft alpha) before reconstruction.
    # Off by default so intentionally hard-edged batch masks are left untouched.
    if refine_mask:
        from sam_wrapper import refine_mask as _refine_mask
        mask = _refine_mask((mask.astype(np.uint8) * 255) if mask.dtype == bool else mask)
        print(f"  Mask refined (soft alpha): {mask.shape}, dtype={mask.dtype}")
    print()
    
    # Initialize pipeline
    print("[INIT] Creating pipeline...")
    t0 = time.perf_counter()
    
    config_path = "checkpoints/hf/pipeline.yaml"
    pipeline = InferencePipelineLowMemory(
        config_path=config_path,
        device="cpu",
        dtype="float16",
        cache_dir=cache_dir,
    )
    
    print(f"[INIT] Pipeline ready in {time.perf_counter() - t0:.1f}s")
    print(f"[INIT] Memory: {get_memory_gb():.1f} GB")
    print()
    
    # Run inference
    _ss_mode = "shortcut-distilled" if ss_distill else "flow-matching+CFG"
    _slat_mode = "shortcut-distilled" if distill else "flow-matching+CFG"
    print(f"[RUN] Stage 1 (sparse structure): {ss_steps} steps · {_ss_mode}")
    print(f"[RUN] Stage 2 (SLAT texture/refine): {inference_steps} steps · {_slat_mode}")
    print(f"[RUN] mesh={output_mesh}, simplify={simplify_ratio}")
    print("-" * 40)
    t_start = time.perf_counter()
    
    output = pipeline.run(
        image,
        mask,
        seed=seed,
        stage1_only=not output_mesh,  # Full pipeline if requesting mesh
        stage1_inference_steps=ss_steps,
        stage2_inference_steps=inference_steps,
        decode_formats=["mesh"] if output_mesh else [],
        simplify_ratio=simplify_ratio,
        load_slat=load_slat,
        texture_bake=texture_bake,
        texture_bake_source=texture_bake_source,
        texture_size=texture_size,
        vertex_color_source=vertex_color_source,
        use_stage1_distillation=ss_distill,
        use_stage2_distillation=distill,
        use_stage2_mps=not args.no_stage2_mps,
        full_res_geometry=full_res_geometry,
    )
    
    t_total = time.perf_counter() - t_start
    print("-" * 40)
    print(f"[DONE] Inference completed in {t_total:.1f}s")
    print(f"[DONE] Peak memory: {get_memory_gb():.1f} GB")
    print()
    
    # Export voxels ALWAYS (raw output before meshing)
    coords = output.get('coords')
    if coords is not None:
        voxel_path = output_path
        if voxel_path.endswith('.glb'):
            voxel_path = voxel_path.replace('.glb', '_voxels.stl')
        elif not voxel_path.endswith('.stl'):
            voxel_path = voxel_path + '_voxels.stl'
        
        print(f"[OUTPUT] Exporting {coords.shape[0]} raw voxels to {voxel_path}")
        export_voxels_to_stl(coords, voxel_path)
    
    # Export smooth mesh if requested
    if output_mesh and "glb" in output and output["glb"] is not None:
        # to_glb already produced a colored/textured trimesh (per-vertex color by
        # default, or a baked UV atlas when --bake). Write both GLB and PLY.
        mesh = output["glb"]
        # Optional: sand off the 64^3 voxel staircase on oblique silhouettes
        # (the 2D mask is full-res/soft; the geometry grid is not).
        if smooth_iterations and smooth_iterations > 0:
            try:
                from mesh_utils import taubin_smooth
                mesh = taubin_smooth(mesh, iterations=smooth_iterations)
                print(f"[MESH] Taubin-smoothed ({smooth_iterations} iterations)")
            except Exception as e:
                print(f"[MESH] smoothing skipped: {e}")
        mesh.export(output_path, file_type='glb')
        print(f"[OUTPUT] Mesh saved to: {output_path}")
        ply_path = os.path.splitext(output_path)[0] + ".ply"
        try:
            mesh.export(ply_path, file_type='ply')
            print(f"[OUTPUT] Mesh saved to: {ply_path}")
        except Exception as e:
            print(f"[OUTPUT] PLY export skipped: {e}")
    elif not output_mesh:
        print("[OUTPUT] Voxel export complete (mesh generation skipped).")
    else:
        print("[ERROR] No GLB mesh generated!")
        return None
    
    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="SAM-3D MPS Pipeline: Single-view or Multi-view 3D reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Single-view arguments
    parser.add_argument(
        "--image", "-i",
        required=False,
        help="Path to input image (required for single-view mode)"
    )
    parser.add_argument(
        "--mask", "-m",
        help="Path to mask file (PNG/JPG)"
    )
    parser.add_argument(
        "--mask-dir",
        help="Directory containing SAM masks"
    )
    parser.add_argument(
        "--mask-index",
        type=int,
        default=0,
        help="Index of mask in mask-dir (default: 0)"
    )
    
    # Multi-view arguments (NEW)
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Enable multi-view mode for improved geometry and pose stability"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory with multiple views (required with --multi-view)"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Directory with masks for each view (optional with --multi-view)"
    )
    parser.add_argument(
        "--view-indices",
        type=str,
        default=None,
        help="Comma-separated view indices (e.g., '0,1,2') to select specific views"
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        choices=["stochastic", "multidiffusion"],
        default="stochastic",
        help="Multi-view fusion mode: 'stochastic' (random view per step) or 'multidiffusion' (fuse all)"
    )
    parser.add_argument(
        "--view-weighting",
        type=str,
        choices=["uniform", "entropy"],
        default="uniform",
        help="View weighting strategy for multi-view fusion"
    )
    parser.add_argument(
        "--num-views-select",
        type=int,
        default=None,
        help="Limit to N best views (optional)"
    )
    
    # Shared arguments
    parser.add_argument(
        "--output", "-o",
        default="outputs/voxels.stl",
        help="Output file path (default: outputs/voxels.stl)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Stage-2 (SLAT texture & refinement) flow-matching steps only. Default 12. "
             "This stage is genuine flow matching and is not distilled; 12 is correct."
    )
    parser.add_argument(
        "--ss-steps",
        type=int,
        default=2,
        help="Sparse-structure (geometry) diffusion steps. This model is shortcut-"
             "distilled; 2 is the shipped default. Values above 4 are unlikely to "
             "improve quality."
    )
    parser.add_argument(
        "--ss-distill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use shortcut-distilled sampling for stage 1 (step-size conditioning, CFG-"
             "free, ~1 eval/step). Required for --ss-steps to be valid; pass --no-ss-"
             "distill to fall back to CFG flow matching (then use ~12 steps). Default on."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Generate smooth mesh (GLB) instead of voxels (STL). Uses chunked decoding for 48GB RAM."
    )
    parser.add_argument(
        "--voxels-only",
        action="store_true",
        help="Only generate voxels (Stage 1), skip mesh decoding"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory to cache intermediate outputs (default: .cache)"
    )
    
    parser.add_argument(
        "--simplify",
        type=float,
        default=None,
        help="Mesh simplification ratio (0.0=none, 0.95=heavy). Default: 0.0 normally, "
             "or 0.9 when --bake (the portable texture rasterizer is slow on full-res meshes). "
             "Pass an explicit value (including 0) to override."
    )
    parser.add_argument(
        "--load-slat",
        type=str,
        default=None,
        help="Load a cached SLAT .pt file (skips stages 0-2, only runs mesh decoding)"
    )
    parser.add_argument(
        "--bake",
        action="store_true",
        help="Bake a UV texture atlas (portable, no CUDA) instead of per-vertex color."
    )
    parser.add_argument(
        "--bake-source",
        choices=["gaussian", "vertex"],
        default="gaussian",
        help="Color source for --bake: 'gaussian' (decodes the Gaussian appearance rep, higher "
             "fidelity) or 'vertex' (mesh vertex colors). Default: gaussian."
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=2048,
        help="Baked texture edge length in pixels (default: 2048)."
    )
    parser.add_argument(
        "--vertex-color-source",
        choices=["gaussian", "mesh"],
        default="gaussian",
        help="For the default (non-bake) path, source of per-vertex color: 'gaussian' "
             "(saturated SH-DC appearance, recommended) or 'mesh' (the decoder's washed-out "
             "vertex head). Default: gaussian."
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Also distill STAGE 2 (SLAT). The released SLAT weights are not shortcut-"
             "distilled, so this is experimental and usually degrades texture quality; "
             "leave it off. Stage 1 is distilled by default (see --ss-distill)."
    )
    parser.add_argument(
        "--no-stage2-mps",
        action="store_true",
        help="Disable MPS acceleration for Stage 2 (SLAT texture & refinement). "
             "Stage 2 runs on MPS (Metal GPU) by default on Apple Silicon for improved speed. "
             "Use this flag to force CPU-only processing if needed."
    )

    parser.add_argument(
        "--refine-mask",
        action="store_true",
        help="Clean and anti-alias (soft-alpha) the mask before reconstruction: fill "
             "pinholes, drop speckles/islands, and feather the jagged SAM boundary into "
             "sub-pixel coverage (survives into the geometry stage). Off by default so "
             "intentionally hard-edged batch masks are left untouched."
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="ITERS",
        help="Taubin-smooth the output mesh by ITERS iterations to sand off the 64^3 "
             "voxel staircase on oblique edges (volume-preserving, keeps thin parts). "
             "0 = off (default). ~10 removes stepping; higher over-rounds sharp corners."
    )
    parser.add_argument(
        "--full-res-geometry",
        action="store_true",
        help="Keep large objects at native 64^3 instead of silently halving them. Big "
             "objects exceed an int32-safe voxel ceiling and are normally factor-2 "
             "downsampled; this prunes interior voxels first (lossless for the surface) "
             "so the count stays under the ceiling. ON BY DEFAULT; disable with the env "
             "var SAM3D_FULL_RES_GEOMETRY=0. Costs more decode memory/time. "
             "(No retraining; the 64^3->256^3 detail ceiling is unchanged.)"
    )

    args = parser.parse_args()
    
    # Multi-view mode
    if args.multi_view:
        if not args.image_dir:
            parser.error("Multi-view mode (--multi-view) requires --image-dir")
        
        print("=" * 70)
        print("SAM-3D MPS Pipeline (Multi-View Mode)")
        print("=" * 70)
        
        # Load multi-view images and masks
        import glob
        image_files = sorted(
            glob.glob(os.path.join(args.image_dir, "*.png")) +
            glob.glob(os.path.join(args.image_dir, "*.jpg")) +
            glob.glob(os.path.join(args.image_dir, "*.jpeg"))
        )
        
        if not image_files:
            parser.error(f"No images found in {args.image_dir}")
        
        if args.view_indices:
            try:
                indices = [int(i.strip()) for i in args.view_indices.split(",")]
                image_files = [image_files[i] for i in indices if i < len(image_files)]
            except (ValueError, IndexError) as e:
                parser.error(f"Invalid --view-indices: {e}")
        
        if len(image_files) < 2:
            parser.error(f"Multi-view requires at least 2 images, found {len(image_files)}")
        
        images, masks = [], []
        for img_path in image_files:
            images.append(load_image(img_path))
            
            if args.masks_dir:
                base = os.path.basename(img_path)
                mask_name = os.path.splitext(base)[0]
                mask_path = os.path.join(args.masks_dir, f"{mask_name}.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(args.masks_dir, f"{mask_name}.jpg")
                if os.path.exists(mask_path):
                    masks.append(load_mask(mask_path))
                else:
                    print(f"[WARN] Mask not found for {base}, using None")
                    masks.append(None)
            else:
                masks.append(None)
        
        print(f"[INFO] Loaded {len(images)} views from {args.image_dir}")
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
        
        if args.simplify is None:
            args.simplify = 0.9 if args.bake else 0.0
        
        # Import multi-view configuration
        from sam3d_objects.pipeline.inference_pipeline import (
            MultiViewFusionConfig, MultiViewWeightingConfig
        )
        from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory
        
        # Create fusion configuration
        fusion_config = MultiViewFusionConfig(
            fusion_mode=args.fusion_mode,
            weighting=MultiViewWeightingConfig(
                enabled=(args.view_weighting != "uniform"),
                mode=args.view_weighting,
                num_views_to_select=args.num_views_select,
            ),
        )
        
        # Initialize pipeline
        pipeline = InferencePipelineLowMemory(
            config_path="checkpoints/hf/pipeline.yaml",
            device="cpu",
            dtype="float16",
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        # Run multi-view reconstruction
        output = pipeline.run_multi_view(
            images, masks,
            seed=args.seed,
            stage1_only=args.voxels_only,
            with_mesh_postprocess=not args.voxels_only,
            with_texture_baking=args.bake,
            use_vertex_color=not args.bake,
            stage1_inference_steps=args.ss_steps,
            stage2_inference_steps=args.steps,
            use_stage1_distillation=args.ss_distill,
            use_stage2_distillation=args.distill,
            use_stage2_mps=not args.no_stage2_mps,
            decode_formats=["mesh"] if not args.voxels_only else None,
            fusion_config=fusion_config,
        )
        
        # Post-processing
        if not args.voxels_only and "glb" in output and output["glb"] is not None:
            result_mesh = output["glb"]
            
            if args.smooth > 0:
                try:
                    from mesh_utils import taubin_smooth
                    print(f"[INFO] Smoothing mesh ({args.smooth} iterations)...")
                    result_mesh = taubin_smooth(result_mesh, iterations=args.smooth)
                except Exception as exc:
                    print(f"[WARN] Smoothing skipped: {exc}")
            
            if args.simplify and args.simplify > 0:
                try:
                    print(f"[INFO] Simplifying mesh to {100*(1-args.simplify):.1f}% vertices...")
                    result_mesh = result_mesh.simplify(args.simplify)
                except Exception as exc:
                    print(f"[WARN] Simplification skipped: {exc}")
            
            if args.bake:
                try:
                    result_mesh.export(args.output, file_type="glb", include_normals=True)
                except Exception as exc:
                    print(f"[ERROR] Failed to export GLB: {exc}")
                    raise
            else:
                result_mesh.export(args.output, file_type="glb")
            
            print(f"[SUCCESS] Multi-view reconstruction saved to {args.output}")
        else:
            print(f"[SUCCESS] Multi-view reconstruction (voxels only) completed")
        
        return
    
    # Single-view mode (existing logic)
    if not args.image and not args.mask_dir:
        parser.error("Must provide either --image (single-view) or --multi-view --image-dir")
    
    if not args.image:
        parser.error("Single-view mode requires --image")
    
    if not args.mask and not args.mask_dir:
        parser.error("Must provide either --mask or --mask-dir")
    
    print("=" * 70)
    print("SAM-3D MPS Pipeline (Single-View Mode)")
    print("=" * 70)
    
    # Determine output mode
    output_mesh = args.mesh and not args.voxels_only
    
    # Set default output path based on mode
    output_path = args.output
    if output_path == "outputs/voxels.stl" and args.mesh:
        output_path = "outputs/output.glb"
    elif output_path == "outputs/output.glb" and args.voxels_only:
        output_path = "outputs/voxels.stl"
    
    run_pipeline(
        image_path=args.image,
        mask_path=args.mask,
        mask_dir=args.mask_dir,
        mask_index=args.mask_index,
        output_path=output_path,
        inference_steps=args.steps,
        ss_steps=args.ss_steps,
        seed=args.seed,
        output_mesh=output_mesh,
        cache_dir=args.cache_dir,
        simplify_ratio=args.simplify,
        load_slat=args.load_slat,
        texture_bake=args.bake,
        texture_bake_source=args.bake_source,
        texture_size=args.texture_size,
        vertex_color_source=args.vertex_color_source,
        distill=args.distill,
        ss_distill=args.ss_distill,
        refine_mask=args.refine_mask,
        smooth_iterations=args.smooth,
        full_res_geometry=args.full_res_geometry,
    )


if __name__ == "__main__":
    main()
