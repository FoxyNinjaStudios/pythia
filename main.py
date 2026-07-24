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
# Use MPS backend (PyTorch native GPU) instead of Metal GPU due to stability issues
os.environ.setdefault('SPARSE_BACKEND', 'mps')
os.environ.setdefault('SPARSE_ATTN_BACKEND', 'sdpa')

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
    layout: bool = False,
    layout_refine: bool = False,
    distill: bool = False,          # stage 2 (SLAT) distillation — SLAT is NOT distilled; keep off
    ss_distill: bool = True,        # stage 1 shortcut distillation — required for shortcut sampling
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
        layout: If True, also emit a scene-placed GLB positioning the object in
            camera space using the predicted pose (written as <output>_placed.glb).
        layout_refine: If True, refine the pose against the pointmap + mask (ICP +
            render-compare) before placement. Slower (runs on CPU).
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
        with_layout_postprocess=layout,
        layout_refine=layout_refine,
        use_stage1_distillation=ss_distill,
        use_stage2_distillation=distill,
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
        mesh.export(output_path, file_type='glb')
        print(f"[OUTPUT] Mesh saved to: {output_path}")
        ply_path = os.path.splitext(output_path)[0] + ".ply"
        try:
            mesh.export(ply_path, file_type='ply')
            print(f"[OUTPUT] Mesh saved to: {ply_path}")
        except Exception as e:
            print(f"[OUTPUT] PLY export skipped: {e}")

        # Scene-placed GLB (object positioned in camera space via the predicted pose).
        placed = output.get("glb_placed")
        if placed is not None:
            placed_path = os.path.splitext(output_path)[0] + "_placed.glb"
            try:
                placed.export(placed_path, file_type='glb')
                iou = output.get("layout_iou")
                iou_str = f" (layout IoU {iou})" if iou is not None else ""
                print(f"[OUTPUT] Placed mesh saved to: {placed_path}{iou_str}")
            except Exception as e:
                print(f"[OUTPUT] Placed GLB export skipped: {e}")
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
        description="SAM-3D MPS Pipeline: Image + Mask -> Voxels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image"
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
        "--layout",
        action="store_true",
        help="Also emit a scene-placed GLB (<output>_placed.glb) positioning the object in "
             "camera space using the model's predicted pose."
    )
    parser.add_argument(
        "--layout-refine",
        action="store_true",
        help="With --layout, refine the pose against the pointmap + mask (ICP + "
             "render-compare) before placement. Slower; runs on CPU."
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Also distill STAGE 2 (SLAT). The released SLAT weights are not shortcut-"
             "distilled, so this is experimental and usually degrades texture quality; "
             "leave it off. Stage 1 is distilled by default (see --ss-distill)."
    )

    args = parser.parse_args()
    
    if not args.mask and not args.mask_dir:
        parser.error("Must provide either --mask or --mask-dir")
    
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
        layout=args.layout,
        layout_refine=args.layout_refine,
        distill=args.distill,
        ss_distill=args.ss_distill,
    )


if __name__ == "__main__":
    main()
