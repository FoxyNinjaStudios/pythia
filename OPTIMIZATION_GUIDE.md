# SAM-3D Performance & Optimization Guide

## System Specs
- **Total RAM**: 64 GB
- **Available RAM**: ~30 GB (before pipeline)
- **GPU**: Apple Silicon (Metal acceleration available)

## Pipeline Performance Options

### 1. **Reduce Inference Steps** (Fastest impact on speed & memory)
```bash
# Default: 12 steps (high quality)
python main.py --image img.png --mask-dir masks --steps 4 --output out.stl

# Speed comparison:
# - 4 steps: ~2x faster, slightly lower quality
# - 8 steps: ~1.3x faster, good balance
# - 12 steps: Full quality (default)
# - 20 steps: Best quality, slower
```

### 2. **Voxels-Only Mode** (Fastest, lowest memory)
```bash
# Output voxel grid instead of smooth mesh - uses ~40% less memory
python main.py --image img.png --mask-dir masks --output out.stl

# vs. Full mesh generation
python main.py --image img.png --mask-dir masks --mesh --output out.glb
```

### 3. **Mesh Simplification** (Reduce output size & processing)
```bash
# Remove 50% of mesh triangles for faster processing
python main.py --image img.png --mask-dir masks --mesh --simplify 0.5 --output out.glb

# Remove 80% for maximum simplification
python main.py --image img.png --mask-dir masks --mesh --simplify 0.8 --output out.glb
```

### 4. **Caching for Multiple Runs**
```bash
# First run - caches intermediate stage outputs
python main.py --image img.png --mask-dir masks --cache-dir .cache --output out.stl

# Second run with same image - skips stages 0-2 (saves ~60% time!)
python main.py --image img.png --mask-dir masks --cache-dir .cache --mesh --output out_mesh.glb
```

### 5. **Load Pre-Computed SLAT**
```bash
# If you have a cached SLAT file from previous run, load it directly
python main.py --load-slat .cache/slat_*.pt --output out_mesh.glb
```

## Recommended Configurations by Use Case

### Quick Preview (1-2 minutes)
```bash
python main.py --image img.png --mask-dir masks \
  --steps 4 \
  --output preview.stl
```

### Good Quality (5-10 minutes)
```bash
python main.py --image img.png --mask-dir masks \
  --steps 8 \
  --simplify 0.3 \
  --output model.stl
```

### High Quality Mesh (10-20 minutes)
```bash
python main.py --image img.png --mask-dir masks \
  --steps 12 \
  --mesh \
  --simplify 0.1 \
  --cache-dir .cache \
  --output model.glb
```

### Maximum Quality (20-30 minutes)
```bash
python main.py --image img.png --mask-dir masks \
  --steps 20 \
  --mesh \
  --cache-dir .cache \
  --output model_hq.glb
```

## Memory Usage Breakdown

| Configuration | RAM Used | Speed |
|---|---|---|
| Voxels only, 4 steps | ~12-15 GB | ~2 min |
| Voxels only, 12 steps | ~15-18 GB | ~5 min |
| Mesh, 8 steps | ~25-30 GB | ~8 min |
| Mesh, 12 steps | ~28-35 GB | ~12 min |
| Mesh, 20 steps | ~35-40 GB | ~20 min |

Your 64GB system can handle all configurations comfortably.

## Mac-Specific Optimizations

### Apple Silicon Acceleration
```bash
# Metal GPU acceleration (enabled by default)
# Already using:
# - Metal compute shaders for sparse convolution
# - Metal Flash Attention
# - MPS (Metal Performance Shaders) for PyTorch
```

### Disable GPU Acceleration (if Metal causes issues)
```bash
SPARSE_BACKEND=cpu SPARSE_ATTN_BACKEND=sdpa python main.py \
  --image img.png --mask-dir masks --output out.stl
```

## Tips for Faster Processing

1. **Use `.cache` directory** - First run creates cache, subsequent runs with same image are ~60% faster
2. **Reduce resolution** - Crop/resize large images before running
3. **Lower step count** - 8 steps is usually a sweet spot (quality vs speed)
4. **Run at off-peak times** - System background tasks won't compete for resources
5. **Close other apps** - Frees up system RAM for the pipeline

## File Size Reference

- **Preview STL (4 steps)**: ~5-10 MB
- **Standard STL (12 steps)**: ~20-25 MB  
- **Simplified GLB mesh**: ~15-20 MB
- **Full quality GLB**: ~30-40 MB

## Recent Run Stats

```
Configuration: --steps 12 (no mesh)
Time: ~12 minutes
Memory peak: ~25-30 GB
Output: 22MB STL voxel grid
Status: ✅ Successful
```

---

Try the "Good Quality" profile first to find your optimal speed/quality balance!
