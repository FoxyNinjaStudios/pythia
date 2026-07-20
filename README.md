# SAM 3D Objects for Apple Silicon

Efficient 3D object reconstruction from images on macOS using Metal Performance Shaders (MPS). Gaussian splatting and color baking are not supported.

**Original Image**
<p align="left">
  <img src="images/shutterstock_stylish_kidsroom_1640806567/image.png" width="600"/>
</p>

<table>
<tr>
<th>Mask</th>
<th>3D Reconstruction</th>
</tr>
<tr>
<td><img src="images/shutterstock_stylish_kidsroom_1640806567/14.png" width="300"/></td>
<td><img src="demo-1.png" width="300"/></td>
</tr>
<tr>
<td><img src="images/shutterstock_stylish_kidsroom_1640806567/0.png" width="300"/></td>
<td><img src="demo-2.png" width="300"/></td>
</tr>
</table>

Using **SAM 3D** by Meta AI:
- [Paper (arXiv)](https://arxiv.org/abs/2511.16624)
- [Official GitHub](https://github.com/facebookresearch/sam-3d-objects)
- [Model Weights (Hugging Face)](https://huggingface.co/facebook/sam-3d-objects)

## Installation

1. **Clone and install dependencies**:
   ```bash
   git clone <this-repo>
   cd Sam3D-MLX
   uv sync
   ```

2. **Download checkpoints** from [Hugging Face](https://huggingface.co/facebook/sam-3d-objects) and place them in `checkpoints/hf/`:
   ```bash
   mkdir -p checkpoints/hf
   # Download pipeline.yaml and all .pt/.safetensors files into checkpoints/hf/
   ```

3. **Configure environment**
   ```fish
   set -x PYTORCH_MPS_HIGH_WATERMARK_RATIO 0.0
   set -x SPARSE_BACKEND mps
   set -x SPARSE_ATTN_BACKEND sdpa
   set -x PYTHONPATH .
   ```

## Usage

**Recommended: Use conda environment directly** (PyTorch3D compatibility)

```bash
conda activate sam-3d-mlx
python main.py \
    --image images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask-dir images/shutterstock_stylish_kidsroom_1640806567 \
    --mask-index 0 \
    --mesh \
    --output outputs/reconstruction.glb
```

**Alternative: Use uv (if uv.lock is available)**
```bash
uv run python main.py \
    --image images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask-dir images/shutterstock_stylish_kidsroom_1640806567 \
    --mesh \
    --output outputs/reconstruction.glb
```

### Key Arguments
| Argument | Description |
|----------|-------------|
| `--image` | Input image path |
| `--mask-dir` / `--mask-index` | SAM-style mask directory and index |
| `--steps` | Diffusion steps (default: 12) |
| `--mesh` | Output a full smooth GLB mesh |
| `--output` | File path for results (.glb, .stl) |

## Structure
```
checkpoints/hf/     # Model weights (download from HuggingFace)
images/             # Example dataset
outputs/            # 3D model results
.cache/             # Intermediate latent files
sam3d_objects/      # Core model logic
```

## What Was Done

This port adapts the original CUDA-based [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) pipeline to run on Apple Silicon:

1. **Removed CUDA dependencies**: Replaced `spconv-cu121`, `xformers`, and other CUDA-specific packages.
2. **[MPS Backend](https://developer.apple.com/metal/pytorch/)**: Rewired model loading and inference to use PyTorch's Metal Performance Shaders backend.
3. **Metal Sparse Convolution**: Custom Metal compute shaders for voxel processing:
   - [`sparse_conv.metal`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/sparse_conv.metal) — GPU kernels
   - [`conv_metal.py`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_metal.py) — PyObjC wrapper
4. **Metal Flash Attention**: GPU-accelerated attention for sparse transformers:
   - [`flash_attn.metal`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/flash_attn.metal) — GPU kernels
   - [`metal_flash_attn.py`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/metal_flash_attn.py) — Python wrapper
5. **[Low-Memory Pipeline](sam3d_objects/pipeline/inference_pipeline_low_memory.py)**: Sequential model loading to fit within 48GB RAM.

## Troubleshooting

### `ImportError: Symbol not found` when using `uv run`

**Problem**: `uv run` creates a fresh `.venv` and installs dependencies, but PyTorch3D's C++ extensions don't link correctly with PyTorch's libraries.

**Solution**: Use the conda environment directly instead:
```bash
# Remove conflicting .venv
rm -rf .venv

# Use conda (recommended)
conda activate sam-3d-mlx
python main.py --image ... --mask-dir ... --output ...
```

**Why**: PyTorch3D has compiled C++ extensions that need to match PyTorch's ABI. The conda environment has PyTorch3D built from source with matching versions.

### Metal GPU segmentation faults

**Problem**: Custom Metal GPU kernels may cause crashes during mesh generation.

**Solution**: The default backend is now MPS (PyTorch's native Metal support), which is stable and recommended. No action needed.

**For advanced users**: If you need different sparse backends:
```bash
# MPS (default, recommended)
SPARSE_BACKEND=mps SPARSE_ATTN_BACKEND=sdpa python main.py ...

# Pure CPU
SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=sdpa python main.py ...
```
