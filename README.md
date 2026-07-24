# SAM 3D Objects for Apple Silicon

Turn a single photo into a 3D object on a Mac. This is a native Apple-Silicon
port of Meta AI's **SAM 3D Objects** — it runs the full single-image
reconstruction pipeline on the GPU via Metal Performance Shaders (MPS) plus
custom Metal compute kernels, with no CUDA required.

It ships two ways:

- **A packaged macOS app** — an interactive desktop application: drop in an
  image, click points to segment the object with SAM, and reconstruct a
  textured 3D model that you can orbit, inspect, and export.
- **A command-line tool** (`main.py`) for scripted / batch reconstruction.

Outputs include watertight **GLB meshes** (per-vertex color or baked UV
texture atlas) and real **3D Gaussian-Splatting `.ply`** files.

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

## Requirements

| | |
|---|---|
| **Hardware** | Apple Silicon Mac, M-series (M1 or later) |
| **Memory** | **24 GB unified memory** |
| **OS** | macOS 13+ |
| **Python** | 3.11 (conda recommended) |

**Not supported:**

- **Intel Macs** — no Metal Performance Shaders (MPS) backend; the pipeline has
  no CPU-only fast path.
- **A-series Macs** (e.g. the MacBook "Neo" class) — insufficient unified memory
  for the working set.
- **iPadOS / iOS** — no general-purpose Python runtime and no on-device
  compilation of the native Metal extensions.

### Memory profile

Sustained usage is ~8–9 GB. Peak stays **under 20 GB** and occurs late in the
pipeline — during mesh decoding and export, *after* the GPU generative stages
have finished and released their buffers. There is no configuration that needs
48 GB; a 24 GB machine has comfortable headroom.

### Why Apple Silicon

The pipeline expands a sparse voxel structure into a high-resolution mesh, and
the transient working set for that step is large. On a discrete-GPU system that
working set must fit in **VRAM** — system RAM cannot substitute, so a card with
too little VRAM simply cannot run the stage regardless of how much RAM the host
has. Consumer NVIDIA cards top out around 24–32 GB of VRAM. Apple's **unified
memory** puts the GPU working set and the CPU-side transients in one pool, so a
24 GB Mac can hold both without a VRAM ceiling.

### Recommended configurations

| Machine | Unified memory | Notes |
|---|---|---|
| **Mac mini (M4 Pro)** | 24 GB | Cheapest supported configuration |
| **MacBook Pro (M-series Pro/Max)** | 24–48 GB | Portable; more headroom for other apps |
| **Mac Studio (M-series Max/Ultra)** | 64 GB+ | Fastest; ample headroom |

## Features

- **Interactive segmentation** — SAM point-prompt masking in the browser UI
  (positive / negative clicks), no manual mask files needed.
- **Single-image 3D reconstruction** — geometry + appearance from one photo.
- **Multiple output formats:**
  - `GLB` mesh with **per-vertex color** (default).
  - `GLB` mesh with a **baked UV texture atlas** (portable, no CUDA / nvdiffrast).
  - Real **3D Gaussian-Splatting `.ply`** for depth-ambiguous / soft previews.
- **Apple-Silicon native** — MPS backend plus hand-written Metal kernels for
  sparse convolution and flash attention.
- **Low-memory pipeline** — sequential stage loading and SLAT caching keep the
  working set small (see [Requirements](#requirements)).
- **Watertight output** — hole filling and floater removal run by default, so the
  exported GLB is closed — suitable for 3D printing, boolean operations, and
  volume computation.
- **Live progress** — streamed pipeline logs and an in-place mask → model
  preview in the web app.

## Two ways to run

### 1. Desktop app (interactive web UI)

The app is a FastAPI server that serves an interactive single-page UI
(`static/index.html`). Start it and open the browser UI:

```bash
conda activate sam-3d
python server.py
# then open http://localhost:8005
```

Workflow: upload an image → click points to segment the object → pick a quality
preset → reconstruct → orbit the result and download **GLB** or **PLY**.

Optional Gaussian-splat export is on by default; disable it with
`SAM3D_SPLAT=0`.

### 2. Command line

```bash
conda activate sam-3d
python main.py \
    --image images/shutterstock_stylish_kidsroom_1640806567/image.png \
    --mask-dir images/shutterstock_stylish_kidsroom_1640806567 \
    --mask-index 0 \
    --mesh \
    --output outputs/reconstruction.glb
```

#### Key Arguments
| Argument | Description |
|----------|-------------|
| `--image` | Input image path |
| `--mask` / `--mask-dir` + `--mask-index` | Object mask (single file or SAM-style directory) |
| `--mesh` | Output a smooth GLB mesh (otherwise voxel STL) |
| `--voxels-only` | Only run stage 1 and export raw voxels (STL); skip mesh decoding |
| `--steps` | Stage-2 (SLAT texture & refinement) flow-matching steps (default: 12). Stage 2 is genuine flow matching and is not distilled. |
| `--ss-steps` | Stage-1 (sparse-structure / geometry) steps (default: 2). This stage is **shortcut-distilled** in the shipped weights, so 2 steps is the intended default; values above 4 rarely help. |
| `--ss-distill` / `--no-ss-distill` | Use shortcut-distilled sampling for stage 1 (step-size conditioning, CFG-free, ~1 eval/step). On by default and required for the low `--ss-steps` to be valid; pass `--no-ss-distill` to fall back to CFG flow matching (then use ~12 steps). |
| `--distill` | Also distill **stage 2** (SLAT). The released SLAT weights are not shortcut-distilled, so this is experimental and usually degrades texture; leave it off. |
| `--simplify` | Mesh decimation ratio (`0.0` = none … `0.95` = heavy) |
| `--vertex-color-source` | `gaussian` (saturated, recommended) or `mesh` |
| `--bake` | Bake a UV texture atlas instead of per-vertex color |
| `--bake-source` | `gaussian` (higher fidelity) or `vertex` |
| `--texture-size` | Baked atlas edge length in px (default: 2048) |
| `--cache-dir` / `--load-slat` | Cache / reuse intermediate SLAT to skip stages 0–2 |
| `--seed` | Random seed for reproducibility (default: 42) |
| `--output` `-o` | Output file (`.glb`, `.stl`) |

## Geometry cleanup

After the mesh is decoded, a post-processing pass runs **by default** to suppress
reconstruction artifacts and keep the output watertight:

- **Hole filling** — small gaps left by decoding/simplification are closed so the
  surface is manifold and closed.
- **Floater removal** — disconnected islands are dropped, keeping the largest
  connected component (the object) and discarding stray voxel debris.

Both passes are guarded: if a step cannot run on a given mesh it is skipped, and
neither can fail the reconstruction job. The result is the watertight GLB used by
the viewer and every export path.

## Mesh editing (web UI)

The web app includes a **stencil-based editor** for trimming unwanted geometry
(background bleed, stray limbs, base plinths) without touching the model code:

1. Drag a stencil primitive — **box, sphere, cylinder, or capsule** — from the
   palette onto the scene.
2. Select it and switch between **move / rotate / scale** tools (Three.js-editor
   style) to position it over the region to delete.
3. Any geometry **inside** a stencil disappears live; stack multiple stencils to
   carve complex regions.

Editing is **non-destructive** — the original mesh is untouched until you export.
A client-side **Fill holes** control re-closes the cut so the edited mesh stays
watertight, and the scene texture is baked into the GLB on download.

## Texture baking

By default the mesh carries **per-vertex color** (`COLOR_0`). That renders
correctly in the in-app viewer and in `model-viewer`, but **macOS Preview, Quick
Look, and USDZ conversion ignore per-vertex color** and show the model grey. For
distribution to those tools, bake the color into a **UV texture atlas** instead:

- **Web UI** — baking runs **client-side** when you download the GLB, so the
  downloaded file displays correctly in Preview / Quick Look out of the box.
- **CLI** — pass `--bake` to write a baked UV atlas (`--bake-source`,
  `--texture-size` tune it). The portable rasterizer needs no CUDA / nvdiffrast.

## Installation

1. **Clone and create the environment** (a conda env is recommended for
   PyTorch3D C++/ABI compatibility):
   ```bash
   git clone https://github.com/FoxyNinjaStudios/pythia
   cd pythia
   conda create -n sam-3d python=3.11
   conda activate sam-3d
   uv pip install -e .        # or: uv sync
   ```

2. **Download checkpoints** from
   [Hugging Face](https://huggingface.co/facebook/sam-3d-objects) into
   `checkpoints/hf/` (the `pipeline.yaml` plus all `.pt` / `.safetensors`
   weights). These weights are governed by Meta's SAM License — see
   [Licensing](#licensing).

3. **Environment variables** (set automatically by `main.py` / `server.py`, but
   useful when running manually):
   ```bash
   export SPARSE_BACKEND=mps
   export SPARSE_ATTN_BACKEND=sdpa
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```

## Structure
```
main.py             # CLI entry point
server.py           # FastAPI web app (segmentation, reconstruct, export)
sam_wrapper.py      # SAM 2 point-prompt segmentation wrapper
static/index.html   # Web UI: segmentation, 3D viewer, stencil editing, client-side baking
splat_export.py     # Optional Gaussian-splat (.ply) export module
texture_baking.py   # UV texture-atlas baking (portable, no CUDA)
sam3d_objects/      # Core model + pipeline (Apple-Silicon port)
checkpoints/hf/     # Model weights (download from Hugging Face)
images/             # Example images + SAM masks
tests/              # Test suite
CHANGELOG.md        # Version history
outputs/            # Reconstruction results (CLI default)
.cache/             # Cached intermediate latents (SLAT)
```

## Upstream and attribution

This project began as a fork of
[`ZimengXiong/Sam3D-Objects-MLX`](https://github.com/ZimengXiong/Sam3D-Objects-MLX),
which produced the original Apple-Silicon port of SAM 3D Objects. The Metal/MPS
foundation — CUDA removal, the MPS backend, the `sparse_conv.metal` and
`flash_attn.metal` kernels, and the low-memory sequential pipeline — **originates
upstream** and is credited there (see *How the port works* below for the exact
split of inherited vs. added work).

> **License status (as of this writing).** The upstream repository has **no
> `LICENSE` file**, no license declaration in its `pyproject.toml`, and no grant
> in its README. Under default copyright law that means **all rights are
> reserved**: GitHub's Terms of Service permit forking and viewing, but do **not**
> grant redistribution or sublicensing rights.
>
> **Consequence — publication is blocked.** Until an explicit license grant is
> obtained from the upstream author (Zimeng Xiong), this project **cannot** be
> released under AGPL-3.0, offer the commercial license described below,
> distribute a packaged application containing the inherited components, or claim
> the inherited port architecture as protected IP. **Do not make this repository
> public and do not distribute a packaged build until this is resolved.**
>
> **Action required / status:** contact Zimeng Xiong to request either an explicit
> license grant or collaboration with credit (contributor / co-author). _Outcome
> of contact: **pending — not yet completed.** Record the result here once it is._

## How the port works

This project layers a full application and pipeline on top of an existing
Apple-Silicon port. Provenance is split as follows (see
[Upstream and attribution](#upstream-and-attribution)).

### Inherited from the upstream port

Originates in
[`ZimengXiong/Sam3D-Objects-MLX`](https://github.com/ZimengXiong/Sam3D-Objects-MLX):

1. **Removed CUDA dependencies** — replaced `spconv-cu121`, `xformers`, and other
   CUDA-only packages.
2. **[MPS backend](https://developer.apple.com/metal/pytorch/)** — model loading
   and inference rewired onto PyTorch's Metal Performance Shaders.
3. **Metal sparse convolution** — custom Metal compute kernels for voxel
   processing:
   - [`sparse_conv.metal`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/sparse_conv.metal)
   - [`conv_metal.py`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_metal.py)
4. **Metal flash attention** — GPU-accelerated attention for the sparse
   transformers:
   - [`flash_attn.metal`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/flash_attn.metal)
   - [`metal_flash_attn.py`](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/attention/metal_flash_attn.py)
5. **[Low-memory pipeline](sam3d_objects/pipeline/inference_pipeline_low_memory.py)**
   — sequential stage loading and chunked decoding.

### Added in this project

Upstream explicitly does **not** support Gaussian splatting or color baking; the
following were built here:

6. **Portable appearance** — real 3D Gaussian-Splatting `.ply` export and a
   PyTorch3D-based UV texture baker, replacing the original CUDA/nvdiffrast
   texturing path.
7. **FastAPI web application** — interactive point-prompt SAM segmentation, live
   progress streaming, and an in-browser 3D viewer.
8. **Depth-stage validity mask** — masks invalid MoGe depth before it conditions
   the geometry stage.
9. **Geometry cleanup** — default hole filling and floater removal for watertight
    output ([Geometry cleanup](#geometry-cleanup)).
10. **Stencil-based mesh editing** — in-UI trimming with client-side hole filling
    and baking ([Mesh editing](#mesh-editing-web-ui)).
11. **Shortcut-distilled stage 1** — the sparse-structure stage ships as a
    *shortcut* model (step-size-conditioned, CFG-free). It is now sampled that way
    by default (`--ss-steps 2 --ss-distill`), decoupled from the stage-2 SLAT step
    count. Earlier revisions of this port ran stage 1 as plain CFG flow matching
    with the stage-2 step count, which both wasted evals and did not match the
    shipped configuration. See the [CHANGELOG](CHANGELOG.md).

## Troubleshooting

### `ImportError: Symbol not found`
PyTorch3D's compiled C++ extensions must match PyTorch's ABI. Use the conda
environment (with PyTorch3D built to match) rather than an ad-hoc `.venv`:
```bash
rm -rf .venv
conda activate sam-3d
python main.py ...
```

### Metal GPU segmentation faults
The default sparse backend is MPS (PyTorch-native Metal), which is stable. If
you experiment with other backends:
```bash
SPARSE_BACKEND=mps    SPARSE_ATTN_BACKEND=sdpa python main.py ...   # default
SPARSE_BACKEND=spconv SPARSE_ATTN_BACKEND=sdpa python main.py ...   # pure CPU
```

### Out of memory
Peak memory occurs **late** in the pipeline — during mesh decode and export,
after the GPU generative stages have released their buffers — so a run can finish
the slow part and still OOM at the end. Notes:

- `--texture-size` does **not** affect peak memory (baking is not the peak).
- `--simplify` (e.g. `--simplify 0.9`) reduces the decoded mesh size and can help.
- Use `--cache-dir` and `--load-slat` to reuse a cached SLAT and skip stages 0–2
  while iterating, so you don't re-run the whole pipeline each time.

### Model download / Hugging Face authentication
The SAM 3D weights are **gated** (Meta's requirement, not this project's): you
need a Meta-approved Hugging Face account and an access token.

1. Request access on the [model page](https://huggingface.co/facebook/sam-3d-objects)
   and wait for approval.
2. Create a token at <https://huggingface.co/settings/tokens>.
3. Authenticate, then download into `checkpoints/hf/`:
   ```bash
   huggingface-cli login          # paste your token
   ```

### No colour in macOS Preview / Quick Look
The default mesh uses per-vertex color (`COLOR_0`), which **Preview and Quick
Look ignore** — the model shows up grey. Export a baked texture instead: use
`--bake` on the CLI, or download from the web UI (baking runs client-side). See
[Texture baking](#texture-baking).

### Confusing checkpoint load failure (zero-byte weight)
An interrupted or truncated download can leave a **zero-byte**
`ss_encoder.safetensors` (or another weight), which fails later with a confusing
deserialization error rather than a clear "file missing". Check file sizes in
`checkpoints/hf/` and re-download any 0-byte file.

## Licensing

> **Note:** the licensing terms below are contingent on **T5 / upstream
> attribution** being resolved. Until an explicit grant is obtained from the
> upstream author, this repository cannot be released under AGPL-3.0 or offered
> under a commercial license. See [Upstream and attribution](#upstream-and-attribution).

This project uses an **open-core** model, similar to MongoDB: the source is
released under a strong copyleft license, while certain components are protected
IP available under a separate commercial license.

- **Open source (AGPL-3.0).** The application source in this repository is
  licensed under the **GNU Affero General Public License v3.0** (see
  [`LICENSE`](LICENSE)). Because the AGPL's network clause applies, if you run a
  modified version of this software to provide a service over a network, you
  must make your complete corresponding source available to the users of that
  service under the same license. **Everything present in this repository is
  AGPL-3.0** — there are no hidden proprietary files here.

- **Protected IP / commercial license.** The proprietary components are **not
  part of this repository.** They exist only in the separately distributed,
  packaged macOS application and are never checked in here. A **commercial
  license** covers those packaged components: it (a) lifts the AGPL's copyleft
  and network source-disclosure obligations for the open-source parts and
  (b) grants rights to the proprietary components for embedding in proprietary
  products. Commercial-licensing enquiries are handled by **Skysong Innovations**
  (Arizona State University's technology-transfer organization), not by an
  individual maintainer.
  <!-- TODO: insert the exact Skysong Innovations intake email / contact URL before publishing. -->

- **Third-party components.** This project depends on the following, each under
  its own license; you must comply with them:

  | Component | Role | License |
  |---|---|---|
  | SAM 2 | Point-prompt segmentation | Apache 2.0 |
  | DINOv2 | Image features | Apache 2.0 |
  | MoGe | Depth estimation | MIT |
  | SAM 3D Objects | Reconstruction model + weights | Meta SAM License |

- **Model weights (Meta SAM License).** The SAM 3D model weights and the code
  under [`sam3d_objects/`](sam3d_objects/LICENSE) are provided by Meta under the
  **SAM License** and remain subject to Meta's terms and acceptable-use policy.
  They are **not** covered by this project's AGPL or commercial grant — you must
  obtain and use them directly under Meta's license. The weights are **downloaded
  at runtime from Hugging Face and are not redistributed by this project.** They
  are **gated**: a Meta-approved Hugging Face account and access token are
  required (this is Meta's requirement — see
  [Troubleshooting](#model-download--hugging-face-authentication)).

- **Upstream port.** Portions of the Metal/MPS foundation originate from
  [`ZimengXiong/Sam3D-Objects-MLX`](https://github.com/ZimengXiong/Sam3D-Objects-MLX),
  which currently has **no license** (all rights reserved). The applicable terms
  will be added here once an upstream grant is obtained; see
  [Upstream and attribution](#upstream-and-attribution).

If you are unsure which license applies to your use case (e.g. shipping the
packaged app, offering a hosted service, or embedding the pipeline in a
proprietary product), please reach out before distribution.

