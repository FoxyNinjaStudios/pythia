# PYTHIA: SAM 3D Objects for Apple Silicon

<img src="images/icon1024.png" width=300>

Turn a single photo into a 3D object on a Mac. This is a native Apple-Silicon port of Meta AI's **SAM 3D Objects**. Segmentation (SAM) and depth (MoGe) run on the GPU via Metal Performance Shaders (MPS); the 3D generative reconstruction stages run on the CPU, with no CUDA required.

It ships two ways:

- **A packaged macOS app**: an interactive desktop application. Drop in an
  image, segment the object with SAM, and reconstruct a textured 3D model that
  you can orbit, inspect, and export.
- **A command-line tool** (`main.py`) for scripted and batch reconstruction.

Outputs include watertight **GLB meshes** (per-vertex color or baked UV texture
atlas), real **3D Gaussian-Splatting `.ply`** files, and - from the web UI -
`glTF`, `USDZ`, `PLY`, `OBJ`, and `STL`, with in-browser mesh cleanup, color
grading, stencil trimming, decimation, and GLB compression.

## Demo

Full pipeline on an M1 Mac - under a minute for simple models:

<img src="images/demo.gif" width="100%">

**Original image**

<p align="left">
  <img src="images/demoimage.jpg" width="100%"/>
</p>

<table>
<tr>
<th>2D mask ("teddy bear")</th>
<th>Depth mask</th>
</tr>
<tr>
<td><img src="images/demo-1.png" width="300"/></td>
<td><img src="images/demo-2.png" width="300"/></td>
</tr>
</table>

**Reconstructed model**

<img src="images/modeltt.gif" width="100%">

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

- **Intel Macs.** SAM segmentation and MoGe depth require the Metal Performance
  Shaders (MPS) backend, which is Apple-Silicon only.
- **A-series Macs** (e.g. the MacBook "Neo" class). Insufficient unified memory
  for the working set.
- **iPadOS / iOS.** No general-purpose Python runtime.

### Memory profile

Sustained usage is around 8 to 9 GB. Peak stays **under 20 GB** and occurs late
in the pipeline, during mesh decoding and export, after the GPU generative
stages have finished and released their buffers. There is no configuration that
needs 48 GB; a 24 GB machine has comfortable headroom.

### Why Apple Silicon

Segmentation (SAM) and depth (MoGe) run on the GPU through Metal Performance
Shaders, so an MPS-capable Apple-Silicon GPU is required. Reconstruction then
expands a sparse voxel structure into a high-resolution mesh on the CPU, and the
transient working set for that step is large. Apple's **unified memory** puts
the GPU segmentation/depth stages and the CPU-side reconstruction transients in
one pool, so a 24 GB Mac holds both comfortably without a separate VRAM ceiling.

### Recommended configurations

| Machine | Unified memory | Notes |
|---|---|---|
| **Mac mini (M4 Pro)** | 24 GB | Cheapest supported configuration |
| **MacBook Pro (M-series Pro/Max)** | 24 to 48 GB | Portable; more headroom for other apps |
| **Mac Studio (M-series Max/Ultra)** | 64 GB+ | Fastest; ample headroom |

## Features

### ⚠️ Multi-view reconstruction (experimental)

Multi-view 3D reconstruction fuses geometry from 2+ images of the same object
for improved shape completeness. It is **implemented and enabled by default** in
the web UI, CLI, and API. Disable it server-side with `--no-multi-view`.

**How fusion works (occupancy voting).** Each view is run independently through
Stage 1 (sparse-structure / geometry). The per-view voxel grids are then fused
in the shared canonical 64³ frame by **occupancy voting**: every view votes at
most once per voxel, and a voxel is kept when enough views agree.

- **2 views:** union (maximum completeness — one view fills what the other
  occludes).
- **3+ views:** majority vote (a voxel must appear in ≥2 views), which rejects
  per-view floaters and reconstruction noise.

Each view is then scored by **geometry agreement** — the fraction of its voxels
that land in the fused consensus. Views that agree with the majority score
higher; outlier or noisy views score lower. These confidence scores drive:

- **View selection** (`--num-views-select N` / `num_views_to_select`): keep only
  the N highest-agreement views before the final vote.
- **Best-view conditioning:** Stage 2 (SLAT texture & appearance) is sampled
  **once** on the fused consensus geometry, conditioned on the single
  highest-agreement view. This is occlusion-aware (the clearest view drives
  appearance) and avoids the cost of running Stage 2 per view.

This replaces the earlier naïve approach (averaging integer voxel indices and
averaging Stage-2 latents), which was geometrically ill-defined and crashed when
views produced different voxel counts.

> **Assumptions & limits.** Voting assumes all views land in the same canonical
> frame (SAM 3D canonicalises per object) and use the same downsample factor.
> True cross-view **pose alignment** (e.g. DUSt3R/MASt3R/VGGT) and
> **MultiDiffusion appearance blending** across views remain future work — the
> current path blends geometry but conditions appearance on the best single
> view. The `--fusion-mode` selector is retained for API compatibility but no
> longer changes the geometry path.

**Web UI:** Toggle "Multi-View" mode in Step 1, upload 2+ images, segment each,
then reconstruct.

**CLI usage:** Multi-view is enabled **by default** whenever you pass
`--image-dir` (multiple inputs); the explicit `--multi-view` flag is optional.
Use `--single-view` to suppress it and force single-view reconstruction.
```bash
# Multi-view is auto-enabled by --image-dir
python main.py --image-dir <images_directory> --masks-dir <masks_directory> --output output.glb

# Select the 3 best-agreement views and disable MPS
python main.py --image-dir <images_directory> --num-views-select 3 --no-stage2-mps --output output.glb

# Suppress multi-view even with --image-dir
python main.py --image-dir <images_directory> --single-view --output output.glb
```

**Multi-view CLI flags:**

| Flag | Description |
|------|-------------|
| `--image-dir` | Directory of input views. Providing it **auto-enables** multi-view. |
| `--masks-dir` | Directory of per-view masks (optional). |
| `--multi-view` | Force multi-view mode (redundant when `--image-dir` is given). |
| `--single-view` | Suppress auto multi-view; force the single-view path. |
| `--view-indices` | Comma-separated indices to select specific views (e.g. `0,1,2`). |
| `--num-views-select` | Keep only the N highest-agreement views (default: all). |
| `--fusion-mode` | `stochastic` or `multidiffusion` (retained for API compatibility; no longer changes the geometry path). |
| `--view-weighting` | `uniform` or `entropy` (view-weighting strategy hint). |

**API usage:**
```bash
POST /reconstruct_multi_view
```
with 2+ images in the request body (rejected with HTTP 403 if the server was
started with `--no-multi-view`). Custom fusion config example:
```bash
curl -X POST http://localhost:8005/reconstruct_multi_view \
  -H "Content-Type: application/json" \
  -d '{
    "images": [...],
    "stage2_mps": true,
    "fusion_config": {
      "view_weighting": "entropy",
      "num_views_to_select": 3
    }
  }'
```

**Performance notes:**
- Stage 1 runs per view (cheap — 2 shortcut-distilled steps each).
- Stage 2 runs **once** on the fused geometry, so total cost is close to a
  single-view Stage 2 plus N× lightweight Stage 1 — faster than the old
  per-view Stage 2 path.
- Geometry voting runs on CPU (no device conflicts); Stage 2 uses MPS by default.

### Stage 2 MPS Acceleration (Default)

Stage 2 (SLAT texture & refinement) runs on Apple Silicon's Metal GPU (MPS) **by default** for improved speed. This uses standard PyTorch gather-scatter operations compatible with MPS—no custom Metal kernels required.

**To disable MPS and use CPU-only:**
```bash
conda activate sam-3d
python main.py --image <path> --mask-dir <masks> --no-stage2-mps --output output.glb
```

**Web API (disable MPS per-request):**
Pass `"stage2_mps": false` in the `/reconstruct` or `/reconstruct_multi_view` POST request to disable for that request:
```json
{
  "image_id": "...",
  "mask_b64": "...",
  "stage2_mps": false
}
```

**Server startup:**
```bash
# MPS enabled (default)
python server.py

# MPS disabled
python server.py --no-stage2-mps
```

**Notes:**
- MPS is enabled by default for best performance on Apple Silicon
- Falls back to CPU automatically if MPS is unavailable
- Does not affect Stage 1 (sparse geometry) or decoding stages
- Uses TRELLIS.2 sparse convolution pattern with gather-scatter operations

### Low-memory pipeline improvements

The low-memory inference pipeline (`InferencePipelineLowMemory`) has been refactored for **robustness and clarity**:

**Device coordination fixes:**
- Fixed Stage 2 MPS device handling to prevent tensor device mismatches
- Centralized device selection logic (new `get_stage2_device()` helper)
- Explicit tensor migration: inputs moved to stage2_device before generation, SLAT moved back to base device after

**Memory & I/O optimizations:**
- Single SLAT save (eliminated redundant double saves)
- Improved model cleanup with proper CPU transition before deletion
- MPS cache clearing with error handling

**Robustness enhancements:**
- Comprehensive error handling around model loading and caching
- Graceful fallbacks (e.g., use vertex color if Gaussian unavailable)
- Better logging with stage-specific prefixes (`[S0]`, `[S1]`, `[S2]`, `[S3]`)
- Clearer per-stage timing output with device info

**Developer experience:**
- Complete docstrings and type hints
- Organized code sections (memory utilities, caching, pipeline logic)
- Clear comments and timing checkpoints throughout

These improvements maintain full **backward compatibility** and don't change the external API or behavior—just the internal robustness and maintainability.

### Reconstruction

- **Interactive segmentation.** Two prompt modes in the browser UI: **text /
  concept** masking with **SAM 3** (describe the object, e.g. "chair") and
  **point-prompt** masking with **SAM 2** (positive / negative clicks). No manual
  mask files needed.
- **Single-image 3D reconstruction.** Geometry and appearance from one photo.
- **Apple-Silicon native.** SAM segmentation and MoGe depth run on the MPS
  backend; the 3-D reconstruction stages run on the CPU. No CUDA required.
- **Fast.** Native-resolution reconstruction in about a minute on M-series
  hardware (see [Geometry resolution](#geometry-resolution)).
- **Low-memory pipeline.** Sequential stage loading and SLAT caching keep the
  working set small (see [Requirements](#requirements)).
- **Watertight output.** Hole filling and floater removal run by default, so the
  exported GLB is closed, suitable for 3D printing, boolean operations, and
  volume computation.
- **Live progress.** Streamed pipeline logs and an in-place mask to model
  preview in the web app.

### Model management (web UI)

A **Models** panel (top-right of the web app) shows the models the app uses and
lets you fetch them without touching the command line:

- **Live status.** Each model reports **Loaded** (in memory), **Downloaded** (on
  disk), or **Missing**, with its on-disk size. The active 2-D segmentation
  model also shows a live "loading into memory" bar (it is loaded on demand, and
  the text model is preloaded as soon as you upload an image).
- **One-click downloads.** Missing models download in the background with a live
  progress indicator; the panel polls until each finishes.
- **Hugging Face access built in.** Paste a Hugging Face access token in the
  panel to authenticate; it is validated and used for gated downloads.
- **Gated-model handling.** **SAM 3** (text segmentation) and **SAM 3D Objects**
  are gated behind Meta approval; if access has not been granted the panel shows
  an actionable message linking straight to the request-access page (approval
  itself still happens on Hugging Face). SAM 2 and MoGe are public and download
  without a token.

### In-browser mesh tuning (web UI)

All edits are **non-destructive** (the source mesh is untouched until export)
and update the 3D preview live:

- **Fill holes.** Cap open boundary loops up to a size threshold so the mesh
  stays watertight.
- **Trim stretched edges.** Drop triangles whose longest edge exceeds a
  threshold (removes background webbing and depth-seam sliver faces).
- **Sand / smooth.** Volume-preserving **Taubin** smoothing passes.
- **Cut fillet.** Round the border of stencil cuts instead of leaving a flat
  rim.
- **Color grading.** Live **saturation**, **contrast**, and **brightness**
  controls applied to the vertex colors.
- **Stencil eraser.** Drag primitive stencils (**box, rounded box, sphere,
  cylinder, capsule, cone, wedge, torus, slab, or half-space plane cut**) from a
  square icon palette onto the model and **move / rotate / scale** them
  (Three.js-editor gizmo); any geometry inside a stencil is erased. Stack
  stencils to carve complex regions.

### Export & optimization (web UI)

- **Seven formats:** `GLB`, `glTF` (embedded), `USDZ` (AR), `PLY`, `OBJ`, `STL`, `3MF` (multi-color 3D printing),
  each with a **live file-size estimate** that reacts to the options below.
- **3MF Multi-color Export.** Export with **per-vertex color quantization** for 3D printing with full-color filament changers (AMS, MMU3, etc.). Native **3MF format** includes embedded color metadata; compressed 35% smaller than raw mesh data. **Color palette:** Adjust quantization from 2–256 distinct colors (typical 8–32 for FDM printers). **Use case:** Multi-material printing in Cura, Prusa Slicer, or Bambu Studio; supports full RGB color mapping onto a single mesh.
- **Simplify mesh.** In-browser **meshoptimizer** decimation (0–90%) that is
  reliable on million-triangle meshes.
- **Bake texture** (on by default). On: color is baked into a `baseColorTexture`
  so the model shows color in **macOS Quick Look / Preview** and USDZ. Off:
  compact **per-vertex color** GLB (like the SAM CLI export) - smallest file,
  colored in `model-viewer` / three.js / Blender.
- **Texture size & format.** 512–4096 px atlas, **JPEG** (smaller) or **PNG**
  (lossless).
- **Texture themes** (optional, web UI only). Apply artistic filters to baked textures:
  **None** (original), **Minecraft** (blocky), **Cartoony** (cell-shaded), **Sketch**
  (pencil), **Low-Poly**, **Watercolor**, **Retro** (8-bit), **Oil Painting**,
  **Comic**, **Pixelated**, **Posterized**, **Grayscale**, **Sepia**, **High Contrast**,
  **Neon**. Live preview in viewport; applies on download when baking is enabled.
- **AI Mesh Cleanup.** The export panel has checkboxes for **point cloud denoising** and **shape completion**. **Denoising is real and deterministic:** it uses **Open3D's Laplacian / statistical smoothing** to refine vertices in-place (neighborhood averaging), removing high-frequency noise while preserving the face topology (no vertex removal, no ML weights required). **Shape completion requires trained SnowflakeNet weights.** If those weights are not present in `checkpoints/ai_cleanup/`, shape completion is **skipped (no-op) and the mesh is returned unchanged** — the app **never** runs an untrained (random-weight) network on your mesh, because doing so would degrade it rather than clean it. When trained SnowflakeNet weights are installed, completion runs on Metal (MPS) on Apple Silicon.

- **Meshopt compress** (opt-in). `EXT_meshopt_compression` +
  `KHR_mesh_quantization` via **glTF-Transform** for dramatically smaller GLBs
  for web/`model-viewer` use.
- **Client-side baking.** Texture baking and compression run entirely in the
  browser at download time - no server round-trip.

### CLI output formats

- `GLB` mesh with **per-vertex color** (default).
- `GLB` mesh with a **baked UV texture atlas** (portable, no CUDA / nvdiffrast).
- Real **3D Gaussian-Splatting `.ply`** for depth-ambiguous / soft previews.
- Raw **voxel `STL`** (`--voxels-only`).

## Two ways to run

### 1. Desktop app (interactive web UI)

The app is a FastAPI server that serves an interactive single-page UI
(`static/index.html`). Start it and it opens the browser UI automatically:

```bash
conda activate sam-3d
python server.py                 # serves on http://localhost:8005 and opens it
python server.py --port 9000     # use a different port
python server.py --silent        # do not auto-open a browser
```

| Flag | Description |
|------|-------------|
| `--port` | TCP port to listen on (default: `8005`). |
| `--silent` | Do not open the client in a browser after startup. |
| `--no-stage2-mps` | Disable MPS acceleration for Stage 2 (SLAT); run CPU-only. Enabled by default on Apple Silicon. |
| `--no-multi-view` | Suppress multi-view: hides the Single/Multi-view toggle in the web UI and rejects `/reconstruct_multi_view` (HTTP 403). Multi-view is enabled by default. |
| `--refine-text-mask` | Enable text-mask refinement for SAM 3 segmentation (off by default). |
| `--ai-part-names` | Enable AI part-naming of segmented meshes via the SmolVLM2 VLM (off by default; downloads a multi-GB model). |
| `--no-client-logs` | Disable client-side console logging in the web UI. |

The server opens the client once the port is actually accepting connections, and
shuts down cleanly on `Ctrl-C` / `SIGTERM` (with a watchdog fallback) so it never
leaves an orphaned process holding the port.

Workflow: upload an image, segment the object, pick a quality preset,
reconstruct, then orbit the result, tune it (mesh cleanup, color grading,
stencil trimming), and download in any of seven formats (**GLB**, **glTF**,
**USDZ**, **PLY**, **OBJ**, **STL**, **3MF**).

The **Models** panel (top-right) reports the download / load status of the
models and can fetch any that are missing - including pasting a Hugging Face
token and downloading the gated SAM 3 / SAM 3D weights - so first-time setup
needs no command-line steps (see [Model management](#model-management-web-ui)).

Optional Gaussian-splat export is on by default; disable it with
`SAM3D_SPLAT=0`.

### 2. Command line

```bash
conda activate sam-3d
python main.py \
    --image <image/path> \
    --mask-dir <mask-path> \
    --mask-index 0 \
    --mesh \
    --output outputs/reconstruction.glb
```

#### Key Arguments

**Input / output**
| Argument | Description |
|----------|-------------|
| `--image` / `-i` | Input image path (single-view mode). |
| `--mask` / `-m` | Mask file (PNG/JPG). |
| `--mask-dir` + `--mask-index` | Object mask from a SAM-style directory (index default: `0`). |
| `--image-dir` | Directory of views — **auto-enables multi-view** (see [Multi-view](#-multi-view-reconstruction-experimental)). |
| `--masks-dir` / `--view-indices` / `--num-views-select` / `--fusion-mode` / `--view-weighting` | Multi-view inputs and fusion controls (see [Multi-view](#-multi-view-reconstruction-experimental)). |
| `--multi-view` / `--single-view` | Force / suppress multi-view mode. |
| `--output` / `-o` | Output file (default: `outputs/voxels.stl`; use `.glb` with `--mesh`). |

**Geometry & stages**
| Argument | Description |
|----------|-------------|
| `--mesh` | Output a smooth GLB mesh (otherwise voxel STL). |
| `--voxels-only` | Only run Stage 1 and export raw voxels (STL); skip mesh decoding. |
| `--steps` | Stage-2 (SLAT texture & refinement) flow-matching steps (default: `12`). Not distilled. |
| `--ss-steps` | Stage-1 (sparse-structure / geometry) steps (default: `2`). Shortcut-distilled; >4 rarely helps. |
| `--ss-distill` / `--no-ss-distill` | Shortcut-distilled Stage-1 sampling (CFG-free, ~1 eval/step). On by default; `--no-ss-distill` falls back to CFG flow matching (then use ~12 steps). |
| `--distill` | Also distill **Stage 2** (SLAT). Experimental; usually degrades texture — leave off. |
| `--full-res-geometry` | Keep large objects at native 64³ (interior-voxel pruning) instead of factor-2 downsampling. **On by default**; disable with `SAM3D_FULL_RES_GEOMETRY=0` (see [Geometry resolution](#geometry-resolution)). |
| `--no-stage2-mps` | Disable MPS for Stage 2; force CPU-only (MPS is default). |
| `--seed` | Random seed (default: `42`). |

**Mesh cleanup & appearance**
| Argument | Description |
|----------|-------------|
| `--simplify` | Mesh decimation ratio (`0.0` = none … `0.95` = heavy). Default `0.0`, or `0.9` with `--bake`. |
| `--smooth ITERS` | Taubin-smooth the output mesh by `ITERS` iterations to sand off the 64³ voxel staircase (default: `0` = off; ~10 removes stepping). |
| `--refine-mask` | Clean and anti-alias the mask before reconstruction (fill pinholes, drop speckles, feather the boundary). Off by default. |
| `--vertex-color-source` | `gaussian` (saturated, recommended) or `mesh`. |
| `--bake` | Bake a UV texture atlas instead of per-vertex color. |
| `--bake-source` | `gaussian` (higher fidelity) or `vertex`. |
| `--texture-size` | Baked atlas edge length in px (default: `2048`). |

**Caching**
| Argument | Description |
|----------|-------------|
| `--cache-dir` | Directory for intermediate outputs (default: `.cache`). |
| `--load-slat` | Load a cached SLAT `.pt` (skips stages 0–2; only runs mesh decoding). |

## Geometry cleanup

After the mesh is decoded, a post-processing pass runs **by default** to
suppress reconstruction artifacts, clean up the silhouette, and keep the output
watertight:

- **Adaptive decimation.** The mesh is decoded via FlexiCubes on a 64³
  sparse-voxel grid, which leaves a ~1-voxel "staircase" on thin features (chair
  legs, table edges). A quadric edge-collapse pass decimates the mesh toward a
  fixed **triangle budget** (~60k) rather than a fixed fraction, so the amount of
  cleanup scales with the raw mesh density. Edge-collapse regularises the voxel
  staircase into clean edges that follow the true surface. Meshes already at or
  below the budget are left untouched, and an explicit `--simplify` /
  `simplify_ratio` still overrides the automatic budget.
- **Taubin smoothing.** A light, volume-preserving **Taubin** pass (5 iterations
  by default) polishes the decimated surface without shrinking thin parts. Kept
  deliberately light because decimation already regularises the silhouette; too
  many passes reintroduce a low-frequency "wave" on straight edges. Raise it via
  the web UI "Sand / smooth" control, `--smooth ITERS` (CLI), or
  `smooth_iterations` (API).
- **Hole filling.** Small gaps left by decoding or simplification are closed so
  the surface is manifold and closed.
- **Floater removal.** Disconnected islands are dropped, keeping the largest
  connected component (the object) and discarding stray voxel debris.

All passes are guarded: if a step cannot run on a given mesh it is skipped, and
none can fail the reconstruction job. The result is the watertight GLB used
by the viewer and every export path.

## Geometry resolution

The geometry stage decodes to a 64 cubed occupancy grid, but before mesh
decoding the sparse structure was previously factor-2 **downsampled** whenever
its surface-voxel count exceeded an int32-indexing ceiling
(`max(int32)/(64*768)` is about 43,691). That halving silently dropped large
objects below native 64 cubed.

`--full-res-geometry` (**on by default**) avoids that with a lossless step: it
prunes fully-interior voxels, which never define the visible surface, so the
remaining surface shell fits under the ceiling and the object stays at native
64 cubed. The silhouette is unchanged; only invisible interior voxels are
removed. Keeping native resolution is also what brings a typical reconstruction
down to about a minute, since the alternative path spends time on a halved grid
that then has to be handled downstream.

Because a native-resolution mesh decode is more expensive on very large objects,
a **size-aware guard** keeps this cheap for typical objects and bounded for
extreme ones:

- Pruned surface shell **at or below `SAM3D_FULL_RES_MAX_COORDS`** (default
  30,000): kept at native 64 cubed.
- Shell **above that budget**: falls back to the halved path so a single huge
  object does not dominate runtime.
- Shell **above the int32 ceiling** (about 42,000) even after pruning: always
  halved.

Tune it to taste: raise `SAM3D_FULL_RES_MAX_COORDS` (e.g. `=42000`) to always
stay native up to the hard ceiling, or set `SAM3D_FULL_RES_GEOMETRY=0` to
restore the original always-halve behavior. The chosen `downsample_factor` is
logged per run (`Downsampled coords from A to B (... downsample_factor=...)`).

## Mesh editing (web UI)

The web app's **Tools** panel provides non-destructive, live-updating controls;
the original mesh is untouched until you export.

**Mesh**

- **Fill holes.** Cap open boundary loops up to a size threshold so the surface
  stays manifold and closed.
- **Trim stretched edges.** Remove triangles whose longest edge exceeds a
  threshold (background webbing, depth-seam slivers).
- **Sand / smooth.** Volume-preserving Taubin smoothing passes.
- **Cut fillet (border radius).** Round stencil-cut borders instead of a flat rim.

**Color**

- **Saturation / Contrast / Brightness.** Live grading of the vertex colors.

**Stencil eraser**

1. Drag a stencil primitive from the square icon palette onto the scene - the
   3D editor opens automatically. Ten shapes are available for different cuts:
   **box**, **rounded box**, **sphere**, **cylinder**, **capsule**, **cone**
   (tapered points), **wedge** (angled ramp cuts), **torus** (rings / grooves),
   **slab** (thin flat cutter), and **plane cut** (an unbounded half-space that
   slices off everything on one side - ideal for removing a base or flattening a
   bottom).
2. Select it and switch between **move / rotate / scale** tools (Three.js-editor
   style) to position it over the region to delete.
3. Any geometry **inside** a stencil disappears live; stack multiple stencils to
   carve complex regions. **Remove** deletes the selected stencil; **Clear all**
   resets them.

Editing is **non-destructive**: the original mesh is untouched until you export.
Hole filling re-closes any cut so the edited mesh stays watertight, and the
chosen export options (baking, simplification, compression) are applied on
download.

## Texture baking & GLB optimization

By default the mesh carries **per-vertex color** (`COLOR_0`). That renders
correctly in the in-app viewer and in `model-viewer`, but **macOS Preview, Quick
Look, and USDZ conversion ignore per-vertex color** and show the model grey. The
**Export** panel lets you choose how color and size are handled:

- **Bake texture** (on by default). Bakes color into a `baseColorTexture` so the
  GLB shows color in Preview / Quick Look / USDZ. Turn it **off** for a compact
  per-vertex-color GLB (smallest file; colored in `model-viewer` / three.js /
  Blender). Baking runs **client-side** at download time.
- **Texture size / format.** 512–4096 px atlas, JPEG (smaller) or PNG (lossless).
- **Texture themes.** Apply artistic filters to the baked texture atlas (web UI only):
  - **None / Original**: Baked texture without filters (default)
  - **Minecraft**: Blocky posterization (64-level quantization) for game-style models
  - **Cartoony**: Cell-shaded effect (32-level posterization) for cartoon aesthetics
  - **Sketch**: Inverted grayscale pencil drawing effect
  - **Low-Poly**: 6× downsampling + upsampling for stylized low-poly look
  - **Watercolor**: Desaturated with brightening for painted wash effect
  - **Retro**: 64×64 extreme downsampling + posterization for 8-bit nostalgia
  - **Oil Painting**: Blurred then posterized for artistic effect
  - **Comic**: Bold 48-level posterization for comic-book style
  - **Pixelated**: 16-pixel grid pixelation effect
  - **Posterized**: Reduced to 16 colors (4 levels per channel)
  - **Grayscale**: Black and white desaturation
  - **Sepia**: Vintage warm-tone coloring
  - **High Contrast**: 1.8× contrast amplification for graphic impact
  - **Neon**: 1.3× saturation boost for electric, vibrant look
  
  Themes are previewed live in the viewport when baking is enabled; the same theme
  is applied on download. When baking is disabled, themes are ignored.
- **Simplify mesh.** meshoptimizer decimation (0–90%) applied before export.
- **Meshopt compress** (opt-in, GLB only). `EXT_meshopt_compression` +
  `KHR_mesh_quantization` via glTF-Transform for much smaller web-ready GLBs;
  needs a modern viewer (not Quick Look).

Every format card shows a **live size estimate** that updates as you change these
options.

**CLI.** Pass `--bake` to write a baked UV atlas (`--bake-source`,
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
   weights). These weights are governed by Meta's SAM License; see
   [Licensing](#licensing). You can also skip this step and fetch the weights
   later from the web app's **Models** panel (paste a Hugging Face token there
   for the gated SAM 3D download; see
   [Model management](#model-management-web-ui)).

3. **Environment variables** (set automatically by `main.py` / `server.py`, but
   useful when running manually):
   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   ```

   Geometry-resolution tuning (optional; see
   [Geometry resolution](#geometry-resolution)):
   ```bash
   export SAM3D_FULL_RES_GEOMETRY=1        # keep large objects native 64 cubed (default); 0 disables
   export SAM3D_FULL_RES_MAX_COORDS=30000  # surface-voxel budget before falling back to the halved path
   ```

   Full reference of the environment variables the app honours:

   | Variable | Default | Effect |
   |----------|---------|--------|
   | `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.0` | PyTorch MPS memory watermark (set automatically). |
   | `SAM3D_SAM_DEVICE` | `mps` | Device for SAM segmentation (set automatically to `mps`). |
   | `SAM3D_MOGE_DEVICE` | `mps` | Device for MoGe depth (set automatically to `mps`). |
   | `SAM3D_SPLAT` | `1` | Gaussian-splat `.ply` export; set `0` to disable. |
   | `SAM3D_FULL_RES_GEOMETRY` | `1` | Keep large objects at native 64³; set `0` to always halve. |
   | `SAM3D_FULL_RES_MAX_COORDS` | `30000` | Surface-voxel budget before falling back to the halved path. |
   | `SAM3D_CHECKPOINTS_DIR` | `checkpoints/ai_cleanup` | Directory for AI mesh-cleanup weights. |
   | `SAM3D_PROFILE` | unset | Set `1` to print mesh-decoder profiling. |

## Structure
```
main.py             # CLI entry point
server.py           # FastAPI web app (segmentation, reconstruct, export)
sam_wrapper.py      # SAM 3 text/concept + SAM 2 point-prompt segmentation wrapper
static/index.html   # Web UI: segmentation, 3D viewer, mesh tuning, color grading, stencil editing, multi-format export (client-side baking / meshopt compression), model download / status panel
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
which produced the original Apple-Silicon port of SAM 3D Objects. That port's
custom Metal reconstruction kernels — a Metal sparse-convolution kernel and a
Metal flash-attention kernel, together with their Python wrappers — have been
**removed from this repository**. Reconstruction now runs on the CPU through a
pure-PyTorch sparse path, and this project **no longer contains upstream-owned
code**.

> **Why they were removed.** The upstream repository has **no `LICENSE` file**,
> no license declaration in its `pyproject.toml`, and no grant in its README.
> Under default copyright law that means **all rights are reserved**: the
> inherited Metal components remained the upstream author's to license and were
> not ours to redistribute. Rather than wait on an explicit grant, those
> components were removed so this project can be published under its own license.
>
> **Credit.** The original Apple-Silicon porting effort — CUDA removal and the
> Metal/MPS reconstruction approach — is credited to Zimeng Xiong (see upstream
> issue #5). If a license grant is obtained later, the optional GPU
> reconstruction path can return as an add-on; until then the project ships
> **CPU-only reconstruction**.

## How the port works

This project layers a full application and pipeline on top of an existing
Apple-Silicon port. Provenance is split as follows (see
[Upstream and attribution](#upstream-and-attribution)).

### Inherited from the upstream port

The upstream port
[`ZimengXiong/Sam3D-Objects-MLX`](https://github.com/ZimengXiong/Sam3D-Objects-MLX)
provided the original CUDA-removal work and a Metal/MPS reconstruction backend.
The custom Metal reconstruction kernels it contributed — a Metal
sparse-convolution kernel and a Metal flash-attention kernel, plus their Python
wrappers — have been **removed** from this repository (see
[Upstream and attribution](#upstream-and-attribution)). Reconstruction now runs
on the CPU through a pure-PyTorch sparse path, while segmentation (SAM) and depth
(MoGe) run on the MPS backend.

### Added in this project

Upstream explicitly does **not** support Gaussian splatting or color baking; the
following were built here:

1. **Portable appearance.** Real 3D Gaussian-Splatting `.ply` export and a
   PyTorch3D-based UV texture baker, replacing the original CUDA/nvdiffrast
   texturing path.
2. **FastAPI web application.** Interactive text/concept (SAM 3) and point-prompt
   (SAM 2) segmentation, live progress streaming, and an in-browser 3D viewer.
3. **Depth-stage validity mask.** Masks invalid MoGe depth before it conditions
   the geometry stage.
4. **Geometry cleanup.** Default hole filling and floater removal for watertight
   output (see [Geometry cleanup](#geometry-cleanup)).
5. **Stencil-based mesh editing.** In-UI trimming with client-side hole filling
    and baking (see [Mesh editing](#mesh-editing-web-ui)).
6. **Shortcut-distilled stage 1.** The sparse-structure stage ships as a
    *shortcut* model (step-size-conditioned, CFG-free). It is now sampled that
    way by default (`--ss-steps 2 --ss-distill`), decoupled from the stage-2
    SLAT step count. Earlier revisions of this port ran stage 1 as plain CFG
    flow matching with the stage-2 step count, which both wasted evals and did
    not match the shipped configuration. See the [CHANGELOG](CHANGELOG.md).
7. **Native-64 cubed geometry with size-aware guard.** Lossless interior-voxel
    pruning keeps large objects at full geometry resolution instead of being
    factor-2 downsampled, with a tunable budget that bounds decode cost (see
    [Geometry resolution](#geometry-resolution)).
8. **In-app model management.** A Models panel that reports each model's
    download / load status and downloads missing weights from Hugging Face,
    including token entry and gated-repo handling for SAM 3D Objects (see
    [Model management](#model-management-web-ui)).
9. **AI mesh cleanup (denoising real; completion gated on weights).** UI controls
    for point cloud denoising and shape completion, integrated into the export
    pipeline with Metal (MPS) acceleration. **Denoising** uses Open3D statistical /
    Laplacian smoothing (deterministic, no weights required). **Shape completion**
    uses **SnowflakeNet** (skip-transformer point deconvolution; MIT; ICCV 2021 /
    TPAMI 2023) and runs **only** when trained weights are present in
    `checkpoints/ai_cleanup/`; otherwise it is **skipped (no-op)** rather than
    running an untrained network that would degrade the mesh. The
    `mesh_cleanup_ai.py` wrappers provide lazy loading and MPS acceleration;
    trained weights are downloaded from their upstream sources, not redistributed
    here.

## Troubleshooting

### `ImportError: Symbol not found`
PyTorch3D's compiled C++ extensions must match PyTorch's ABI. Use the conda
environment (with PyTorch3D built to match) rather than an ad-hoc `.venv`:
```bash
rm -rf .venv
conda activate sam-3d
python main.py ...
```

### CPU reconstruction, Metal segmentation & depth
Interactive **SAM segmentation** and **MoGe depth** run on Metal (MPS), while the
heavy 3-D reconstruction (sparse structure, SLAT, mesh decode) runs on the CPU.
This is automatic; no flags are needed:
```bash
python server.py
python main.py ...
```
SAM 2/3 load on Metal (`SAM3D_SAM_DEVICE` is set for them) and the MoGe depth
model runs on Metal too (`SAM3D_MOGE_DEVICE`), so segmentation and the depth
preview stay fast and interactive while the reconstruction stages run on the
CPU. On the CLI `main.py`, which has no segmentation stage, only MoGe runs on
Metal.


### Out of memory
Peak memory occurs **late** in the pipeline, during mesh decode and export,
after the GPU generative stages have released their buffers, so a run can finish
the slow part and still OOM at the end. Notes:

- `--texture-size` does **not** affect peak memory (baking is not the peak).
- `--simplify` (e.g. `--simplify 0.9`) reduces the decoded mesh size and can help.
- Use `--cache-dir` and `--load-slat` to reuse a cached SLAT and skip stages 0
  to 2 while iterating, so you do not re-run the whole pipeline each time.

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
   Or do both from the web app's **Models** panel: paste the token to
   authenticate and click **Download** on SAM 3D Objects (see
   [Model management](#model-management-web-ui)).

### No color in macOS Preview / Quick Look
The default mesh uses per-vertex color (`COLOR_0`), which **Preview and Quick
Look ignore**, so the model shows up grey. Export a baked texture instead: use
`--bake` on the CLI, or download from the web UI (baking runs client-side). See
[Texture baking](#texture-baking--glb-optimization).

### Confusing checkpoint load failure (zero-byte weight)
An interrupted or truncated download can leave a **zero-byte**
`ss_encoder.safetensors` (or another weight), which fails later with a confusing
deserialization error rather than a clear "file missing". Check file sizes in
`checkpoints/hf/` and re-download any 0-byte file.

## Licensing

This project uses an **open-core** model, similar to MongoDB: the source is
released under a strong copyleft license, while certain components are protected
IP available under a separate commercial license.

- **Open source (AGPL-3.0).** The first-party application source in this
  repository is licensed under the **GNU Affero General Public License v3.0**
  (see [`LICENSE`](LICENSE)). Because the AGPL's network clause applies, if you
  run a modified version of this software to provide a service over a network,
  you must make your complete corresponding source available to the users of
  that service under the same license.

- **What is actually in this repository.** Two license regimes coexist in the
  tree, and this is deliberate:
  - **First-party application code** (web app, splatting, baking, geometry work,
    editing, AI cleanup UI/architecture): **AGPL-3.0**. This includes the
    PyTorch model wrapper code in `mesh_cleanup_ai.py` that provides lazy loading
    and Metal (MPS) acceleration. The actual trained model weights are obtained
    from the open-source third-party sources listed below and are not
    redistributed with this project.
  - **`sam3d_objects/`**: under Meta's **SAM License** (see
    [`sam3d_objects/LICENSE`](sam3d_objects/LICENSE)), not AGPL.

  No proprietary Foxy Ninja components are checked in. Those exist only in the
  separately distributed, packaged macOS application.

- **Protected IP / commercial license.** The proprietary components are **not
  part of this repository.** A **commercial license** covers those packaged
  components: it (a) lifts the AGPL's copyleft and network source-disclosure
  obligations for the open-source parts and (b) grants rights to the proprietary
  components for embedding in proprietary products. Commercial-licensing
  enquiries are handled by **Skysong Innovations** (Arizona State University's
  technology-transfer organization), not by an individual maintainer.
  <!-- TODO: insert the exact Skysong Innovations intake email / contact URL before publishing. -->

- **Third-party components.** This project depends on the following, each under
  its own license; you must comply with them:

  | Component | Role | License |
  |---|---|---|
  | SAM 2 | Point-prompt segmentation | Apache 2.0 |
  | SAM 3 | Text / concept segmentation | Meta SAM License |
  | DINOv2 | Image features | Apache 2.0 |
  | MoGe | Depth estimation | MIT |
  | SAM 3D Objects | Reconstruction model + weights | Meta SAM License |
  | Point Transformer V3 | AI point cloud denoising (modern) | MIT (Pointcept/PointTransformerV3) |
  | SnowflakeNet (SPD) | AI point cloud completion (modern) | MIT (AllenXiangX/SnowflakeNet, ICCV 2021/TPAMI 2023) |
  | Point Completion Network (PCN) | AI point cloud denoising (legacy fallback) | MIT (wentaoyuan/pcn) |
  | 3D Shape VAE | AI geometry completion (legacy fallback) | MIT (autonomousvision/occupancy-networks) |

- **Model weights (Meta SAM License).** The SAM 3D model weights and the code
  under [`sam3d_objects/`](sam3d_objects/LICENSE) are provided by Meta under the
  **SAM License** and remain subject to Meta's terms and acceptable-use policy.
  They are **not** covered by this project's AGPL or commercial grant; you must
  obtain and use them directly under Meta's license. The weights are
  **downloaded at runtime from Hugging Face and are not redistributed by this
  project.** They are **gated**: a Meta-approved Hugging Face account and access
  token are required (this is Meta's requirement; see
  [Troubleshooting](#model-download--hugging-face-authentication)).

- **SnowflakeNet (MIT).** The point cloud completion model (SnowflakeNet, "Snowflake Point Deconvolution with Skip-Transformer") comes from [`AllenXiangX/SnowflakeNet`](https://github.com/AllenXiangX/SnowflakeNet) (ICCV 2021 oral; extended in TPAMI 2023). The repository is **MIT-licensed in its entirety** (a single `LICENSE` file; its README states "This project is open sourced under MIT license"), and it distributes its **pretrained models within that same project under that MIT license** — there is no separate or more-restrictive weights license stated. The pretrained weights are **not** included or redistributed in this repository and must be downloaded manually from [Google Drive](https://drive.google.com/drive/folders/1mdA-6ZwzXAbaWJ6fmfL9-gl3aGTGTWyR) (or [Baidu backup](https://pan.baidu.com/s/10tkqJfMdWO9GkzXSBSNlIw), password: oy5c) and placed in `checkpoints/ai_cleanup/`. Note that these weights were **trained on third-party datasets** (e.g. ShapeNet / PCN), whose own dataset terms may apply to specific downstream uses.

- **Upstream port.** The custom Metal reconstruction kernels that originated in
  [`ZimengXiong/Sam3D-Objects-MLX`](https://github.com/ZimengXiong/Sam3D-Objects-MLX)
  (which has **no license**; all rights reserved) have been **removed** from
  this repository, so no upstream-owned code ships here. Reconstruction runs on
  the CPU. See [Upstream and attribution](#upstream-and-attribution).

If you are unsure which license applies to your use case (e.g. shipping the
packaged app, offering a hosted service, or embedding the pipeline in a
proprietary product), please reach out before distribution.