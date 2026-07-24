# Changelog

All notable changes to this Apple Silicon port are documented here.

## [Unreleased]

### Fixed — stage-1 sampling now uses the shipped shortcut-distilled configuration

The sparse-structure stage (stage 1) of SAM 3D Objects ships as a **shortcut**
model: it is distilled to sample with step-size conditioning and *without*
classifier-free guidance, so it produces good geometry in very few steps. This
port was **not** using that configuration. Stage 1 was driven by the stage-2
`--steps` value and run as plain CFG flow matching, which:

- ran far more function evaluations than the distilled model needs, and
- did not match the sampler the weights were trained for.

Stage 1 and stage 2 are now fully decoupled and stage 1 defaults to the shipped
shortcut config.

#### Added
- `--ss-steps` (default `2`) — stage-1 (sparse-structure) step count. This is the
  shipped shortcut default; values above ~4 rarely improve geometry.
- `--ss-distill` / `--no-ss-distill` (default **on**) — toggle shortcut-distilled
  sampling for stage 1. Required for the low `--ss-steps` to be valid; `--no-ss-distill`
  falls back to CFG flow matching (use ~12 steps then).
- Per-stage wall-clock timing in `InferencePipelineLowMemory.run()`: MoGe depth,
  stage 1 (sparse structure), stage 2 (SLAT), mesh decode, and export/bake are
  each timed and logged as a `[TIMING]` summary table, with step count, sampler
  mode, and an NFE estimate per generative stage plus peak RSS. The timings are
  also attached to the result dict (`stage_timings`, `peak_rss_gb`) so callers
  and benchmarks can read them without parsing logs. This works identically on the
  CLI (`main.py`) and the FastAPI server (`server.py`) paths.
- `benchmark_wo013.py` — a resumable, subprocess-isolated harness that sweeps
  stage-1 step counts (`ss=12/4/2/1`, stage 2 fixed at 12) across a set of objects
  and emits a Markdown results table.

#### Changed
- `--steps` now applies **only to stage 2** (SLAT texture & refinement), which is
  genuine flow matching and is not distilled. Its default is unchanged (`12`).
- `server.py`: stage-1 distillation is decoupled from `--distill` via a new
  `ss_distill` request field (default **on**), so the server path also uses the
  shipped shortcut configuration for stage 1.

#### Notes
- `--distill` still exists but is scoped to **stage 2** only. The released SLAT
  weights are not shortcut-distilled, so it remains experimental and off by default.
- Backward compatibility: an explicit `--steps N` still works and applies to stage 2.

### Changed — memory footprint

- Documented and corrected the memory requirement. Earlier documentation claimed
  **~48 GB** of unified memory; measured usage is **sustained ~8–9 GB with peak
  under 20 GB** (peak occurs late, during mesh decode/export, after the GPU stages
  release their buffers). The supported minimum is now stated as **24 GB**.

### Added — web UI editing & appearance

- **Stencil-based mesh editing** in the web UI: drag box / sphere / cylinder /
  capsule stencils into the scene, move/rotate/scale them, and geometry inside any
  stencil is removed. Editing is non-destructive until export.
- **Client-side hole filling** after stencil editing, so an edited mesh stays
  watertight.
- **Client-side texture baking** on GLB download, so downloaded files display
  correctly in macOS Preview, Quick Look, and USDZ conversion (which ignore
  per-vertex `COLOR_0`).

### Changed — naming

- Renamed the recommended conda environment from `sam-3d-mlx` to **`sam-3d`** and
  updated all documented commands. The project does **not** use Apple's MLX
  framework (it uses PyTorch MPS + hand-written Metal kernels); the old name was
  misleading. The repository directory is `pythia`.

### Documentation

- Added **Requirements**, **memory profile**, **Why Apple Silicon**, and
  **recommended configurations** sections.
- Added **Geometry cleanup**, **Mesh editing**, and **Texture baking** sections.
- Added an **Upstream and attribution** section recording that the upstream port
  (`ZimengXiong/Sam3D-Objects-MLX`) has no license, and split *How the port works*
  into inherited vs. added work.
- Rewrote the **Licensing** section: commercial enquiries routed to Skysong
  Innovations (ASU technology transfer), clarified that proprietary components are
  not in this repository, added a third-party license table, and documented the
  gated, runtime-downloaded Meta weights.
- Fixed the web-app port in the README (`8005`, not `8000`).

### Not present (documentation accuracy)

- The `--depth-prep`, `--depth-prep-quality`, `--no-clean`, and `--clean-ratio`
  CLI flags referenced in some planning docs are **not implemented** in this
  revision and were intentionally left out of the README. Geometry cleanup (hole
  filling + floater removal) runs **unconditionally** as part of mesh
  post-processing; there is currently no flag to disable it or tune a component
  ratio, and there is no separate depth-edge–trimming stage.

