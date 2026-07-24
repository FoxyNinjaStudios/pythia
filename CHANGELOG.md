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
