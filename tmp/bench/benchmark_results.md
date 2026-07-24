# WO-013 T4 — Stage-1 shortcut-sampling benchmark

All configs use the shipped shortcut-distilled stage-1 sampler (`--ss-distill`, on by default); only `--ss-steps` changes. Stage 2 (SLAT) is fixed at 12 flow-matching steps. Times are wall-clock seconds from the pipeline's per-stage instrumentation. Peak RSS is the worker process `ru_maxrss`. Quality is a manual 0–5 score of the exported GLB.

| Object | Config | ss-steps | Stage 1 (s) | Stage 2 (s) | Total (s) | Peak RSS (GB) | Verts | Faces | Quality (0–5) |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| kid_box | D_ss1 | 1 | 40.2 | 434.5 | 637.0 | 13.7 | 770712 | 1541778 | |
