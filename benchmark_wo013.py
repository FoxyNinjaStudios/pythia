#!/usr/bin/env python3
"""
WO-013 T4 — Stage-1 shortcut-sampling benchmark harness.

Sweeps the sparse-structure (stage 1) step count across 4 configurations for a
set of objects, reusing the exact CLI code path (`main.run_pipeline`, which runs
`InferencePipelineLowMemory`). Stage 2 (SLAT) is held fixed at 12 steps so the
only variable is the stage-1 cost.

Each (config, object) run executes in an isolated subprocess so that a segfault
(seen occasionally on Apple Silicon) only kills that one run, and so peak RSS is
measured per-run. The worker reads the per-stage timings the pipeline now attaches
to its result (`stage_timings`, `peak_rss_gb` — see WO-013 T3) and writes them to
a JSON file. The parent aggregates the JSONs into a Markdown table.

Usage
-----
Full sweep (4 configs x 5 objects = 20 runs; slow on CPU):
    python benchmark_wo013.py

Reduced smoke run (1 object, all configs):
    python benchmark_wo013.py --objects kid_box

Custom timeout / results dir / re-run:
    python benchmark_wo013.py --timeout 1800 --results-dir tmp/bench --force

Just (re)build the table from existing JSON results:
    python benchmark_wo013.py --table-only

The "quality" column is left blank for manual scoring (0-5) after inspecting the
exported GLBs in tmp/bench/glb/.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ── Configurations (WO-013 T4) ───────────────────────────────────────────────
# All use the shipped shortcut-distilled stage-1 sampler (ss_distill=True); only
# the stage-1 step count changes. Stage 2 (SLAT) is fixed at 12 flow-matching
# steps and never distilled (its released weights are not shortcut-distilled).
CONFIGS = {
    # name: (ss_steps, stage2_steps, ss_distill, distill)
    "A_ss12": (12, 12, True, False),   # high-step shortcut (upper bound on stage-1 quality)
    "B_ss4":  (4,  12, True, False),
    "C_ss2":  (2,  12, True, False),   # shipped default
    "D_ss1":  (1,  12, True, False),
}

# ── Objects (image dir + SAM mask index) ─────────────────────────────────────
# Index 0 is the first/most-prominent SAM mask in each folder; override per-object
# with --object-index if a different mask isolates the subject better.
OBJECTS = {
    "kid_box":        ("images/kid_box", 0),
    "human_object":   ("images/human_object", 0),
    "socrates":       ("images/2560px-David_-_The_Death_of_Socrates", 0),
    "shutter_1243":   ("images/shutterstock_1243680295", 0),
    "shutter_1980":   ("images/shutterstock_1980085646", 0),
}

REPO = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Worker: run one (config, object) and dump a JSON result.
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(config_name: str, object_name: str, mask_index: int, out_dir: str):
    from main import run_pipeline

    ss_steps, stage2_steps, ss_distill, distill = CONFIGS[config_name]
    image_dir, default_index = OBJECTS[object_name]
    idx = mask_index if mask_index >= 0 else default_index

    glb_dir = os.path.join(out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)
    output_path = os.path.join(glb_dir, f"{config_name}__{object_name}.glb")

    t0 = time.perf_counter()
    output = run_pipeline(
        image_path=os.path.join(image_dir, "image.png"),
        mask_dir=image_dir,
        mask_index=idx,
        output_path=output_path,
        inference_steps=stage2_steps,
        ss_steps=ss_steps,
        ss_distill=ss_distill,
        distill=distill,
        seed=42,
        output_mesh=True,
        vertex_color_source="gaussian",
    )
    wall = time.perf_counter() - t0

    if output is None:
        raise RuntimeError("run_pipeline returned None (no mesh produced)")

    timings = dict(output.get("stage_timings", {}))
    peak_rss = output.get("peak_rss_gb")

    verts = faces = None
    glb = output.get("glb")
    if glb is not None:
        try:
            verts = int(glb.vertices.shape[0])
            faces = int(glb.faces.shape[0])
        except Exception:
            pass

    result = {
        "config": config_name,
        "object": object_name,
        "mask_index": idx,
        "ss_steps": ss_steps,
        "stage2_steps": stage2_steps,
        "ss_distill": ss_distill,
        "distill": distill,
        "stage_timings": timings,
        "stage1_s": timings.get("stage1_sparse_structure"),
        "stage2_s": timings.get("stage2_slat"),
        "total_s": timings.get("total"),
        "wall_s": wall,
        "peak_rss_gb": peak_rss,
        "vertices": verts,
        "faces": faces,
        "glb": output_path,
    }

    res_path = os.path.join(out_dir, f"{config_name}__{object_name}.json")
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[WORKER] Wrote {res_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Parent: orchestrate subprocesses and build the table.
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_table(results_dir: str) -> str:
    rows = []
    for obj in OBJECTS:
        for cfg in CONFIGS:
            p = os.path.join(results_dir, f"{cfg}__{obj}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                rows.append(json.load(f))

    lines = []
    lines.append("# WO-013 T4 — Stage-1 shortcut-sampling benchmark\n")
    lines.append(
        "All configs use the shipped shortcut-distilled stage-1 sampler "
        "(`--ss-distill`, on by default); only `--ss-steps` changes. Stage 2 (SLAT) "
        "is fixed at 12 flow-matching steps. Times are wall-clock seconds from the "
        "pipeline's per-stage instrumentation. Peak RSS is the worker process "
        "`ru_maxrss`. Quality is a manual 0–5 score of the exported GLB.\n"
    )
    lines.append(
        "| Object | Config | ss-steps | Stage 1 (s) | Stage 2 (s) | Total (s) | "
        "Peak RSS (GB) | Verts | Faces | Quality (0–5) |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|"
    )
    for r in rows:
        lines.append(
            "| {obj} | {cfg} | {ss} | {s1} | {s2} | {tot} | {rss} | {v} | {f} | |".format(
                obj=r["object"],
                cfg=r["config"],
                ss=r["ss_steps"],
                s1=_fmt(r.get("stage1_s")),
                s2=_fmt(r.get("stage2_s")),
                tot=_fmt(r.get("total_s")),
                rss=_fmt(r.get("peak_rss_gb")),
                v=_fmt(r.get("vertices"), 0),
                f=_fmt(r.get("faces"), 0),
            )
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="WO-013 stage-1 shortcut benchmark")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", help=argparse.SUPPRESS)
    ap.add_argument("--object", dest="object_name", help=argparse.SUPPRESS)
    ap.add_argument("--object-index", type=int, default=-1,
                    help="SAM mask index for the object (default: per-object default)")
    ap.add_argument("--objects", nargs="*", default=list(OBJECTS.keys()),
                    help="Subset of objects to run (default: all)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()),
                    help="Subset of configs to run (default: all)")
    ap.add_argument("--results-dir", default="tmp/bench",
                    help="Where per-run JSON + GLB + table are written")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-run subprocess timeout in seconds (default 3600)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if a result JSON already exists")
    ap.add_argument("--table-only", action="store_true",
                    help="Only rebuild the Markdown table from existing JSONs")
    args = ap.parse_args()

    # Worker branch: run exactly one combination in this process.
    if args.worker:
        run_worker(args.config, args.object_name, args.object_index, args.results_dir)
        return

    os.makedirs(args.results_dir, exist_ok=True)
    table_path = os.path.join(args.results_dir, "benchmark_results.md")

    if args.table_only:
        with open(table_path, "w") as f:
            f.write(build_table(args.results_dir))
        print(f"[BENCH] Wrote {table_path}")
        return

    objects = [o for o in args.objects if o in OBJECTS]
    configs = [c for c in args.configs if c in CONFIGS]
    bad = [o for o in args.objects if o not in OBJECTS] + [c for c in args.configs if c not in CONFIGS]
    if bad:
        print(f"[BENCH] Unknown objects/configs ignored: {bad}")

    total = len(objects) * len(configs)
    print(f"[BENCH] Running {total} combinations "
          f"({len(configs)} configs x {len(objects)} objects), timeout {args.timeout}s each.")

    n = 0
    for obj in objects:
        for cfg in configs:
            n += 1
            res_path = os.path.join(args.results_dir, f"{cfg}__{obj}.json")
            if os.path.exists(res_path) and not args.force:
                print(f"[BENCH] ({n}/{total}) skip {cfg}__{obj} (exists)")
                continue
            print(f"[BENCH] ({n}/{total}) run  {cfg}__{obj} …")
            cmd = [
                sys.executable, os.path.join(REPO, "benchmark_wo013.py"),
                "--worker", "--config", cfg, "--object", obj,
                "--object-index", str(args.object_index),
                "--results-dir", args.results_dir,
            ]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=REPO, timeout=args.timeout)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
                print(f"[BENCH] ({n}/{total}) TIMEOUT after {args.timeout}s")
            dt = time.perf_counter() - t0
            if not ok:
                # Record the failure so the table shows a gap and the sweep is resumable.
                with open(res_path, "w") as f:
                    json.dump({
                        "config": cfg, "object": obj, "error": True,
                        "wall_s": dt, "ss_steps": CONFIGS[cfg][0],
                        "stage2_steps": CONFIGS[cfg][1],
                    }, f, indent=2)
                print(f"[BENCH] ({n}/{total}) FAILED {cfg}__{obj} ({dt:.0f}s)")
            else:
                print(f"[BENCH] ({n}/{total}) done {cfg}__{obj} ({dt:.0f}s)")

            # Rebuild the table incrementally so partial sweeps are still useful.
            with open(table_path, "w") as f:
                f.write(build_table(args.results_dir))

    print(f"[BENCH] Complete. Table: {table_path}")


if __name__ == "__main__":
    main()
