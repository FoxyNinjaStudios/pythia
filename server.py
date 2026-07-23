"""
server.py  –  FastAPI web server for the SAM-3D interactive demo.

Endpoints
---------
GET  /                          → serve static/index.html
POST /upload                    → save image, return image_id
POST /segment                   → SAM point segmentation → mask (base64 PNG)
POST /reconstruct               → launch async 3-D reconstruction, return job_id
GET  /status/{job_id}           → SSE stream of progress events
GET  /result/{job_id}           → download the final GLB

Run
---
    conda activate sam-3d-mlx
    python server.py
  or
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── environment must be set before any ML imports ─────────────────────────────
os.environ.setdefault("SPARSE_BACKEND",      "mps")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("OMP_NUM_THREADS",     "14")
os.environ.setdefault("MKL_NUM_THREADS",     "14")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
from PIL import Image as PILImage

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── directory setup ────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("tmp/uploads")
RESULT_DIR = Path("tmp/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("sam3d.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Memory sampling  (drives the live RAM graph shown during reconstruction)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    _PROC = psutil.Process()

    def _mem_rss_gb() -> float:
        """Current process resident set size, in GB (live, goes up and down)."""
        return _PROC.memory_info().rss / 1e9

    def _sys_used_gb() -> float:
        return psutil.virtual_memory().used / 1e9

    def _sys_total_gb() -> float:
        return psutil.virtual_memory().total / 1e9
except Exception:  # psutil unavailable → fall back to peak RSS from resource
    import resource

    def _mem_rss_gb() -> float:
        # ru_maxrss is bytes on macOS, KB on Linux; treat as macOS bytes here.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

    def _sys_used_gb() -> float:
        return 0.0

    def _sys_total_gb() -> float:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe log buffer  (used by /logs SSE endpoint)
# ─────────────────────────────────────────────────────────────────────────────

_log_lines: List[str] = []
_log_lock  = threading.Lock()
_LOG_MAX   = 2000
_ANSI_RE   = re.compile(r'\x1b\[[0-9;]*[mK]')


class _LogBufferHandler(logging.Handler):
    """Appends every log record to _log_lines (thread-safe, bounded)."""
    def emit(self, record: logging.LogRecord):
        line = self.format(record)
        with _log_lock:
            _log_lines.append(line)
            if len(_log_lines) > _LOG_MAX:
                del _log_lines[0]


class _StdoutTee:
    """
    Tees sys.stdout into _log_lines so print() calls from the pipeline
    appear in the /logs SSE stream alongside log records.
    """
    def __init__(self, original):
        self._orig = original
        self._buf  = ""

    def write(self, text: str):
        self._orig.write(text)
        self._buf += _ANSI_RE.sub("", text)   # strip ANSI colour codes
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                with _log_lock:
                    _log_lines.append(line)
                    if len(_log_lines) > _LOG_MAX:
                        del _log_lines[0]

    def flush(self):            self._orig.flush()
    def fileno(self):           return self._orig.fileno()
    def isatty(self):           return False


sys.stdout = _StdoutTee(sys.stdout)

_buf_handler = _LogBufferHandler()
_buf_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logging.getLogger().addHandler(_buf_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Job state
# ─────────────────────────────────────────────────────────────────────────────

class JobState:
    def __init__(self, job_id: str):
        self.job_id      = job_id
        self.progress    = 0
        self.message     = "Queued"
        self.done        = False
        self.error: Optional[str] = None
        self.result_path: Optional[str] = None
        self._queues: List[asyncio.Queue] = []
        # Live memory tracking (for the RAM graph). mem_series is the full
        # history so a late subscriber can still redraw the whole curve.
        self.mem_series: List[dict] = []
        self.peak_gb: float   = 0.0
        self.sys_total_gb: float = _sys_total_gb()
        self._mem_t0: float   = time.time()

    def start_mem(self):
        """Reset the memory clock to the moment reconstruction actually begins."""
        self._mem_t0 = time.time()
        self.mem_series.clear()
        self.peak_gb = 0.0

    def sample_mem(self):
        """Capture one memory reading and broadcast it to subscribers."""
        rss = _mem_rss_gb()
        self.peak_gb = max(self.peak_gb, rss)
        point = {
            "t":   round(time.time() - self._mem_t0, 2),
            "rss": round(rss, 3),
            "sys": round(_sys_used_gb(), 3),
        }
        self.mem_series.append(point)
        if len(self.mem_series) > 5000:      # keep bounded for very long runs
            del self.mem_series[0]
        self._broadcast({"mem": point, "peak": round(self.peak_gb, 3)})

    def update(self, message: str, progress: int):
        self.message  = message
        self.progress = progress
        self._broadcast({"progress": progress, "message": message})

    def complete(self, result_path: str):
        self.done        = True
        self.result_path = result_path
        self.progress    = 100
        self.message     = "complete"
        self._broadcast({"progress": 100, "message": "complete", "done": True,
                         "peak": round(self.peak_gb, 3)})

    def fail(self, error: str):
        self.done  = True
        self.error = error
        self._broadcast({"progress": -1, "message": f"error: {error}", "done": True})

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    def _broadcast(self, payload: dict):
        for q in list(self._queues):
            q.put_nowait(payload)


jobs: Dict[str, JobState] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    image_id:        str
    positive_points: List[Dict[str, float]]
    negative_points: Optional[List[Dict[str, float]]] = []


class ReconstructRequest(BaseModel):
    image_id:  str
    mask_b64:  str   # base64-encoded PNG mask
    # Quality preset step counts (stage-1 = coarse shape, stage-2 = latent refine).
    # Client sends these from the Fast/Medium/Slow presets; default is Fast (8/8).
    stage1_steps: int = 8
    stage2_steps: int = 8
    # Layout placement: also produce a scene-placed GLB positioning the object in
    # camera space via the predicted pose. layout_refine adds ICP + render-compare
    # pose refinement (slower, CPU-only).
    layout: bool = False
    layout_refine: bool = False
    # Shortcut-model distillation: sample both flow stages CFG-free with step-size
    # conditioning (~1 eval/step). Much faster with few steps; needs distilled weights.
    distill: bool = False


class DepthRequest(BaseModel):
    image_id: str
    mask_b64: str   # base64-encoded PNG mask


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    logger.info("SAM-3D server starting…")
    asyncio.get_event_loop().run_in_executor(None, _preload_sam)
    yield

app = FastAPI(title="SAM-3D Interactive Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    img  = PILImage.open(io.BytesIO(data)).convert("RGB")
    image_id = str(uuid.uuid4())
    img.save(UPLOAD_DIR / f"{image_id}.png")
    return {"image_id": image_id, "width": img.width, "height": img.height}


# ── Segment ────────────────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(req: SegmentRequest):
    img_path = UPLOAD_DIR / f"{req.image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    image = np.array(PILImage.open(img_path).convert("RGB"))

    loop = asyncio.get_event_loop()
    mask = await loop.run_in_executor(
        None,
        lambda: _sam_predict(image, req.positive_points, req.negative_points),
    )
    mask = smooth_mask(mask)

    from sam_wrapper import mask_to_base64_png
    return {"mask_b64": mask_to_base64_png(mask)}


# ── Depth preview ──────────────────────────────────────────────────────────────

@app.post("/depth")
async def depth(req: DepthRequest):
    """Greyscale depth map of the masked object (best-effort, for the preview)."""
    img_path = UPLOAD_DIR / f"{req.image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    image = np.array(PILImage.open(img_path).convert("RGB"))
    from sam_wrapper import base64_png_to_mask
    mask = base64_png_to_mask(req.mask_b64)

    loop = asyncio.get_event_loop()
    depth_b64 = await loop.run_in_executor(None, lambda: _depth_to_png(image, mask))
    return {"depth_b64": depth_b64}


# ── Reconstruct ────────────────────────────────────────────────────────────────

@app.post("/reconstruct")
async def reconstruct(req: ReconstructRequest):
    img_path = UPLOAD_DIR / f"{req.image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    # Save mask
    mask_bytes = base64.b64decode(req.mask_b64)
    mask_img   = PILImage.open(io.BytesIO(mask_bytes)).convert("L")
    mask_path  = UPLOAD_DIR / f"{req.image_id}_mask.png"
    mask_img.save(mask_path)

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobState(job_id)

    # Run pipeline in a background thread (non-blocking)
    # Clamp to a sane range so a bad client value can't hang the machine.
    stage1_steps = max(1, min(int(req.stage1_steps), 100))
    stage2_steps = max(1, min(int(req.stage2_steps), 100))

    asyncio.get_event_loop().run_in_executor(
        None,
        _run_reconstruction_sync,
        job_id,
        str(img_path),
        str(mask_path),
        stage1_steps,
        stage2_steps,
        bool(req.layout),
        bool(req.layout_refine),
        bool(req.distill),
    )

    return {"job_id": job_id}


# ── Status SSE ─────────────────────────────────────────────────────────────────

@app.get("/status/{job_id}")
async def status_stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    async def event_gen():
        # Immediately send current state (including any memory history so a late
        # subscriber can redraw the full RAM curve alongside the result).
        yield _sse({
            "progress":  job.progress,
            "message":   job.message,
            "done":      job.done,
            "mem_series": job.mem_series,
            "sys_total": round(job.sys_total_gb, 2),
            "peak":      round(job.peak_gb, 3),
        })
        if job.done:
            return

        q = job.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield _sse(event)
                    if event.get("done"):
                        break
                except asyncio.TimeoutError:
                    yield _sse({"ping": True})  # keep-alive
        finally:
            job.unsubscribe(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Log streaming SSE ─────────────────────────────────────────────────────────

@app.get("/logs")
async def log_stream():
    """SSE stream of all Python log messages. Front-end console subscribes here."""
    with _log_lock:
        cursor = len(_log_lines)   # start from current tail, skip old history

    async def gen():
        nonlocal cursor
        # Send last 50 lines as backlog
        with _log_lock:
            backlog = _log_lines[max(0, cursor - 50):cursor]
        for line in backlog:
            yield _sse({"line": line})

        while True:
            await asyncio.sleep(0.25)
            with _log_lock:
                new  = _log_lines[cursor:]
                cursor = len(_log_lines)
            for line in new:
                yield _sse({"line": line})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Result download ────────────────────────────────────────────────────────────

@app.get("/result/{job_id}")
async def get_result(job_id: str, format: str = "glb"):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if not job.done:
        raise HTTPException(202, "Not ready yet")
    if job.error:
        raise HTTPException(500, job.error)
    fmt_raw = str(format).lower()
    if fmt_raw == "placed":
        # Scene-placed GLB (object positioned in camera space).
        path = Path(job.result_path).with_name(Path(job.result_path).stem + "_placed.glb")
        if not path.exists():
            raise HTTPException(404, "Placed GLB not available")
        return FileResponse(
            str(path),
            media_type="model/gltf-binary",
            filename="reconstruction_placed.glb",
        )
    fmt = "ply" if fmt_raw == "ply" else "glb"
    path = Path(job.result_path).with_suffix(f".{fmt}")
    if not path.exists():
        raise HTTPException(404, f"{fmt.upper()} not available")
    media_type = "text/plain" if fmt == "ply" else "model/gltf-binary"
    return FileResponse(
        str(path),
        media_type=media_type,
        filename=f"reconstruction.{fmt}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (run in ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _sam_predict(image, positive_points, negative_points):
    from sam_wrapper import predict_mask
    return predict_mask(image, positive_points, negative_points)


def smooth_mask(
    mask: np.ndarray,
    close_frac: float = 0.006,
    open_frac: float = 0.004,
    blur_frac: float = 0.004,
    keep_largest: bool = True,
) -> np.ndarray:
    """Clean a SAM mask before reconstruction.

    Fills pinholes, removes speckles, optionally keeps only the largest blob,
    and smooths the jagged boundary (blur -> re-threshold). Kernel sizes are a
    fraction of the image's shorter side so the amount of smoothing is
    resolution-independent. Returns a uint8 0/255 mask (same as the input).
    """
    import cv2

    m = (np.asarray(mask) > 127).astype(np.uint8) * 255
    if m.ndim == 3:
        m = m[..., -1]

    short = max(1, min(m.shape[:2]))

    def _odd(frac: float, lo: int = 3) -> int:
        k = int(round(short * frac))
        k = max(lo, k)
        return k | 1  # force odd

    # 1) close then open: fill pinholes, then drop speckles
    if close_frac > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_frac), _odd(close_frac)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_frac > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_frac), _odd(open_frac)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    # 2) keep only the largest connected component (drops stray islands)
    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 2:  # background + >1 blob
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = np.where(labels == largest, 255, 0).astype(np.uint8)

    # 3) blur -> threshold: round the jaggies
    if blur_frac > 0:
        ksize = _odd(blur_frac)
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
        m = (m > 127).astype(np.uint8) * 255

    if not m.any():
        # Never hand back an empty mask if the original had something.
        return (np.asarray(mask) > 127).astype(np.uint8) * 255
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Depth preview  – run MoGe once to show a greyscale depth map of the object
# while the (much slower) 3-D reconstruction runs.
# ─────────────────────────────────────────────────────────────────────────────

_depth_pipeline = None
_depth_lock = threading.Lock()


def _get_depth_pipeline():
    """Lazily build (and cache) a lightweight pipeline used only for MoGe depth.

    Construction only loads the small preprocessors; the MoGe weights are loaded
    on first ``compute_pointmap`` call and unloaded again right after so they do
    not compete for memory with the reconstruction pipeline.
    """
    global _depth_pipeline
    with _depth_lock:
        if _depth_pipeline is None:
            from sam3d_objects.pipeline.inference_pipeline_low_memory import (
                InferencePipelineLowMemory,
            )
            _depth_pipeline = InferencePipelineLowMemory(
                config_path="checkpoints/hf/pipeline.yaml",
                device="cpu",
                dtype="float16",
                cache_dir=".cache",
            )
        return _depth_pipeline


def _depth_to_png(image_rgb: np.ndarray, mask: np.ndarray) -> Optional[str]:
    """Return a base64 RGBA PNG: greyscale depth inside the mask, transparent out.

    Near = bright, far = dark. Never raises — returns ``None`` on any failure so
    the client simply keeps showing the plain mask.
    """
    try:
        import cv2

        pipe = _get_depth_pipeline()

        m = (np.asarray(mask) > 127)
        if m.ndim == 3:
            m = m[..., -1]
        if not m.any():
            return None

        # MoGe expects an RGBA image (mask carried in the alpha channel).
        rgba = pipe.merge_image_and_mask(image_rgb, m.astype(np.uint8) * 255)

        # Serialise MoGe access: it is not safe to run the same model from two
        # threads, and the reconstruction pipeline uses its own instance.
        with _depth_lock:
            point_map = pipe.compute_pointmap(rgba)
            try:
                z = point_map["pointmap"][2].detach().cpu().numpy().astype(np.float32)
            finally:
                # Free the MoGe weights before the heavy reconstruction stages.
                pipe._unload_depth_model()

        # Resize the mask to the pointmap grid if MoGe changed resolution.
        if z.shape != m.shape:
            m = cv2.resize(m.astype(np.uint8), (z.shape[1], z.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0

        vals = z[m]
        if vals.size == 0 or not np.isfinite(vals).any():
            return None
        vals = vals[np.isfinite(vals)]

        # Robust normalise within the object (ignore outlier tails).
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
        # Map depth through a full-colour scale (turbo): near = warm, far = cool.
        depth_u8 = ((1.0 - norm) * 255.0).astype(np.uint8)  # near = high end
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)  # BGR
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Compose RGBA: coloured depth where masked, fully transparent elsewhere.
        h, w = norm.shape
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[..., :3] = colored
        out[..., 3] = np.where(m, 255, 0).astype(np.uint8)

        buf = io.BytesIO()
        PILImage.fromarray(out, mode="RGBA").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:  # never break the preview
        logger.warning("Depth preview failed: %s", exc)
        return None


class _PipelineLogHandler(logging.Handler):
    """Redirect pipeline log messages to SSE progress updates."""

    def __init__(self, job: JobState):
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        if "STAGE 0" in msg or "depth" in msg.lower():
            self.job.update("Estimating depth…", 15)
        elif "STAGE 1" in msg:
            self.job.update("Generating sparse voxels…", 30)
        elif "STAGE 2" in msg:
            self.job.update("Refining latent structure…", 55)
        elif "STAGE 3" in msg or "decod" in msg.lower():
            self.job.update("Decoding 3-D mesh…", 70)
        elif "postprocess" in msg.lower():
            self.job.update("Post-processing mesh…", 80)


def _run_reconstruction_sync(
    job_id: str,
    img_path: str,
    mask_path: str,
    stage1_steps: int = 8,
    stage2_steps: int = 8,
    layout: bool = False,
    layout_refine: bool = False,
    distill: bool = False,
):
    """Full reconstruction pipeline – runs in a thread, updates job SSE queue."""
    job = jobs[job_id]
    # Attach log handler so pipeline stages drive the progress bar
    root_logger = logging.getLogger()
    handler = _PipelineLogHandler(job)
    root_logger.addHandler(handler)

    # Live RAM sampler: poll process memory ~2×/s and stream it to the client
    # so the UI can draw a memory graph during (and after) reconstruction.
    job.start_mem()
    _mem_stop = threading.Event()

    def _mem_sampler():
        while not _mem_stop.is_set():
            try:
                job.sample_mem()
            except Exception:
                pass
            _mem_stop.wait(0.5)

    _mem_thread = threading.Thread(target=_mem_sampler, name=f"mem-{job_id}", daemon=True)
    _mem_thread.start()

    try:
        image = np.array(PILImage.open(img_path).convert("RGB"))
        mask  = np.array(PILImage.open(mask_path).convert("L"))

        job.update("Loading pipeline…", 5)

        from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory
        pipeline = InferencePipelineLowMemory(
            config_path="checkpoints/hf/pipeline.yaml",
            device="cpu",
            dtype="float16",
            cache_dir=".cache",
        )

        job.update("Running 3-D reconstruction…", 12)

        # When the optional splat module is on, ask the decoder for the Gaussian
        # appearance rep too so we can export a real 3DGS .ply (see splat_export).
        import splat_export
        _decode_formats = ["mesh", "gaussian"] if splat_export.enabled() else ["mesh"]

        output = pipeline.run(
            image,
            mask,
            seed=42,
            stage1_only=False,
            # Step counts come from the client quality preset (Fast/Medium/Slow).
            # Stage 1 generates the coarse sparse-voxel shape; more steps give it
            # more confident geometry and noticeably less hallucinated wrinkling
            # on depth-ambiguous (grazing) surfaces like sofa arms. Slower, but
            # the main quality lever for single-view side geometry.
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=_decode_formats,
            simplify_ratio=0.0,
            vertex_color_source="gaussian",
            with_layout_postprocess=layout,
            layout_refine=layout_refine,
            use_stage1_distillation=distill,
            use_stage2_distillation=distill,
        )

        # The pipeline already returns a per-vertex-colored GLB (to_glb colors each
        # vertex from the Gaussian appearance field). Vertex color is more accurate
        # than the UV bake here, so just export it directly.
        result_mesh = output["glb"]

        result_path = str(RESULT_DIR / f"{job_id}.glb")
        result_ply = str(RESULT_DIR / f"{job_id}.ply")
        job.update("Exporting GLB + PLY…", 95)
        result_mesh.export(result_path, file_type="glb")

        # Scene-placed GLB (object positioned in camera space via predicted pose).
        placed_mesh = output.get("glb_placed")
        if placed_mesh is not None:
            try:
                placed_path = str(RESULT_DIR / f"{job_id}_placed.glb")
                placed_mesh.export(placed_path, file_type="glb")
                iou = output.get("layout_iou")
                logger.info(
                    f"[JOB {job_id}] Placed GLB exported"
                    + (f" (layout IoU {iou})" if iou is not None else "")
                )
            except Exception as exc:
                logger.warning(f"[JOB {job_id}] Placed GLB export failed: {exc}")

        # PLY: prefer a real Gaussian splat (optional module); otherwise fall back
        # to exporting the mesh as a .ply so the download link always works.
        if not splat_export.export_splat(output, result_ply):
            try:
                result_mesh.export(result_ply, file_type="ply")
            except Exception as exc:
                logger.warning(f"[JOB {job_id}] PLY export failed: {exc}")

        job.complete(result_path)
        logger.info(f"[JOB {job_id}] Done → {result_path}")

    except Exception as exc:
        job.fail(str(exc))
        logger.error(f"[JOB {job_id}] Failed:\n{traceback.format_exc()}")
    finally:
        _mem_stop.set()
        job.sample_mem()          # one final reading so the graph ends at the peak
        root_logger.removeHandler(handler)


# ─────────────────────────────────────────────────────────────────────────────
# Startup  – pre-download SAM weights in background
# ─────────────────────────────────────────────────────────────────────────────

def _preload_sam():
    try:
        from sam_wrapper import ensure_sam_weights
        ensure_sam_weights()
    except Exception as exc:
        logger.warning(f"SAM weight preload failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8005,
        workers=1,       # must be 1 – ML models are not fork-safe
        reload=False,
    )
