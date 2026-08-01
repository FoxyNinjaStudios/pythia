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
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── environment must be set before any ML imports ─────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS",     "14")
os.environ.setdefault("MKL_NUM_THREADS",     "14")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Reconstruction runs on the CPU. The upstream native-Metal reconstruction
# kernels were removed, so the 3-D generative pipeline (sparse structure, SLAT,
# mesh decode) is CPU-only; SAM segmentation and MoGe depth still run on Metal
# (MPS). Leave CPU headroom so the asyncio event loop (SSE progress + keep-alive
# pings) keeps running while the CPU-bound 3-D stages run. If every core is
# saturated the loop starves, no bytes reach the browser's EventSource, and the
# UI shows "Connection lost" even though the job finishes and writes the GLB.
_cpu_threads = max(1, (os.cpu_count() or 8) - 2)
os.environ["OMP_NUM_THREADS"] = str(_cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(_cpu_threads)

import numpy as np
import torch
from PIL import Image as PILImage

# Disable MPS detection for the 3-D generative pipeline so every
# `torch.backends.mps.is_available()` fallback lands on the CPU. Must run before
# any pipeline module reads torch.backends.mps. We first record the real MPS
# availability so SAM and MoGe can still load on Metal via SAM3D_SAM_DEVICE /
# SAM3D_MOGE_DEVICE.
torch.set_num_threads(_cpu_threads)  # keep a couple of cores for the event loop
_real_mps = torch.backends.mps.is_available()
if _real_mps:
    os.environ["SAM3D_SAM_DEVICE"] = "mps"    # keep SAM segmentation on Metal
    os.environ["SAM3D_MOGE_DEVICE"] = "mps"   # keep MoGe depth on Metal
torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
logging.getLogger("sam3d.server").warning(
    "3-D reconstruction on CPU; SAM + MoGe on Metal."
)

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── directory setup ────────────────────────────────────────────────────────────
# All checkpoints, caches and I/O live under a single app root (PYTHIA_HOME,
# default ~/Downloads/pythia) so a packaged build run from any directory still
# finds and stores its data in one predictable place. Point the Hugging Face
# hub cache there too, *before* any huggingface_hub / transformers import.
import paths
paths.configure_hf_cache()
paths.ensure_dirs()

UPLOAD_DIR = paths.UPLOAD_DIR
RESULT_DIR = paths.RESULT_DIR

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
        # Source image / mask paths, so post-processing steps (e.g. functional
        # part segmentation) can be run on demand after reconstruction finishes.
        self.img_path: Optional[str] = None
        self.mask_path: Optional[str] = None
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
        # Progress/memory events are produced by worker threads (reconstruction
        # runs in a ThreadPoolExecutor, memory sampling in its own thread). An
        # asyncio.Queue is NOT thread-safe, so hop onto the event-loop thread to
        # enqueue — otherwise the SSE getter never wakes and the browser stalls.
        loop = _main_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._deliver, payload)
        else:
            self._deliver(payload)

    def _deliver(self, payload: dict):
        for q in list(self._queues):
            q.put_nowait(payload)


jobs: Dict[str, JobState] = {}

# The main asyncio event loop, captured at startup so worker threads can hand
# SSE events back to it safely (see JobState._broadcast).
_main_loop: Optional[asyncio.AbstractEventLoop] = None


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    image_id:        str
    positive_points: List[Dict[str, float]] = []
    negative_points: Optional[List[Dict[str, float]]] = []
    # Optional SAM 3 text/concept prompt. When non-empty, segmentation runs in
    # text mode (SAM 3) and the point prompts are ignored.
    text:            Optional[str] = None


class ReconstructRequest(BaseModel):
    image_id:  str
    mask_b64:  str   # base64-encoded PNG mask
    # Quality preset step counts (stage-1 = coarse shape, stage-2 = latent refine).
    # Client sends these from the Fast/Medium/Slow presets; default is Fast (8/8).
    stage1_steps: int = 8
    stage2_steps: int = 8
    # Shortcut-model distillation: sample the flow stages CFG-free with step-size
    # conditioning (~1 eval/step). Much faster with few steps; needs distilled weights.
    # Stage 1 (sparse structure) IS shortcut-distilled in the shipped weights, so it is
    # on by default; stage 2 (SLAT) is genuine flow matching, so `distill` (stage 2) is off.
    ss_distill: bool = True
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
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    logger.info("SAM-3D server starting…")
    asyncio.get_event_loop().run_in_executor(None, _preload_sam)
    yield

# Resolve the static directory both in development and when frozen into a
# PyInstaller executable (which unpacks bundled data under sys._MEIPASS).
if hasattr(sys, "_MEIPASS"):
    base_path = sys._MEIPASS          # PyInstaller runtime extraction dir
else:
    base_path = os.path.abspath(".")  # normal development run
static_dir = os.path.join(base_path, "static")

app = FastAPI(title="SAM-3D Interactive Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


# ── Model status ────────────────────────────────────────────────────────────────

def _human_size(num_bytes: int) -> str:
    """Format a byte count as a short human-readable string."""
    if not num_bytes:
        return "—"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < step:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} PB"


# ── Model download orchestration ────────────────────────────────────────────────
# Some weights live on Hugging Face and one repo (facebook/sam-3d-objects) is
# gated behind Meta approval, so downloads are driven from the UI: the user pastes
# an access token, then triggers a per-model download that runs in a background
# thread while the sidebar polls /models for progress.

_downloads = {}          # model_id -> {"state","message","progress"}
_downloads_lock = threading.Lock()
_hf_user = None          # HF username once a token is validated (in-memory only)
_hf_token = None         # HF token kept in memory only, never written to disk

# Per-model download source. ``gated`` marks repos that need Meta approval.
# ``dl_bytes`` is the approximate total transfer size, shown in the progress label.
_MODEL_META = {
    "sam2":  {"repo": None,                      "gated": False, "dl_bytes": 898_000_000},
    "sam3":  {"repo": "facebook/sam3",           "gated": True,  "dl_bytes": 3_400_000_000},
    "sam3d": {"repo": "facebook/sam-3d-objects", "gated": True,  "dl_bytes": 13_106_000_000},
    "moge":  {"repo": "Ruicheng/moge-vitl",      "gated": False, "dl_bytes": 1_340_000_000},
}


def _set_dl(model_id: str, **kw) -> None:
    with _downloads_lock:
        rec = _downloads.setdefault(model_id, {"state": "idle", "message": "", "progress": None})
        rec.update(kw)


def _get_dl(model_id: str) -> dict:
    with _downloads_lock:
        rec = _downloads.get(model_id)
        return dict(rec) if rec else {"state": "idle", "message": "", "progress": None}


def _hf_auth() -> dict:
    """Report whether an in-memory Hugging Face token is set (no network call)."""
    return {"authenticated": bool(_hf_token), "user": _hf_user}


def _friendly_download_error(model_id: str, exc: Exception) -> str:
    """Turn an HF download exception into an actionable message for the UI."""
    text = str(exc)
    lowered = text.lower()
    meta = _MODEL_META.get(model_id, {})
    repo = meta.get("repo")
    gated_hint = False
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        if isinstance(exc, GatedRepoError):
            gated_hint = True
        elif isinstance(exc, RepositoryNotFoundError):
            gated_hint = True  # gated repos 404 to unauthenticated users
    except Exception:
        pass
    if not gated_hint and ("401" in text or "403" in text or "gated" in lowered
                           or "awaiting" in lowered or "access" in lowered
                           or "unauthorized" in lowered or "authentication" in lowered):
        gated_hint = True
    if gated_hint and meta.get("gated") and repo:
        return (f"Access not granted yet. Request approval at "
                f"https://huggingface.co/{repo} and make sure your token above is "
                f"from the approved account.")
    if "token" in lowered or "credential" in lowered:
        return "A valid Hugging Face token is required — add one above and retry."
    return f"Download failed: {text}"


def _path_size(path: Path) -> int:
    """Total bytes at ``path`` — the file size, or the sum of a directory tree."""
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for p in path.rglob("*"):
                try:
                    if p.is_file():
                        total += p.stat().st_size
                except OSError:
                    pass
            return total
    except OSError:
        pass
    return 0


def _start_progress_sampler(model_id: str, watch_path, total_bytes: int,
                            label: str) -> threading.Event:
    """Stream real download progress by polling the destination's growing size.

    Returns a ``threading.Event``; call ``.set()`` on it to stop the sampler
    (do this once the download finishes). Progress is reported as the fraction of
    ``total_bytes`` written since the sampler started, so pre-existing partial
    files don't skew the bar. Works for both single-file (SAM 2) and Hugging Face
    snapshot (dir tree) downloads without needing a library-specific callback.
    """
    watch_path = Path(watch_path)
    baseline = _path_size(watch_path)
    stop = threading.Event()

    def _run() -> None:
        while not stop.is_set():
            got = max(0, _path_size(watch_path) - baseline)
            if total_bytes > 0:
                frac = min(0.999, got / total_bytes)
                msg = f"{label}… {_human_size(got)} / {_human_size(total_bytes)}"
            else:
                frac = None
                msg = f"{label}… {_human_size(got)}"
            _set_dl(model_id, state="running", message=msg, progress=frac)
            stop.wait(0.5)

    threading.Thread(target=_run, name=f"dlprog-{model_id}", daemon=True).start()
    return stop


def _download_worker(model_id: str) -> None:
    global _hf_token, _hf_user
    try:
        meta = _MODEL_META.get(model_id, {})
        size_str = _human_size(meta.get("dl_bytes", 0))
        if model_id == "sam2":
            stop = _start_progress_sampler(
                "sam2", paths.SAM2_CHECKPOINT_PATH,
                meta.get("dl_bytes", 0), "Downloading SAM 2.1 Hiera-L")
            try:
                from sam_wrapper import ensure_sam_weights
                ensure_sam_weights()
            finally:
                stop.set()
        elif model_id == "moge":
            repo_dir = paths.HF_HUB_DIR / ("models--" + meta["repo"].replace("/", "--"))
            stop = _start_progress_sampler(
                "moge", repo_dir, meta.get("dl_bytes", 0), "Downloading MoGe ViT-L")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=meta["repo"])
            finally:
                stop.set()
        elif model_id == "sam3":
            from huggingface_hub import snapshot_download
            token = _hf_token
            if not token:
                raise PermissionError("Add a Hugging Face token for the gated model first.")
            repo_dir = paths.HF_HUB_DIR / ("models--" + meta["repo"].replace("/", "--"))
            stop = _start_progress_sampler(
                "sam3", repo_dir, meta.get("dl_bytes", 0), "Downloading SAM 3 weights")
            try:
                snapshot_download(repo_id=meta["repo"], token=token)
            finally:
                stop.set()
        elif model_id == "sam3d":
            from huggingface_hub import snapshot_download
            token = _hf_token
            if not token:
                raise PermissionError("Add a Hugging Face token for the gated model first.")
            dest = paths.HF_WEIGHTS_DIR
            stop = _start_progress_sampler(
                "sam3d", dest, meta.get("dl_bytes", 0), "Downloading SAM 3D Objects weights")
            try:
                snapshot_download(
                    repo_id=meta["repo"],
                    local_dir=str(dest),
                    allow_patterns=["*.ckpt", "*.pt", "*.safetensors", "*.yaml", "*.json"],
                    token=token,
                )
                # The repo nests its weights under a ``checkpoints/`` subfolder, but
                # the app loads them flat from checkpoints/hf/. Move any nested files
                # up so the pipeline (and the status check) find them.
                nested = dest / "checkpoints"
                if nested.is_dir():
                    for f in nested.iterdir():
                        if f.is_file():
                            target = dest / f.name
                            if target.exists():
                                target.unlink()
                            shutil.move(str(f), str(target))
                    shutil.rmtree(nested, ignore_errors=True)
            finally:
                stop.set()
        else:
            raise ValueError(f"Unknown model '{model_id}'")
        _set_dl(model_id, state="done", message="Download complete.", progress=1.0)
    except Exception as exc:
        _set_dl(model_id, state="error",
                message=_friendly_download_error(model_id, exc), progress=None)
        logger.warning("Download of %s failed: %s", model_id, exc)
    finally:
        # The token is single-use: drop it from memory once the gated download
        # has run (whether it succeeded or failed) so it is never retained.
        if model_id in ("sam3d", "sam3"):
            _hf_token = None
            _hf_user = None


def _purge_hf_repo(repo_id: str) -> int:
    """Delete every cached revision of an HF repo. Returns bytes reclaimed."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hashes, freed = [], 0
        for repo in info.repos:
            if repo.repo_id == repo_id:
                freed += int(repo.size_on_disk)
                hashes.extend(rev.commit_hash for rev in repo.revisions)
        if hashes:
            info.delete_revisions(*hashes).execute()
        return freed
    except Exception as exc:
        logger.warning("HF cache purge for %s failed: %s", repo_id, exc)
        return 0


def _purge_model(model_id: str) -> int:
    """Delete a model's downloaded weights and unload it from memory.

    Returns the number of bytes reclaimed. Runs in a thread executor.
    """
    global _depth_pipeline
    freed = 0
    if model_id == "sam2":
        try:
            import sam_wrapper
            p = Path(sam_wrapper.SAM2_CHECKPOINT_PATH)
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
            sam_wrapper._predictor = None          # drop the in-memory predictor
        except Exception as exc:
            logger.warning("SAM 2 purge failed: %s", exc)
    elif model_id == "sam3d":
        hf = paths.HF_WEIGHTS_DIR
        if hf.exists():
            for f in list(hf.glob("*.ckpt")) + list(hf.glob("*.pt")) + list(hf.glob("*.safetensors")):
                try:
                    freed += f.stat().st_size
                    f.unlink()
                except Exception as exc:
                    logger.warning("Could not delete %s: %s", f, exc)
    elif model_id == "sam3":
        meta = _MODEL_META.get(model_id, {})
        if meta.get("repo"):
            freed += _purge_hf_repo(meta["repo"])
        try:
            import sam_wrapper
            sam_wrapper.unload_text_model()         # drop the in-memory SAM 3 model
        except Exception as exc:
            logger.warning("SAM 3 unload failed: %s", exc)
    elif model_id == "moge":
        meta = _MODEL_META.get(model_id, {})
        if meta.get("repo"):
            freed += _purge_hf_repo(meta["repo"])
        _depth_pipeline = None                      # drop the in-memory depth pipeline
    return freed


def _files_size(paths) -> tuple[bool, int]:
    """Return (all_present, total_bytes) for a list of checkpoint files."""
    total = 0
    all_present = True
    for p in paths:
        p = Path(p)
        if p.exists() and p.stat().st_size > 0:
            total += p.stat().st_size
        else:
            all_present = False
    return all_present, total


def _moge_cache() -> tuple[bool, int]:
    """Detect whether the MoGe (Ruicheng/moge-vitl) weights are in the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == "Ruicheng/moge-vitl" and repo.size_on_disk > 0:
                return True, int(repo.size_on_disk)
    except Exception:
        pass
    return False, 0


def _sam3_cache() -> tuple[bool, int]:
    """Detect whether the SAM 3 (facebook/sam3) weights are in the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == "facebook/sam3" and repo.size_on_disk > 0:
                return True, int(repo.size_on_disk)
    except Exception:
        pass
    return False, 0


def _models_status() -> list:
    """Report per-model download / load state for the three pipeline models."""
    # 1) SAM 2.1 Hiera-L (interactive point-prompt segmentation)
    try:
        import sam_wrapper
        sam2_path = Path(sam_wrapper.SAM2_CHECKPOINT_PATH)
        # Point-prompt segmentation keeps SAM 2 resident (_predictor). The text
        # mode uses a separate model (SAM 3, reported below), so SAM 2 is only
        # "loaded" when its own predictor is in memory.
        sam2_loaded = getattr(sam_wrapper, "_predictor", None) is not None
        sam3_loaded = getattr(sam_wrapper, "_sam3_model", None) is not None
    except Exception:
        sam2_path = Path(paths.SAM2_CHECKPOINT_PATH)
        sam2_loaded = False
        sam3_loaded = False
    sam2_down = sam2_path.exists() and sam2_path.stat().st_size > 0
    sam2_size = sam2_path.stat().st_size if sam2_down else 0

    # 1b) SAM 3 (text / concept segmentation). Gated HF repo (facebook/sam3),
    # cached under the shared HF hub dir; loaded on demand (and on upload) for
    # the default text-prompt mode.
    sam3_down, sam3_size = _sam3_cache()

    # 2) SAM 3D Objects reconstruction stack (weights under checkpoints/hf/).
    # Only the weights the inference pipeline actually loads are checked — the
    # encoders (ss_encoder.*, slat_encoder.ckpt) are training-only and one of
    # them ships as a 0-byte placeholder, so requiring them would wrongly report
    # the model as missing even though reconstruction works.
    hf = paths.HF_WEIGHTS_DIR
    sam3d_files = [
        hf / "ss_generator.ckpt",
        hf / "ss_decoder.ckpt",
        hf / "slat_generator.ckpt",
        hf / "slat_decoder_mesh.ckpt",
        hf / "slat_decoder_gs.ckpt",
    ]
    sam3d_down, sam3d_size = _files_size(sam3d_files)
    # The low-memory pipeline is (re)built per job and streams weights stage by
    # stage, then released in the job's ``finally``. It is only resident in RAM
    # while a reconstruction is actually running.
    sam3d_running = any(not j.done for j in jobs.values())
    sam3d_loaded = sam3d_running

    # 3) MoGe ViT-L (monocular depth / pointmap prior). The depth weights are
    # loaded on demand and unloaded right after each depth call, so they are
    # only resident in RAM during the (brief) depth computation itself.
    moge_down, moge_size = _moge_cache()
    moge_loaded = _depth_weights_loaded

    def entry(mid, name, role, downloaded, loaded, size):
        status = "loaded" if (downloaded and loaded) else (
            "downloaded" if downloaded else "not_downloaded"
        )
        meta = _MODEL_META.get(mid, {})
        return {
            "id": mid,
            "name": name,
            "role": role,
            "downloaded": downloaded,
            "loaded": loaded,
            "status": status,
            "size": _human_size(size),
            "gated": bool(meta.get("gated")),
            "repo": meta.get("repo"),
            "download": _get_dl(mid),
        }

    return [
        entry("sam2", "SAM 2.1 Hiera-L", "Interactive segmentation",
              sam2_down, sam2_loaded, sam2_size),
        entry("sam3", "SAM 3", "Text / concept segmentation",
              sam3_down, sam3_loaded, sam3_size),
        entry("sam3d", "SAM 3D Objects", "3-D reconstruction",
              sam3d_down, sam3d_loaded, sam3d_size),
        entry("moge", "MoGe ViT-L", "Depth / pointmap prior",
              moge_down, moge_loaded, moge_size),
    ]


@app.get("/models")
async def models_status():
    return {"models": _models_status(), "hf": _hf_auth()}


@app.get("/seg_status")
async def seg_status():
    """Load state of the active 2-D segmentation model (drives the Step-2 bar)."""
    try:
        from sam_wrapper import get_seg_load_status
        return get_seg_load_status()
    except Exception as exc:
        return {"state": "idle", "model": None, "message": "", "progress": None,
                "error": str(exc)}


class DownloadRequest(BaseModel):
    token: Optional[str] = None


@app.post("/models/{model_id}/download")
async def download_model(model_id: str, req: Optional[DownloadRequest] = None):
    """Kick off a background download for one model; poll /models for progress.

    Gated models accept a Hugging Face token in the body; it is validated,
    stashed in memory for this single download, and never persisted.
    """
    global _hf_user, _hf_token
    if model_id not in _MODEL_META:
        raise HTTPException(404, "Unknown model.")
    meta = _MODEL_META[model_id]
    if meta.get("gated"):
        token = ((req.token if req else None) or "").strip()
        if not token:
            raise HTTPException(400, "A Hugging Face token is required for this model.")
        try:
            from huggingface_hub import HfApi
            info = HfApi().whoami(token=token)      # raises on an invalid token
            _hf_user = info.get("name") if isinstance(info, dict) else None
            _hf_token = token                       # in-memory only; not persisted
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(401, f"Invalid token: {exc}")
    with _downloads_lock:
        cur = _downloads.get(model_id)
        if cur and cur.get("state") == "running":
            return {"state": "running"}
        _downloads[model_id] = {"state": "running", "message": "Starting…", "progress": None}
    threading.Thread(
        target=_download_worker, args=(model_id,),
        name=f"dl-{model_id}", daemon=True,
    ).start()
    return {"state": "running"}


@app.post("/models/{model_id}/purge")
async def purge_model(model_id: str):
    """Delete a model's downloaded weights and unload it from memory."""
    if model_id not in _MODEL_META:
        raise HTTPException(404, "Unknown model.")
    with _downloads_lock:
        cur = _downloads.get(model_id)
        if cur and cur.get("state") == "running":
            raise HTTPException(409, "A download is in progress for this model.")
    if model_id == "sam3d" and any(not j.done for j in jobs.values()):
        raise HTTPException(409, "A reconstruction is running; try again when it finishes.")
    freed = await asyncio.get_event_loop().run_in_executor(None, _purge_model, model_id)
    with _downloads_lock:
        _downloads.pop(model_id, None)
    return {"ok": True, "freed": _human_size(freed)}




# ── Upload ─────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    img  = PILImage.open(io.BytesIO(data)).convert("RGB")

    # Normalize uploads to a width of 1024, preserving aspect ratio.
    target_width = 1024
    if img.width != target_width:
        target_height = max(1, round(img.height * target_width / img.width))
        img = img.resize((target_width, target_height), PILImage.LANCZOS)

    image_id = str(uuid.uuid4())
    img.save(UPLOAD_DIR / f"{image_id}.png")

    # Text/concept segmentation is the default UI mode, so start loading the SAM 3
    # text model now that an image is in play. Runs in a worker thread so the
    # upload response is not blocked; the model is ready by the first segment.
    try:
        asyncio.get_event_loop().run_in_executor(None, _load_text_seg_model)
    except Exception:
        pass

    return {"image_id": image_id, "width": img.width, "height": img.height}


# ── Segment ────────────────────────────────────────────────────────────────────

@app.post("/segment")
async def segment(req: SegmentRequest):
    img_path = UPLOAD_DIR / f"{req.image_id}.png"
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    image = np.array(PILImage.open(img_path).convert("RGB"))

    text = (req.text or "").strip()
    loop = asyncio.get_event_loop()
    if text:
        mask = await loop.run_in_executor(None, lambda: _sam_predict_text(image, text))
    else:
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
    jobs[job_id].img_path  = str(img_path)
    jobs[job_id].mask_path = str(mask_path)

    # Free every 2-D segmentation model (SAM 2 points + SAM 3 text) before the
    # memory-hungry low-memory 3D pipeline starts, so they don't compete for
    # unified memory. The point-prompt path otherwise leaves SAM 2.1's feature
    # pyramid pinned and reconstruction runs several times slower under memory
    # compression than the text-prompt path.
    try:
        from sam_wrapper import unload_segmentation_models
        unload_segmentation_models()
    except Exception:
        pass

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
        bool(req.distill),
        bool(req.ss_distill),
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

@app.api_route("/result/{job_id}", methods=["GET", "HEAD"])
async def get_result(job_id: str, format: str = "glb"):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if not job.done:
        raise HTTPException(202, "Not ready yet")
    if job.error:
        raise HTTPException(500, job.error)
    fmt_raw = str(format).lower()
    fmt = "ply" if fmt_raw == "ply" else "glb"
    path = Path(job.result_path).with_suffix(f".{fmt}")
    if not path.exists():
        raise HTTPException(404, f"{fmt.upper()} not available")
    media_type = "text/plain" if fmt == "ply" else "model/gltf-binary"
    if fmt == "glb":
        # Inject a matte, non-metallic material so downloads / Quick Look don't
        # show the shiny-metal look glTF viewers apply to material-less meshes.
        # (The on-disk file stays vertex-coloured so segmentation can read it.)
        raw = path.read_bytes()
        try:
            from texture_baking import _patch_glb_metallic

            raw = _patch_glb_metallic(raw)
        except Exception as exc:
            logger.warning(f"[JOB {job_id}] metallic patch on serve skipped: {exc}")
        return Response(
            content=raw,
            media_type=media_type,
            headers={"Content-Disposition": 'attachment; filename="reconstruction.glb"'},
        )
    return FileResponse(
        str(path),
        media_type=media_type,
        filename=f"reconstruction.{fmt}",
    )


# ── Functional part segmentation (post-processing) ─────────────────────────────
# Runs on demand *after* reconstruction and writes a SEPARATE multi-object GLB
# ({job_id}_parts.glb), leaving the primary single-mesh result untouched so the
# in-browser editing tools keep working. Each part is a named GLB object that
# downstream tools can recolour / texture independently.

@app.post("/segment_parts/{job_id}")
async def segment_parts(
    job_id: str,
    n_colors: int = 6,
    bake: bool = False,
    texture_size: int = 2048,
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if not job.done or job.error:
        raise HTTPException(202, "Reconstruction not finished")
    if not job.result_path or not Path(job.result_path).exists():
        raise HTTPException(404, "Reconstructed mesh not available")

    loop = asyncio.get_event_loop()
    try:
        out_path, n_parts = await loop.run_in_executor(
            None,
            _run_part_segmentation_sync,
            job_id,
            job.result_path,
            job.img_path,
            job.mask_path,
            max(2, min(int(n_colors), 16)),
            bool(bake),
            max(256, min(int(texture_size), 4096)),
        )
    except Exception as exc:
        logger.exception(f"[JOB {job_id}] part segmentation failed")
        raise HTTPException(500, f"Part segmentation failed: {exc}")

    return {"job_id": job_id, "parts": n_parts, "url": f"/result_parts/{job_id}"}


@app.api_route("/result_parts/{job_id}", methods=["GET", "HEAD"])
async def get_result_parts(job_id: str):
    path = RESULT_DIR / f"{job_id}_parts.glb"
    if not path.exists():
        raise HTTPException(404, "Segmented parts not available")
    return FileResponse(
        str(path),
        media_type="model/gltf-binary",
        filename="reconstruction_parts.glb",
    )


# ── Photo-colour re-projection (post-processing) ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Worker functions (run in ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _sam_predict(image, positive_points, negative_points):
    from sam_wrapper import predict_mask
    return predict_mask(image, positive_points, negative_points)


def _sam_predict_text(image, text):
    from sam_wrapper import predict_mask_text
    return predict_mask_text(image, text)


def _load_text_seg_model():
    """Load the SAM 3 text/concept model (worker-thread target for /upload)."""
    try:
        from sam_wrapper import preload_text_model
        preload_text_model()
    except Exception as exc:
        logger.warning("Text seg model preload failed: %s", exc)


def _run_part_segmentation_sync(
    job_id: str,
    result_path: str,
    img_path: Optional[str],
    mask_path: Optional[str],
    n_colors: int,
    bake: bool = False,
    texture_size: int = 2048,
):
    """Post-processing: split the reconstructed mesh into functional parts.

    Loads the finished GLB and clusters its per-vertex colour into functional
    parts, writing a separate multi-object GLB. Runs in a worker thread; never
    touches the primary {job_id}.glb output. Parts keep the mesh's accurate
    per-vertex colour (fast, and identical in quality to the unsegmented
    download). ``bake`` optionally UV-bakes a texture atlas per part, but that
    rasterises on CPU (pytorch3d has no Metal/MPS backend) and is very slow for
    several parts, so it is off by default.
    """
    import trimesh

    from part_segmentation import segment_mesh_parts

    image = np.array(PILImage.open(img_path).convert("RGB")) if img_path else None
    mask  = np.array(PILImage.open(mask_path).convert("L")) if mask_path else None
    mesh  = trimesh.load(result_path, force="mesh")

    scene = segment_mesh_parts(
        mesh, image, mask,
        n_colors=n_colors,
        bake=bake,
        texture_size=texture_size,
    )
    out_path = str(RESULT_DIR / f"{job_id}_parts.glb")
    raw = scene.export(file_type="glb")
    # Inject a matte, non-metallic material (and, for vertex-coloured parts, wire
    # it to every primitive) so the parts render with true colour instead of the
    # shiny grey-metal default — matching the unsegmented download.
    from texture_baking import _patch_glb_metallic
    raw = _patch_glb_metallic(raw)
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path, len(scene.geometry)


def _apply_photo_vertex_color(mesh, image: np.ndarray, mask: Optional[np.ndarray]):
    """Re-project the original full-resolution photo onto a mesh as per-vertex colour.

    The reconstruction colours the mesh from the model's (Gaussian) appearance
    field, which is soft because the model conditions on a 518 px view. This
    re-samples the camera-facing front surface from the ORIGINAL full-resolution
    photo (sharper, more vibrant colour), while occluded / back / side faces keep
    the mesh's own reconstructed per-vertex colour so hidden regions stay
    plausible.

    The reconstructed mesh is dense (100k+ vertices), so the photo colour is
    written straight to per-vertex COLOR_0 rather than a UV texture atlas — this
    skips xatlas UV unwrapping (minutes on CPU for a high-poly mesh; no GPU/Metal
    path) and the CPU rasteriser, finishing in ~1 s. Geometry is unchanged.

    Returns a new vertex-coloured trimesh (same geometry); returns the input mesh
    unchanged if anything goes wrong.
    """
    from texture_baking import bake_mesh_texture

    # The mesh's existing per-vertex colour is the trustworthy base layer for
    # faces the photo can't see. Model colours live in the decoder's z-up frame,
    # so un-rotate the GLB (y-up) vertices with (x, -z, y) to match.
    model_vertices = None
    model_colors = None
    try:
        vcol = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
        if vcol.ndim == 2 and vcol.shape[0] == len(mesh.vertices) and vcol.shape[1] >= 3:
            model_colors = vcol[:, :3]
            v = np.asarray(mesh.vertices, dtype=np.float32)
            model_vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    except Exception:
        model_vertices = None
        model_colors = None

    return bake_mesh_texture(
        mesh, image, mask, texture_size=2048,
        model_vertices=model_vertices if model_colors is not None else None,
        model_colors=model_colors,
        model_colors_have_hue=model_colors is not None,
        as_vertex_colors=True,
    )


def smooth_mask(
    mask: np.ndarray,
    close_frac: float = 0.006,
    open_frac: float = 0.004,
    blur_frac: float = 0.004,
    keep_largest: bool = True,
) -> np.ndarray:
    """Clean and anti-alias a SAM mask before reconstruction.

    Fills pinholes, removes speckles, keeps only the largest blob, and rebuilds
    the jagged boundary as an anti-aliased *soft alpha* (signed-distance-field
    smoothstep) instead of a hard 0/255 edge. The low-memory pipeline reads the
    mask as a raw, non-binarized alpha channel, so that sub-pixel coverage
    survives into the geometry stage and cleans up the silhouette. ``blur_frac``
    controls the feather half-width. Returns a uint8 0..255 (soft) mask.
    """
    from sam_wrapper import refine_mask

    return refine_mask(
        mask,
        close_frac=close_frac,
        open_frac=open_frac,
        feather_frac=blur_frac,
        keep_largest=keep_largest,
        soft=True,
    )



# ─────────────────────────────────────────────────────────────────────────────
# Depth preview  – run MoGe once to show a greyscale depth map of the object
# while the (much slower) 3-D reconstruction runs.
# ─────────────────────────────────────────────────────────────────────────────

_depth_pipeline = None
_depth_lock = threading.Lock()
# True only while the MoGe weights are actually resident in RAM (they are loaded
# on demand and unloaded again right after each depth call).
_depth_weights_loaded = False


def _preimport_hydra_targets() -> None:
    """Pre-import modules that are instantiated by Hydra via string ``_target_``.

    Hydra resolves targets like ``sam3d_objects.pipeline.depth_models.moge.MoGe``
    with a ``getattr``-then-``import_module`` walk. In a PyInstaller onefile that
    walk can fail if the submodule has not yet been imported (it isn't an
    attribute of its parent package). Importing the modules here registers them
    on their parents so Hydra can always locate them. Best-effort and idempotent.
    """
    try:
        import sam3d_objects.pipeline.depth_models.moge  # noqa: F401
    except Exception as exc:  # pragma: no cover - only matters in frozen builds
        logger.debug("depth_models.moge pre-import skipped: %s", exc)
    try:
        import moge.model.v1  # noqa: F401
    except Exception as exc:  # pragma: no cover
        logger.debug("moge.model.v1 pre-import skipped: %s", exc)


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
            # In a frozen (PyInstaller) build Hydra resolves the depth model from
            # its dotted string target. Force-import the concrete modules first so
            # they are registered as attributes of their parent packages —
            # otherwise Hydra's getattr-then-import_module lookup can fail inside
            # the onefile bundle when the depth preview runs before anything else
            # has imported them. (No-op in a normal source checkout.)
            _preimport_hydra_targets()
            _depth_pipeline = InferencePipelineLowMemory(
                config_path=str(paths.PIPELINE_CONFIG_PATH),
                device="cpu",
                dtype="float16",
                cache_dir=str(paths.PIPELINE_CACHE_DIR),
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

        # One heavy model in memory at a time: drop the 2-D segmentation models
        # before MoGe runs (segmentation is finished once we have a mask here).
        try:
            from sam_wrapper import unload_segmentation_models
            unload_segmentation_models()
        except Exception:
            pass

        # Serialise MoGe access: it is not safe to run the same model from two
        # threads, and the reconstruction pipeline uses its own instance.
        global _depth_weights_loaded
        with _depth_lock:
            _depth_weights_loaded = True
            try:
                point_map = pipe.compute_pointmap(rgba)
                z = point_map["pointmap"][2].detach().cpu().numpy().astype(np.float32)
            finally:
                # Free the MoGe weights before the heavy reconstruction stages.
                pipe._unload_depth_model()
                _depth_weights_loaded = False

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
    distill: bool = False,
    ss_distill: bool = True,
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

    pipeline = None
    try:
        image = np.array(PILImage.open(img_path).convert("RGB"))
        mask  = np.array(PILImage.open(mask_path).convert("L"))

        job.update("Loading pipeline…", 5)

        from sam3d_objects.pipeline.inference_pipeline_low_memory import InferencePipelineLowMemory
        _preimport_hydra_targets()
        pipeline = InferencePipelineLowMemory(
            config_path=str(paths.PIPELINE_CONFIG_PATH),
            device="cpu",
            dtype="float16",
            cache_dir=str(paths.PIPELINE_CACHE_DIR),
        )

        job.update("Running 3-D reconstruction…", 12)

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
            decode_formats=["mesh"],
            simplify_ratio=0.0,
            vertex_color_source="gaussian",
            use_stage1_distillation=ss_distill,
            use_stage2_distillation=distill,
        )

        # The pipeline returns a per-vertex-coloured GLB: to_glb colours each
        # vertex from the model's Gaussian appearance field. These colours are
        # coherent across the whole surface (front, sides and hidden faces all
        # look plausible), so we keep them as-is. A previous full-resolution
        # photo re-projection made the front sharper but left an incoherent
        # photo/model blend on grazing / occluded faces (blotches), so the photo
        # overlay is intentionally disabled here.
        result_mesh = output["glb"]

        # Sand off the 64^3 voxel staircase on oblique silhouettes. The 2D mask is
        # full-res/soft; the geometry grid is not, so the step lives in the mesh.
        # Volume-preserving Taubin keeps thin parts (legs) while removing stepping.
        try:
            from mesh_utils import taubin_smooth

            result_mesh = taubin_smooth(result_mesh, iterations=10)
        except Exception as exc:
            logger.warning(f"[JOB {job_id}] mesh smoothing skipped: {exc}")

        result_path = str(RESULT_DIR / f"{job_id}.glb")
        job.update("Exporting GLB…", 95)
        # Keep the on-disk mesh vertex-coloured (no material): trimesh drops the
        # per-vertex COLOR_0 on reload when a material is present, and the colour
        # is what the colour-based part segmentation reads back. The matte
        # (non-metallic) material that removes the viewer "shine" is injected
        # only when the GLB is served for download (see /result).
        result_mesh.export(result_path, file_type="glb")

        job.complete(result_path)
        logger.info(f"[JOB {job_id}] Done → {result_path}")

    except Exception as exc:
        job.fail(str(exc))
        logger.error(f"[JOB {job_id}] Failed:\n{traceback.format_exc()}")
    finally:
        _mem_stop.set()
        job.sample_mem()          # one final reading so the graph ends at the peak
        root_logger.removeHandler(handler)
        # Unload the 3-D reconstruction model now the job is done so its weights
        # and working buffers are released back to the OS.
        pipeline = None
        import gc
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Startup  – pre-download SAM weights in background
# ─────────────────────────────────────────────────────────────────────────────

def _preload_sam():
    # Ensure the SAM 2 point checkpoint is on disk so the secondary point mode is
    # ready without a first-click download. The segmentation models themselves
    # are loaded on demand (text model on image upload, point model on first
    # point click) so idle memory stays low.
    try:
        from sam_wrapper import ensure_sam_weights
        ensure_sam_weights()
    except Exception as exc:
        logger.warning(f"SAM 2 weight preload failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Pass the app object directly (not the "server:app" import string) so this
    # works when frozen into a PyInstaller executable, where the "server" module
    # can't be re-imported by Uvicorn.
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        workers=1,       # must be 1 – ML models are not fork-safe
        reload=False,
    )
