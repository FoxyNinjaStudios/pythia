"""Central filesystem layout for the Pythia app.

Everything the app downloads or writes lives under a single root so a packaged
(PyInstaller) build launched from any working directory still finds — and stores
— its data in one predictable place. The root defaults to ``~/Downloads/pythia``
and can be overridden with the ``PYTHIA_HOME`` environment variable.

Layout::

    <PYTHIA_HOME>/
        checkpoints/
            hf/                     SAM 3D weights + pipeline YAML configs
            sam2.1_hiera_large.pt   SAM 2.1 point-prompt weights
        cache/
            huggingface/            Hugging Face hub cache (SAM 3, MoGe)
            pipeline/               InferencePipeline intermediate (SLAT) cache
        images/                     input images
        outputs/
            uploads/                uploaded source images
            results/                exported GLB results
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_home() -> Path:
    """Return the app data root (``PYTHIA_HOME`` env var or ~/Downloads/pythia)."""
    env = os.environ.get("PYTHIA_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Downloads" / "pythia").resolve()


PYTHIA_HOME = _resolve_home()

# ── Checkpoints / weights ──────────────────────────────────────────────────────
CHECKPOINTS_DIR = PYTHIA_HOME / "checkpoints"
HF_WEIGHTS_DIR = CHECKPOINTS_DIR / "hf"                    # SAM 3D weights + configs
SAM2_CHECKPOINT_PATH = CHECKPOINTS_DIR / "sam2.1_hiera_large.pt"
PIPELINE_CONFIG_PATH = HF_WEIGHTS_DIR / "pipeline.yaml"

# ── Caches ─────────────────────────────────────────────────────────────────────
CACHE_DIR = PYTHIA_HOME / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"                  # HF hub downloads (SAM 3, MoGe)
HF_HUB_DIR = HF_CACHE_DIR / "hub"
PIPELINE_CACHE_DIR = CACHE_DIR / "pipeline"               # SLAT intermediate cache

# ── I/O ────────────────────────────────────────────────────────────────────────
IMAGES_DIR = PYTHIA_HOME / "images"
OUTPUTS_DIR = PYTHIA_HOME / "outputs"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"
RESULT_DIR = OUTPUTS_DIR / "results"


def configure_hf_cache() -> None:
    """Point the Hugging Face hub cache under ``PYTHIA_HOME`` and cache aggressively.

    Uses ``setdefault`` so an explicit value set by the user still wins. Must run
    before ``huggingface_hub`` / ``transformers`` are imported so their
    module-level cache/offline constants pick up these values.

    Once a model's weights are on disk we never want to touch the network again —
    no ETag HEAD checks, no re-validation, no re-downloads. So the app defaults to
    Hugging Face *offline* mode: every model load is served straight from the
    local cache. The only time we need the network is an explicit download from
    the Models tab, which wraps its ``snapshot_download`` in :func:`hf_online`.
    """
    HF_HUB_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_DIR))
    # Serve cached weights without ever revalidating them against the Hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # No telemetry pings on import/load.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


from contextlib import contextmanager


@contextmanager
def hf_online():
    """Temporarily allow Hugging Face Hub network access inside this block.

    The app runs in offline mode by default (see :func:`configure_hf_cache`) so
    cached weights are never revalidated. Explicit downloads must opt back into
    the network for their duration. ``huggingface_hub`` snapshots the offline
    flag into a module constant at import time, so we patch that constant too —
    flipping the env var alone would not take effect for an already-imported
    module.
    """
    prev_env = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    patched = []
    for mod_name in ("huggingface_hub.constants", "transformers.utils.hub"):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "HF_HUB_OFFLINE"):
                patched.append((mod, "HF_HUB_OFFLINE", mod.HF_HUB_OFFLINE))
                mod.HF_HUB_OFFLINE = False
        except Exception:
            pass
    try:
        yield
    finally:
        for mod, attr, old in patched:
            try:
                setattr(mod, attr, old)
            except Exception:
                pass
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v



def ensure_dirs() -> None:
    """Create the writable directories the app needs at runtime."""
    for d in (HF_WEIGHTS_DIR, HF_HUB_DIR, PIPELINE_CACHE_DIR,
              IMAGES_DIR, UPLOAD_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)
