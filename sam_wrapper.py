"""
sam_wrapper.py  –  SAM 2.1 (Segment Anything Model 2) wrapper.

Uses SAM2.1 Hiera-Large (~900 MB checkpoint, 224M params) for the highest
mask quality / cleanest silhouettes.
Downloads weights on first use.
Re-uses a single SAM2ImagePredictor instance across requests.
"""

from __future__ import annotations

import io
import os
import base64
import threading
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image as PILImage


def _sam_device() -> str:
    """Device SAM should run on.

    Normally MPS when available. Reconstruction is CPU-only (the launcher hides
    MPS from PyTorch so the 3-D stages fall back to CPU), so the launcher sets
    ``SAM3D_SAM_DEVICE`` to keep SAM segmentation on Metal.
    """
    forced = os.environ.get("SAM3D_SAM_DEVICE")
    if forced:
        return forced
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

# SAM 2 weights live under the shared app root (PYTHIA_HOME, default
# ~/Downloads/pythia) so a packaged build stores them alongside every other
# checkpoint. Also route the Hugging Face cache there for the SAM 3 text model.
import paths
paths.configure_hf_cache()
SAM2_CHECKPOINT_PATH = paths.SAM2_CHECKPOINT_PATH
_SAM2_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    print(f"\r[SAM2] Downloading… {pct}%", end="", flush=True)


def ensure_sam_weights() -> Path:
    """Download SAM2.1 Hiera-Large weights if they are not already present."""
    if SAM2_CHECKPOINT_PATH.exists():
        return SAM2_CHECKPOINT_PATH
    SAM2_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SAM2] Downloading SAM2.1 Hiera-L weights to {SAM2_CHECKPOINT_PATH} (~900 MB)…")
    urllib.request.urlretrieve(_SAM2_DOWNLOAD_URL, SAM2_CHECKPOINT_PATH, reporthook=_reporthook)
    print()
    print("[SAM2] Download complete.")
    return SAM2_CHECKPOINT_PATH


# ---------------------------------------------------------------------------
# 2-D segmentation model load status  (drives the Step-2 loading bar)
# ---------------------------------------------------------------------------
# Only one 2-D model is ever resident (SAM 2 points OR SAM 3 text); this record
# lets the UI show a "loading into memory" progress bar while a model is being
# brought in on demand. ``progress`` is None for an indeterminate bar (weight
# loading has no cheap byte-level callback), 1.0 once ready.

_seg_load_status = {"state": "idle", "model": None, "message": "", "progress": None}
_seg_load_lock = threading.Lock()


def _set_seg_load(model, state, message, progress=None) -> None:
    with _seg_load_lock:
        _seg_load_status.update(
            model=model, state=state, message=message, progress=progress
        )


def get_seg_load_status() -> dict:
    """Current 2-D segmentation model load state (thread-safe snapshot)."""
    with _seg_load_lock:
        return dict(_seg_load_status)


# ---------------------------------------------------------------------------
# Lazy singleton predictor
# ---------------------------------------------------------------------------

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # One 2-D model in memory at a time: drop the SAM 3 text model before
        # bringing the SAM 2 point model in (they are never used simultaneously).
        unload_text_model()
        _set_seg_load("point", "loading", "Loading SAM 2.1 point model…")
        try:
            ckpt = ensure_sam_weights()
            device = _sam_device()
            print(f"[SAM2] Loading SAM2.1 Hiera-L on {device}…")
            model = build_sam2(_SAM2_CONFIG, ckpt_path=str(ckpt), device=device)
            _predictor = SAM2ImagePredictor(model)
            print("[SAM2] Ready.")
            _set_seg_load("point", "ready", "SAM 2.1 point model ready", progress=1.0)
        except Exception as exc:
            _set_seg_load("point", "error", f"Failed to load point model: {exc}")
            raise
    return _predictor


def release_sam_memory() -> None:
    """Drop SAM's cached per-image embeddings before a memory-hungry reconstruction.

    ``set_image`` leaves SAM 2.1 Hiera-L's multi-scale feature pyramid pinned on
    the singleton predictor. On Apple unified memory that resident cache squeezes
    the low-memory 3D pipeline into memory compression/swap, slowing every stage.
    Clearing it (weights stay loaded for the next click) restores full headroom so
    the point-prompt path reconstructs as fast as the text path.
    """
    global _predictor
    try:
        if _predictor is not None:
            _predictor.reset_predictor()   # clears set_image() feature cache, keeps weights
    except Exception:
        pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _free_torch_memory() -> None:
    """Run a GC pass and release cached Metal (MPS) allocations.

    The launcher hides MPS from PyTorch so the 3-D stages fall back to the CPU,
    which makes ``torch.backends.mps.is_available()`` return ``False`` even
    though SAM's weights really do live on Metal. So we empty the MPS cache
    unconditionally (guarded) rather than gating on ``is_available()``.
    """
    import gc

    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def unload_point_model() -> None:
    """Fully unload the SAM 2 point-prompt predictor (weights + caches).

    Unlike :func:`release_sam_memory` (which only drops the per-image feature
    cache and keeps the weights resident), this drops the whole predictor so its
    weights are freed from unified memory.
    """
    global _predictor, _amg
    was_loaded = _predictor is not None
    _predictor = None
    _amg = None
    _free_torch_memory()
    if was_loaded and get_seg_load_status().get("model") == "point":
        _set_seg_load(None, "idle", "")


def unload_text_model() -> None:
    """Fully unload the SAM 3 text/concept model (model + processor)."""
    global _sam3_model, _sam3_processor
    was_loaded = _sam3_model is not None
    _sam3_model = None
    _sam3_processor = None
    _free_torch_memory()
    if was_loaded and get_seg_load_status().get("model") == "text":
        _set_seg_load(None, "idle", "")


def unload_segmentation_models() -> None:
    """Unload every 2-D segmentation model (SAM 2 points + SAM 3 text).

    Called right before the memory-hungry 3-D reconstruction starts so the two
    segmentation models do not compete with it for unified memory.
    """
    unload_point_model()
    unload_text_model()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid of SAM mask logits -> per-pixel probability."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -30.0, 30.0)))


def predict_mask(
    image: np.ndarray,
    positive_points: List[Dict[str, float]],
    negative_points: Optional[List[Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Predict a segmentation mask from point prompts.

    Parameters
    ----------
    image           : (H, W, 3) uint8 RGB image
    positive_points : list of {x, y} dicts – foreground prompts
    negative_points : list of {x, y} dicts – background prompts (optional)

    Returns
    -------
    (H, W) uint8 mask  (255 = foreground, 0 = background)
    """
    if not positive_points:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    predictor = _get_predictor()
    predictor.set_image(image)

    coords, labels = [], []
    for pt in positive_points:
        coords.append([float(pt["x"]), float(pt["y"])])
        labels.append(1)
    for pt in (negative_points or []):
        coords.append([float(pt["x"]), float(pt["y"])])
        labels.append(0)

    masks, scores, _ = predictor.predict(
        point_coords=np.array(coords, dtype=np.float32),
        point_labels=np.array(labels, dtype=np.int32),
        multimask_output=True,
        return_logits=True,          # keep per-pixel confidence, not just a hard mask
    )

    best = int(np.argmax(scores))
    # masks[best] are the raw mask logits at full resolution; sigmoid -> per-pixel
    # foreground probability (confidence).
    prob = _sigmoid(masks[best])

    # SAM's mask decoder is only ~256x256, so ``prob`` is a smooth confidence field
    # bilinearly upsampled to full resolution. Hard-thresholding it per pixel
    # (``prob >= 0.5``) is what turns that smooth ramp into a coarse ~4px staircase
    # on oblique edges. Instead we return SAM's confidence *directly* as an
    # anti-aliased 0..255 alpha: the visible 0.5 boundary is unchanged, but the
    # sub-pixel coverage along the silhouette is preserved. The low-memory pipeline
    # reads the mask as a raw (non-binarized) alpha channel, so this anti-aliasing
    # survives all the way into reconstruction. Downstream cleanup (``refine_mask``)
    # keeps this soft edge rather than re-thresholding it.
    return (np.clip(prob, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Automatic "everything" masks — part discovery for functional segmentation
# ---------------------------------------------------------------------------
#
# SAM 2's automatic mask generator samples a grid of point prompts over the
# whole image and returns every distinct region it finds. For a single object
# those regions are its *parts* (a car's wheels / windows / body, a chair's
# cushion / frame). We reuse the already-loaded SAM 2 weights (no extra model),
# then hand the masks to ``part_segmentation`` which lifts them onto the mesh.

_amg = None


def _get_amg(points_per_side: int = 32):
    """Lazily build a SAM2AutomaticMaskGenerator from the loaded predictor's model."""
    global _amg
    if _amg is None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        model = _get_predictor().model
        # Tuned for part discovery rather than whole-scene panoptic output:
        # a denser grid finds small parts; the area/quality filters drop noise.
        _amg = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            min_mask_region_area=200,
            output_mode="binary_mask",
        )
    return _amg


def generate_auto_masks(
    image: np.ndarray,
    points_per_side: int = 32,
) -> List[Dict]:
    """
    Automatically discover candidate part masks over an image.

    Parameters
    ----------
    image           : (H, W, 3) uint8 RGB image
    points_per_side : density of the sampling grid (more = finer parts)

    Returns
    -------
    list of dicts, each with keys ``segmentation`` ((H, W) bool), ``area`` (int),
    ``predicted_iou`` and ``stability_score`` (floats). Sorted largest-first.
    """
    amg = _get_amg(points_per_side)
    masks = amg.generate(image)
    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks


# ---------------------------------------------------------------------------
# SAM 3 – text / concept ("promptable concept segmentation") — optional path
# ---------------------------------------------------------------------------
#
# SAM 2 (above) needs point clicks. SAM 3 additionally accepts a short text
# phrase ("chair", "blue mug") and segments every matching instance. The web UI
# exposes this as an optional mode; for single-object reconstruction we collapse
# the returned instances to the single highest-confidence one. Weights come from
# the gated Hugging Face repo ``facebook/sam3`` (same Meta auth flow as SAM 3D)
# and are loaded lazily so this model is only pulled/held when text mode is used.

import os

_SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
_sam3_model = None
_sam3_processor = None
# Serialises SAM 3 loading so an on-upload preload and an explicit text segment
# arriving close together can't both start a (multi-GB) load at the same time.
_sam3_load_lock = threading.Lock()


def _sam3_cached() -> bool:
    """True if the (gated) SAM 3 repo already has a snapshot in the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        return any(r.repo_id == _SAM3_MODEL_ID for r in scan_cache_dir().repos)
    except Exception:
        return False


def _get_sam3():
    """Lazily load the SAM 3 model + processor (singleton)."""
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        return _sam3_model, _sam3_processor
    with _sam3_load_lock:
        # Re-check inside the lock: another thread may have finished loading while
        # we were waiting for it.
        if _sam3_model is not None:
            return _sam3_model, _sam3_processor
        from transformers import Sam3Model, Sam3Processor

        # One 2-D model in memory at a time: drop the SAM 2 point model before
        # bringing the SAM 3 text model in (they are never used simultaneously).
        unload_point_model()
        _set_seg_load("text", "loading", "Loading SAM 3 text model…")
        try:
            device = _sam_device()
            # facebook/sam3 is a gated repo. When the weights are already cached we
            # load them with local_files_only=True so transformers does NOT reach out
            # to the Hub for optional side files (e.g. chat_template.json) — that call
            # 401s for a gated repo without a token even though the model itself is
            # present locally. If a token is available we pass it and allow network.
            token = (os.environ.get("HF_TOKEN")
                     or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                     or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
            local_only = _sam3_cached() and not token
            kw = {"local_files_only": local_only}
            if token:
                kw["token"] = token

            print(f"[SAM3] Loading {_SAM3_MODEL_ID} on {device}… "
                  f"({'offline (cached)' if local_only else 'first run downloads gated weights'})")
            _sam3_processor = Sam3Processor.from_pretrained(_SAM3_MODEL_ID, **kw)
            _sam3_model = Sam3Model.from_pretrained(_SAM3_MODEL_ID, **kw).to(device).eval()
            print("[SAM3] Ready.")
            _set_seg_load("text", "ready", "SAM 3 text model ready", progress=1.0)
        except Exception as exc:
            _set_seg_load("text", "error", f"Failed to load text model: {exc}")
            raise
    return _sam3_model, _sam3_processor


def note_text_preload_starting() -> bool:
    """Decide whether an on-upload SAM 3 preload should run, and flag the bar.

    Returns True only when the weights are downloaded but not yet resident — i.e.
    there is something to load and it can be loaded offline (no blocking gated
    download). In that case the segmentation-load bar is switched to ``loading``
    *synchronously* here so the UI shows it immediately, before the worker thread
    (which then holds the GIL while building the model) can starve the event loop.

    Returns False when the model is already in memory or not downloaded, so the
    caller can skip dispatching a pointless / blocking load.
    """
    if _sam3_model is not None:
        return False
    if not _sam3_cached():
        return False
    _set_seg_load("text", "loading", "Loading SAM 3 text model…")
    return True


def preload_text_model() -> None:
    """Eagerly load the SAM 3 text/concept model — but only if it is downloaded.

    Text segmentation is the default UI mode, so loading its weights up front
    makes the first text prompt instant. This is fired on image upload, so it
    must NOT block: for an un-cached gated repo, ``from_pretrained`` would try a
    multi-GB download (which also needs a token) on a worker thread, holding the
    GIL and starving the event loop so the whole server stalls. Downloading is
    done explicitly from the Models tab (real progress + token handling), so we
    only preload when the weights are already on disk.
    """
    if not _sam3_cached():
        # Nothing resident and nothing to load without a network download — leave
        # the segmentation-load bar idle. The Models tab shows the Download step.
        if _sam3_model is None and get_seg_load_status().get("model") != "point":
            _set_seg_load(None, "idle", "")
        return
    _get_sam3()


def predict_mask_text(
    image: np.ndarray,
    text: str,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """
    Predict a segmentation mask from a text (concept) prompt using SAM 3.

    Parameters
    ----------
    image           : (H, W, 3) uint8 RGB image
    text            : concept phrase, e.g. "chair" or "blue mug"
    score_threshold : minimum instance confidence to keep

    Returns
    -------
    (H, W) uint8 mask (255 = foreground, 0 = background). Empty if the phrase is
    blank or SAM 3 finds no matching instance. The mask is returned hard-edged;
    the downstream ``smooth_mask``/``refine_mask`` pass anti-aliases it to the
    same soft-alpha contract as the point-prompt path.
    """
    text = (text or "").strip()
    H, W = image.shape[:2]
    if not text:
        return np.zeros((H, W), dtype=np.uint8)

    model, processor = _get_sam3()
    device = next(model.parameters()).device

    pil = PILImage.fromarray(image)
    inputs = processor(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=0.5,
        target_sizes=[(H, W)],
    )[0]

    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Text prompts return every matching instance; reconstruction wants a single
    # object, so keep the highest-confidence one.
    best = int(torch.as_tensor(scores).argmax())
    m = np.asarray(masks[best].detach().to("cpu").numpy())
    return (m > 0.5).astype(np.uint8) * 255


def _refine_mask(prob: np.ndarray, image: Optional[np.ndarray] = None) -> np.ndarray:
    """Clean a SAM mask into a solid, smooth-edged silhouette using SAM's own
    per-pixel confidence.

    Instead of thresholding SAM's hard boolean mask (whose boundary is a noisy
    pixel staircase), we work with the model's foreground *probability* field
    ``prob = sigmoid(mask_logits)``. The object boundary is taken as the 0.5
    confidence isocontour of that smooth field, which follows the object far more
    cleanly than a per-pixel argmax and naturally suppresses the low-confidence
    fringe pixels along the silhouette.

    The SAM-3D model is trained with heavy boundary *dilation* augmentation
    (``perturb_mask_boundary``: ``p_dilate=0.8`` vs ``p_erode=0.1``), so it wants
    a mask that fully covers the object. We therefore keep the confident interior
    intact and only do lossless cleanup — largest connected component, hole fill,
    and confidence-field smoothing — without eroding or morphological opening
    (which would nibble thin parts such as chair legs).

    When the source ``image`` is provided it is currently unused: the SAM mask
    decoder is only ~256px, so its 0.5 isocontour is a coarse staircase once
    upsampled to full resolution. We remove that staircase with *morphological
    anti-aliasing* — blurring the confidence field and re-taking the 0.5
    isocontour — which yields smooth, curved silhouettes. This runs in mask
    space only, so it never latches onto busy background texture the way an
    image-guided edge filter does (which caused the ragged, wobbly edges).

    Parameters
    ----------
    prob  : (H, W) float array – SAM foreground probability in [0, 1]
            (a boolean/uint8 mask is also accepted and treated as 0/1).
    image : (H, W, 3) uint8 RGB source image used as the guided-filter guide.
            Optional; if omitted, only confidence-field smoothing is applied.

    Returns
    -------
    (H, W) uint8 mask (255 = foreground, 0 = background).
    """
    prob = np.asarray(prob, dtype=np.float32)
    if prob.size and prob.max() > 1.0:      # tolerate 0..255 input
        prob = prob / 255.0

    h, w = prob.shape
    core = (prob >= 0.5).astype(np.uint8)
    if core.sum() == 0:
        return np.zeros((h, w), np.uint8)

    try:
        import cv2
    except Exception:
        return (core * 255)

    # 1. Largest connected component only (drops low-confidence stray islands).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        core = (labels == largest).astype(np.uint8)

    # 2. Fill interior holes (flood-fill the background, invert).
    ff = core.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 1)
    filled = (core | (1 - ff)).astype(np.uint8)

    # 3. Build a clean confidence field limited to this object: zero out any
    #    confidence outside the kept component, and force filled holes to full
    #    confidence so they stay solid.
    conf = prob * filled.astype(np.float32)
    conf[(filled == 1) & (core == 0)] = 1.0

    # 4. Smooth the silhouette with morphological anti-aliasing. Blur the
    #    confidence field and re-take the 0.5 isocontour: this turns the coarse
    #    ~256px SAM decoder staircase into smooth curves. Because straight edges
    #    are preserved by a symmetric blur + 0.5 threshold, thin parts such as
    #    chair legs keep their width while corners/steps are rounded off. Doing
    #    this in mask space (not image-guided) avoids snapping to floor cracks
    #    and other background texture.
    sigma = max(2.0, min(h, w) / 150.0)
    conf = cv2.GaussianBlur(conf, (0, 0), sigmaX=sigma)
    m = (conf >= 0.5).astype(np.uint8)

    # 5. Seal hairline notches, then re-extract a single solid silhouette
    #    (largest component + hole fill). A small elliptical close rounds
    #    concave nicks without nibbling thin parts.
    k = max(3, int(min(h, w) / 200))
    k += 1 - (k & 1)  # force odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n2 > 2:
        largest2 = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        m = (lab2 == largest2).astype(np.uint8)
    ff2 = m.copy()
    ff2_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff2, ff2_mask, (0, 0), 1)
    m = (m | (1 - ff2)).astype(np.uint8)

    # 6. Contour polish for a demo-clean silhouette: replace the boundary with a
    #    Gaussian-smoothed version of itself (periodic smoothing of the contour
    #    points), which removes the last residual stair-steps and gives the
    #    smooth curved edges the SAM demo produces. A modest sigma rounds the
    #    staircase without collapsing thin parts such as chair legs.
    m = _smooth_contour(m, sigma=max(1.5, min(h, w) / 350.0))

    return (m * 255)


def _smooth_contour(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a binary mask's outline with periodic Gaussian filtering.

    Each external contour is treated as a closed curve; its x/y coordinates are
    low-pass filtered (wrap-around) and the result is re-filled. ``sigma`` is in
    contour-point (~pixel) units.
    """
    import cv2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask

    rad = int(max(2, round(sigma * 3)))
    kernel = cv2.getGaussianKernel(2 * rad + 1, sigma).ravel()
    out = np.zeros_like(mask)
    for cnt in cnts:
        pts = cnt[:, 0, :].astype(np.float32)
        n = len(pts)
        if n < max(12, 2 * rad + 1):        # too small to smooth safely
            cv2.drawContours(out, [cnt], -1, 1, thickness=cv2.FILLED)
            continue
        xs = np.pad(pts[:, 0], (rad, rad), mode="wrap")
        ys = np.pad(pts[:, 1], (rad, rad), mode="wrap")
        sx = np.convolve(xs, kernel, mode="same")[rad:-rad]
        sy = np.convolve(ys, kernel, mode="same")[rad:-rad]
        smooth = np.stack([sx, sy], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(out, [smooth], -1, 1, thickness=cv2.FILLED)
    return out



def mask_to_base64_png(mask: np.ndarray) -> str:
    """Encode a (H, W) uint8 mask as a base64 PNG string."""
    buf = io.BytesIO()
    PILImage.fromarray(mask).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def base64_png_to_mask(b64: str) -> np.ndarray:
    """Decode a base64 PNG string to a (H, W) uint8 numpy mask."""
    data = base64.b64decode(b64)
    return np.array(PILImage.open(io.BytesIO(data)).convert("L"))


def refine_mask(
    mask: np.ndarray,
    close_frac: float = 0.006,
    open_frac: float = 0.004,
    feather_frac: float = 0.004,
    keep_largest: bool = True,
    soft: bool = True,
) -> np.ndarray:
    """Clean and (optionally) anti-alias a SAM mask before reconstruction.

    Topology cleanup (fill pinholes, drop speckles/islands, fill interior holes)
    runs on a binary working copy — those operations genuinely need a hard mask.
    The returned *edge* is anti-aliased so the ~256px SAM decoder staircase does
    not survive as a hard per-pixel jag on oblique edges:

    * If the input already carries a soft (anti-aliased) alpha — e.g. SAM's own
      sub-pixel confidence field from ``predict_mask`` — that ramp is
      *preserved*: the cleaned silhouette selects which region to keep, but the
      boundary values come straight from SAM. This is the important case: the
      anti-aliasing is produced at mask-generation time and must not be thrown
      away here.
    * If the input is genuinely hard-edged (e.g. a binary file mask from the
      CLI), the boundary is rebuilt as fractional coverage via a signed-distance
      field (the SDF-text-rendering trick): coverage = smoothstep across a
      +/- feather-pixel ramp centred on the silhouette.

    Either way the low-memory pipeline reads the mask as a *raw, non-binarized*
    alpha channel, so this soft coverage survives into the geometry stage.

    Kernel and feather sizes are a fraction of the image's shorter side, so the
    amount of smoothing is resolution-independent. Returns a uint8 0..255 mask
    (soft/anti-aliased by default; pass ``soft=False`` for the legacy hard
    binary output).
    """
    import cv2

    src = np.asarray(mask).astype(np.float32)
    if src.ndim == 3:
        src = src[..., -1]
    if src.size and src.max() <= 1.0:  # tolerate 0..1 input
        src = src * 255.0
    binary = (src > 127).astype(np.uint8) * 255

    short = max(1, min(binary.shape[:2]))

    def _odd(frac: float, lo: int = 3) -> int:
        k = int(round(short * frac))
        return max(lo, k) | 1

    # 1) close then open: fill pinholes, then shave speckles.
    if close_frac > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_frac),) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if open_frac > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_frac),) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    # 2) keep only the largest connected component (drop stray islands).
    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if n > 2:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    # 2b) fill interior holes so the silhouette is solid (flood-fill background).
    #     Pad a 1px background border first so the flood seed at (0,0) is always
    #     background even when the object touches the frame corner (otherwise the
    #     flood leaks and the whole frame is marked solid).
    fg = (binary > 0).astype(np.uint8)
    bordered = np.pad(fg, 1, mode="constant", constant_values=0)
    ff = bordered.copy()
    ff_mask = np.zeros((bordered.shape[0] + 2, bordered.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (0, 0), 1)  # fill exterior background
    holes = (ff == 0).astype(np.uint8)     # background unreachable from border = holes
    region = ((bordered | holes)[1:-1, 1:-1]).astype(np.uint8)  # solid silhouette

    if not region.any():
        # Never hand back an empty mask if the original had something in it.
        return (src > 127).astype(np.uint8) * 255

    if not soft:
        return region * 255

    feather = max(1.5, short * feather_frac)

    # Does the *input* already carry anti-aliased (fractional) coverage? SAM's
    # confidence field does; a hard binary file mask does not.
    frac_px = int(((src > 8) & (src < 247)).sum())
    input_is_soft = frac_px > max(64, int(0.0005 * src.size))

    if input_is_soft:
        # 3a) Preserve SAM's own sub-pixel ramp. Keep the soft values in a band
        #     around the cleaned silhouette (dilate the solid region by the
        #     feather so the outer <0.5 side of the ramp survives), zero distant
        #     halos / dropped islands, and force filled holes to solid.
        rad = max(1, int(round(feather)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        keep = cv2.dilate(region, k)
        out = src * keep
        out[(region == 1) & (src < 127)] = 255.0  # solidify filled holes
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    # 3b) Hard input: synthesise anti-aliasing with a signed distance field.
    #     distanceTransform gives the distance to the nearest opposite pixel;
    #     inside minus outside is a signed field (>0 inside, <0 outside, 0 on the
    #     silhouette). A smoothstep over +/- ``feather`` pixels yields fractional
    #     coverage — no re-thresholding, so the anti-aliasing is preserved.
    dist_in = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - region, cv2.DIST_L2, 3)
    sdf = dist_in - dist_out
    coverage = np.clip(0.5 + sdf / (2.0 * feather), 0.0, 1.0)
    return (coverage * 255.0 + 0.5).astype(np.uint8)
