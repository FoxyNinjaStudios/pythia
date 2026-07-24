"""
sam_wrapper.py  –  SAM 2.1 (Segment Anything Model 2) wrapper.

Uses SAM2.1 Hiera-Large (~900 MB checkpoint, 224M params) for the highest
mask quality / cleanest silhouettes.
Downloads weights on first use.
Re-uses a single SAM2ImagePredictor instance across requests.
"""

from __future__ import annotations

import io
import base64
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

SAM2_CHECKPOINT_PATH = Path("checkpoints/sam2.1_hiera_large.pt")
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
# Lazy singleton predictor
# ---------------------------------------------------------------------------

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = ensure_sam_weights()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[SAM2] Loading SAM2.1 Hiera-L on {device}…")
        model = build_sam2(_SAM2_CONFIG, ckpt_path=str(ckpt), device=device)
        _predictor = SAM2ImagePredictor(model)
        print("[SAM2] Ready.")
    return _predictor


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
