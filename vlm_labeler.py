"""
vlm_labeler.py  –  SmolVLM2 visual namer for automatic part labels.

The part splitter (`part_segmentation.py`) discovers parts by colour clustering,
but the *names* used to come from a hardcoded per-class vocabulary
(``chair → legs/seat/pillow``). That is not "a label from the image": an object
with no entry got no names, and the candidate words were fixed in advance.

This module removes the fixed vocabulary. Each discovered part is rendered with
the part highlighted, and **SmolVLM2-2.2B-Instruct** — a small open (Apache-2.0)
vision-language model — is asked, in open vocabulary, what that highlighted part
is called. The label therefore comes from the rendered image, not a lookup table.

Everything is lazy and defensive: the ~4.5 GB model is only pulled/held when
labeling is requested, runs on Metal (MPS) with a CPU fallback, and any failure
(weights absent, load/inference error) returns ``None`` so the caller falls back
to the geometric / colour heuristic. Weights are the public repo
``HuggingFaceTB/SmolVLM2-2.2B-Instruct`` (no gating, downloaded at runtime, not
redistributed).

History: this used Moondream2 (``vikhyatk/moondream2``), but that pinned build
is broken on current PyTorch/transformers — its vision encoder returns an empty
kv_cache, so the LM hallucinates from text alone. SmolVLM2 is a maintained,
first-party ``transformers`` model (no remote code) and works correctly on MPS.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# Public Apache-2.0 model, native transformers support (no trust_remote_code).
_MODEL_ID = os.environ.get("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

_model = None
_processor = None
# Serialises the (multi-GB) load so two concurrent label requests can't both
# start building the model at once.
_load_lock = threading.Lock()


def _device() -> str:
    """Prefer Metal (MPS) on Apple Silicon, else CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def available() -> bool:
    """True when the SmolVLM2 weights are already in the local HF cache.

    Used so labeling is skipped (rather than triggering a blocking multi-GB
    download) when the model has not been fetched yet."""
    try:
        from huggingface_hub import scan_cache_dir

        return any(r.repo_id == _MODEL_ID for r in scan_cache_dir().repos)
    except Exception:
        return False


def _get_model():
    """Lazily load SmolVLM2 (singleton). Loads on MPS, falls back to CPU."""
    global _model, _processor
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dev = _device()
        print(f"[VLM] Loading {_MODEL_ID} on {dev}…")
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            _MODEL_ID,
            dtype=torch.float32,
        )
        model = model.to(dev).eval()
        _model = model
        print("[VLM] Ready.")
        return _model


def unload() -> None:
    """Drop the in-memory SmolVLM2 model and free Metal (MPS) memory.

    The VLM is only used to name parts at export time; keeping its ~4.5 GB
    resident afterwards would compete with the next reconstruction's peak on
    unified memory. Callers unload it right after naming (mirroring how MoGe is
    released after each depth call) so it is never held during reconstruction."""
    global _model, _processor
    if _model is None:
        return
    _model = None
    _processor = None
    try:
        import gc

        gc.collect()
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def name_image(image: np.ndarray, question: str) -> Optional[str]:
    """Ask SmolVLM2 ``question`` about ``image`` and return its short answer.

    Parameters
    ----------
    image    : (H, W, 3) uint8 RGB image (a render with the part highlighted).
    question : the open-vocabulary prompt, e.g. "What part … is highlighted?"

    Returns the model's raw answer string, or ``None`` on any failure. Decoding
    is greedy (``do_sample=False``) and capped at a few tokens so the answer is
    a short, deterministic noun phrase.
    """
    from PIL import Image

    model = _get_model()
    if model is None or _processor is None:
        return None
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    dev = _device()

    try:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=prompt, images=[pil], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        ans = _processor.batch_decode(new_tokens, skip_special_tokens=True)
        text = ans[0].strip() if ans else ""
        return text or None
    except Exception:
        return None
