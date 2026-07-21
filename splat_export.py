"""
splat_export.py  –  optional Gaussian-splat (.ply) export module.

The reconstruction pipeline decodes a Gaussian appearance representation
alongside the mesh (it is used to colour the mesh vertices). This module lets
the server *also* save that Gaussian as a real 3D Gaussian-Splatting ``.ply``
— a soft point representation that avoids the flexicubes mesh artifacts
(degenerate faces, holey grazing sides) and is nicer to preview for
depth-ambiguous single-view inputs.

It is intentionally a standalone, easy-to-disable module:

    * Turn it OFF with the environment variable ``SAM3D_SPLAT=0`` (also accepts
      ``false``/``no``/``off``). Default is ON.
    * When OFF, ``enabled()`` returns ``False`` and the server keeps its previous
      behaviour (exporting the mesh as a ``.ply`` instead of the splat).
    * ``export_splat()`` never raises — on any failure it logs and returns
      ``False`` so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("sam3d.splat")

_OFF_VALUES = {"0", "false", "no", "off", ""}


def enabled() -> bool:
    """Whether real Gaussian-splat export is turned on (env ``SAM3D_SPLAT``)."""
    return os.environ.get("SAM3D_SPLAT", "1").strip().lower() not in _OFF_VALUES


def extract_gaussian(output: dict) -> Optional[Any]:
    """Pull the decoded Gaussian object out of a pipeline ``run()`` result.

    Returns the ``Gaussian`` (with ``.save_ply``) or ``None`` if the pipeline
    did not produce one.
    """
    gs = output.get("gs")
    if gs is not None:
        return gs
    gaussian = output.get("gaussian")
    if isinstance(gaussian, (list, tuple)) and len(gaussian) > 0:
        return gaussian[0]
    return gaussian


def export_splat(output: dict, path: str) -> bool:
    """Export the pipeline's Gaussian representation as a 3DGS ``.ply``.

    Parameters
    ----------
    output : dict returned by ``InferencePipelineLowMemory.run`` (expects a
             ``"gs"`` / ``"gaussian"`` entry).
    path   : destination ``.ply`` file path.

    Returns
    -------
    bool  – ``True`` if a splat file was written, ``False`` otherwise (disabled,
            no Gaussian available, or an error occurred).
    """
    if not enabled():
        return False

    gs = extract_gaussian(output)
    if gs is None:
        logger.warning("[SPLAT] No Gaussian in pipeline output; skipping splat export.")
        return False

    if not hasattr(gs, "save_ply"):
        logger.warning("[SPLAT] Gaussian object has no save_ply(); skipping.")
        return False

    try:
        gs.save_ply(path)
        n = int(gs.get_xyz.shape[0]) if hasattr(gs, "get_xyz") else -1
        logger.info(f"[SPLAT] Wrote Gaussian splat ({n} gaussians) → {path}")
        return True
    except Exception as exc:  # never break the run over an optional artifact
        logger.warning(f"[SPLAT] Splat export failed: {exc}")
        return False
