"""
mesh_utils.py – lightweight, geometry-only mesh helpers (trimesh + numpy).

Kept dependency-light on purpose (no torch / pytorch3d) so both the CLI
(``main.py``) and the web server (``server.py``) can import it cheaply.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh


def taubin_smooth(
    mesh: "trimesh.Trimesh",
    iterations: int = 10,
    lamb: float = 0.5,
    nu: float = 0.53,
) -> "trimesh.Trimesh":
    """Volume-preserving Taubin (λ/μ) smoothing of a mesh's *geometry*.

    The SAM-3D geometry is extracted with FlexiCubes on a 64³ sparse-voxel
    grid, so oblique silhouettes come out as a ~1-voxel staircase regardless of
    how clean the 2D mask is (the mask is full-resolution; the grid is not).
    Taubin smoothing is a low-pass filter that alternates a positive-weight
    Laplacian pass (``lamb``, shrinks) with a negative-weight pass (``nu``,
    re-inflates), so it sands off that voxel stair-step *without* the volume
    loss / shrinkage of plain Laplacian smoothing — thin parts such as chair
    legs keep their width while the stepping dissolves.

    Only vertex positions change; topology and per-vertex colours (``COLOR_0``)
    are untouched, so appearance is preserved. Best-effort: any failure returns
    the mesh unchanged rather than breaking export.

    Args:
        mesh: a ``trimesh.Trimesh`` (per-vertex coloured or plain).
        iterations: number of λ/μ pass pairs. 0 disables. ~8–15 removes voxel
            stepping; higher over-rounds sharp corners.
        lamb: positive Laplacian weight (shrink pass).
        nu: negative Laplacian weight (inflate pass); should be slightly larger
            than ``lamb`` for volume preservation.

    Returns:
        The smoothed mesh (same object; smoothing is applied in place).
    """
    if mesh is None or iterations <= 0:
        return mesh
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return mesh

    try:
        import trimesh.smoothing as _smoothing

        # filter_taubin needs a watertight-ish, indexed mesh; keep colours by
        # operating in place (vertex indices are unchanged).
        _smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=int(iterations))
    except Exception:
        # Never let cosmetic smoothing break a reconstruction/export.
        return mesh

    return mesh
