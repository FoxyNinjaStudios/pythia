"""SAM-based mesh part segmentation (render → segment → back-project).

Colour clustering (:mod:`part_segmentation`) is the reliable fallback, but on a
mesh whose colour is *continuous* (e.g. a near-grey sofa, a photographic
gradient) it has no clean signal to cut on and the part boundaries come out
jagged — the pillow/cushion seam is a good example.

This module segments the mesh **semantically** instead of by colour:

1. Decimate the mesh (open3d) so it can be rasterised quickly on CPU.
2. Render the object un-lit from several fixed views (front / back / sides /
   top / bottom) using PyTorch3D, so SAM sees the true surface colour.
3. Run SAM 2's automatic ("everything") mask generator on each render to get
   clean, semantic 2-D part masks.
4. Back-project every mask onto the *full* mesh: project each vertex into the
   view and keep it if a depth test says it is the front-most surface there.
5. Fuse the per-view masks into a single per-vertex part label (a greedy
   front-view-seeded accumulation), turn that into per-face labels, clean it up
   with the same face-graph smoothing used by the colour path, and split.

The public entry point :func:`split_mesh_by_sam` is a drop-in replacement for
``part_segmentation.split_mesh_by_color`` — it returns ``(submeshes, colors)``.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from part_segmentation import (
    _dominant_rgb,
    _merge_small_components,
    _smooth_face_labels,
)

# Default multi-view rig: four side views around the equator plus top and
# bottom. (elev, azim) in degrees. Covers almost every visible surface.
_DEFAULT_VIEWS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 90.0),
    (0.0, 180.0),
    (0.0, 270.0),
    (89.0, 0.0),
    (-89.0, 0.0),
)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _normalise(verts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre ``verts`` on their bbox centre and scale to a 0.9-radius sphere so
    a unit orthographic camera frames the whole object with a small margin.
    Returns ``(verts_norm, centre, radius)``."""
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    centre = (lo + hi) * 0.5
    v = verts - centre
    radius = float(np.linalg.norm(v, axis=1).max()) or 1.0
    return (v / radius * 0.9).astype(np.float32), centre, radius


def _decimate(
    mesh: trimesh.Trimesh, target_faces: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate ``mesh`` to roughly ``target_faces`` triangles for fast
    rasterisation, carrying per-vertex colour along. Returns
    ``(verts, faces, colors01)`` where ``colors01`` is float RGB in [0, 1].
    Falls back to the original mesh when it is already small or open3d is
    unavailable."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    try:
        colors01 = (
            np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        )
    except Exception:
        colors01 = np.full((len(verts), 3), 0.75)

    if len(faces) <= target_faces:
        return verts.astype(np.float32), faces, colors01.astype(np.float32)

    try:
        import open3d as o3d

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces),
        )
        m.vertex_colors = o3d.utility.Vector3dVector(colors01)
        m = m.simplify_quadric_decimation(int(target_faces))
        dv = np.asarray(m.vertices, dtype=np.float32)
        df = np.asarray(m.triangles, dtype=np.int64)
        dc = (
            np.asarray(m.vertex_colors, dtype=np.float32)
            if len(m.vertex_colors) == len(dv)
            else np.full((len(dv), 3), 0.75, dtype=np.float32)
        )
        if len(df) == 0:
            raise RuntimeError("decimation produced no faces")
        return dv, df, dc
    except Exception as exc:  # pragma: no cover - depends on optional open3d
        print(f"[SAM-PARTS] Decimation unavailable ({exc}); rendering full mesh.")
        return verts.astype(np.float32), faces, colors01.astype(np.float32)


# ---------------------------------------------------------------------------
# rendering + back-projection
# ---------------------------------------------------------------------------
def _render_and_backproject(
    dec_verts_n: np.ndarray,
    dec_faces: np.ndarray,
    dec_colors: np.ndarray,
    full_verts_n: np.ndarray,
    views: Tuple[Tuple[float, float], ...],
    image_size: int,
    points_per_side: int,
    verbose: bool,
) -> List[Tuple[np.ndarray, int]]:
    """Render each view, run SAM, and back-project the masks onto the full mesh.

    Returns a list (one entry per view) of ``(vert_mask, n_masks)`` where
    ``vert_mask`` is an ``(N,)`` int array giving, for each full-mesh vertex, the
    **local** mask index it falls in for that view (``-1`` if not visible / not
    covered). Local indices are ``0..n_masks-1``."""
    import torch
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        MeshRasterizer,
        RasterizationSettings,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    import sam_wrapper

    dv = torch.from_numpy(dec_verts_n)
    df = torch.from_numpy(dec_faces)
    dc = torch.from_numpy(dec_colors)
    fv = torch.from_numpy(full_verts_n)
    meshes = Meshes(verts=[dv], faces=[df])
    raster = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    H = W = image_size

    out: List[Tuple[np.ndarray, int]] = []
    for vi, (elev, azim) in enumerate(views):
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
        cam = FoVOrthographicCameras(R=R, T=T)
        frag = MeshRasterizer(cameras=cam, raster_settings=raster)(meshes)
        ptf = frag.pix_to_face[0, ..., 0]          # (H, W) face idx, -1 bg
        bary = frag.bary_coords[0, ..., 0, :]      # (H, W, 3)
        zbuf = frag.zbuf[0, ..., 0].numpy()        # (H, W) view-space depth

        # Un-lit RGB: barycentric-interpolate the decimated vertex colours so
        # SAM sees the true surface colour (no shading gradients to over-cut).
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)
        valid = ptf >= 0
        if valid.any():
            fidx = ptf[valid]                      # (n,)
            vcols = dc[df[fidx]]                    # (n, 3, 3)
            b = bary[valid].unsqueeze(-1)          # (n, 3, 1)
            col = (b * vcols).sum(dim=1)           # (n, 3)
            rgb[valid.numpy()] = (
                (col.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            )

        # SAM everything-masks on the render.
        try:
            masks = sam_wrapper.generate_auto_masks(rgb, points_per_side=points_per_side)
        except Exception as exc:
            print(f"[SAM-PARTS] view {vi}: SAM failed ({exc}); skipping view.")
            out.append((np.full(len(full_verts_n), -1, np.int32), 0))
            continue

        # Rasterise masks into one label image. Assign largest first so smaller
        # (finer) masks overwrite and win → the finest part at each pixel.
        label_img = np.full((H, W), -1, dtype=np.int32)
        for mi, m in enumerate(masks):
            label_img[m["segmentation"]] = mi

        # Project full-mesh vertices into this view; keep the front-most ones.
        ndc = cam.transform_points_ndc(fv[None])[0].numpy()          # (N, 3)
        vz = cam.get_world_to_view_transform().transform_points(fv[None])[0, :, 2].numpy()
        ix = np.round((1 - ndc[:, 0]) / 2 * (W - 1)).astype(np.int64)
        iy = np.round((1 - ndc[:, 1]) / 2 * (H - 1)).astype(np.int64)
        inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ixc = np.clip(ix, 0, W - 1)
        iyc = np.clip(iy, 0, H - 1)
        zb = zbuf[iyc, ixc]
        # Visible = in front-most surface (depth match within a tolerance that
        # scales with the coarse decimated geometry).
        visible = inb & (zb > 0) & (vz <= zb + 0.03)
        vert_mask = np.full(len(full_verts_n), -1, dtype=np.int32)
        vert_mask[visible] = label_img[iyc[visible], ixc[visible]]
        if verbose:
            print(
                f"[SAM-PARTS] view {vi} (elev={elev:.0f},azim={azim:.0f}): "
                f"{len(masks)} masks, {int((vert_mask >= 0).sum())} verts covered"
            )
        out.append((vert_mask, len(masks)))
    return out


# ---------------------------------------------------------------------------
# multi-view fusion
# ---------------------------------------------------------------------------
def _fuse_views(
    per_view: List[Tuple[np.ndarray, int]],
    n_verts: int,
    overlap_thresh: float = 0.3,
) -> np.ndarray:
    """Fuse per-view mask assignments into a single per-vertex part label.

    Greedy, front-view-seeded accumulation: process views in order of how many
    vertices they cover (most first). The first view establishes the main parts;
    each later mask is either merged into the existing part it overlaps most (if
    that overlap exceeds ``overlap_thresh`` of the mask) or starts a new part.
    Already-labelled vertices are never overwritten, so early (well-seen) views
    win and later views only *extend* parts onto newly-visible surface.

    Returns an ``(N,)`` int array of part ids (``-1`` = never covered)."""
    coverage = [int((vm >= 0).sum()) for vm, _ in per_view]
    order = sorted(range(len(per_view)), key=lambda i: coverage[i], reverse=True)

    vert_part = np.full(n_verts, -1, dtype=np.int32)
    part_verts: List[np.ndarray] = []   # boolean membership per part

    for vi in order:
        vert_mask, n_masks = per_view[vi]
        for mi in range(n_masks):
            sel = vert_mask == mi
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            # Overlap of this mask with each existing part (fraction of the mask).
            best_part, best_ov = -1, 0.0
            for pi, pv in enumerate(part_verts):
                inter = int(np.count_nonzero(sel & pv))
                if inter == 0:
                    continue
                ov = inter / n_sel
                if ov > best_ov:
                    best_ov, best_part = ov, pi

            unlabeled = sel & (vert_part < 0)
            if not unlabeled.any():
                continue
            if best_part >= 0 and best_ov >= overlap_thresh:
                vert_part[unlabeled] = best_part
                part_verts[best_part] = part_verts[best_part] | unlabeled
            else:
                new_id = len(part_verts)
                vert_part[unlabeled] = new_id
                part_verts.append(unlabeled.copy())
    return vert_part


def _vertex_to_face_labels(faces: np.ndarray, vert_part: np.ndarray) -> np.ndarray:
    """Majority-vote each face's three vertex part labels (ignoring ``-1``).
    Faces whose vertices are all unlabelled stay ``-1``."""
    fl = vert_part[faces]                       # (F, 3)
    a, b, c = fl[:, 0], fl[:, 1], fl[:, 2]
    # Any label shared by >=2 vertices wins; otherwise take the first labelled.
    out = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))
    # If the "winner" is -1 but another vertex is labelled, prefer the label.
    need = out < 0
    if need.any():
        alt = np.where(a >= 0, a, np.where(b >= 0, b, c))
        out = np.where(need, alt, out)
    return out.astype(np.int32)


def _fill_unlabeled_faces(
    mesh: trimesh.Trimesh, face_labels: np.ndarray, iters: int = 32
) -> np.ndarray:
    """Flood unlabelled (``-1``) faces from their labelled neighbours via
    iterated majority vote over the face-adjacency graph."""
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return face_labels
    a, b = adj[:, 0], adj[:, 1]
    labels = face_labels.astype(np.int32).copy()
    K = int(labels.max()) + 1 if labels.max() >= 0 else 1
    for _ in range(iters):
        missing = labels < 0
        if not missing.any():
            break
        votes = np.zeros((len(labels), K), dtype=np.int32)
        la, lb = labels[a], labels[b]
        okb = lb >= 0
        np.add.at(votes, (a[okb], lb[okb]), 1)
        oka = la >= 0
        np.add.at(votes, (b[oka], la[oka]), 1)
        has = votes.sum(axis=1) > 0
        upd = missing & has
        if not upd.any():
            break
        labels[upd] = votes[upd].argmax(axis=1).astype(np.int32)
    if (labels < 0).any():                       # isolated islands: dominant part
        labels[labels < 0] = int(np.bincount(labels[labels >= 0]).argmax())
    return labels


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def split_mesh_by_sam(
    mesh: trimesh.Trimesh,
    *,
    views: Tuple[Tuple[float, float], ...] = _DEFAULT_VIEWS,
    image_size: int = 512,
    render_faces: int = 20000,
    points_per_side: int = 32,
    min_part_frac: float = 0.01,
    verbose: bool = False,
) -> Tuple[List[trimesh.Trimesh], List[np.ndarray]]:
    """Segment ``mesh`` into semantic parts by rendering it, running SAM 2's
    automatic mask generator on each render, and back-projecting the masks.

    Drop-in replacement for ``part_segmentation.split_mesh_by_color``: returns
    ``(submeshes, colors)`` where each ``colors[i]`` is the uint8 RGB dominant
    colour of ``submeshes[i]``. Raises on hard failure so callers can fall back
    to colour clustering.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0 or len(verts) == 0:
        return [mesh], [_dominant_rgb(mesh)]

    try:
        vcol = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    except Exception:
        vcol = None

    # Shared normalisation frame for both the decimated render and the full-mesh
    # projection (they MUST use the same transform).
    full_verts_n, centre, radius = _normalise(verts)
    dec_verts, dec_faces, dec_colors = _decimate(mesh, render_faces)
    dec_verts_n = ((dec_verts - centre) / radius * 0.9).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_view = _render_and_backproject(
            dec_verts_n, dec_faces, dec_colors, full_verts_n,
            views, image_size, points_per_side, verbose,
        )

    if not any(n > 0 for _, n in per_view):
        raise RuntimeError("SAM produced no masks on any view")

    vert_part = _fuse_views(per_view, len(verts))
    n_parts_raw = int(vert_part.max()) + 1 if vert_part.max() >= 0 else 0
    print(f"[SAM-PARTS] Fused {len(views)} views → {n_parts_raw} raw part(s).")
    if n_parts_raw <= 1:
        return [mesh], [_dominant_rgb(mesh)]

    face_labels = _vertex_to_face_labels(faces, vert_part)
    face_labels = _fill_unlabeled_faces(mesh, face_labels)

    # Clean up: smooth over the face graph and dissolve tiny islands (same
    # machinery the colour path uses to avoid speckle parts).
    K = int(face_labels.max()) + 1
    face_labels = _smooth_face_labels(mesh, face_labels, K, iters=8)

    # Merge parts smaller than min_part_frac into the neighbour they border most.
    min_faces = max(1, int(min_part_frac * len(faces)))
    counts = np.bincount(face_labels, minlength=K)
    small_labels = [l for l in range(K) if 0 < counts[l] < min_faces]
    if small_labels:
        # Reassign small parts to their most-common labelled neighbour.
        adj = np.asarray(mesh.face_adjacency)
        for _ in range(4):
            changed = False
            counts = np.bincount(face_labels, minlength=int(face_labels.max()) + 1)
            for l in list(range(len(counts))):
                if 0 < counts[l] < min_faces:
                    fmask = face_labels == l
                    # neighbour labels across the border
                    a, b = adj[:, 0], adj[:, 1]
                    border = (face_labels[a] == l) ^ (face_labels[b] == l)
                    nbl = np.where(face_labels[a[border]] == l,
                                   face_labels[b[border]], face_labels[a[border]])
                    nbl = nbl[nbl != l]
                    if len(nbl):
                        face_labels[fmask] = int(np.bincount(nbl).argmax())
                        changed = True
            if not changed:
                break

    min_comp = max(1, int(0.01 * len(faces)))
    face_labels = _merge_small_components(mesh, face_labels, min_comp)

    present = [int(l) for l in np.unique(face_labels)]
    if len(present) <= 1:
        print("[SAM-PARTS] Parts merged into one — single object.")
        return [mesh], [_dominant_rgb(mesh)]

    face_groups = [np.where(face_labels == l)[0] for l in present]
    submeshes = mesh.submesh(face_groups, only_watertight=False, append=False)

    if vcol is not None and len(vcol) == len(verts):
        colors = [
            np.median(vcol[faces[grp].reshape(-1)], axis=0).astype(np.uint8)
            for grp in face_groups
        ]
    else:
        colors = [_dominant_rgb(s) for s in submeshes]

    print(f"[SAM-PARTS] Split mesh into {len(submeshes)} semantic part(s).")
    return list(submeshes), colors
