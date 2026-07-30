"""
part_segmentation.py  –  Functional part segmentation for SAM-3D meshes.

The reconstructed mesh is a single watertight `trimesh.Trimesh` with an accurate
continuous per-vertex colour but no semantic labels. Downstream texturing tools
need the object broken into functional parts (a car's tyres / glass / body, a
chair's cushion / wood frame) so each can be recoloured and textured on its own.

Approach – **colour clustering** (the most reliable cue we have):

1. Read the mesh's per-vertex RGB colour.
2. Cluster the colours in perceptual Lab space into ``n_colors`` groups
   (k-means). Parts that differ in colour separate cleanly; all four wheels fall
   into one "tyre" group, which is what re-texturing wants.
3. Assign each face the majority colour label of its vertices; merge tiny
   speckle clusters into the nearest-colour group.
4. Split the mesh into one named sub-mesh per colour group and return a
   ``trimesh.Scene`` (a multi-object GLB), optionally baking a UV atlas per part.

No camera / pose data and no second model are required — only the mesh colour.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import trimesh


def _color_cluster_labels(
    vertex_colors: np.ndarray,
    n_colors: int,
) -> tuple[np.ndarray, int]:
    """
    Cluster per-vertex RGB colours into ``n_colors`` groups in perceptual Lab
    space and return a per-vertex integer label plus the number of clusters
    actually used.

    Colour is the most reliable cue for functional parts on these meshes: a
    car's tyres, glass and body – or a chair's cushion and wood frame – occupy
    distinct colour regions, so clustering the vertex colours groups each part
    together (and groups all four wheels into one "tyre" part, which is what
    downstream re-texturing wants).
    """
    import cv2
    from scipy.cluster.vq import kmeans2

    rgb = np.asarray(vertex_colors, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)

    k = int(max(2, min(n_colors, len(lab))))
    # Deterministic k-means++ init; a couple of restarts for stability.
    best_labels, best_centroids, best_inertia = None, None, np.inf
    with warnings.catch_warnings():
        # Fewer distinct colours than k leaves empty clusters — benign, we
        # renumber below.
        warnings.simplefilter("ignore")
        for seed in (0, 1, 2):
            centroids, labels = kmeans2(
                lab, k, minit="++", seed=seed, missing="warn"
            )
            d = lab - centroids[labels]
            inertia = float((d * d).sum())
            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels, centroids

    labels = np.asarray(best_labels, dtype=np.int32)
    # Renumber to a dense 0..n-1 range (some centroids may capture no vertices).
    used = np.unique(labels)
    remap = {int(u): i for i, u in enumerate(used)}
    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
    return labels, len(used)


def segment_mesh_parts(
    mesh: trimesh.Trimesh,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    *,
    n_colors: int = 6,
    min_part_frac: float = 0.02,
    bake: bool = True,
    texture_size: int = 2048,
    **_legacy,
) -> trimesh.Scene:
    """
    Split a reconstructed mesh into functional parts **by colour** and return a
    multi-object GLB scene.

    Colour is the most reliable signal available: the reconstructed mesh already
    carries an accurate per-vertex colour, so parts that differ in colour (tyres
    vs glass vs body, cushion vs wood frame) separate cleanly without needing a
    camera pose or a second segmentation model.

    Parameters
    ----------
    mesh          : reconstructed `trimesh.Trimesh` (GLB Y-up, per-vertex colour)
    image         : (H, W, 3) uint8 RGB input photo — only used for baking
    mask          : (H, W) object mask — only used for baking
    n_colors      : number of colour clusters (functional parts) to split into
    min_part_frac : merge clusters smaller than this fraction of faces into the
                    nearest-colour cluster (removes speckle parts)
    bake          : bake a UV texture atlas per part (matches the unsegmented
                    download quality); when False parts keep per-vertex colour
    texture_size  : baked atlas edge length in px (when ``bake`` is True)

    Returns
    -------
    trimesh.Scene with one named ``part_NN`` geometry per part. Falls back to a
    single ``part_00`` (the whole mesh) if only one colour is found.
    """
    faces = np.asarray(mesh.faces)

    # Per-vertex RGB is required for colour clustering.
    try:
        vcol = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    except Exception as exc:
        print(f"[PARTS] Mesh has no per-vertex colour ({exc}); exporting single object.")
        return _scene_from_parts([mesh], image, mask, bake, texture_size)

    if len(vcol) != len(mesh.vertices) or len(faces) == 0:
        print("[PARTS] Missing vertex colours / faces; exporting single object.")
        return _scene_from_parts([mesh], image, mask, bake, texture_size)

    vert_labels, n_clusters = _color_cluster_labels(vcol, n_colors)
    print(f"[PARTS] Colour clustering → {n_clusters} colour group(s).")
    if n_clusters <= 1:
        print("[PARTS] Only one colour found — exporting mesh as a single object.")
        return _scene_from_parts([mesh], image, mask, bake, texture_size)

    # Assign each face the majority colour label of its three vertices.
    # Vectorised (a pure-Python per-face loop would take minutes / appear to
    # hang on a real 100k+-face mesh): with three labels a "majority" is any
    # label that appears at least twice; if all three differ, keep the first.
    fl = vert_labels[faces]            # (F, 3)
    a, b, c = fl[:, 0], fl[:, 1], fl[:, 2]
    face_labels = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))
    face_labels = face_labels.astype(np.int32)

    # Merge tiny colour clusters into the nearest remaining cluster (by mean
    # colour) so we do not emit speckle parts.
    min_faces = max(1, int(min_part_frac * len(faces)))
    counts = np.bincount(face_labels, minlength=n_clusters)
    big = [lbl for lbl in range(n_clusters) if counts[lbl] >= min_faces]
    if not big:
        big = [int(np.argmax(counts))]
    if len(big) < n_clusters:
        # Mean RGB per cluster for nearest-neighbour merging.
        means = np.zeros((n_clusters, 3), np.float64)
        for lbl in range(n_clusters):
            m = face_labels == lbl
            if m.any():
                means[lbl] = vcol[faces[m].reshape(-1)].mean(axis=0)
        big_arr = np.array(big)
        for lbl in range(n_clusters):
            if lbl in big:
                continue
            j = big_arr[np.argmin(((means[big_arr] - means[lbl]) ** 2).sum(axis=1))]
            face_labels[face_labels == lbl] = j

    present = [lbl for lbl in np.unique(face_labels)]
    if len(present) <= 1:
        print("[PARTS] Colours merged into one part — exporting single object.")
        return _scene_from_parts([mesh], image, mask, bake, texture_size)

    face_groups = [np.where(face_labels == lbl)[0] for lbl in present]
    submeshes = mesh.submesh(face_groups, only_watertight=False, append=False)

    print(f"[PARTS] Split mesh into {len(submeshes)} colour parts.")
    return _scene_from_parts(submeshes, image, mask, bake, texture_size)


def _solid_part_geometry(sub: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    """Return ``sub`` with a matte material whose baseColorFactor is the part's
    dominant (median) colour, and with the per-vertex colours removed.

    Why not keep the per-vertex COLOR_0 and a white material?  Compliant viewers
    (three.js, model-viewer) multiply white × COLOR_0 and show colour, but macOS
    Quick Look / RealityKit *ignore* COLOR_0 when a material is present and
    render the white baseColorFactor — the model comes out fully white.  Because
    every part is a colour cluster (nearly uniform), baking its dominant colour
    into the material's baseColorFactor shows correctly in every viewer and is
    exactly what downstream per-part recolouring wants.
    """
    from trimesh.visual.material import PBRMaterial

    try:
        vc = np.asarray(sub.visual.vertex_colors, dtype=np.float64)[:, :3]
        col = np.median(vc, axis=0) / 255.0
    except Exception:
        col = np.array([0.8, 0.8, 0.8])

    mat = PBRMaterial(
        name=name,
        baseColorFactor=[float(col[0]), float(col[1]), float(col[2]), 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    geom = sub.copy()
    geom.visual = trimesh.visual.TextureVisuals(material=mat)
    return geom


def _scene_from_parts(
    submeshes: List[trimesh.Trimesh],
    image: np.ndarray,
    mask: np.ndarray,
    bake: bool,
    texture_size: int,
) -> trimesh.Scene:
    """Assemble named part sub-meshes into a Scene.

    Default (fast): each part gets a matte material tinted with its dominant
    colour so it renders correctly in every viewer (see ``_solid_part_geometry``).
    When ``bake`` is set, a per-part UV texture atlas is baked instead (keeps
    within-part colour variation but is much slower)."""
    scene = trimesh.Scene()
    for i, sub in enumerate(submeshes):
        name = f"part_{i:02d}"
        geom = None
        if bake:
            try:
                from texture_baking import bake_mesh_texture

                # Feed the part's own reconstructed per-vertex colours as the
                # trustworthy base layer.  The image refines only the visible
                # front faces; occluded / back faces then keep the real object
                # colour (matching the unsegmented baked download) instead of a
                # flat average.  model colours live in the decoder's z-up frame:
                # un-rotate the GLB (y-up) vertices with (x, -z, y).
                v = np.asarray(sub.vertices, dtype=np.float32)
                model_vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
                try:
                    model_colors = np.asarray(
                        sub.visual.vertex_colors, dtype=np.float32
                    )[:, :3]
                except Exception:
                    model_colors = None

                geom = bake_mesh_texture(
                    sub, image, mask, texture_size=texture_size,
                    model_vertices=model_vertices if model_colors is not None else None,
                    model_colors=model_colors,
                    model_colors_have_hue=model_colors is not None,
                )
            except Exception as exc:
                print(f"[PARTS] {name}: texture bake failed ({exc}); using solid colour")
                geom = None

        if geom is None:
            geom = _solid_part_geometry(sub, name)

        scene.add_geometry(geom, geom_name=name)
        print(f"[PARTS] {name}: {len(geom.faces)} faces, {len(geom.vertices)} verts"
              f"{' (baked)' if bake and isinstance(geom.visual, trimesh.visual.TextureVisuals) and getattr(geom.visual.material, 'baseColorTexture', None) is not None else ' (solid)'}")
    return scene

