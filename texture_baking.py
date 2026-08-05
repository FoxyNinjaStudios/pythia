"""
texture_baking.py  –  UV-based texture baking for SAM-3D meshes on Apple Silicon.

Strategy
--------
1. UV-parametrize the post-processed mesh with xatlas.
2. Rasterize the *UV mesh* (vertices placed at their (u,v) coordinates on a flat Z=0 plane)
   using PyTorch3D's MeshRasterizer.  Each texel that falls inside a UV triangle gets a
   face-id and barycentric weights.
3. For every valid texel:
      a. Interpolate the 3-D world position from the barycentric weights.
      b. Project that world position back to the original image using the known frontal-view
         orthographic projection:
               image_x  = vertex.X  (right/left)
               image_y  = -vertex.Z  (up/down; Z = -Y_original in the Y-up mesh)
      c. Sample the input image at that projected pixel.
4. Write sampled colors into the texture atlas; cv2.inpaint fills any unfilled holes.
5. Return a trimesh.Trimesh with PBRMaterial carrying the baked texture.

No CUDA, nvdiffrast, or Gaussian splatting required.
"""

from __future__ import annotations

import json
import struct
import numpy as np
import cv2
import torch
import trimesh
import xatlas
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_uv_mesh(uvs: np.ndarray) -> np.ndarray:
    """
    Embed UV coordinates as 3-D vertices on a flat Z=0 plane.
    U in [0,1] → X in [-1,1],  V in [0,1] → Y in [-1,1]  (V=0 → bottom → Y=-1).
    PyTorch3D's orthographic camera looks along -Z, so the flat plane is fully visible.
    """
    verts = np.zeros((len(uvs), 3), dtype=np.float32)
    verts[:, 0] = uvs[:, 0] * 2.0 - 1.0   # U  → X
    verts[:, 1] = uvs[:, 1] * 2.0 - 1.0   # V  → Y  (bottom=−1, top=+1)
    verts[:, 2] = 1.0                       # Z=1 puts mesh between znear=0.01 and zfar=100
    return verts


def _barycentric_coords(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    Vectorised barycentric coordinates of points ``p`` w.r.t. triangles (a, b, c).

    All inputs are (N, 2); returns (N, 3) weights (w_a, w_b, w_c) that sum to 1,
    matching the vertex order (a→faces[:,0], b→faces[:,1], c→faces[:,2]).
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=1).astype(np.float32)


def _rasterize_uv_space_fast(
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_size: int,
):
    """
    Fast, pure-CPU replacement for the PyTorch3D UV-space rasteriser.

    PyTorch3D has no Metal/MPS backend; on Apple Silicon its rasteriser runs the
    naive CPU C++ kernel which is O(faces × texels) and effectively hangs at large
    atlas sizes (2048² × ~250k faces ≈ 10¹² ops).  UV rasterisation is just flat 2-D
    triangle fill, so we do it directly: OpenCV fills a face-id map (each triangle
    tagged with its index — a tight C loop), then barycentric weights are computed
    analytically at the covered texel centres in one vectorised pass.  Runs in a
    couple of seconds instead of hanging.

    The UV→pixel mapping reproduces the orientation of the previous
    ``FoVOrthographicCameras`` set-up so the baked atlas stays consistent with the
    exported UVs (PyTorch3D NDC: +X→left, +Y→up).

    Returns
    -------
    pix_to_face : LongTensor (TS, TS)  – face index per texel (−1 = background)
    bary_coords : FloatTensor (TS, TS, 3) – barycentric coordinates
    """
    ts = int(texture_size)

    # UV (u, v) in [0, 1] → atlas pixel (col x, row y), matching PyTorch3D's
    # FoVOrthographicCameras output: NDC +X points left and +Y points up, so both
    # axes are flipped relative to a naive mapping.
    px = (1.0 - uvs[:, 0]) * (ts - 1)   # column
    py = (1.0 - uvs[:, 1]) * (ts - 1)   # row
    pts = np.stack([px, py], axis=1).astype(np.float32)   # (V', 2)

    # Fill a face-id map: each triangle painted with its own index. Non-overlapping
    # UV charts mean draw order is irrelevant; sub-pixel triangles simply drop out
    # (the caller pads + inpaints atlas gutters afterwards).
    face_id = np.full((ts, ts), -1, dtype=np.int32)
    tri = pts[faces]                                       # (F, 3, 2)
    tri_int = np.round(tri).astype(np.int32)
    for i in range(len(faces)):
        cv2.fillConvexPoly(face_id, tri_int[i], int(i), lineType=cv2.LINE_8)

    valid = face_id >= 0
    ys, xs = np.where(valid)
    bary_map = np.zeros((ts, ts, 3), dtype=np.float32)
    if len(ys):
        fids = face_id[valid]
        a = pts[faces[fids, 0]]
        b = pts[faces[fids, 1]]
        c = pts[faces[fids, 2]]
        centres = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32)
        bary_map[ys, xs] = _barycentric_coords(centres, a, b, c)

    return (
        torch.from_numpy(face_id.astype(np.int64)),
        torch.from_numpy(bary_map),
    )


def _rasterize_uv_space(
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_size: int,
    device: torch.device = torch.device("cpu"),
):
    """
    Rasterize the UV-space mesh (fast CPU path; ``device`` kept for signature
    compatibility).

    Returns
    -------
    pix_to_face : LongTensor (TS, TS)  – face index per texel (−1 = background)
    bary_coords : FloatTensor (TS, TS, 3) – barycentric coordinates
    """
    return _rasterize_uv_space_fast(uvs, faces, texture_size)


def _project_vertices_via_pointmap(
    vertices: np.ndarray,
    pointmap: np.ndarray,
    mask: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-correct vertex→pixel mapping using the pipeline's pointmap.

    The pointmap (H, W, 3) stores the 3D camera-space coordinate of each pixel
    (as computed by MoGe depth estimation, then transformed to PyTorch3D convention).

    The mesh vertices live in GLB Y-up space.  Before they can be matched against
    the pointmap they must be:
      1. Un-rotated from GLB back to the decoder's local space
         (inverse of to_glb rotation: local_x=glb_x, local_y=-glb_z, local_z=glb_y)
      2. Transformed from local space to camera/pointmap space using the pose
         decoder's scale + translation:
             cam = scale * local + translation

    For each transformed vertex we find the nearest pixel in the pointmap via
    KD-tree lookup.
    """
    from scipy.spatial import cKDTree

    img_h, img_w = pointmap.shape[:2]

    # Step 1: Invert GLB rotation → decoder local space
    local_x = vertices[:, 0]
    local_y = -vertices[:, 2]
    local_z = vertices[:, 1]
    verts_local = np.stack([local_x, local_y, local_z], axis=1)  # (V, 3)

    # Step 2: Apply pose decoder's scale + translation → pointmap (camera) space
    if scale is not None:
        s = np.asarray(scale, dtype=np.float32).ravel()[:3]
        verts_local = verts_local * s
    if translation is not None:
        t = np.asarray(translation, dtype=np.float32).ravel()[:3]
        verts_local = verts_local + t

    verts_in_pm_space = verts_local

    print(f"[TEXTURE] Vertex range in pointmap space: "
          f"X=[{verts_in_pm_space[:,0].min():.3f},{verts_in_pm_space[:,0].max():.3f}] "
          f"Y=[{verts_in_pm_space[:,1].min():.3f},{verts_in_pm_space[:,1].max():.3f}] "
          f"Z=[{verts_in_pm_space[:,2].min():.3f},{verts_in_pm_space[:,2].max():.3f}]")

    # Build KD-tree from pointmap pixels (only foreground if mask given)
    pm_flat = pointmap.reshape(-1, 3)  # (H*W, 3)
    pixel_ys, pixel_xs = np.mgrid[:img_h, :img_w]
    pixel_ys_flat = pixel_ys.reshape(-1).astype(np.float32)
    pixel_xs_flat = pixel_xs.reshape(-1).astype(np.float32)

    if mask is not None:
        # Resize mask to pointmap resolution if dimensions differ
        if mask.shape[:2] != (img_h, img_w):
            import cv2
            mask_u8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
            mask_resized = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
        fg = mask_resized.reshape(-1) > 127
        pm_fg = pm_flat[fg]
        px_fg = pixel_xs_flat[fg]
        py_fg = pixel_ys_flat[fg]
    else:
        pm_fg = pm_flat
        px_fg = pixel_xs_flat
        py_fg = pixel_ys_flat

    # Filter out invalid pointmap entries (zeros/NaN)
    valid = np.isfinite(pm_fg).all(axis=1) & (np.abs(pm_fg).max(axis=1) < 100.0)
    pm_valid = pm_fg[valid]
    px_valid = px_fg[valid]
    py_valid = py_fg[valid]

    print(f"[TEXTURE] Pointmap KD-tree: {len(pm_valid)} valid pixels")
    print(f"[TEXTURE] Pointmap range: "
          f"X=[{pm_valid[:,0].min():.3f},{pm_valid[:,0].max():.3f}] "
          f"Y=[{pm_valid[:,1].min():.3f},{pm_valid[:,1].max():.3f}] "
          f"Z=[{pm_valid[:,2].min():.3f},{pm_valid[:,2].max():.3f}]")

    tree = cKDTree(pm_valid)
    dists, indices = tree.query(verts_in_pm_space, k=1)

    px_f = px_valid[indices].astype(np.float32)
    py_f = py_valid[indices].astype(np.float32)

    # Report quality
    median_dist = np.median(dists)
    p90_dist = np.percentile(dists, 90)
    print(f"[TEXTURE] Pointmap match: median dist={median_dist:.4f}, "
          f"p90={p90_dist:.4f}, max={dists.max():.4f}")

    return px_f, py_f, dists.astype(np.float32)


def _transfer_model_colors(
    target_vertices: np.ndarray,
    model_vertices: np.ndarray,
    model_colors: np.ndarray,
) -> np.ndarray:
    """
    Transfer the model's generated per-vertex colors onto arbitrary target
    vertices via nearest-neighbour lookup in the decoder's local (z-up) space.

    SAM 3D is a *generative* model: the mesh decoder predicts a plausible
    per-vertex RGB colour for the ENTIRE object (front, back, sides, underside),
    not just the visible surface.  These colours (``mesh.vertex_attrs[:, :3]``,
    already passed through a sigmoid so in [0, 1]) are the model's texture
    prediction and form the correct base texture everywhere the input image
    cannot see.

    Post-processing (decimate / hole-fill / weld) remeshes the object, so the
    raw colours no longer align 1-to-1 with the exported GLB vertices.  We
    therefore resample them by nearest neighbour.

    Parameters
    ----------
    target_vertices : (M, 3) GLB Y-up vertices (post-processed / UV mesh)
    model_vertices  : (N, 3) raw decoder z-up vertices
    model_colors    : (N, 3) [0, 1] (or [0, 255]) RGB from ``vertex_attrs``

    Returns
    -------
    (M, 3) float32 colours in [0, 255]
    """
    from scipy.spatial import cKDTree

    # Un-rotate GLB (y-up) → decoder (z-up): inverse of to_glb's (x, z, -y).
    lx = target_vertices[:, 0]
    ly = -target_vertices[:, 2]
    lz = target_vertices[:, 1]
    tgt_local = np.stack([lx, ly, lz], axis=1).astype(np.float32)

    mv = np.asarray(model_vertices, dtype=np.float32)
    tree = cKDTree(mv)
    _, idx = tree.query(tgt_local, k=1)

    colors = np.asarray(model_colors, dtype=np.float32)[idx]
    if colors.max() <= 1.0 + 1e-6:
        colors = colors * 255.0
    print(f"[TEXTURE] Transferred model vertex colours: "
          f"mean R={colors[:,0].mean():.1f} G={colors[:,1].mean():.1f} "
          f"B={colors[:,2].mean():.1f}")
    return np.clip(colors, 0.0, 255.0).astype(np.float32)


def _project_vertices_to_image(
    vertices: np.ndarray,
    img_h: int,
    img_w: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project Y-up GLB mesh vertices onto the original image plane.

    GLB coordinate system (after to_glb() rotation from original space):
      new_x = old_x  (horizontal)
      new_y = old_z  (height — GLB is Y-up, this is the tallest axis)
      new_z = -old_y (depth — camera sits at +Z looking toward −Z)

    Frontal projection from +Z camera:  proj_x = X,  proj_y = Y (Y-up → top of image)
    Image Y goes down, so high GLB-Y → small image-py (handled in the mapping below).

    Key fix: when a mask is provided, the normalised [-1,1] range is mapped
    onto the mask's tight bounding box rather than the full image.
    """
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    cx = (lo[0] + hi[0]) / 2.0
    cy = (lo[1] + hi[1]) / 2.0
    rx = (hi[0] - lo[0]) / 2.0 + 1e-8   # X (horizontal) half-range
    ry = (hi[1] - lo[1]) / 2.0 + 1e-8   # Y (height)     half-range

    # Use UNIFORM scale to preserve mesh aspect ratio
    r = max(rx, ry)

    proj_x = (vertices[:, 0] - cx) / r    # [-1, 1] (may not fill full range if rx < ry)
    proj_y = (vertices[:, 1] - cy) / r    # [-1, 1] bottom→top  (GLB Y-up)

    if mask is not None and np.any(mask > 127):
        # Align projection to the foreground bounding box in the image
        rows = np.where(np.any(mask > 127, axis=1))[0]
        cols = np.where(np.any(mask > 127, axis=0))[0]
        y_min, y_max = float(rows[0]),  float(rows[-1])
        x_min, x_max = float(cols[0]),  float(cols[-1])
        print(f"[TEXTURE] mask bbox  x=[{int(x_min)}, {int(x_max)}]  "
              f"y=[{int(y_min)}, {int(y_max)}]")

        # Preserve aspect ratio: use uniform pixel-per-unit scale
        mask_w = x_max - x_min
        mask_h = y_max - y_min
        # proj range after uniform scale: proj_x in [-rx/r, rx/r], proj_y in [-ry/r, ry/r]
        # The actual range used is [-1, 1] from the uniform r = max(rx, ry)
        # Map [-1,1] → mask bbox, fitting the larger dimension and centering the smaller
        scale = min(mask_w, mask_h) / 2.0   # pixels per normalized unit
        cx_img = (x_min + x_max) / 2.0
        cy_img = (y_min + y_max) / 2.0
        px_f = proj_x * scale + cx_img
        py_f = -proj_y * scale + cy_img   # negate: GLB Y-up → image Y-down
    else:
        scale = min(img_w - 1, img_h - 1) / 2.0
        cx_img = (img_w - 1) / 2.0
        cy_img = (img_h - 1) / 2.0
        px_f = proj_x * scale + cx_img
        py_f = -proj_y * scale + cy_img

    px_f = np.clip(px_f, 0.0, float(img_w - 1))
    py_f = np.clip(py_f, 0.0, float(img_h - 1))
    return px_f, py_f


def _bilinear_sample(image: np.ndarray, px_f: np.ndarray, py_f: np.ndarray) -> np.ndarray:
    """
    Vectorised bilinear sampling of image at floating-point (px_f, py_f) coordinates.

    Parameters
    ----------
    image  : (H, W, 3) uint8
    px_f   : (N,) float – column coords
    py_f   : (N,) float – row coords

    Returns
    -------
    colors : (N, 3) float32
    """
    h, w = image.shape[:2]
    x0 = np.floor(px_f).astype(np.int32).clip(0, w - 2)
    y0 = np.floor(py_f).astype(np.int32).clip(0, h - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = (px_f - x0).astype(np.float32)[:, None]   # (N,1)
    dy = (py_f - y0).astype(np.float32)[:, None]

    img = image.astype(np.float32)
    c00 = img[y0, x0]   # (N,3)
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]

    return (1 - dy) * ((1 - dx) * c00 + dx * c10) + dy * ((1 - dx) * c01 + dx * c11)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bake_texture_from_image(
    vertices: np.ndarray,
    faces: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    texture_size: int = 1024,
    pointmap: np.ndarray | None = None,
    pm_translation: np.ndarray | None = None,
    pm_scale: np.ndarray | None = None,
    model_vertices: np.ndarray | None = None,
    model_colors: np.ndarray | None = None,
    model_colors_have_hue: bool = False,
    as_vertex_colors: bool = False,
    sparse_geometry_confidence: np.ndarray | None = None,
    slat_confidence: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    UV-unwrap a mesh and bake a texture atlas.

    When ``as_vertex_colors`` is True the whole UV pipeline (xatlas unwrap +
    rasterisation) is skipped: the projected/blended colours are written straight
    to per-vertex COLOR_0 instead.  On dense meshes (100k+ vertices) this looks
    virtually identical to a texture atlas but is ~100× faster, because xatlas
    parametrisation of a high-poly mesh alone can take minutes on CPU (and there
    is no GPU/Metal path for it).

    Following SAM 3D's generative approach, the texture is composed of two
    sources:
      * The model's own per-vertex colour prediction (``model_colors``), which
        covers the ENTIRE object including surfaces the camera never sees.
        This is the base texture.
      * The sharp input image, projected via the MoGe pointmap, which refines
        only the confidently-visible front surface for photo-real fidelity.

    Parameters
    ----------
    vertices       : (V, 3) float32 – mesh vertices in Y-up space (from to_glb)
    faces          : (F, 3) int32   – mesh face indices
    image          : (H, W, 3) uint8 – original RGB input image
    mask           : (H, W) uint8   – foreground mask (optional; pixels outside → grey)
    texture_size   : int            – NxN size of output texture atlas
    pointmap       : (H, W, 3) float32 – per-pixel 3D coords from MoGe (optional)
    pm_translation : (3,) float32  – pose decoder translation (local→camera)
    pm_scale       : (3,) float32  – pose decoder scale (local→camera)
    model_vertices : (N, 3) float32 – raw decoder (z-up) vertices for the model
                                      colour prediction (optional)
    model_colors   : (N, 3) float   – model per-vertex RGB in [0,1] or [0,255]
    sparse_geometry_confidence : (V,) float32 – per-vertex confidence from Stage 1
                                      sparse geometry reconstruction [0,1] (optional)
    slat_confidence : (V,) float32 – per-vertex confidence from Stage 2 SLAT 
                                      texture refinement [0,1] (optional)

    Returns
    -------
    trimesh.Trimesh with UV coordinates and a PBRMaterial with the baked baseColorTexture.
    """
    # Normalise a boolean mask to uint8 0/255 so downstream `> 127` checks work.
    if mask is not None and mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    if as_vertex_colors:
        # Fast path: operate directly on the original vertices/faces, no UV seams.
        print(f"[TEXTURE] Vertex-colour photo bake ({len(vertices)} vertices, "
              f"no UV unwrap)...")
        vertices_uv = vertices.astype(np.float32)
        faces_uv    = faces.astype(np.int64)
        uvs         = None
    else:
        print(f"[TEXTURE] UV-unwrapping {len(vertices)} vertices with xatlas...")
        vmapping, faces_uv, uvs = xatlas.parametrize(vertices, faces)
        vertices_uv = vertices[vmapping]          # (V', 3) – expanded for UV seams
        faces_uv    = faces_uv.astype(np.int64)   # (F, 3)
        uvs         = uvs.astype(np.float32)      # (V', 2) in [0, 1]
        print(f"[TEXTURE]   → {len(vertices_uv)} UV vertices, {len(faces_uv)} faces")

    # ── Base texture: the model's generative per-vertex colour ────────────
    # This is SAM 3D's predicted appearance for the whole object.  It is the
    # fallback everywhere the input image cannot provide a reliable colour.
    model_base = None
    if model_vertices is not None and model_colors is not None:
        model_base = _transfer_model_colors(vertices_uv, model_vertices, model_colors)

    # ── Step A: per-vertex image color (bilinear) ─────────────────────────
    img_h, img_w = image.shape[:2]
    pm_dists = None  # KD-tree match distances (only with pointmap)

    # Use pointmap for perspective-correct projection if available
    if pointmap is not None:
        print("[TEXTURE] Using pointmap for perspective-correct projection")
        px_f, py_f, pm_dists = _project_vertices_via_pointmap(
            vertices_uv, pointmap, mask,
            translation=pm_translation, scale=pm_scale,
        )
        # Scale from pointmap resolution to image resolution
        pm_h, pm_w = pointmap.shape[:2]
        if pm_h != img_h or pm_w != img_w:
            px_f = px_f * (img_w / pm_w)
            py_f = py_f * (img_h / pm_h)
    else:
        px_f, py_f = _project_vertices_to_image(vertices_uv, img_h, img_w, mask)

    # Pre-fill background pixels with nearest foreground pixel's color so that
    # any vertex projecting outside the exact mask silhouette (back/side faces)
    # picks up a plausible object color instead of the scene background.
    sample_image = image
    if mask is not None:
        from scipy.ndimage import distance_transform_edt
        fg_mask_2d = mask > 127
        if fg_mask_2d.any() and not fg_mask_2d.all():
            print("[TEXTURE] Extending foreground colors into background...")
            _, nearest = distance_transform_edt(~fg_mask_2d, return_indices=True)
            sample_image = image.copy()
            bg_r, bg_c = np.where(~fg_mask_2d)
            sample_image[bg_r, bg_c] = image[nearest[0][bg_r, bg_c],
                                             nearest[1][bg_r, bg_c]]

    vertex_colors = _bilinear_sample(sample_image, px_f, py_f)   # (V', 3) float32

    # ── Visibility-weighted blending (Meta-style) ─────────────────────────
    # Instead of binary front/back, use smooth cosine weight:
    # Visibility weighting based on camera direction.
    # The camera looks along +Z in pointmap space, which maps to -Y in GLB
    # (to_glb: glb_y = orig_z).  A front-facing vertex has its outward normal
    # pointing toward the camera, i.e. normal_Y < 0 in GLB space.
    # cos_vis = dot(normal, toward_camera) = -normal_Y
    face_verts = vertices_uv[faces_uv]                      # (F, 3, 3)
    e1 = face_verts[:, 1] - face_verts[:, 0]
    e2 = face_verts[:, 2] - face_verts[:, 0]
    face_normals = np.cross(e1, e2)                         # (F, 3)

    vert_normals = np.zeros_like(vertices_uv)               # (V', 3)
    for i in range(3):
        np.add.at(vert_normals, faces_uv[:, i], face_normals)
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    vert_normals /= norms

    # Camera direction: +Z in pointmap → +Y in GLB.  Front faces point
    # toward camera, so their normal has a negative Y component in GLB.
    cos_vis = -vert_normals[:, 1]                           # [-1, 1]
    
    # ── Adaptive visibility weighting (Improvement #2) ──────────────────
    # Compute local surface curvature to adaptively weight grazing surfaces.
    # Sharp features (high curvature) need harsher visibility falloff to avoid
    # color bleeding; smooth surfaces benefit from softer weighting.
    curvature = np.zeros(len(vertices_uv), dtype=np.float32)
    for i, j, k in faces_uv:
        v0, v1, v2 = vertices_uv[[i, j, k]]
        # Mean curvature magnitude via edge dihedral angles
        e1_norm = np.linalg.norm(v1 - v0)
        e2_norm = np.linalg.norm(v2 - v1)
        e3_norm = np.linalg.norm(v0 - v2)
        if e1_norm > 1e-8 and e2_norm > 1e-8 and e3_norm > 1e-8:
            # Simplified: use triangle angle as proxy for local curvature
            # Acute angles → high curvature
            face_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
            if face_area > 1e-8:
                c = 1.0 / (face_area + 1e-8)
                np.add.at(curvature, [i, j, k], c)
    
    # Normalize curvature per vertex
    vert_deg = np.zeros(len(vertices_uv), dtype=np.int32)
    for i in [0, 1, 2]:
        np.add.at(vert_deg, faces_uv[:, i], 1)
    vert_deg = np.maximum(vert_deg, 1)
    curvature = curvature / vert_deg
    # Normalize to [0, 1] range for weighting
    if curvature.max() > 1e-8:
        curvature = curvature / (curvature.max() + 1e-8)
    
    # Adaptive visibility exponent: sharp edges use 0.5, smooth surfaces use 0.35
    # This allows grazing surfaces on smooth regions to retain more image detail
    vis_exp = 0.35 + 0.15 * curvature  # Range [0.35, 0.5]
    
    # Smooth weight: 0 for back-facing, ramps up for front-facing
    # gamma < 1 keeps more image detail on grazing faces
    visibility_weight = np.clip(cos_vis, 0.0, 1.0) ** vis_exp  # (V',)
    print(f"[TEXTURE] Adaptive visibility exp: min={vis_exp.min():.3f}, "
          f"max={vis_exp.max():.3f}, mean={vis_exp.mean():.3f}")

    # ── Occlusion (z-buffer) test ─────────────────────────────────────────
    # A vertex can be front-FACING yet still be HIDDEN behind another part of
    # the object (seat underside, rear legs, back of the backrest curve).
    # Without a depth test those hidden vertices sample dark shadow/gap pixels
    # from the photo and bake in as dark blotches.  Build a front-camera depth
    # buffer (GLB +Z points toward the camera, so larger Z = closer) and drop
    # the image colour for any vertex that is not the front-most surface at its
    # projected pixel — it then falls back to the model's base colour.
    from scipy.ndimage import maximum_filter
    depth = vertices_uv[:, 2].astype(np.float32)
    zmin, zmax = float(depth.min()), float(depth.max())
    xi = np.clip(px_f.astype(np.int32), 0, img_w - 1)
    yi = np.clip(py_f.astype(np.int32), 0, img_h - 1)
    zbuf = np.full((img_h, img_w), -np.inf, dtype=np.float32)
    np.maximum.at(zbuf, (yi, xi), depth)
    # Grow the recorded front surface a few pixels so near-coincident samples
    # compare against the same surface (vertex splatting is sparse).
    zbuf = maximum_filter(zbuf, size=7, mode='nearest')
    front_depth = zbuf[yi, xi]
    occ_tol = max((zmax - zmin) * 0.02, 1e-6)
    occluded = depth < (front_depth - occ_tol)
    visibility_weight[occluded] = 0.0
    n_occ = int(occluded.sum())
    print(f"[TEXTURE] Occluded (hidden) vertices: {n_occ} / {len(occluded)} "
          f"({100*n_occ/max(len(occluded),1):.1f}%) → use base colour")

    # Incorporate pointmap match quality: high KD-tree distance → unreliable color
    # (Improvement #1: Enhanced pointmap confidence scoring)
    if pm_dists is not None:
        dist_median = np.median(pm_dists)
        dist_std = np.std(pm_dists)
        dist_q75 = np.percentile(pm_dists, 75)
        
        # Adaptive threshold: use 75th percentile + 1 std dev for better separation
        # of good vs poor matches, rather than fixed 1.5× median
        dist_threshold = max(dist_q75 + dist_std, dist_median * 1.2, 0.01)
        
        # Smooth confidence falloff using sigmoid-like curve instead of hard clip
        # Poor matches at threshold → confidence ≈ 0.5; 2× threshold → confidence ≈ 0.1
        pm_conf = 1.0 / (1.0 + (pm_dists / (dist_threshold + 1e-8)) ** 2)
        
        # Vertices far from image center are less reliable
        # (more likely to be on silhouettes or heavily foreshortened)
        img_center_x, img_center_y = img_w / 2.0, img_h / 2.0
        dist_from_center = np.sqrt((px_f - img_center_x) ** 2 + (py_f - img_center_y) ** 2)
        max_dist_from_center = np.sqrt(img_center_x ** 2 + img_center_y ** 2)
        center_conf = 1.0 - (dist_from_center / (max_dist_from_center + 1e-8)) ** 2
        center_conf = np.clip(center_conf, 0.5, 1.0)  # Don't penalize too heavily
        
        visibility_weight = visibility_weight * pm_conf * center_conf
        n_poor = int((pm_dists > dist_threshold).sum())
        print(f"[TEXTURE] Pointmap quality: median={dist_median:.4f}, "
              f"threshold={dist_threshold:.4f}, poor matches={n_poor} "
              f"({100*n_poor/max(len(pm_dists),1):.1f}%)")

    n_front = int((cos_vis > 0).sum())
    print(f"[TEXTURE] front-facing vertices: {n_front} / {len(cos_vis)} "
          f"({100*n_front/max(len(cos_vis),1):.1f}%)")

    # ── Integrate reconstruction quality confidence (Improvement #5) ──────
    # Use sparse geometry and SLAT stage confidence to modulate texture weight.
    # High-confidence regions from the generative pipeline should trust image
    # texture more; low-confidence regions should rely more on model colors.
    
    # Auto-generate confidence if not provided (Improvement #3 default)
    if sparse_geometry_confidence is None:
        # Heuristic: vertices near object center are more confident
        # (interior details) vs silhouette vertices (ambiguous edges)
        sparse_geometry_confidence = estimate_sparse_geometry_confidence_from_vertices(
            vertices_uv, falloff_dist=1.0
        )
        print(f"[TEXTURE] Stage 1: Generated heuristic sparse confidence "
              f"(center-based falloff)")
    
    if slat_confidence is None:
        # Heuristic: smooth surfaces are more reliably refined
        # Use inverse of curvature as proxy for refinement confidence
        slat_confidence = 1.0 - np.clip(curvature, 0.0, 1.0)
        print(f"[TEXTURE] Stage 2: Generated heuristic SLAT confidence "
              f"(inverse curvature)")
    
    # Apply confidence weighting
    if sparse_geometry_confidence is not None and len(sparse_geometry_confidence) == len(vertices_uv):
        # Soft weighting: confident regions get more image detail
        # Confidence in [0,1] → weight multiplier in [0.5, 1.0]
        # (always keep at least 50% image texture for confident regions)
        sparse_conf = 0.5 + 0.5 * np.clip(sparse_geometry_confidence, 0.0, 1.0)
        visibility_weight = visibility_weight * sparse_conf
        n_high_conf = int((sparse_geometry_confidence > 0.7).sum())
        print(f"[TEXTURE] Stage 1 sparse confidence: high={n_high_conf} "
              f"({100*n_high_conf/len(sparse_geometry_confidence):.1f}%), "
              f"mean={sparse_geometry_confidence.mean():.3f}")
    
    if slat_confidence is not None and len(slat_confidence) == len(vertices_uv):
        # Stage 2 SLAT refinement confidence: also modulates image texture trust
        slat_conf = 0.5 + 0.5 * np.clip(slat_confidence, 0.0, 1.0)
        visibility_weight = visibility_weight * slat_conf
        n_high_slat = int((slat_confidence > 0.7).sum())
        print(f"[TEXTURE] Stage 2 SLAT confidence: high={n_high_slat} "
              f"({100*n_high_slat/len(slat_confidence):.1f}%), "
              f"mean={slat_confidence.mean():.3f}")

    # ── Base colour for the blend ─────────────────────────────────────────
    # Prefer the model's generative per-vertex colour (covers the whole object
    # with plausible, spatially-varying texture).  Fall back to a flat average
    # of the reliable front-facing samples only if the model colour is absent.
    if mask is not None:
        px_i = np.clip(px_f.astype(np.int32), 0, mask.shape[1] - 1)
        py_i = np.clip(py_f.astype(np.int32), 0, mask.shape[0] - 1)
        in_mask = (mask[py_i, px_i] > 127)
        # strongly front-facing, inside the mask, and not occluded
        reliable = (cos_vis > 0.3) & in_mask & ~occluded
    else:
        reliable = (cos_vis > 0.3) & ~occluded

    if model_base is not None and model_colors_have_hue:
        # Trustworthy full-object colours (e.g. Gaussian-splat SH DC): the hue is
        # correct everywhere, so use them directly as the per-vertex base.  This
        # matches how the reference demo colours hidden surfaces.
        base_color = model_base
        print(f"[TEXTURE] Using model appearance colours directly as base "
              f"(mean R={base_color[:,0].mean():.1f} "
              f"G={base_color[:,1].mean():.1f} B={base_color[:,2].mean():.1f})")
    elif model_base is not None and reliable.any():
        # The SAM 3D mesh-decoder colour head encodes useful *luminance*
        # variation (light/dark shading that follows the geometry) but its hue
        # is unreliable — it drifts into pastel / rainbow artefacts.  So we keep
        # ONLY the luminance of the model colours and use it to modulate the
        # object's dominant (median) observed colour.  Hidden faces therefore
        # take on the real object palette (e.g. tan) with plausible shading,
        # and no rainbow hue noise.
        ref = vertex_colors[reliable]                     # observed object colours
        obj_color = np.median(ref, axis=0)                # dominant object colour
        lum = (0.299 * model_base[:, 0]
               + 0.587 * model_base[:, 1]
               + 0.114 * model_base[:, 2])
        lum = lum / (lum.mean() + 1e-5)
        lum = np.clip(lum, 0.7, 1.3)                       # limit shading swing
        base_color = np.clip(obj_color[None, :] * lum[:, None], 0.0, 255.0)
        print(f"[TEXTURE] Object-palette base (obj={obj_color.astype(int)}) → "
              f"mean R={base_color[:,0].mean():.1f} "
              f"G={base_color[:,1].mean():.1f} B={base_color[:,2].mean():.1f}")
    elif model_base is not None:
        base_color = model_base                       # (V', 3) per-vertex
    else:
        if reliable.any():
            base_color = np.broadcast_to(
                vertex_colors[reliable].mean(axis=0), vertex_colors.shape
            )
        else:
            base_color = np.broadcast_to(
                vertex_colors.mean(axis=0), vertex_colors.shape
            )

    # Blend: front/visible → sharp image colour, elsewhere → model texture.
    w = visibility_weight[:, None]  # (V', 1)
    vertex_colors = w * vertex_colors + (1.0 - w) * base_color

    if mask is not None:
        n_fg = int(in_mask.sum())
        print(f"[TEXTURE] foreground vertices: {n_fg} / {len(in_mask)} "
              f"({100*n_fg/max(len(in_mask),1):.1f}%)")
    print(f"[TEXTURE] vertex color mean  R={vertex_colors[:,0].mean():.1f} "
          f"G={vertex_colors[:,1].mean():.1f}  B={vertex_colors[:,2].mean():.1f}")

    # ── Fast path: emit per-vertex colours and skip the UV atlas entirely ──
    if as_vertex_colors:
        rgba = np.zeros((len(vertices_uv), 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(vertex_colors, 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        mesh_out = trimesh.Trimesh(
            vertices=vertices_uv,
            faces=faces_uv,
            vertex_colors=rgba,
            process=False,
        )
        print(f"[TEXTURE] Vertex colours assigned (no atlas).")
        return mesh_out

    # ── Step B: rasterize UV space ────────────────────────────────────────
    # PyTorch3D's rasterize_meshes C++ kernel requires CPU tensors.
    device = torch.device("cpu")
    print(f"[TEXTURE] Rasterizing UV space ({texture_size}x{texture_size}) on cpu...")
    pix_to_face, bary_coords = _rasterize_uv_space(uvs, faces_uv, texture_size, device)

    valid = pix_to_face >= 0   # (TS, TS) boolean
    print(f"[TEXTURE] Valid texels: {int(valid.sum())} / {texture_size*texture_size} "
          f"({100*valid.float().mean().item():.1f}%)")

    # ── Step C: fill texture atlas ────────────────────────────────────────
    texture_np = np.full((texture_size, texture_size, 3), 128, dtype=np.float32)

    if valid.any():
        fids  = pix_to_face[valid].numpy()       # (N,)  int
        bary  = bary_coords[valid].numpy()       # (N, 3) float

        # (N, 3, 3)  – 3D positions of 3 face vertices for each hit texel
        face_vert_colors = vertex_colors[faces_uv[fids]]  # (N, 3, 3)
        texel_colors = (bary[:, :, None] * face_vert_colors).sum(axis=1)  # (N, 3)

        texel_ys, texel_xs = np.where(valid.numpy())
        texture_np[texel_ys, texel_xs] = texel_colors

    texture_np = np.clip(texture_np, 0, 255).astype(np.uint8)

    # ── Step D: pad UV islands, then inpaint the remaining gutter ─────────
    # Grow each chart's filled colours a few px into the surrounding gutter so
    # that bilinear filtering / mipmapping at chart borders no longer bleeds the
    # dark inpainted background inward (the dark rims around every chart).
    from scipy.ndimage import distance_transform_edt
    valid_np = valid.numpy()
    if valid_np.any() and (~valid_np).any():
        dist, (iy, ix) = distance_transform_edt(
            ~valid_np, return_distances=True, return_indices=True,
        )
        PAD_BAND = 6
        grow = (~valid_np) & (dist <= PAD_BAND)
        texture_np[grow] = texture_np[iy[grow], ix[grow]]
        inpaint_mask = ((~valid_np) & (dist > PAD_BAND)).astype(np.uint8)
    else:
        inpaint_mask = (~valid_np).astype(np.uint8)
    texture_np = cv2.inpaint(texture_np, inpaint_mask, 8, cv2.INPAINT_TELEA)

    # ── Step D2: light denoise — a small median filter removes isolated
    #    speckle from noisy KD-tree matches; a gentle bilateral smooths the
    #    remaining grain while keeping UV-chart edges crisp.
    texture_np = cv2.medianBlur(texture_np, 3)
    texture_np = cv2.bilateralFilter(texture_np, 5, 20, 20)

    # ── Debug: save atlas PNG so we can inspect colours ──────────────────────
    try:
        PILImage.fromarray(texture_np).save("/tmp/texture_atlas_debug.png")
        print("[TEXTURE] DEBUG atlas → /tmp/texture_atlas_debug.png")
    except Exception:
        pass
    print(f"[TEXTURE] Texture atlas built.")

    # ── Step E: build trimesh with PBR material ───────────────────────────
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        metallicFactor=0.0,           # non-metallic so base colour shows correctly
        baseColorTexture=PILImage.fromarray(texture_np),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    mesh_out = trimesh.Trimesh(
        vertices=vertices_uv,
        faces=faces_uv,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )
    return mesh_out


def bake_vertex_colors(
    trimesh_obj: trimesh.Trimesh,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    pointmap: np.ndarray | None = None,
    pm_translation: np.ndarray | None = None,
    pm_scale: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Assign per-vertex colors from the image via projection.

    Skips the entire UV/atlas pipeline — vertex colors are interpolated across
    faces directly by the GPU. With 100K+ vertex meshes this gives smooth,
    seam-free results and is much faster than UV baking.
    """
    vertices = np.array(trimesh_obj.vertices, dtype=np.float32)
    faces = np.array(trimesh_obj.faces, dtype=np.int32)
    img_h, img_w = image.shape[:2]

    print(f"[TEXTURE] Vertex-color mode: {len(vertices)} verts, {len(faces)} faces")

    # Project vertices to image
    if pointmap is not None:
        print("[TEXTURE] Using pointmap for perspective-correct projection")
        px_f, py_f, _pm_dists = _project_vertices_via_pointmap(
            vertices, pointmap, mask,
            translation=pm_translation, scale=pm_scale,
        )
        # Scale from pointmap resolution to image resolution
        pm_h, pm_w = pointmap.shape[:2]
        if pm_h != img_h or pm_w != img_w:
            px_f = px_f * (img_w / pm_w)
            py_f = py_f * (img_h / pm_h)
    else:
        px_f, py_f = _project_vertices_to_image(vertices, img_h, img_w, mask)

    # Pre-fill background with nearest foreground color
    sample_image = image
    if mask is not None:
        from scipy.ndimage import distance_transform_edt
        fg_mask_2d = mask > 127
        if fg_mask_2d.any() and not fg_mask_2d.all():
            print("[TEXTURE] Extending foreground colors into background...")
            _, nearest = distance_transform_edt(~fg_mask_2d, return_indices=True)
            sample_image = image.copy()
            bg_r, bg_c = np.where(~fg_mask_2d)
            sample_image[bg_r, bg_c] = image[nearest[0][bg_r, bg_c],
                                             nearest[1][bg_r, bg_c]]

    # Sample colors
    vertex_colors = _bilinear_sample(sample_image, px_f, py_f)  # (V, 3) float32

    # Visibility masking: back-facing → average foreground color
    face_verts = vertices[faces]  # (F, 3, 3)
    e1 = face_verts[:, 1] - face_verts[:, 0]
    e2 = face_verts[:, 2] - face_verts[:, 0]
    face_normals = np.cross(e1, e2)  # (F, 3)

    vert_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    vert_normals /= norms

    # Visibility-weighted blending
    # Camera looks along +Z in pointmap → +Y in GLB.
    # Front-facing = normal pointing toward camera = normal_Y < 0 in GLB.
    cos_vis = -vert_normals[:, 1]
    visibility_weight = np.clip(cos_vis, 0.0, 1.0) ** 0.5

    n_front = int((cos_vis > 0).sum())
    print(f"[TEXTURE] front-facing: {n_front} / {len(vertices)} "
          f"({100*n_front/len(vertices):.1f}%)")

    if mask is not None:
        px_i = px_f.astype(np.int32)
        py_i = py_f.astype(np.int32)
        in_mask = (mask[py_i, px_i] > 127)
        reliable = (cos_vis > 0.3) & in_mask
    else:
        reliable = cos_vis > 0.3

    if reliable.any():
        avg_color = vertex_colors[reliable].mean(axis=0)
    else:
        avg_color = vertex_colors.mean(axis=0)

    w = visibility_weight[:, None]
    vertex_colors = w * vertex_colors + (1.0 - w) * avg_color[None, :]

    print(f"[TEXTURE] vertex color mean  R={vertex_colors[:,0].mean():.1f} "
          f"G={vertex_colors[:,1].mean():.1f}  B={vertex_colors[:,2].mean():.1f}")

    # Build RGBA vertex colors (uint8)
    rgba = np.zeros((len(vertices), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(vertex_colors, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255

    mesh_out = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=rgba,
        process=False,
    )
    print(f"[TEXTURE] Vertex colors assigned.")
    return mesh_out


def bake_mesh_texture(
    trimesh_obj: trimesh.Trimesh,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    texture_size: int = 1024,
    pointmap: np.ndarray | None = None,
    pm_translation: np.ndarray | None = None,
    pm_scale: np.ndarray | None = None,
    model_vertices: np.ndarray | None = None,
    model_colors: np.ndarray | None = None,
    model_colors_have_hue: bool = False,
    as_vertex_colors: bool = False,
    sparse_geometry_confidence: np.ndarray | None = None,
    slat_confidence: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Apply UV texture baking to an existing trimesh.Trimesh object.

    Convenience wrapper around bake_texture_from_image.
    
    Additional parameters
    ---------------------
    sparse_geometry_confidence : (V,) float32, optional
        Per-vertex confidence from Stage 1 sparse geometry [0,1]
    slat_confidence : (V,) float32, optional
        Per-vertex confidence from Stage 2 SLAT refinement [0,1]
    """
    vertices = np.array(trimesh_obj.vertices, dtype=np.float32)
    faces    = np.array(trimesh_obj.faces,    dtype=np.int32)
    return bake_texture_from_image(
        vertices, faces, image, mask, texture_size, pointmap,
        pm_translation=pm_translation, pm_scale=pm_scale,
        model_vertices=model_vertices, model_colors=model_colors,
        model_colors_have_hue=model_colors_have_hue,
        as_vertex_colors=as_vertex_colors,
        sparse_geometry_confidence=sparse_geometry_confidence,
        slat_confidence=slat_confidence,
    )


# ---------------------------------------------------------------------------
# GLB export with metallicFactor patch
# ---------------------------------------------------------------------------

def _patch_glb_metallic(glb_bytes: bytes) -> bytes:
    """
    Force every mesh in a GLB to use a matte, non-metallic material.

    Two problems this fixes:

    * Trimesh silently drops ``metallicFactor=0.0`` (falsy) from the JSON. The
      glTF 2.0 default is 1.0 (fully metallic), so the mesh renders as a grey
      mirror.
    * A vertex-coloured mesh is exported with **no material at all**; viewers
      then fall back to the spec defaults (metallic=1.0, roughness=1.0), giving
      the model an unwanted metallic shine.

    So we (a) ensure ``metallicFactor=0.0`` and ``roughnessFactor=1.0`` on every
    existing material, and (b) if there are no materials, inject a matte white
    material and assign it to every mesh primitive (baseColorFactor white simply
    passes the per-vertex COLOR_0 through unchanged).
    """
    JSON_TYPE = 0x4E4F534A  # b'JSON'

    # Parse chunks
    off, chunks = 12, []
    while off < len(glb_bytes):
        clen  = struct.unpack_from("<I", glb_bytes, off)[0]
        ctype = struct.unpack_from("<I", glb_bytes, off + 4)[0]
        cdata = bytearray(glb_bytes[off + 8 : off + 8 + clen])
        chunks.append([clen, ctype, cdata])
        off += 8 + clen

    for chunk in chunks:
        if chunk[1] != JSON_TYPE:
            continue
        j = json.loads(bytes(chunk[2]).rstrip(b" "))
        changed = False

        materials = j.get("materials")
        if materials:
            for mat in materials:
                pbr = mat.setdefault("pbrMetallicRoughness", {})
                if pbr.get("metallicFactor") != 0.0:
                    pbr["metallicFactor"] = 0.0
                    changed = True
                if "roughnessFactor" not in pbr:
                    pbr["roughnessFactor"] = 1.0
                    changed = True
        else:
            # No material → inject a matte one and wire every primitive to it.
            mat_index = 0
            j["materials"] = [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }]
            for mesh in j.get("meshes", []):
                for prim in mesh.get("primitives", []):
                    prim["material"] = mat_index
            changed = True

        if changed:
            new_json = json.dumps(j, separators=(",", ":" )).encode("utf-8")
            pad = (4 - len(new_json) % 4) % 4
            new_json += b" " * pad
            chunk[0] = len(new_json)
            chunk[2] = bytearray(new_json)
        break

    body  = b"".join(struct.pack("<II", c[0], c[1]) + bytes(c[2]) for c in chunks)
    total = struct.pack("<I", 12 + len(body))
    return glb_bytes[:8] + total + body


def export_textured_glb(mesh: trimesh.Trimesh, path: str) -> None:
    """
    Export a UV-textured trimesh to a GLB file, ensuring metallicFactor=0.0
    is present in the material so model-viewer renders colours correctly.
    """
    raw = mesh.export(file_type="glb")
    patched = _patch_glb_metallic(raw)
    with open(path, "wb") as f:
        f.write(patched)
    print(f"[TEXTURE] GLB exported ({len(patched)//1024} KB) → {path}")


# ---------------------------------------------------------------------------
# Confidence metric helpers (Texture Mapping Improvements)
# ---------------------------------------------------------------------------

def compute_surface_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex surface curvature from triangle dihedral angles.
    
    Returns normalized curvature in [0, 1] where:
    - 0 = very smooth surface (spheres, cylinders)
    - 1 = very sharp features (edges, corners)
    
    Used to adaptively weight visibility falloff: smooth surfaces preserve
    grazing-angle detail, sharp edges suppress noise artifacts.
    """
    curvature = np.zeros(len(vertices), dtype=np.float32)
    
    for i, j, k in faces:
        v0, v1, v2 = vertices[[i, j, k]]
        
        # Triangle area (related to sharpness of vertices)
        face_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5
        if face_area > 1e-8:
            # Curvature proxy: inverse of area (sharp features have small areas)
            c = 1.0 / (face_area + 1e-8)
            np.add.at(curvature, [i, j, k], c)
    
    # Normalize per vertex by face count
    vert_deg = np.zeros(len(vertices), dtype=np.int32)
    for i in [0, 1, 2]:
        np.add.at(vert_deg, faces[:, i], 1)
    vert_deg = np.maximum(vert_deg, 1)
    curvature = curvature / vert_deg
    
    # Normalize to [0, 1]
    if curvature.max() > 1e-8:
        curvature = curvature / (curvature.max() + 1e-8)
    
    return curvature


def create_uniform_confidence(num_vertices: int, confidence_level: float = 0.8) -> np.ndarray:
    """
    Create a uniform confidence array for testing/fallback.
    
    Parameters
    ----------
    num_vertices : int
        Number of vertices in the mesh
    confidence_level : float
        Confidence value in [0, 1] (default: 0.8 = high confidence)
    
    Returns
    -------
    np.ndarray of shape (num_vertices,) with uniform confidence
    """
    return np.full(num_vertices, confidence_level, dtype=np.float32)


def estimate_sparse_geometry_confidence_from_vertices(
    vertices: np.ndarray,
    center: np.ndarray | None = None,
    falloff_dist: float = 1.0,
) -> np.ndarray:
    """
    Estimate sparse geometry confidence based on vertex distance from object center.
    
    Simple heuristic: vertices near the geometric center are more confident
    (interior details), vertices on the silhouette are less confident (ambiguous).
    
    This is a placeholder; real implementations should use actual Stage 1
    occupancy probability maps.
    
    Parameters
    ----------
    vertices : (V, 3) np.ndarray
        Mesh vertices
    center : (3,) np.ndarray or None
        Object center (computed as mean if None)
    falloff_dist : float
        Distance scale for confidence falloff
    
    Returns
    -------
    (V,) np.ndarray with confidence in [0, 1]
    """
    if center is None:
        center = vertices.mean(axis=0)
    
    # Distance from center
    dists = np.linalg.norm(vertices - center[None, :], axis=1)
    
    # Confidence falloff: peak at center, drops toward edges
    # Using Gaussian-like falloff
    sigma = falloff_dist
    confidence = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def blend_confidence_arrays(*confidence_arrays) -> np.ndarray:
    """
    Blend multiple confidence arrays via multiplication (conservative).
    
    When combining multiple confidence sources, multiplication ensures that
    if any source is uncertain, the combined confidence is reduced.
    
    Parameters
    ----------
    *confidence_arrays : np.ndarray
        Variable number of (V,) confidence arrays in [0, 1]
    
    Returns
    -------
    (V,) np.ndarray with blended confidence in [0, 1]
    
    Example
    -------
    >>> sparse_conf = ...   # Stage 1 confidence
    >>> slat_conf = ...     # Stage 2 confidence
    >>> combined = blend_confidence_arrays(sparse_conf, slat_conf)
    """
    if not confidence_arrays:
        return None
    
    result = confidence_arrays[0].copy()
    for conf in confidence_arrays[1:]:
        if conf is not None:
            result = result * np.clip(conf, 0.0, 1.0)
    
    return result.astype(np.float32)
