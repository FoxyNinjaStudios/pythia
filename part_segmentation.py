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
    luminance_weight: float = 0.35,
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

    ``luminance_weight`` scales the L (lightness) axis down relative to the a/b
    (chroma) axes. The reconstructed colour has strong baked-in shading, so with
    equal weighting k-means bands the surface by *brightness* (splitting one blue
    cushion into a light half and a dark half). Down-weighting L makes actual
    hue differences (blue seat vs yellow base) drive the clustering instead.
    """
    import cv2
    from scipy.cluster.vq import kmeans2

    rgb = np.asarray(vertex_colors, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    # Emphasise chroma over lightness so shading does not fake colour splits.
    lab[:, 0] *= float(luminance_weight)

    # Auto mode (n_colors is None / <= 0): over-cluster, then merge perceptually
    # close colours so the number of parts emerges from the data instead of
    # being forced (forcing k splits one real colour into extra segments).
    auto = n_colors is None or int(n_colors) <= 0
    k = 12 if auto else int(n_colors)
    k = int(max(2, min(k, len(lab))))
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
    centroids = np.asarray(best_centroids, dtype=np.float64)

    if auto:
        # Agglomeratively merge cluster centroids that are closer than a
        # perceptual threshold (in the same chroma-weighted Lab space used for
        # clustering). This collapses brightness bands of one colour back
        # together and yields the natural number of distinct colours.
        labels, _ = _merge_close_centroids(labels, centroids, threshold=26.0)

    # Renumber to a dense 0..n-1 range (some centroids may capture no vertices).
    used = np.unique(labels)
    remap = {int(u): i for i, u in enumerate(used)}
    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
    return labels, len(used)


def _merge_close_centroids(
    labels: np.ndarray,
    centroids: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    """Agglomeratively merge k-means clusters whose centroids are within
    ``threshold`` (size-weighted, in the clustering's colour space). Used by the
    auto colour-count mode so brightness bands of one colour collapse back into
    a single part. ``centroids`` count is small (<= 12) so the O(k^2) search is
    cheap.
    """
    labels = labels.astype(np.int32).copy()
    K = len(centroids)
    cent = centroids.astype(np.float64).copy()
    sizes = np.bincount(labels, minlength=K).astype(np.float64)
    alive = sizes > 0

    while True:
        idxs = np.where(alive)[0]
        if len(idxs) <= 1:
            break
        best_d, best_a, best_b = np.inf, -1, -1
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = int(idxs[i]), int(idxs[j])
                d = float(np.linalg.norm(cent[a] - cent[b]))
                if d < best_d:
                    best_d, best_a, best_b = d, a, b
        if best_d >= threshold:
            break
        na, nb = sizes[best_a], sizes[best_b]
        cent[best_a] = (cent[best_a] * na + cent[best_b] * nb) / (na + nb)
        sizes[best_a] = na + nb
        alive[best_b] = False
        labels[labels == best_b] = best_a

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
    label: Optional[str] = None,
    use_sam3: bool = False,
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
    trimesh.Scene with one named geometry per part. When ``label`` names a known
    object class (chair/table/lamp), parts get functional names (``legs``,
    ``seat``, ``pillow`` …). With ``use_sam3`` those names come from running SAM
    3 text segmentation on renders of the parts (true semantics); otherwise they
    come from a geometric heuristic, falling back to dominant colour. The scene's
    root node is named after ``label`` (e.g. ``chair``) instead of the generic
    ``world`` frame.
    """
    submeshes, _colors = split_mesh_into_parts(
        mesh, n_colors=n_colors, min_part_frac=min_part_frac
    )
    return _scene_from_parts(
        submeshes, image, mask, bake, texture_size, label, use_sam3
    )


def split_mesh_into_parts(
    mesh: trimesh.Trimesh,
    *,
    n_colors: int = 6,
    min_part_frac: float = 00.02,
    method: str = "color",
) -> tuple[list[trimesh.Trimesh], list[np.ndarray]]:
    """Split a mesh into parts and return ``(submeshes, colors)``.

    ``method``:
      * ``"color"`` (default) – colour clustering over the whole mesh
        (:func:`split_mesh_by_color`). This is **holistic**: every vertex is
        labelled from one global colour model, so a part is identical on the
        front and the back of the object and boundaries follow real colour
        edges (see the edge-aware smoothing in :func:`_smooth_face_labels`).
      * ``"sam"``   – semantic segmentation only (render the mesh from several
        views, run SAM 2's automatic mask generator, back-project the masks).
        SAM is correct *per view* but each view is segmented independently, so
        the fused 3-D labelling is not guaranteed to match front-to-back.
      * ``"auto"`` – try SAM first and fall back to colour clustering.

    Colour is the default because it is consistent across the whole surface;
    SAM remains available for callers that explicitly want per-view semantics.
    """
    if method in ("auto", "sam"):
        try:
            from sam_part_segmentation import split_mesh_by_sam

            subs, cols = split_mesh_by_sam(
                mesh, min_part_frac=max(min_part_frac, 0.01)
            )
            if len(subs) > 1 or method == "sam":
                return subs, cols
            print("[PARTS] SAM found a single part; falling back to colour.")
        except Exception as exc:
            if method == "sam":
                raise
            print(f"[PARTS] SAM segmentation unavailable ({exc}); using colour.")
    return split_mesh_by_color(
        mesh, n_colors=n_colors, min_part_frac=min_part_frac
    )


def _dominant_rgb(mesh: trimesh.Trimesh) -> np.ndarray:
    """Median per-vertex RGB of ``mesh`` as a uint8 (3,) array (grey fallback)."""
    try:
        vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3]
        return np.median(vc, axis=0).astype(np.uint8)
    except Exception:
        return np.array([200, 200, 200], dtype=np.uint8)


def _face_mean_lab(vcol: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face mean colour in perceptual Lab space, ``(F, 3)`` float.

    Used to make label smoothing edge-aware: the Lab distance across a shared
    edge measures the colour gradient there, so a large distance marks a colour
    edge that boundary smoothing should not cross.
    """
    import cv2

    rgb = (
        vcol[faces].astype(np.float64).mean(axis=1).clip(0, 255).astype(np.uint8)
    ).reshape(-1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def _smooth_face_labels(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    n_labels: int,
    iters: int = 8,
    face_colors: Optional[np.ndarray] = None,
    edge_sigma: float = 18.0,
    edge_floor: float = 0.6,
) -> np.ndarray:
    """Spatially smooth per-face colour labels over the mesh's face-adjacency
    graph (iterated majority vote).

    The reconstructed mesh carries *continuous* photographic colour, so raw
    colour clustering fragments each colour group into hundreds of tiny
    disconnected speckle patches — which, once split into separate objects,
    look like the mesh is "full of holes". Relabelling every face to the
    majority colour of itself + its neighbours collapses that speckle into
    contiguous regions while leaving the real colour boundaries in place.

    When ``face_colors`` (per-face colour, e.g. Lab) is supplied the vote is
    **edge-aware**: neighbours separated by a strong colour edge vote a little
    less for each other, so part boundaries prefer real colour transitions.
    Colour is the primary signal here, though: ``edge_floor`` keeps most of the
    smoothing (never below the floor) even across a colour edge, so similar
    colours are grouped aggressively and the contour stays smooth instead of
    tracing every jagged wobble of the colour edge.
    """
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return face_labels
    a, b = adj[:, 0], adj[:, 1]
    F = len(face_labels)
    labels = face_labels.astype(np.int32).copy()
    idx = np.arange(F)
    # Per-edge affinity: mostly uniform (aggressive colour grouping) with a mild
    # preference for real colour edges. ``edge_floor`` is the minimum affinity
    # kept even across a strong colour edge, so grouping never fully stops and
    # the contour is straightened by the majority vote rather than following the
    # ragged colour boundary.
    if face_colors is not None:
        fc = np.asarray(face_colors, dtype=np.float64)
        d = np.linalg.norm(fc[a] - fc[b], axis=1)
        floor = float(edge_floor)
        w = floor + (1.0 - floor) * np.exp(-((d / float(edge_sigma)) ** 2))
    else:
        w = np.ones(len(a), dtype=np.float64)
    for _ in range(iters):
        votes = np.zeros((F, n_labels), dtype=np.float64)
        votes[idx, labels] += 1.0               # a face votes for itself
        np.add.at(votes, (a, labels[b]), w)     # + each neighbour's label
        np.add.at(votes, (b, labels[a]), w)     #   weighted by colour-edge gate
        new = votes.argmax(axis=1).astype(np.int32)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _face_chroma(face_lab: np.ndarray) -> np.ndarray:
    """Per-face chroma (colourfulness) from Lab: ``sqrt(a*^2 + b*^2)``.

    OpenCV's Lab stores ``a*``/``b*`` offset by 128, so a neutral grey/white
    face sits at ``(128, 128)`` → chroma ≈ 0, while a saturated colour has large
    chroma. Chroma is illumination-robust: a *shaded* white cushion stays
    near-neutral (low chroma) even though its lightness drops, whereas a rust
    frame stays chromatic. That makes chroma the reliable signal for telling a
    neutral part from a coloured one regardless of shading.
    """
    fc = np.asarray(face_lab, dtype=np.float64)
    return np.sqrt((fc[:, 1] - 128.0) ** 2 + (fc[:, 2] - 128.0) ** 2)


def _neutral_chroma_tau(face_chroma: np.ndarray) -> float:
    """Chroma threshold separating neutral (grey/white/black) faces from
    chromatic ones, via Otsu on the chroma histogram, clamped to a perceptually
    achromatic band ``[10, 20]``.

    Clamping keeps "neutral" meaning genuinely grey (near-zero ``a*``/``b*``) so
    the gate never fires between two *different saturated colours* (whose Otsu
    split could land high); it only ever protects truly achromatic material.
    """
    import cv2

    ch8 = (np.clip(face_chroma, 0.0, 60.0) / 60.0 * 255.0).astype(np.uint8)
    t, _ = cv2.threshold(ch8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(float(t) / 255.0 * 60.0, 10.0, 20.0))


def _make_chroma_gate(
    face_lab: np.ndarray,
    face_labels: np.ndarray,
    n_labels: int,
):
    """Build a constraint that forbids labels from crossing the neutral↔chromatic
    colour boundary, or ``None`` when the mesh has no such mix.

    A reconstructed mesh bakes ambient/AO shading into its vertex colours, so a
    neutral part (e.g. a white cushion) fades and picks up a little of an
    adjacent chromatic part (a rust chair) in the crevice between them. Plain Lab
    clustering then *erodes* the shaded edge of the neutral part into the
    coloured part, because the shaded face's lightness pulls it toward the
    coloured centroid. Chroma does not lie, though: those faces stay near-neutral.

    The returned ``constrain(labels)`` re-homes every face that landed on the
    wrong side of the chroma threshold: a near-neutral face stuck in a chromatic
    part is moved to the nearest neutral part (by lightness, which is what
    separates a white cushion from black legs), and a chromatic face stuck in a
    neutral part is moved to the nearest chromatic part (by full-Lab colour).

    Returns ``None`` unless the seed has *both* a clearly neutral part and a
    clearly chromatic one (margin), so meshes that are all-neutral or
    all-colourful are left untouched.
    """
    fc = np.asarray(face_lab, dtype=np.float64)
    chroma = _face_chroma(fc)
    tau = _neutral_chroma_tau(chroma)

    def _centroid_sides(labels: np.ndarray):
        cen = np.zeros((n_labels, 3), dtype=np.float64)
        cen_chroma = np.full(n_labels, np.nan)
        for l in range(n_labels):
            m = labels == l
            if m.any():
                cen[l] = fc[m].mean(axis=0)
                cen_chroma[l] = _face_chroma(cen[l : l + 1])[0]
        neutral = [l for l in range(n_labels) if cen_chroma[l] < tau]
        chrom = [l for l in range(n_labels) if cen_chroma[l] >= tau + 2.0]
        return cen, neutral, chrom

    # Only enable the gate when the seed genuinely mixes neutral and chromatic
    # material (otherwise there is nothing to protect and we risk splitting a
    # single colour).
    cen0, neutral0, chrom0 = _centroid_sides(np.asarray(face_labels, np.int32))
    if not neutral0 or not chrom0:
        return None

    # Baseline neutrality: a face is neutral if its chroma is below tau.
    face_neutral = chroma < tau

    # Hue-direction refinement. Some coloured parts are only *weakly* saturated
    # on a shaded reconstruction — e.g. a light-blue garment on a grey body whose
    # chroma magnitude overlaps the body's shaded fur — yet their *hue* is
    # consistently offset from the body's (blue vs the body's warm grey). Chroma
    # magnitude alone then wrongly calls those faces neutral and the coloured
    # part comes out incomplete. Project each face's a*b* onto the axis running
    # from the neutral centroid toward the chromatic centroid: a face that sits
    # well along that axis is on the chromatic *hue* side even when its chroma is
    # low, so treat it as chromatic. Faces with a random/opposite hue (ordinary
    # body speckle) project near zero or negative and stay neutral, so this only
    # rescues genuinely mis-hued faces and leaves neutral-vs-neutral parts (a
    # white cushion vs black legs, both near the neutral centroid) untouched.
    neutral_ab = cen0[np.array(neutral0), 1:3].mean(axis=0)
    chrom_ab = cen0[np.array(chrom0), 1:3].mean(axis=0)
    axis = chrom_ab - neutral_ab
    axis_len = float(np.linalg.norm(axis))
    if axis_len > 1e-6:
        axis_u = axis / axis_len
        proj = (fc[:, 1:3] - neutral_ab[None, :]) @ axis_u
        # Absolute a*b* distance along the chromatic hue axis, capped so a very
        # short axis (barely-coloured mesh) does not over-trigger.
        proj_tau = min(6.0, 0.25 * axis_len)
        face_neutral = face_neutral & (proj < proj_tau)

    def constrain(labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int32).copy()
        cen, neutral, chrom = _centroid_sides(labels)
        if not neutral or not chrom:
            return labels
        neutral_arr, chrom_arr = np.array(neutral), np.array(chrom)
        # Near-neutral face trapped in a chromatic part → nearest neutral part
        # by lightness (L separates white cushion from black legs).
        bad = face_neutral & np.isin(labels, chrom_arr)
        if bad.any():
            d = np.abs(fc[bad, 0:1] - cen[neutral_arr, 0][None, :])
            labels[bad] = neutral_arr[d.argmin(axis=1)]
        # Chromatic face trapped in a neutral part → nearest chromatic part by
        # full-Lab colour.
        bad = (~face_neutral) & np.isin(labels, neutral_arr)
        if bad.any():
            d = np.linalg.norm(fc[bad, None, :] - cen[chrom_arr][None, :, :], axis=2)
            labels[bad] = chrom_arr[d.argmin(axis=1)]
        return labels

    return constrain


def _mrf_refine_labels(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    face_lab: np.ndarray,
    n_labels: int,
    lam: float = 24.0,
    edge_sigma: float = 14.0,
    iters: int = 20,
    edge_floor: float = 0.6,
    constrain=None,
) -> np.ndarray:
    """Crisp, contour-regularised part boundaries via an MRF / graph-cut-style
    energy minimisation (iterated conditional modes).

    Plain label smoothing has no *data* term, so faces in the blurry colour
    transition between two parts (e.g. a white pillow fading into a rust chair on
    the reconstructed surface) get decided purely by their noisy neighbourhood
    and the boundary comes out ragged. Here each face pays:

    * a **data cost** = Lab distance to each part's mean colour (anchors every
      face to the part whose colour it is closest to, in *full* Lab so lightness
      counts — that is what separates a white cushion from a mid-tone frame).
      **Colour is the signal**: this term alone decides which part a face joins.
    * a **smoothness cost** = a mostly-uniform Potts penalty for disagreeing
      with each neighbour, with only a mild preference for real colour edges
      (``edge_floor`` keeps the penalty from vanishing at the edge).

    Minimising data + λ·smoothness makes each face take its nearest part colour
    and then straightens the contour (short boundaries are cheaper). Because the
    smoothness is near-uniform rather than edge-gated, the cut is a clean, smooth
    line through the blurry colour-transition band instead of tracing the jagged
    colour edge. λ (``lam``) trades contour smoothness against colour fit.
    """
    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0 or n_labels <= 1:
        return face_labels
    a, b = adj[:, 0], adj[:, 1]
    fc = np.asarray(face_lab, dtype=np.float64)
    F = len(face_labels)
    labels = face_labels.astype(np.int32).copy()
    if constrain is not None:
        labels = constrain(labels)

    # Mostly-uniform edge affinities: ``edge_floor`` is kept even across a strong
    # colour edge so the contour is straightened by the smoothness term, with a
    # mild extra pull toward real colour edges. This groups similar colours
    # aggressively instead of snapping the boundary to every jag of the edge.
    contrast = np.linalg.norm(fc[a] - fc[b], axis=1)
    floor = float(edge_floor)
    w = floor + (1.0 - floor) * np.exp(-((contrast / float(edge_sigma)) ** 2))

    # Fixed data cost: Lab distance from every face to each part's mean colour.
    centroids = np.zeros((n_labels, 3), dtype=np.float64)
    for l in range(n_labels):
        m = labels == l
        if m.any():
            centroids[l] = fc[m].mean(axis=0)
    data = np.linalg.norm(fc[:, None, :] - centroids[None, :, :], axis=2)  # (F,K)

    for _ in range(iters):
        # Smoothness: for label l, sum of neighbour affinities that already
        # carry l. Choosing l then avoids paying the Potts penalty for those
        # edges, so cost = data - lam * (agreeing neighbour weight).
        w_same = np.zeros((F, n_labels), dtype=np.float64)
        np.add.at(w_same, (a, labels[b]), w)
        np.add.at(w_same, (b, labels[a]), w)
        new = (data - lam * w_same).argmin(axis=1).astype(np.int32)
        if constrain is not None:
            new = constrain(new)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _merge_small_components(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    min_comp_faces: int,
) -> np.ndarray:
    """Dissolve small disconnected same-colour islands into the surrounding
    region so each colour part is a few large, contiguous patches (no speckle
    holes). Each connected component smaller than ``min_comp_faces`` is
    reassigned *in full* to the label its border touches most, repeated until
    the labelling is stable.
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        return face_labels

    adj = np.asarray(mesh.face_adjacency)
    if len(adj) == 0:
        return face_labels
    a, b = adj[:, 0], adj[:, 1]
    F = len(face_labels)
    labels = face_labels.astype(np.int32).copy()
    K = int(labels.max()) + 1

    for _ in range(8):
        same = labels[a] == labels[b]
        e = adj[same]
        if len(e) == 0:
            break
        g = coo_matrix(
            (np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(F, F)
        )
        n_comp, comp = connected_components(g, directed=False)
        sizes = np.bincount(comp, minlength=n_comp)
        small = sizes < min_comp_faces
        if not small.any():
            break

        # Cross-component boundary edges vote the label on the *large* side to
        # the small component so the whole island flips at once.
        cross = comp[a] != comp[b]
        ca, cb = a[cross], b[cross]
        comp_a, comp_b = comp[ca], comp[cb]
        lab_a, lab_b = labels[ca], labels[cb]
        sa, sb = small[comp_a], small[comp_b]

        votes = np.zeros((n_comp, K), dtype=np.int64)
        m1 = sa & ~sb                       # a small, b large
        np.add.at(votes, (comp_a[m1], lab_b[m1]), 1)
        m2 = sb & ~sa                       # b small, a large
        np.add.at(votes, (comp_b[m2], lab_a[m2]), 1)

        has = votes.sum(axis=1) > 0
        target_for_comp = np.where(has, votes.argmax(axis=1), -1)
        target = target_for_comp[comp]
        change = (target >= 0) & (target != labels)
        if not change.any():
            break
        labels[change] = target[change].astype(np.int32)
    return labels


def _cut_faces_along_label_boundary(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    smooth_iters: int = 6,
):
    """Re-tessellate only the faces that straddle a part boundary so the cut
    follows a smooth iso-contour instead of the ragged per-triangle staircase.

    The per-face labelling gives a boundary that zig-zags along whole triangle
    edges (the sawtooth "teeth"). Here we:

    1. Convert the (already spatially-smoothed) per-face labels to a per-*vertex*
       label by area-weighted majority of each vertex's incident faces.
    2. Build a per-part membership field and Laplacian-smooth it a few times so
       the boundary between two parts is a smooth iso-line rather than the hard
       per-vertex step.
    3. For every mesh edge whose two endpoints land in different parts, insert a
       single seam vertex at the iso-crossing (``ind_a == ind_b``). Because the
       cut vertex lives on the *shared* edge and is created once (keyed by the
       sorted edge), both triangles that share that edge use the *same* vertex —
       so the two resulting parts share an identical boundary loop and the mesh
       stays watertight (no T-junctions / cracks).
    4. Re-triangulate each straddling triangle (marching-triangles 1-vs-2 case,
       plus a rare 3-way triple-point case) and assign each sub-face to the part
       of the original corner(s) it covers.

    Returns ``(vertices, faces, vertex_colors, new_face_labels)`` ready to be
    split by :func:`trimesh.Trimesh.submesh`. Falls back (raises) to let the
    caller keep the plain per-face cut if anything is degenerate.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    try:
        VC = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)  # (V,4)
    except Exception:
        VC = np.ones((len(V), 4), dtype=np.float64) * 255.0
    n_v = len(V)
    K = int(face_labels.max()) + 1

    # (1) Per-vertex label: area-weighted majority of incident face labels.
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    vote = np.zeros((n_v, K), dtype=np.float64)
    for j in range(3):
        np.add.at(vote, (F[:, j], face_labels), areas)
    vlabel = vote.argmax(axis=1).astype(np.int32)

    # (2) Per-part membership field, Laplacian-smoothed over the edge graph so
    #     the iso-contour is smooth.
    ind = np.zeros((n_v, K), dtype=np.float64)
    ind[np.arange(n_v), vlabel] = 1.0
    e = np.asarray(mesh.edges_unique)
    if len(e) and smooth_iters > 0:
        e0, e1 = e[:, 0], e[:, 1]
        deg = np.bincount(e0, minlength=n_v) + np.bincount(e1, minlength=n_v)
        denom = (1.0 + deg)[:, None]
        for _ in range(int(smooth_iters)):
            s = ind.copy()
            np.add.at(s, e0, ind[e1])
            np.add.at(s, e1, ind[e0])
            ind = s / denom

    # (3) Insert one seam vertex per boundary edge (shared → watertight).
    new_pos: list[np.ndarray] = []
    new_col: list[np.ndarray] = []
    edge_cut: dict[tuple[int, int], int] = {}

    def cut_vertex(u: int, w: int) -> int:
        key = (u, w) if u < w else (w, u)
        hit = edge_cut.get(key)
        if hit is not None:
            return hit
        a, b = key
        la, lb = int(vlabel[a]), int(vlabel[b])
        # Crossing where membership of la equals membership of lb along a→b.
        d0 = ind[a, la] - ind[a, lb]      # > 0 (a is la)
        d1 = ind[b, la] - ind[b, lb]      # < 0 (b is lb)
        denom = d0 - d1
        t = 0.5 if abs(denom) < 1e-9 else float(np.clip(d0 / denom, 0.05, 0.95))
        idx = n_v + len(new_pos)
        new_pos.append(V[a] + t * (V[b] - V[a]))
        new_col.append(VC[a] + t * (VC[b] - VC[a]))
        edge_cut[key] = idx
        return idx

    out_faces: list[tuple[int, int, int]] = []
    out_labels: list[int] = []
    fl = vlabel[F]  # (Nf,3) per-corner labels

    uniform = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
    for fi in np.nonzero(uniform)[0]:
        out_faces.append((int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])))
        out_labels.append(int(fl[fi, 0]))

    for fi in np.nonzero(~uniform)[0]:
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        la, lb, lc = int(fl[fi, 0]), int(fl[fi, 1]), int(fl[fi, 2])
        if la == lb or lb == lc or la == lc:
            # 1-vs-2: identify the odd corner `o` (differs from the other two).
            if la == lb:
                o, p, q, lo, lpq = c, a, b, lc, la
            elif lb == lc:
                o, p, q, lo, lpq = a, b, c, la, lb
            else:  # la == lc
                o, p, q, lo, lpq = b, a, c, lb, la
            op = cut_vertex(o, p)
            oq = cut_vertex(o, q)
            # odd corner's little triangle
            out_faces.append((o, op, oq)); out_labels.append(lo)
            # remaining quad (op, p, q, oq) → two triangles
            out_faces.append((op, p, q)); out_labels.append(lpq)
            out_faces.append((op, q, oq)); out_labels.append(lpq)
        else:
            # 3-way triple point (rare): cut all edges + a centroid vertex.
            pab = cut_vertex(a, b)
            pbc = cut_vertex(b, c)
            pca = cut_vertex(c, a)
            cen = n_v + len(new_pos)
            new_pos.append((V[a] + V[b] + V[c]) / 3.0)
            new_col.append((VC[a] + VC[b] + VC[c]) / 3.0)
            out_faces.append((a, pab, cen)); out_labels.append(la)
            out_faces.append((a, cen, pca)); out_labels.append(la)
            out_faces.append((b, pbc, cen)); out_labels.append(lb)
            out_faces.append((b, cen, pab)); out_labels.append(lb)
            out_faces.append((c, pca, cen)); out_labels.append(lc)
            out_faces.append((c, cen, pbc)); out_labels.append(lc)

    if new_pos:
        V2 = np.vstack([V, np.asarray(new_pos, dtype=np.float64)])
        C2 = np.vstack([VC, np.asarray(new_col, dtype=np.float64)])
    else:
        V2, C2 = V, VC
    F2 = np.asarray(out_faces, dtype=np.int64)
    L2 = np.asarray(out_labels, dtype=np.int32)
    C2 = np.clip(np.rint(C2), 0, 255).astype(np.uint8)
    return V2, F2, C2, L2


def split_mesh_by_color(
    mesh: trimesh.Trimesh,
    *,
    n_colors: int = 6,
    min_part_frac: float = 0.02,
) -> tuple[list[trimesh.Trimesh], list[np.ndarray]]:
    """Cluster the mesh's per-vertex colour and split it into one sub-mesh per
    colour group.

    This is the single source of truth for colour segmentation used by BOTH the
    multi-object GLB export (:func:`segment_mesh_parts`) and the per-colour 3MF
    export, so the two formats always produce the same colour groups.

    Returns
    -------
    (submeshes, colors) : each ``colors[i]`` is the uint8 RGB dominant colour of
    ``submeshes[i]``. Falls back to ``([mesh], [dominant])`` when the mesh has no
    usable per-vertex colour or resolves to a single colour.
    """
    faces = np.asarray(mesh.faces)

    # Per-vertex RGB is required for colour clustering.
    try:
        vcol = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    except Exception as exc:
        print(f"[PARTS] Mesh has no per-vertex colour ({exc}); single object.")
        return [mesh], [_dominant_rgb(mesh)]

    if len(vcol) != len(mesh.vertices) or len(faces) == 0:
        print("[PARTS] Missing vertex colours / faces; single object.")
        return [mesh], [_dominant_rgb(mesh)]

    vert_labels, n_clusters = _color_cluster_labels(vcol, n_colors)
    print(f"[PARTS] Colour clustering → {n_clusters} colour group(s).")
    if n_clusters <= 1:
        print("[PARTS] Only one colour found — single object.")
        return [mesh], [_dominant_rgb(mesh)]

    # Assign each face the majority colour label of its three vertices.
    # Vectorised (a pure-Python per-face loop would take minutes / appear to
    # hang on a real 100k+-face mesh): with three labels a "majority" is any
    # label that appears at least twice; if all three differ, keep the first.
    fl = vert_labels[faces]            # (F, 3)
    a, b, c = fl[:, 0], fl[:, 1], fl[:, 2]
    face_labels = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))
    face_labels = face_labels.astype(np.int32)

    # The mesh colour is continuous (photographic), so the raw per-face labels
    # are heavily speckled. Smooth them over the face-adjacency graph so each
    # colour group becomes contiguous regions instead of hundreds of tiny
    # disconnected patches (which would look like the mesh is full of holes).
    # Feed the per-face Lab colour so the smoothing is edge-aware and snaps the
    # part borders onto real colour transitions (colour-edge detection).
    face_lab = _face_mean_lab(vcol, faces)
    face_labels = _smooth_face_labels(
        mesh, face_labels, n_clusters, iters=8, face_colors=face_lab
    )
    # Guard the neutral↔chromatic colour boundary: on a shaded reconstruction the
    # crevice between a neutral part (e.g. a white cushion) and a chromatic part
    # (a rust chair) bakes in a little colour bleed, so lightness-driven
    # clustering erodes the shaded edge of the neutral part into the coloured
    # one. This constraint keeps genuinely near-neutral faces in the neutral
    # part (and vice-versa) so the boundary stays complete. ``None`` (no-op) when
    # the mesh isn't a neutral+chromatic mix, leaving all-colour meshes untouched.
    constrain = _make_chroma_gate(face_lab, face_labels, n_clusters)
    if constrain is not None:
        face_labels = constrain(face_labels)
    # Regularise the part contours: an MRF/graph-cut pass (colour data term +
    # contrast-weighted boundary penalty) turns the ragged colour-cluster seams
    # into clean cuts that stay locked on the real colour edge.
    face_labels = _mrf_refine_labels(
        mesh, face_labels, face_lab, n_clusters, constrain=constrain
    )
    if constrain is not None:
        face_labels = _smooth_face_labels(
            mesh, face_labels, n_clusters, iters=6, face_colors=face_lab
        )
        face_labels = constrain(face_labels)
        # Final contour cleanup. The per-face chroma gate above decides colour
        # face-by-face, so along an ambiguous shaded crease (e.g. where the white
        # pillow meets the rust seat) it leaves single-face "teeth" poking across
        # the part boundary. A short *uniform* (edge-agnostic) majority vote
        # straightens the contour and dissolves those isolated teeth without
        # moving the bulk boundary — and crucially without re-applying the gate,
        # which is exactly what stamped the teeth back in.
        face_labels = _smooth_face_labels(
            mesh, face_labels, n_clusters, iters=2, face_colors=None
        )

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

    # Dissolve small disconnected islands so each part is a few large contiguous
    # patches rather than speckle (the main cause of the "holey" look).
    min_comp = max(1, int(0.01 * len(faces)))
    face_labels = _merge_small_components(mesh, face_labels, min_comp)

    present = [int(lbl) for lbl in np.unique(face_labels)]
    if len(present) <= 1:
        print("[PARTS] Colours merged into one part — single object.")
        return [mesh], [_dominant_rgb(mesh)]

    # Re-tessellate the part boundaries so the cut follows a smooth iso-contour
    # instead of the ragged per-triangle staircase. This inserts shared seam
    # vertices (one per boundary edge) so the parts share an identical boundary
    # and the mesh stays watertight. Guarded: on any failure keep the plain
    # per-face cut, which is always valid.
    cut_mesh = mesh
    try:
        V2, F2, C2, L2 = _cut_faces_along_label_boundary(mesh, face_labels)
        cut_mesh = trimesh.Trimesh(
            vertices=V2, faces=F2, vertex_colors=C2, process=False
        )
        faces = F2
        vcol = C2[:, :3]
        face_labels = L2
        present = [int(lbl) for lbl in np.unique(face_labels)]
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[PARTS] Boundary re-tessellation skipped ({exc}); per-face cut.")

    face_groups = [np.where(face_labels == lbl)[0] for lbl in present]
    submeshes = cut_mesh.submesh(face_groups, only_watertight=False, append=False)

    # Dominant (median) colour of each group — the flat colour each part /
    # 3MF object is painted with, so GLB objects and 3MF colours match.
    colors = [
        np.median(vcol[faces[grp].reshape(-1)], axis=0).astype(np.uint8)
        for grp in face_groups
    ]

    print(f"[PARTS] Split mesh into {len(submeshes)} colour parts.")
    return list(submeshes), colors


def _smooth_part_vertex_colors(sub: "trimesh.Trimesh", iterations: int = 3, lam: float = 0.5) -> None:
    """De-speckle a part's per-vertex colours in place (before baking).

    The Gaussian appearance field leaves high-frequency **salt-and-pepper**
    colour noise across the surface — isolated near-black / near-white vertices
    that show up as speckle when baked into a texture. A Laplacian (mean) pass
    only smears those outliers into their neighbours; a **median** filter over
    the 1-ring neighbourhood rejects them outright while preserving the part's
    real colour gradient and edges. We run a few median passes (kills speckle),
    then one light Laplacian pass (removes any residual graininess).
    Operates on ``sub.visual.vertex_colors``; no-op if there are none."""
    try:
        vc = np.asarray(sub.visual.vertex_colors)
    except Exception:
        return
    if vc is None or len(vc) == 0 or len(sub.faces) == 0:
        return
    V = len(sub.vertices)
    rgb = vc[:, :3].astype(np.float32)
    alpha = vc[:, 3:4] if vc.shape[1] == 4 else np.full((V, 1), 255, np.uint8)

    faces = np.asarray(sub.faces)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    # Undirected 1-ring neighbours plus a self-loop, de-duplicated.
    i = np.concatenate([edges[:, 0], edges[:, 1], np.arange(V)])
    j = np.concatenate([edges[:, 1], edges[:, 0], np.arange(V)])
    pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    i, j = pairs[:, 0], pairs[:, 1]

    # Group neighbours per vertex into a padded (V, maxdeg) index matrix so the
    # per-vertex median can be computed vectorised with np.nanmedian.
    order = np.argsort(i, kind="stable")
    i, j = i[order], j[order]
    counts = np.bincount(i, minlength=V)
    maxdeg = int(counts.max()) if len(counts) else 0
    if maxdeg == 0:
        return
    starts = np.zeros(V + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    pos = np.arange(len(i)) - starts[i]
    nbr = np.full((V, maxdeg), -1, dtype=np.int64)
    nbr[i, pos] = j
    nbr_mask = nbr >= 0

    # Median passes: replace each vertex colour with the median of its 1-ring
    # (including itself). Median rejects the isolated outliers cleanly.
    for _ in range(int(iterations)):
        gathered = np.where(nbr_mask[..., None], rgb[nbr], np.nan)  # (V, maxdeg, 3)
        rgb = np.nanmedian(gathered, axis=1).astype(np.float32)

    # One light Laplacian (mean) pass to remove residual graininess.
    deg = counts.astype(np.float32)
    deg[deg == 0] = 1.0
    from scipy.sparse import coo_matrix
    adj = coo_matrix((np.ones(len(i), np.float32), (i, j)), shape=(V, V)).tocsr()
    nbr_mean = (adj @ rgb) / deg[:, None]
    rgb = rgb * (1.0 - lam) + nbr_mean * lam

    out_rgb = np.clip(rgb, 0, 255).round().astype(np.uint8)
    sub.visual.vertex_colors = np.concatenate([out_rgb, alpha], axis=1)


def _decimate_part_for_bake(sub: "trimesh.Trimesh", target_faces: int = 12000) -> "trimesh.Trimesh":
    """Quadric-decimate a part to ``target_faces`` before baking (colour-aware).

    Reconstructed parts are extremely dense (100k+ tiny triangles). Baking such a
    mesh is both **slow** (xatlas unwrap + a per-face rasterisation loop scale
    with face count — tens of seconds per part) and **speckled** (with only a
    handful of texels per triangle, adjacent UV islands bleed together under the
    viewer's bilinear/mip filtering). Decimating first gives large triangles with
    many texels each: the bake is ~100× faster and essentially speckle-free. The
    surface is locally smooth, so the decimation keeps the visible colour detail.

    Returns the decimated mesh (with per-vertex colours carried over), or ``sub``
    unchanged if it is already small enough or Open3D is unavailable."""
    try:
        faces = np.asarray(sub.faces)
        if len(faces) <= target_faces:
            return sub
        import open3d as o3d

        me = o3d.geometry.TriangleMesh()
        me.vertices = o3d.utility.Vector3dVector(np.asarray(sub.vertices, dtype=np.float64))
        me.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        vc = np.asarray(sub.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        me.vertex_colors = o3d.utility.Vector3dVector(vc)
        dec = me.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))

        V = np.asarray(dec.vertices)
        F = np.asarray(dec.triangles)
        if len(F) == 0 or len(V) == 0:
            return sub
        C = np.clip(np.asarray(dec.vertex_colors) * 255.0, 0, 255).astype(np.uint8)
        out = trimesh.Trimesh(vertices=V, faces=F, process=False)
        out.visual.vertex_colors = np.concatenate(
            [C, np.full((len(C), 1), 255, np.uint8)], axis=1
        )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[PARTS] decimation skipped ({exc})")
        return sub


def _vertex_color_part_geometry(sub: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    """Return ``sub`` keeping its per-vertex colours with a matte white material.

    This matches what the in-app preview and the unsegmented GLB look like:
    three.js / model-viewer multiply the white baseColorFactor by COLOR_0 and
    Gouraud-interpolate across each triangle, so the surface reads smooth. (The
    UV-texture bake path, by contrast, samples the noisy per-vertex colours
    per-texel and comes out speckled.) The colours are Laplacian-smoothed first
    to knock down the Gaussian field's high-frequency mottling.

    Trade-off: macOS Quick Look / RealityKit ignore COLOR_0 when a material is
    present and show the white baseColorFactor — so this path is for
    three.js/model-viewer/Blender, not Quick Look. The solid path
    (``_solid_part_geometry``) is the Quick Look-safe option."""
    geom = sub.copy()
    _smooth_part_vertex_colors(geom)
    vc = np.asarray(geom.visual.vertex_colors)
    # Pure COLOR_0, NO material: trimesh silently DROPS the per-vertex colours on
    # GLB export as soon as a material is attached (it re-classes the visual as a
    # texture), which is what turned the parts grey. We therefore leave a bare
    # ``ColorVisuals`` here and let the server's ``_patch_glb_metallic`` inject the
    # matte white material (baseColorFactor=white × COLOR_0) at serve time — the
    # same path the smooth unsegmented GLB uses.
    geom.visual = trimesh.visual.ColorVisuals(mesh=geom, vertex_colors=vc)
    return geom


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


_BASIC_COLORS = [
    ("black", (20, 20, 20)),   ("white", (240, 240, 240)),
    ("grey", (128, 128, 128)), ("red", (200, 40, 40)),
    ("orange", (230, 130, 40)),("brown", (120, 75, 45)),
    ("rust", (155, 75, 45)),   ("tan", (205, 175, 135)),
    ("yellow", (220, 200, 50)),("green", (50, 150, 65)),
    ("blue", (50, 80, 185)),   ("purple", (120, 55, 150)),
    ("pink", (230, 135, 165)),
]


def slugify_label(label: Optional[str]) -> str:
    """Filesystem/identifier-safe slug from a free-text concept (``"Blue Chair"``
    → ``"blue_chair"``). Returns ``""`` for empty/None input."""
    import re

    s = re.sub(r"[^a-z0-9]+", "_", (label or "").strip().lower()).strip("_")
    return s[:48]


def _color_name(rgb: np.ndarray) -> str:
    """Nearest basic colour name for a uint8 RGB triple."""
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return min(
        _BASIC_COLORS,
        key=lambda nc: (r - nc[1][0]) ** 2 + (g - nc[1][1]) ** 2 + (b - nc[1][2]) ** 2,
    )[0]


_SEATING = ("armchair", "chair", "sofa", "couch", "loveseat", "bench",
            "stool", "recliner", "throne", "seat", "ottoman", "settee")
_TABLE = ("coffee table", "dining table", "table", "desk", "nightstand", "workbench")
_LAMP = ("floor lamp", "desk lamp", "lamp", "lampshade")


def _concept_class(concept: Optional[str]) -> Optional[str]:
    """Map a free-text concept to a coarse part-vocabulary class."""
    c = (concept or "").lower()
    if any(k in c for k in _SEATING):
        return "seating"
    if any(k in c for k in _TABLE):
        return "table"
    if any(k in c for k in _LAMP):
        return "lamp"
    return None


def _part_features(submeshes: List[trimesh.Trimesh]) -> List[dict]:
    """Per-part geometry features normalised within the whole object's bbox
    (GLB is Y-up): vertical position ``y`` (0 bottom … 1 top), vertical/height
    extents, horizontal footprint, area fraction, verticality and connected-
    component count. Used to assign functional part names heuristically."""
    allv = np.vstack([np.asarray(s.vertices, dtype=np.float64) for s in submeshes])
    lo, hi = allv.min(0), allv.max(0)
    span = np.maximum(hi - lo, 1e-6)
    tot_area = sum(float(s.area) for s in submeshes) + 1e-9
    feats = []
    for s in submeshes:
        V = np.asarray(s.vertices, dtype=np.float64)
        cen = V.mean(0)
        yext = (V[:, 1].max() - V[:, 1].min()) / span[1]
        xext = (V[:, 0].max() - V[:, 0].min()) / span[0]
        zext = (V[:, 2].max() - V[:, 2].min()) / span[2]
        horiz = max(xext, zext)
        try:
            ncomp = len(s.split(only_watertight=False))
        except Exception:
            ncomp = 1
        feats.append({
            "y": float((cen[1] - lo[1]) / span[1]),
            "yext": float(yext),
            "horiz": float(horiz),
            "rel": float(s.area) / tot_area,
            "vert": float(yext / max(horiz, 1e-6)),
            "ncomp": int(ncomp),
            "color": _dominant_rgb(s),
        })
    return feats


def _name_seating(feats: List[dict]) -> List[Optional[str]]:
    """Assign chair/sofa-style functional names by vertical position and size:
    lowest part → ``legs``, a tall part on top → ``backrest``, a small upper
    part → ``pillow``, the largest remaining → ``seat``."""
    n = len(feats)
    names: List[Optional[str]] = [None] * n
    remaining = set(range(n))

    low = min(remaining, key=lambda i: feats[i]["y"])
    if feats[low]["y"] < 0.45:
        names[low] = "legs"
        remaining.discard(low)

    if remaining:
        top = max(remaining, key=lambda i: feats[i]["y"])
        if feats[top]["y"] > 0.55 and feats[top]["vert"] > 0.7 and feats[top]["rel"] > 0.15:
            names[top] = "backrest"
            remaining.discard(top)

    if remaining:
        cand = [i for i in remaining if feats[i]["rel"] < 0.30 and feats[i]["y"] > 0.35]
        if cand:
            p = min(cand, key=lambda i: feats[i]["rel"])
            names[p] = "pillow"
            remaining.discard(p)

    if remaining:
        s = max(remaining, key=lambda i: feats[i]["rel"])
        names[s] = "seat"
        remaining.discard(s)
    return names


def _name_table(feats: List[dict]) -> List[Optional[str]]:
    """Table/desk: lowest thin part → ``legs``, the flat part on top → ``top``."""
    n = len(feats)
    names: List[Optional[str]] = [None] * n
    remaining = set(range(n))
    low = min(remaining, key=lambda i: feats[i]["y"])
    if feats[low]["y"] < 0.5:
        names[low] = "legs"
        remaining.discard(low)
    if remaining:
        t = max(remaining, key=lambda i: feats[i]["y"])
        names[t] = "top"
        remaining.discard(t)
    return names


def _name_lamp(feats: List[dict]) -> List[Optional[str]]:
    """Lamp: lowest → ``base``, top → ``shade``, a tall middle part → ``stem``."""
    n = len(feats)
    names: List[Optional[str]] = [None] * n
    remaining = set(range(n))
    low = min(remaining, key=lambda i: feats[i]["y"])
    names[low] = "base"
    remaining.discard(low)
    if remaining:
        top = max(remaining, key=lambda i: feats[i]["y"])
        names[top] = "shade"
        remaining.discard(top)
    if remaining:
        m = max(remaining, key=lambda i: feats[i]["vert"])
        names[m] = "stem"
        remaining.discard(m)
    return names


# SAM 3 text prompts per object class. Each entry is ``(prompt, part_name)``:
# the phrase sent to SAM 3 and the clean name stored for the part it lands on.
_SAM3_PART_PROMPTS = {
    "seating": [
        ("seat cushion", "cushion"),
        ("pillow", "pillow"),
        ("backrest", "backrest"),
        ("arm rest", "armrest"),
        ("chair legs", "legs"),
        ("seat", "seat"),
    ],
    "table": [
        ("table top", "top"),
        ("table legs", "legs"),
        ("drawer", "drawer"),
    ],
    "lamp": [
        ("lampshade", "shade"),
        ("lamp base", "base"),
        ("lamp stem", "stem"),
    ],
}

# 3/4 front, 3/4 back-ish, side and a low view (to expose legs / base).
_SAM3_VIEWS = ((12.0, 30.0), (12.0, 150.0), (12.0, -90.0), (-35.0, 20.0))


def _combined_parts_mesh(
    submeshes: List[trimesh.Trimesh],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate parts into one mesh, tagging every face with its part index.

    Returns ``(V, F, C01, face_part)`` where ``C01`` is float RGB in [0, 1] and
    ``face_part[f]`` is the index of the sub-mesh face ``f`` came from — so a
    render's ``pix_to_face`` maps straight back to the owning part."""
    V, F, C, P = [], [], [], []
    off = 0
    for i, s in enumerate(submeshes):
        v = np.asarray(s.vertices, dtype=np.float32)
        f = np.asarray(s.faces, dtype=np.int64) + off
        try:
            c = np.asarray(s.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        except Exception:
            c = np.full((len(v), 3), 0.75, dtype=np.float32)
        V.append(v)
        F.append(f)
        C.append(c)
        P.append(np.full(len(f), i, dtype=np.int32))
        off += len(v)
    return (
        np.vstack(V),
        np.vstack(F),
        np.vstack(C).astype(np.float32),
        np.concatenate(P),
    )


def _render_parts_faceids(
    V: np.ndarray,
    F: np.ndarray,
    C01: np.ndarray,
    face_part: np.ndarray,
    views: tuple,
    image_size: int,
) -> List[tuple]:
    """Render the combined mesh from ``views`` (PyTorch3D, CPU).

    Returns one ``(rgb, part_id)`` per view: ``rgb`` is an (H, W, 3) uint8 image
    of the true surface colour (so SAM 3 sees the object as-is) and ``part_id``
    is an (H, W) int map of which part owns each pixel (``-1`` = background)."""
    import torch
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        MeshRasterizer,
        RasterizationSettings,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    # Centre + scale to a 0.9-radius sphere so an orthographic camera frames it.
    lo, hi = V.min(0), V.max(0)
    cen = (lo + hi) * 0.5
    v = V - cen
    radius = float(np.linalg.norm(v, axis=1).max()) or 1.0
    Vn = (v / radius * 0.9).astype(np.float32)

    dv = torch.from_numpy(Vn)
    df = torch.from_numpy(F.astype(np.int64))
    dc = torch.from_numpy(C01)
    fp = torch.from_numpy(face_part.astype(np.int64))
    meshes = Meshes(verts=[dv], faces=[df])
    raster = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    H = W = image_size

    out = []
    for elev, azim in views:
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
        cam = FoVOrthographicCameras(R=R, T=T)
        frag = MeshRasterizer(cameras=cam, raster_settings=raster)(meshes)
        ptf = frag.pix_to_face[0, ..., 0]
        bary = frag.bary_coords[0, ..., 0, :]
        valid = ptf >= 0
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)
        pid = np.full((H, W), -1, dtype=np.int32)
        if valid.any():
            fidx = ptf[valid]
            vcols = dc[df[fidx]]
            b = bary[valid].unsqueeze(-1)
            col = (b * vcols).sum(dim=1)
            vm = valid.numpy()
            rgb[vm] = (col.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            pid[vm] = fp[fidx].numpy()
        out.append((rgb, pid))
    return out


def _sam3_semantic_names(
    submeshes: List[trimesh.Trimesh],
    concept: Optional[str],
    views: tuple = _SAM3_VIEWS,
    image_size: int = 512,
    min_frac: float = 0.12,
    verbose: bool = True,
) -> List[Optional[str]]:
    """Label parts semantically with SAM 3 (render → text-segment → back-project).

    The parts are rendered from a few views; for every part-name prompt in the
    object's vocabulary (``"chair legs"``, ``"seat"``, ``"pillow"`` …) SAM 3's
    text/concept segmentation is run on each render, and its mask is back-
    projected via the per-pixel ``part_id`` buffer to whichever 3-D part it
    covers most. Each prompt is assigned (greedily, uniquely) to its best-
    covered part above ``min_frac`` of that part's visible pixels.

    Returns an ``Optional[str]`` per part (``None`` where SAM 3 was not confident
    or the object class has no vocabulary). Any failure — no cached SAM 3
    weights, no PyTorch3D, a render/inference error — returns all ``None`` so the
    caller falls back to the geometric heuristic."""
    n = len(submeshes)
    cls = _concept_class(concept)
    prompts = _SAM3_PART_PROMPTS.get(cls or "")
    if not prompts or n == 0:
        return [None] * n
    try:
        import sam_wrapper

        V, F, C01, face_part = _combined_parts_mesh(submeshes)
        renders = _render_parts_faceids(V, F, C01, face_part, views, image_size)

        overlap = {name: np.zeros(n, dtype=np.int64) for _, name in prompts}
        part_px = np.zeros(n, dtype=np.int64)
        for rgb, pid in renders:
            for p in range(n):
                part_px[p] += int(np.count_nonzero(pid == p))
            for prompt, name in prompts:
                try:
                    m = sam_wrapper.predict_mask_text(rgb, prompt)
                except Exception as exc:
                    if verbose:
                        print(f"[SAM3-PARTS] '{prompt}' inference failed: {exc}")
                    continue
                if m is None or int(m.max()) == 0:
                    continue
                mm = m > 0
                for p in range(n):
                    overlap[name][p] += int(np.count_nonzero(mm & (pid == p)))

        cand = []
        for name, ov in overlap.items():
            for p in range(n):
                if part_px[p] <= 0:
                    continue
                frac = ov[p] / part_px[p]
                if frac >= min_frac:
                    cand.append((frac, name, p))
        cand.sort(reverse=True)

        names: List[Optional[str]] = [None] * n
        used = set()
        for frac, name, p in cand:
            if names[p] is None and name not in used:
                names[p] = name
                used.add(name)
                if verbose:
                    print(f"[SAM3-PARTS] part {p} → {name} (coverage {frac:.0%})")
        if verbose and not used:
            print("[SAM3-PARTS] no confident part labels; using geometric names.")
        return names
    except Exception as exc:
        if verbose:
            print(f"[SAM3-PARTS] labeling skipped ({exc}); using geometric names.")
        return [None] * n


def _highlight_part_image(
    rgb: np.ndarray,
    pid: np.ndarray,
    part: int,
    dim: float = 0.35,
    alpha: float = 0.55,
) -> np.ndarray:
    """Composite a render that visually isolates one part for the VLM.

    The target part is tinted red (blended over its true colour) while the rest
    of the object is darkened, so Moondream's attention is drawn to the part
    being asked about; the white background is left untouched."""
    out = rgb.astype(np.float32)
    region = pid == part
    other = (pid >= 0) & ~region
    out[other] *= float(dim)
    red = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    a = float(alpha)
    out[region] = out[region] * (1.0 - a) + red * a
    return out.clip(0, 255).astype(np.uint8)


# Words dropped from a VLM answer because they describe the whole object / are
# filler, not a part name.
_VLM_STOPWORDS = {"the", "a", "an", "this", "that", "is", "it", "of", "part",
                  "highlighted", "red", "in", "colour", "color", "object", "model"}


def _clean_part_phrase(answer: Optional[str], obj_label: str) -> Optional[str]:
    """Turn Moondream's free-text answer into a clean, unique part slug.

    Strips filler / prefixes, keeps the first few descriptive words, drops the
    object noun itself (so "chair leg" → "leg" but "seat cushion" is kept), and
    rejects answers that are just the whole object (e.g. "a chair") or empty.

    Also rejects degenerate small-VLM output: repeated tokens
    ("unter unter"), overlong merged fragments ("herselfisticistic…"), and
    vowel-less non-words — these fall back to the colour-name label instead of
    being saved as gibberish part names."""
    import re

    if not answer:
        return None
    words = re.findall(r"[a-z]+", str(answer).lower())
    obj = re.findall(r"[a-z]+", obj_label.lower())
    # Drop filler and the object's own noun(s); keep at most three words.
    kept = [w for w in words if w not in _VLM_STOPWORDS and w not in obj]
    if not kept:
        # Answer was only the object / filler — not a distinct part.
        return None

    # Repeated words ("unter unter", "maint maint") are degenerate sampling
    # loops, not a part name — reject before any collapsing.
    if len(kept) > 1 and len(set(kept)) < len(kept):
        return None
    kept = kept[:3]

    def _degenerate(w: str) -> bool:
        # Long merged fragment, an absurdly long single token, a vowel-less
        # non-word, a repeated 3+ char chunk ("isticistic"), or a tiny fragment
        # ("st") that can't be a real part name.
        if len(w) > 12 or len(w) < 3:
            return True
        if len(w) > 4 and not re.search(r"[aeiouy]", w):
            return True
        if re.search(r"(.{3,})\1", w):
            return True
        return False

    if any(_degenerate(w) for w in kept):
        return None
    return slugify_label(" ".join(kept)) or None


def _moondream_semantic_names(
    submeshes: List[trimesh.Trimesh],
    concept: Optional[str],
    views: tuple = _SAM3_VIEWS,
    image_size: int = 512,
    min_px_frac: float = 0.02,
    verbose: bool = True,
) -> List[Optional[str]]:
    """Name parts in **open vocabulary** by asking Moondream2 about renders.

    Each part is rendered highlighted (best of ``views`` by visible pixels) and
    Moondream2 is asked what that highlighted part of the object is called. The
    label therefore comes from the image, not a fixed per-class prompt list.

    Returns an ``Optional[str]`` per part (``None`` where the model is not
    downloaded, a part is too occluded to name, or the answer was just the
    object). Any failure returns all ``None`` so the caller falls back to the
    geometric / colour heuristic."""
    n = len(submeshes)
    if n == 0:
        return []
    try:
        import vlm_labeler

        if not vlm_labeler.available():
            if verbose:
                print("[VLM-PARTS] SmolVLM2 not downloaded; using geometric names.")
            return [None] * n

        V, F, C01, face_part = _combined_parts_mesh(submeshes)
        renders = _render_parts_faceids(V, F, C01, face_part, views, image_size)

        # Pick, per part, the view where it shows the most pixels (least occluded).
        best_view = [-1] * n
        best_px = [0] * n
        for vi, (_rgb, pid) in enumerate(renders):
            for p in range(n):
                c = int(np.count_nonzero(pid == p))
                if c > best_px[p]:
                    best_px[p] = c
                    best_view[p] = vi

        obj_label = (concept or "").strip() or "object"
        total = image_size * image_size
        names: List[Optional[str]] = [None] * n
        for p in range(n):
            vi = best_view[p]
            if vi < 0 or best_px[p] < min_px_frac * total:
                continue  # never visible enough to name reliably
            rgb, pid = renders[vi]
            img = _highlight_part_image(rgb, pid, p)
            question = (
                f"This is a {obj_label}. Name the red-highlighted part. "
                f"Answer with one common noun, like 'leaf', 'trunk', 'pot', "
                f"'leg', or 'seat'. One word only."
            )
            try:
                ans = vlm_labeler.name_image(img, question)
            except Exception as exc:
                if verbose:
                    print(f"[VLM-PARTS] part {p} query failed: {exc}")
                continue
            nm = _clean_part_phrase(ans, obj_label)
            if nm:
                names[p] = nm
                if verbose:
                    print(f"[VLM-PARTS] part {p} → {nm!r}  (from {ans!r})")
        if verbose and not any(names):
            print("[VLM-PARTS] no confident labels; using geometric names.")
        return names
    except Exception as exc:
        if verbose:
            print(f"[VLM-PARTS] labeling skipped ({exc}); using geometric names.")
        return [None] * n
    finally:
        # Release the ~4 GB VLM as soon as naming is done so it is never held
        # resident into a later reconstruction (which peaks unified memory).
        try:
            import vlm_labeler

            vlm_labeler.unload()
        except Exception:
            pass


def _semantic_part_names(
    submeshes: List[trimesh.Trimesh],
    concept: Optional[str] = None,
    use_sam3: bool = False,
) -> List[str]:
    """Assign a readable name to every part.

    When ``use_sam3`` is set, **Moondream2** is asked, in open vocabulary, what
    each rendered part is called — the label comes from the image, not a fixed
    per-class word list (see :func:`_moondream_semantic_names`). Parts the VLM
    can't name (or if the model isn't downloaded) fall back to a geometric
    heuristic keyed off the object class (chair/table/lamp), inferred from each
    part's vertical position, size and shape. Anything still unnamed — and every
    part of an unknown object — falls back to its dominant colour name
    (``rust``, ``white`` …). Duplicate names are disambiguated with a numeric
    suffix so every part name stays unique."""
    n = len(submeshes)
    if n == 0:
        return []
    feats = _part_features(submeshes)

    sam_names: List[Optional[str]] = [None] * n
    if use_sam3:
        sam_names = _moondream_semantic_names(submeshes, concept)

    cls = _concept_class(concept)
    if cls == "seating":
        geo = _name_seating(feats)
    elif cls == "table":
        geo = _name_table(feats)
    elif cls == "lamp":
        geo = _name_lamp(feats)
    else:
        geo = [None] * n

    # Prefer the VLM's open-vocabulary labels; only use the geometric guess for
    # parts the VLM didn't name, and never reuse a name the VLM already assigned.
    used_sam = {s for s in sam_names if s}
    names: List[Optional[str]] = []
    for i in range(n):
        nm = sam_names[i]
        if nm is None and geo[i] not in used_sam:
            nm = geo[i]
        names.append(nm)

    for i in range(n):
        if names[i] is None:
            names[i] = _color_name(feats[i]["color"])

    seen: dict = {}
    out: List[str] = []
    for nm in names:
        if nm in seen:
            seen[nm] += 1
            out.append(f"{nm}_{seen[nm]}")
        else:
            seen[nm] = 1
            out.append(nm)
    return out


def _scene_from_parts(
    submeshes: List[trimesh.Trimesh],
    image: np.ndarray,
    mask: np.ndarray,
    bake: bool,
    texture_size: int,
    label: Optional[str] = None,
    use_sam3: bool = False,
) -> trimesh.Scene:
    """Assemble named part sub-meshes into a Scene.

    Default (fast): each part gets a matte material tinted with its dominant
    colour so it renders correctly in every viewer (see ``_solid_part_geometry``).
    When ``bake`` is set, a per-part UV texture atlas is baked from the part's
    own reconstructed per-vertex colours (keeps within-part colour variation but
    is slower). The atlas is derived **only** from the generated mesh — the input
    photo is never sampled — so parts keep the true object colour on every face.

    ``label`` (the object concept, e.g. ``"chair"``) drives functional part
    names and the scene's root-node name, so the exported GLB has a meaningful
    root (``chair``) instead of the generic ``world`` frame. ``use_sam3`` runs
    Moondream2 on renders of the parts to assign open-vocabulary semantic names
    from the image instead of the geometric heuristic (see
    :func:`_moondream_semantic_names`)."""
    base_frame = slugify_label(label) or "model"
    part_names = _semantic_part_names(submeshes, label, use_sam3=use_sam3)
    scene = trimesh.Scene(base_frame=base_frame)
    for i, sub in enumerate(submeshes):
        name = part_names[i]
        geom = None
        if bake:
            # "Bake texture (Quick Look)": bake a UV texture atlas from the
            # part's own per-vertex colours. Baking a real ``baseColorTexture``
            # is what makes the segmented GLB show colour in EVERY viewer —
            # including macOS Quick Look / Preview / USDZ, which ignore per-vertex
            # COLOR_0 and render a plain material grey otherwise.
            #
            # The bake uses a flat cell-per-triangle GRID atlas (see
            # ``bake_vertex_color_texture``): every triangle gets its own solid
            # cell and all three UVs point at the cell centre, so there are no
            # UV-chart seams or gutters to bleed — no speckle, in any viewer. It
            # needs no decimation (it's O(faces) and never packs charts together)
            # and no texture-space de-speckle (each cell is already one flat
            # colour). Colours are still lightly smoothed first to tame per-vertex
            # salt-and-pepper noise from the reconstruction.
            try:
                from texture_baking import bake_vertex_color_texture

                _smooth_part_vertex_colors(sub)
                geom = bake_vertex_color_texture(sub, texture_size=texture_size)
            except Exception as exc:
                print(f"[PARTS] {name}: texture bake failed ({exc}); using per-vertex colour")
                geom = None

        if geom is None:
            # Default (and bake fallback): keep the part's own per-vertex COLOR_0,
            # exactly like the in-app preview and the unsegmented GLB. No UV atlas
            # is created, so there is no chart-gutter / front-back colour bleed —
            # the surface renders with the clean, smooth reconstruction colour in
            # three.js / model-viewer / Blender. (macOS Quick Look ignores COLOR_0
            # and shows grey; enable "bake" for a Quick Look-safe texture atlas.)
            geom = _vertex_color_part_geometry(sub, name)

        scene.add_geometry(geom, geom_name=name)
        _baked = (
            bake and geom is not None
            and isinstance(geom.visual, trimesh.visual.TextureVisuals)
            and getattr(geom.visual.material, "baseColorTexture", None) is not None
        )
        print(f"[PARTS] {name}: {len(geom.faces)} faces, {len(geom.vertices)} verts"
              f"{' (baked)' if _baked else ' (vertex-colour)'}")
    return scene

