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
    submeshes, _colors = split_mesh_into_parts(
        mesh, n_colors=n_colors, min_part_frac=min_part_frac
    )
    return _scene_from_parts(submeshes, image, mask, bake, texture_size)


def split_mesh_into_parts(
    mesh: trimesh.Trimesh,
    *,
    n_colors: int = 6,
    min_part_frac: float = 0.02,
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

    face_groups = [np.where(face_labels == lbl)[0] for lbl in present]
    submeshes = mesh.submesh(face_groups, only_watertight=False, append=False)

    # Dominant (median) colour of each group — the flat colour each part /
    # 3MF object is painted with, so GLB objects and 3MF colours match.
    colors = [
        np.median(vcol[faces[grp].reshape(-1)], axis=0).astype(np.uint8)
        for grp in face_groups
    ]

    print(f"[PARTS] Split mesh into {len(submeshes)} colour parts.")
    return list(submeshes), colors


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
    When ``bake`` is set, a per-part UV texture atlas is baked from the part's
    own reconstructed per-vertex colours (keeps within-part colour variation but
    is slower). The atlas is derived **only** from the generated mesh — the input
    photo is never sampled — so parts keep the true object colour on every face."""
    scene = trimesh.Scene()
    for i, sub in enumerate(submeshes):
        name = f"part_{i:02d}"
        geom = None
        if bake:
            try:
                from texture_baking import bake_vertex_color_texture

                # Bake the atlas purely from the part's own generated-mesh
                # per-vertex colours (no image projection), so every face keeps
                # the true object colour instead of a photo-tinted front.
                geom = bake_vertex_color_texture(sub, texture_size=texture_size)
            except Exception as exc:
                print(f"[PARTS] {name}: texture bake failed ({exc}); using solid colour")
                geom = None

        if geom is None:
            geom = _solid_part_geometry(sub, name)

        scene.add_geometry(geom, geom_name=name)
        print(f"[PARTS] {name}: {len(geom.faces)} faces, {len(geom.vertices)} verts"
              f"{' (baked)' if bake and isinstance(geom.visual, trimesh.visual.TextureVisuals) and getattr(geom.visual.material, 'baseColorTexture', None) is not None else ' (solid)'}")
    return scene

