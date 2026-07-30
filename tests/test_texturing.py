"""Tests for GLB texturing: vertex-color passthrough and portable UV baking.

These exercise the pure geometry/color paths in ``postprocessing_utils`` on
synthetic meshes — no model weights, no CUDA, runs in a second.
"""
import numpy as np
import torch
import pytest
import trimesh

from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils as pu


class FakeMesh:
    """Minimal stand-in for MeshExtractResult (what to_glb consumes)."""

    def __init__(self, vertices, faces, vertex_attrs=None):
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32)
        self.faces = torch.as_tensor(faces, dtype=torch.int64)
        self.vertex_attrs = (
            torch.as_tensor(vertex_attrs, dtype=torch.float32)
            if vertex_attrs is not None
            else None
        )


def unit_cube():
    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


# --------------------------------------------------------------------------- #
# _resample_vertex_colors
# --------------------------------------------------------------------------- #

def test_resample_identity_when_unchanged():
    v = np.random.RandomState(0).rand(20, 3).astype(np.float32)
    c = np.random.RandomState(1).rand(20, 3).astype(np.float32)
    out = pu._resample_vertex_colors(v, c, v)
    assert out.shape == (20, 4)
    assert out.dtype == np.uint8
    assert (out[:, 3] == 255).all()
    # RGB preserved to within uint8 quantization
    assert np.allclose(out[:, :3] / 255.0, np.clip(c, 0, 1), atol=1.0 / 255 + 1e-6)


def test_resample_nearest_after_reindex():
    # Two clusters; a decimated set of query points near each cluster must pick
    # up that cluster's color.
    v = np.array([[0, 0, 0], [0.01, 0, 0], [5, 5, 5], [5.01, 5, 5]], np.float32)
    c = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], np.float32)
    q = np.array([[0.004, 0, 0], [5.004, 5, 5]], np.float32)
    out = pu._resample_vertex_colors(v, c, q)[:, :3]
    assert tuple(out[0]) == (255, 0, 0)
    assert tuple(out[2 - 1]) == (0, 0, 255)


def test_resample_accepts_255_range():
    v = np.zeros((3, 3), np.float32)
    v[1, 0] = 1.0
    v[2, 0] = 2.0
    c = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.float32)
    out = pu._resample_vertex_colors(v, c, v)[:, :3]
    assert tuple(out[0]) == (255, 0, 0)
    assert tuple(out[1]) == (0, 255, 0)


# --------------------------------------------------------------------------- #
# rasterize_uv_barycentric
# --------------------------------------------------------------------------- #

def test_rasterizer_barycentric_partition():
    uvs = np.array([[0, 0], [1, 0], [0, 1]], np.float64)
    faces = np.array([[0, 1, 2]], np.int64)
    fid, bary = pu.rasterize_uv_barycentric(uvs, faces, 32)
    covered = fid >= 0
    assert covered.sum() > 0
    # roughly half the atlas for a right triangle over the unit square
    assert 0.35 < covered.mean() < 0.65
    # barycentric weights are a partition of unity and non-negative
    b = bary[covered]
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-5)
    assert (b >= -1e-6).all()


def test_rasterizer_empty_atlas():
    uvs = np.zeros((3, 2), np.float64)  # degenerate triangle → no coverage
    faces = np.array([[0, 1, 2]], np.int64)
    fid, _ = pu.rasterize_uv_barycentric(uvs, faces, 16)
    assert (fid == -1).all()


# --------------------------------------------------------------------------- #
# bake_texture_portable (vertex-color source, Phase 2)
# --------------------------------------------------------------------------- #

def test_portable_bake_vertex_color():
    # single triangle filling most of UV space, three primary-colored corners
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    uvs = np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95]], np.float64)
    vc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    tex = pu.bake_texture_portable(verts, faces, uvs, 64, vert_colors=vc, inpaint=False)
    assert tex.shape == (64, 64, 3)
    assert tex.dtype == np.uint8
    # texture is not blank and contains saturated color near a corner
    assert tex.max() > 200
    # near the red corner (uv ~ (0.05,0.05) → col ~3, row ~60) red dominates
    patch = tex[57:63, 1:7].reshape(-1, 3).mean(0)
    assert patch[0] > patch[1] and patch[0] > patch[2]


def test_portable_bake_inpaint_fills_seams():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    uvs = np.array([[0.1, 0.1], [0.5, 0.1], [0.1, 0.5]], np.float64)
    vc = np.ones((3, 3), np.float32)
    tex = pu.bake_texture_portable(verts, faces, uvs, 48, vert_colors=vc, inpaint=True)
    # inpaint leaves no fully-black hole inside the atlas
    assert (tex.reshape(-1, 3).sum(1) == 0).mean() < 0.05


# --------------------------------------------------------------------------- #
# to_glb integration (Phase 1 vertex-color passthrough)
# --------------------------------------------------------------------------- #

def test_to_glb_attaches_vertex_color_no_postprocess():
    v, f = unit_cube()
    attrs = np.zeros((v.shape[0], 6), np.float32)
    attrs[:, 0] = 1.0  # solid red (sigmoid-range [0,1])
    mesh = FakeMesh(v, f, attrs)
    glb = pu.to_glb(
        None, mesh,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=True,
    )
    assert isinstance(glb, trimesh.Trimesh)
    assert hasattr(glb.visual, "vertex_colors")
    vc = np.asarray(glb.visual.vertex_colors)
    assert vc.shape[0] == glb.vertices.shape[0]
    # predominantly red
    assert vc[:, 0].mean() > 200 and vc[:, 1].mean() < 60


def test_to_glb_gray_when_no_color_requested():
    v, f = unit_cube()
    mesh = FakeMesh(v, f, None)
    glb = pu.to_glb(
        None, mesh,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=True,  # requested, but no attrs available
    )
    # no vertex_attrs → no color attached, but still a valid mesh
    assert isinstance(glb, trimesh.Trimesh)
    assert glb.vertices.shape[0] > 0


def test_to_glb_portable_vertex_bake_produces_texture():
    v, f = unit_cube()
    attrs = np.zeros((v.shape[0], 6), np.float32)
    attrs[:, 1] = 1.0  # green
    mesh = FakeMesh(v, f, attrs)
    glb = pu.to_glb(
        None, mesh,
        simplify=0.0,
        with_mesh_postprocess=False,
        with_texture_baking=True,
        bake_backend="portable",
        bake_source="vertex",
        texture_size=128,
    )
    assert isinstance(glb.visual, trimesh.visual.TextureVisuals)
    img = np.asarray(glb.visual.material.baseColorTexture)
    assert img.shape[:2] == (128, 128)
    # green channel dominates the baked atlas
    assert img[..., 1].mean() > img[..., 0].mean()
    assert img[..., 1].mean() > img[..., 2].mean()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
