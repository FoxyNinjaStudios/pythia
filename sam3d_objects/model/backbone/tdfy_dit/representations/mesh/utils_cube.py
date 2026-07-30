# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch

cube_corners = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.int,
)
cube_neighbor = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)
cube_edges = torch.tensor(
    [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4],
    dtype=torch.long,
    requires_grad=False,
)


def construct_dense_grid(res, device="cuda"):
    """construct a dense grid based on resolution"""
    res_v = res + 1
    vertsid = torch.arange(res_v**3, device=device)
    coordsid = vertsid.reshape(res_v, res_v, res_v)[:res, :res, :res].flatten()
    cube_corners_bias = (
        cube_corners[:, 0] * res_v + cube_corners[:, 1]
    ) * res_v + cube_corners[:, 2]
    cube_fx8 = coordsid.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device)
    verts = torch.stack(
        [vertsid // (res_v**2), (vertsid // res_v) % res_v, vertsid % res_v], dim=1
    )
    return verts, cube_fx8


def construct_voxel_grid(coords):
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    # torch.unique with dim is not implemented on MPS, use CPU fallback
    original_device = verts.device
    if verts.device.type == "mps":
        verts_cpu = verts.cpu()
        verts_unique, inverse_indices = torch.unique(verts_cpu, dim=0, return_inverse=True)
        verts_unique = verts_unique.to(original_device)
        inverse_indices = inverse_indices.to(original_device)
    else:
        verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce="mean"):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    device = value.device
    M = value.shape[2]  # number of channels

    # Fast CPU path: scatter_reduce is slower on CPU and has known MPS placeholder issues.
    # We implement mean/sum via index_add + bincount.
    if cubes.device.type == "cpu" and reduce in {"mean", "sum"}:
        idx = cubes.reshape(-1).to(dtype=torch.int64)
        src = value.reshape(-1, M)
        out = torch.zeros((num_verts, M), device=src.device, dtype=src.dtype)
        out.index_add_(0, idx, src)
        if reduce == "mean":
            counts = torch.bincount(idx, minlength=num_verts).clamp_min(1).to(out.dtype)
            out = out / counts.unsqueeze(1)
        return out

    reduced = torch.zeros(num_verts, M, device=cubes.device, dtype=value.dtype)
    return torch.scatter_reduce(
        reduced,
        0,
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1),
        value.flatten(0, 1),
        reduce=reduce,
        include_self=False,
    )


def sparse_cube2verts(coords, feats, training=True):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    if training:
        con_loss = torch.mean((feats - new_feats[cubes]) ** 2)
    else:
        con_loss = 0.0
    return new_coords, new_feats, con_loss


def get_dense_attrs(coords: torch.Tensor, feats: torch.Tensor, res: int, sdf_init=True):
    F = feats.shape[-1]
    # Building a 4D tensor and doing advanced indexing is slow on CPU for large grids.
    # Use a flat buffer with linear indices for much lower overhead.
    total = res * res * res
    dense = torch.zeros((total, F), device=feats.device, dtype=feats.dtype)
    if sdf_init:
        dense[:, 0] = 1

    coords_i = coords.to(dtype=torch.int64)
    idx = coords_i[:, 0] * (res * res) + coords_i[:, 1] * res + coords_i[:, 2]
    dense[idx] = feats
    return dense


def get_defomed_verts(v_pos: torch.Tensor, deform: torch.Tensor, res):
    # Ensure v_pos is on same device as deform for MPS compatibility
    v_pos = v_pos.to(device=deform.device, dtype=deform.dtype)
    return (v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)).to(
        deform.dtype
    )
