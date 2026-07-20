# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
import itertools
import torch
import torch.nn as nn
from . import SparseTensor

__all__ = ["SparseDownsample", "SparseUpsample", "SparseSubdivide"]


_SUBDIV_OFFSETS_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def _get_subdivide_offsets(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (dim, device.type, device.index, dtype)
    cached = _SUBDIV_OFFSETS_CACHE.get(key)
    if cached is not None:
        return cached

    # Offsets are all binary combinations in {0,1}^dim, with a leading batch offset of 0.
    offsets_xyz = list(itertools.product([0, 1], repeat=dim))
    offsets = torch.tensor(offsets_xyz, dtype=torch.int32)
    offsets = torch.cat([torch.zeros((offsets.shape[0], 1), dtype=torch.int32), offsets], dim=1)
    offsets = offsets.to(device=device, dtype=dtype)
    _SUBDIV_OFFSETS_CACHE[key] = offsets
    return offsets


class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """

    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]]):
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(
            factor
        ), "Input coordinates must have the same dimension as the downsample factor."

        coord = list(input.coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i + 1] = coord[i + 1] // f

        MAX = [coord[i + 1].max().item() + 1 for i in range(DIM)]
        # Create offset tensor on same device as input
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1], device=input.coords.device), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)
        # Ensure idx is on same device as features for scatter_reduce
        idx = idx.to(device=input.feats.device)

        new_feats = torch.scatter_reduce(
            torch.zeros(
                code.shape[0],
                input.feats.shape[1],
                device=input.feats.device,
                dtype=input.feats.dtype,
            ),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce="mean",
        )
        new_coords = torch.stack(
            [code // OFFSET[0]]
            + [(code // OFFSET[i + 1]) % MAX[i] for i in range(DIM)],
            dim=-1,
        )
        out = SparseTensor(
            new_feats,
            new_coords,
            input.shape,
        )
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f"upsample_{factor}_coords", input.coords)
        out.register_spatial_cache(f"upsample_{factor}_layout", input.layout)
        out.register_spatial_cache(f"upsample_{factor}_idx", idx)

        return out


class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """

    def __init__(self, factor: Union[int, Tuple[int, int, int], List[int]]):
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(
            factor
        ), "Input coordinates must have the same dimension as the upsample factor."

        new_coords = input.get_spatial_cache(f"upsample_{factor}_coords")
        new_layout = input.get_spatial_cache(f"upsample_{factor}_layout")
        idx = input.get_spatial_cache(f"upsample_{factor}_idx")
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError(
                "Upsample cache not found. SparseUpsample must be paired with SparseDownsample."
            )
        new_feats = input.feats[idx]
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out


class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """

    def __init__(self):
        super(SparseSubdivide, self).__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        import time
        start = time.perf_counter()
        
        DIM = input.coords.shape[-1] - 1
        device = input.coords.device
        dtype = input.coords.dtype
        n_coords = _get_subdivide_offsets(DIM, device, dtype)
        factor = n_coords.shape[0]

        # Safer on MPS than broadcast integer multiply: do the original in-place scaling.
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0)

        new_feats = input.feats.unsqueeze(1).expand(
            input.feats.shape[0], factor, *input.feats.shape[1:]
        )
        out = SparseTensor(
            new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape
        )
        # Element-wise scale update (tuple * int would repeat elements)
        out._scale = tuple([s * 2 for s in input._scale])
        out._spatial_cache = input._spatial_cache
        
        elapsed = time.perf_counter() - start
        in_size = input.feats.shape[0]
        out_size = out.feats.shape[0]
        print(f"[TIMING] SparseSubdivide: {elapsed*1000:.1f}ms (in: {in_size} -> out: {out_size} voxels)")
        return out
