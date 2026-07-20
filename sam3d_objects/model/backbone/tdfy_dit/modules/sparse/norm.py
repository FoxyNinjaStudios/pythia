# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import time
from . import SparseTensor
from . import DEBUG

# Global timing stats
_TIMING_ENABLED = True
_timing_stats = {}

def log_timing(name: str, elapsed: float):
    if _TIMING_ENABLED:
        if name not in _timing_stats:
            _timing_stats[name] = {"count": 0, "total": 0.0}
        _timing_stats[name]["count"] += 1
        _timing_stats[name]["total"] += elapsed
        if _timing_stats[name]["count"] % 10 == 0:  # Log every 10 calls
            avg = _timing_stats[name]["total"] / _timing_stats[name]["count"]
            print(f"[TIMING] {name}: {elapsed*1000:.1f}ms (avg: {avg*1000:.1f}ms, calls: {_timing_stats[name]['count']})")

__all__ = [
    "SparseGroupNorm",
    "SparseLayerNorm",
    "SparseGroupNorm32",
    "SparseLayerNorm32",
]


class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        start = time.perf_counter()
        
        # Move weights to input device for MPS GPU acceleration
        if self.weight is not None and self.weight.device != input.feats.device:
            self.weight.data = self.weight.data.to(input.feats.device)
        if self.bias is not None and self.bias.device != input.feats.device:
            self.bias.data = self.bias.data.to(input.feats.device)
        
        # Optimized: Process all batches together using padding
        batch_size = input.shape[0]
        if batch_size == 1:
            # Single batch - no loop needed
            bfeats = input.feats[input.layout[0]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            nfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        else:
            # Multi-batch - use original loop (vectorization requires equal sizes)
            nfeats = torch.zeros_like(input.feats)
            for k in range(batch_size):
                if DEBUG:
                    assert (
                        input.coords[input.layout[k], 0] == k
                    ).all(), f"SparseGroupNorm: batch index mismatch"
                bfeats = input.feats[input.layout[k]]
                bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
                bfeats = super().forward(bfeats)
                bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
                nfeats[input.layout[k]] = bfeats
        
        elapsed = time.perf_counter() - start
        log_timing("SparseGroupNorm", elapsed)
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)


class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
