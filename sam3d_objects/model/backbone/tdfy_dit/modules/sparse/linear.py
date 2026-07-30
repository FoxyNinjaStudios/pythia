# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from . import SparseTensor

__all__ = ["SparseLinear"]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        # Move weights to input device for GPU acceleration
        feats = input.feats
        weight = self.weight.to(device=feats.device, dtype=feats.dtype)
        bias = self.bias.to(device=feats.device, dtype=feats.dtype) if self.bias is not None else None
        result = torch.nn.functional.linear(feats, weight, bias)
        return input.replace(result)
