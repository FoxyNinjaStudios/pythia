# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move weights to input device for GPU acceleration
        if self.weight is not None and self.weight.device != x.device:
            weight = self.weight.to(device=x.device)
            bias = self.bias.to(device=x.device) if self.bias is not None else None
            return nn.functional.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type(x.dtype)
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move weights to input device for GPU acceleration
        if self.weight.device != x.device:
            weight = self.weight.to(device=x.device)
            bias = self.bias.to(device=x.device) if self.bias is not None else None
            return nn.functional.group_norm(x.float(), self.num_groups, weight, bias, self.eps).type(x.dtype)
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x
