# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from types import SimpleNamespace
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations import MeshExtractResult
from ...representations.mesh import SparseFeatures2Mesh
import os
import time
from safetensors.torch import load_file
from loguru import logger


def _force_gc():
    """Aggressive garbage collection for memory management."""
    gc.collect()
    gc.collect()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except:
            pass


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """

    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels), sp.SparseSiLU()
        )

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3d(
                channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"
            ),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(
                sp.SparseConv3d(
                    self.out_channels,
                    self.out_channels,
                    3,
                    indice_key=f"res_{self.out_resolution}",
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(
                channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}"
            )

    def forward(self, x: sp.SparseTensor, delete_input: bool = False) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
            delete_input: If True, delete input tensor after use to save memory.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        # The subdivide operation is purely a nearest-neighbor expansion.
        # Running it twice (once for the main path, once for the skip) doubles runtime.
        # We subdivide `x` once and reuse it for both paths.

        # Sync MPS to free intermediate buffers before heavy subdivision
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        x_sub = self.sub(x)

        # Sync again after subdivision (helps peak memory on MPS)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        
        # Delete original input after subdivision if requested
        if delete_input:
            del x
            _force_gc()
        
        h = self.act_layers(x_sub)
        h = self.out_layers(h)
        
        # Sync after convolutions
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
        
        h = h + self.skip_connection(x_sub)
        
        # Delete subdivided skip connection
        del x_sub
        
        return h



class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        device="cpu"  # Changed from cuda for CPU compatibility
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeatures2Mesh(
            res=self.resolution * 4, use_color=self.rep_config.get("use_color", False), device=device
        )
        self.out_channels = self.mesh_extractor.feats_channels
        self.upsample = nn.ModuleList(
            [
                SparseSubdivideBlock3d(
                    channels=model_channels,
                    resolution=resolution,
                    out_channels=model_channels // 4,
                ),
                SparseSubdivideBlock3d(
                    channels=model_channels // 4,
                    resolution=resolution * 2,
                    out_channels=model_channels // 8,
                ),
            ]
        )
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)

    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        # Stability: Run transformer on CPU to avoid MPS matrix multiplication crashes
        # 25K voxels is small enough for CPU transformer (takes ~10-15s)
        device = next(self.parameters()).device
        
        profile = os.environ.get("SAM3D_PROFILE") == "1"
        t0 = time.perf_counter() if profile else 0.0

        logger.info(f"[LOW-MEM] Running transformer on CPU for stability...")
        # Avoid moving the entire decoder to CPU (very expensive). We only move the
        # transformer torso to CPU, keep upsampling + out_layer on the target device.
        if not hasattr(self, "_transformer_pinned_cpu"):
            self._transformer_pinned_cpu = False

        if not self._transformer_pinned_cpu:
            self.input_layer.to("cpu")
            if getattr(self, "pe_mode", None) == "ape" and hasattr(self, "pos_embedder"):
                self.pos_embedder.to("cpu")
            self.blocks.to("cpu")
            self.convert_to_fp32()
            self._transformer_pinned_cpu = True

        x_cpu = x.cpu().float()

        with torch.no_grad():
            h_cpu = super(SLatMeshDecoder, self).forward(x_cpu)

            if profile:
                logger.info(f"[PROFILE] Transformer(CPU): {(time.perf_counter() - t0):.3f}s")
                t0 = time.perf_counter()

            # Move features back to target device (MPS) for upsampling
            h = h_cpu.to(device)

            logger.info(f"[LOW-MEM] Transformer complete. Running upsampling on GPU...")

            for block in self.upsample:
                h = block(h)

            h = self.out_layer(h)

        if profile:
            logger.info(f"[PROFILE] Upsample+OutLayer(MPS): {(time.perf_counter() - t0):.3f}s")
            t0 = time.perf_counter()
            
        # Stability: Run representation conversion (FlexiCubes extraction) on CPU
        # to avoid MPS-specific index_add_ bugs and NaN generation.
        # 1.6M voxels is fast on CPU (~5-10s).
        logger.info(f"[LOW-MEM] Moving features to CPU for final mesh extraction...")

        coords_cpu = h.coords.detach().cpu()
        feats_cpu = h.feats.detach().cpu().float()

        self.mesh_extractor.device = "cpu"

        outputs: List[MeshExtractResult] = []
        for i in range(h.shape[0]):
            sl = h.layout[i]
            coords_i = coords_cpu[sl].clone().contiguous()
            coords_i[:, 0] = 0
            feats_i = feats_cpu[sl].contiguous()
            cubefeats_i = SimpleNamespace(coords=coords_i, feats=feats_i)
            outputs.append(self.mesh_extractor(cubefeats_i, training=self.training))

        if profile:
            logger.info(f"[PROFILE] MeshExtract(CPU): {(time.perf_counter() - t0):.3f}s")

        return outputs



class SLatMeshDecoderTdfyWrapper(SLatMeshDecoder):
    def __init__(self, *args, **kwargs):
        pretrained_ckpt_path = kwargs.pop("pretrained_ckpt_path", None)
        super().__init__(*args, **kwargs)
        if pretrained_ckpt_path is not None and os.path.exists(pretrained_ckpt_path):
            logger.info(
                f"Loading pretrained slat decoder gs from {pretrained_ckpt_path}"
            )
            self.load_state_dict(load_file(pretrained_ckpt_path))
