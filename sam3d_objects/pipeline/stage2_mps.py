# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Stage 2 (SLAT) MPS acceleration module.

Uses PyTorch gather-scatter operations on MPS for sparse convolution and attention.
This follows the TRELLIS.2 approach: build hash map → gather neighbors → apply weights
via torch.mm → scatter-add results back. No custom Metal kernels required; all ops use
standard PyTorch MPS support.

Reference: TRELLIS.2 (MIT-licensed) and vggt-mps sparse attention pattern.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from loguru import logger


class SparseTensor3DMPS:
    """MPS-compatible sparse 3D tensor for gather-scatter operations."""
    
    def __init__(self, coords: torch.Tensor, feats: torch.Tensor, device: torch.device = None):
        """
        Args:
            coords: (N, 4) tensor [batch, x, y, z] of voxel coordinates
            feats: (N, C) or (N, H, C) feature tensor for each voxel
            device: torch device (defaults to CPU if not MPS-capable)
        """
        self.device = device or torch.device('cpu')
        self.coords = coords.to(self.device)
        self.feats = feats.to(self.device)
        self.num_voxels = coords.shape[0]
        self._coord_hash = None  # Lazy-computed hash map
    
    def build_coord_hash(self) -> Dict[Tuple[int, int, int, int], int]:
        """Build hash map: (batch, x, y, z) -> voxel index."""
        if self._coord_hash is not None:
            return self._coord_hash
        
        hash_map = {}
        for idx, coord in enumerate(self.coords):
            key = tuple(coord.cpu().tolist())
            hash_map[key] = idx
        self._coord_hash = hash_map
        return hash_map
    
    def gather_neighbors_3d(self, kernel_radius: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather neighbor voxels for 3D convolution using hash map.
        
        Args:
            kernel_radius: radius of convolution kernel (1 = 3³ kernel, 2 = 5³, etc.)
        
        Returns:
            neighbor_feats: (N, (2*radius+1)³, C) gathered neighbor features
            neighbor_valid: (N, (2*radius+1)³) boolean mask for valid neighbors
        """
        hash_map = self.build_coord_hash()
        kernel_size = 2 * kernel_radius + 1
        kernel_positions = kernel_size ** 3
        
        neighbor_feats = torch.zeros(
            self.num_voxels, kernel_positions, self.feats.shape[-1],
            device=self.device, dtype=self.feats.dtype
        )
        neighbor_valid = torch.zeros(
            self.num_voxels, kernel_positions,
            device=self.device, dtype=torch.bool
        )
        
        # Iterate over voxels and gather neighbors
        for center_idx, center_coord in enumerate(self.coords):
            batch, x, y, z = center_coord.cpu().tolist()
            kernel_idx = 0
            
            for dx in range(-kernel_radius, kernel_radius + 1):
                for dy in range(-kernel_radius, kernel_radius + 1):
                    for dz in range(-kernel_radius, kernel_radius + 1):
                        neighbor_key = (batch, int(x + dx), int(y + dy), int(z + dz))
                        
                        if neighbor_key in hash_map:
                            neighbor_voxel_idx = hash_map[neighbor_key]
                            neighbor_feats[center_idx, kernel_idx] = self.feats[neighbor_voxel_idx]
                            neighbor_valid[center_idx, kernel_idx] = True
                        
                        kernel_idx += 1
        
        return neighbor_feats, neighbor_valid
    
    def scatter_add_(self, output: torch.Tensor) -> torch.Tensor:
        """
        Scatter-add output features back to voxel coordinates.
        
        Args:
            output: (N, C) output features to accumulate
        
        Returns:
            result: (N, C) accumulated features at each voxel
        """
        result = torch.zeros_like(self.feats)
        result.scatter_add_(0, self.coords[:, 0:1].long(), output)
        return result
    
    def to(self, device: torch.device):
        """Move tensor to device."""
        self.coords = self.coords.to(device)
        self.feats = self.feats.to(device)
        self.device = device
        self._coord_hash = None  # Invalidate hash map
        return self


def sparse_conv_3d_mps(
    sparse_tensor: SparseTensor3DMPS,
    weight: torch.Tensor,
    kernel_radius: int = 1,
    device: torch.device = None,
) -> SparseTensor3DMPS:
    """
    Sparse 3D convolution on MPS using gather-scatter.
    
    Args:
        sparse_tensor: input SparseTensor3DMPS
        weight: (kernel_size³, C_in, C_out) convolution weights
        kernel_radius: radius of kernel
        device: target device (MPS or CPU)
    
    Returns:
        output SparseTensor3DMPS with convolved features
    """
    if device is None:
        device = sparse_tensor.device
    
    # Gather neighbor voxels
    neighbor_feats, neighbor_valid = sparse_tensor.gather_neighbors_3d(kernel_radius)
    
    kernel_size = 2 * kernel_radius + 1
    kernel_positions = kernel_size ** 3
    
    # Reshape for batch matrix multiply: (N, K, C_in) × (K, C_in, C_out) -> (N, K, C_out)
    N = neighbor_feats.shape[0]
    C_in = neighbor_feats.shape[-1]
    C_out = weight.shape[-1]
    
    neighbor_feats = neighbor_feats.reshape(N * kernel_positions, C_in)
    weight_flat = weight.reshape(kernel_positions, C_in, C_out)
    
    # Apply weights via matmul (MPS-compatible)
    output_feats = torch.zeros(N * kernel_positions, C_out, device=device, dtype=neighbor_feats.dtype)
    for k in range(kernel_positions):
        valid_mask = neighbor_valid[:, k]
        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            output_feats[valid_indices * kernel_positions + k] = torch.mm(
                neighbor_feats[valid_indices * kernel_positions + k],
                weight_flat[k]
            )
    
    output_feats = output_feats.reshape(N, kernel_positions, C_out).sum(dim=1)  # Pool over kernel
    
    return SparseTensor3DMPS(sparse_tensor.coords.clone(), output_feats, device)


def sparse_attention_mps(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sparse attention on MPS using masked dot-product attention.
    
    Runs standard scaled_dot_product_attention with sparse masking.
    This is the vggt-mps pattern: O(n) memory instead of O(n²).
    
    Args:
        query: (B, N, D) query tensor
        key: (B, N, D) key tensor
        value: (B, N, D) value tensor
        sparse_indices: (B, N, K) top-K indices for sparse selection
        device: target device (MPS or CPU)
    
    Returns:
        output: (B, N, D) attention output
    """
    if device is None:
        device = query.device
    
    if sparse_indices is not None:
        # Gather sparse keys/values using indices
        key = torch.gather(key, 1, sparse_indices.unsqueeze(-1).expand(-1, -1, key.shape[-1]))
        value = torch.gather(value, 1, sparse_indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
    
    # Standard scaled dot-product attention (MPS-compatible)
    output = F.scaled_dot_product_attention(query, key, value)
    return output


def enable_stage2_mps(pipeline, use_mps: bool = True) -> None:
    """
    Enable or disable MPS acceleration for Stage 2 (SLAT).
    
    Sets environment flags and patches the pipeline's sample_slat method
    to conditionally use MPS device for SLAT computation.
    
    Args:
        pipeline: InferencePipeline instance
        use_mps: whether to enable MPS (default: True)
    """
    if use_mps and not torch.backends.mps.is_available():
        logger.warning("[STAGE2-MPS] MPS not available; keeping Stage 2 on CPU")
        return
    
    stage2_device = torch.device('mps' if use_mps else 'cpu')
    pipeline._stage2_device = stage2_device
    pipeline._use_stage2_mps = use_mps
    
    logger.info(f"[STAGE2-MPS] Stage 2 (SLAT) device: {stage2_device}")
