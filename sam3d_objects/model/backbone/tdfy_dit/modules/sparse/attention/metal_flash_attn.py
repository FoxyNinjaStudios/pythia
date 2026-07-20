# Metal Flash Attention for SAM-3D
# Uses PyObjC to interface with Metal compute shaders for efficient attention
#
# Replaces O(N²) attention mask with tiled algorithm

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

__all__ = [
    'metal_flash_attention',
    'metal_masked_sdpa',
    'is_metal_attention_available',
]

# ============================================================================
# METAL FRAMEWORK INITIALIZATION
# ============================================================================

_metal_device = None
_metal_library = None
_metal_command_queue = None
_metal_attn_functions = {}

def is_metal_attention_available() -> bool:
    """Check if Metal Flash Attention is available."""
    # Allow disabling Metal via environment variable
    if os.environ.get('DISABLE_METAL_GPU') == '1':
        return False
    
    try:
        _ensure_metal_attn_initialized()
        return _metal_device is not None
    except Exception:
        return False

def _ensure_metal_attn_initialized():
    """Lazy initialization of Metal attention framework."""
    global _metal_device, _metal_library, _metal_command_queue, _metal_attn_functions
    
    if _metal_device is not None:
        return
    
    try:
        import Metal
        
        _metal_device = Metal.MTLCreateSystemDefaultDevice()
        if _metal_device is None:
            raise RuntimeError("No Metal GPU device found")
        
        shader_path = Path(__file__).parent / "flash_attn.metal"
        if not shader_path.exists():
            raise FileNotFoundError(f"Metal shader not found: {shader_path}")
        
        source = shader_path.read_text()
        options = Metal.MTLCompileOptions.alloc().init()
        library, error = _metal_device.newLibraryWithSource_options_error_(
            source, options, None
        )
        if error:
            raise RuntimeError(f"Metal compilation error: {error}")
        _metal_library = library
        
        _metal_command_queue = _metal_device.newCommandQueue()
        
        for kernel_name in ["flash_attention_block_diag", "flash_attention_dense", 
                           "flash_attention_tiled"]:
            fn = _metal_library.newFunctionWithName_(kernel_name)
            if fn is None:
                print(f"[METAL ATTN] Warning: Kernel '{kernel_name}' not found")
                continue
            
            pipeline, error = _metal_device.newComputePipelineStateWithFunction_error_(
                fn, None
            )
            if error:
                raise RuntimeError(f"Pipeline error for {kernel_name}: {error}")
            _metal_attn_functions[kernel_name] = pipeline
        
        print(f"[METAL ATTN] Initialized Flash Attention with device: {_metal_device.name()}")
        
    except ImportError:
        raise ImportError(
            "PyObjC Metal framework not found. Install with: pip install pyobjc-framework-Metal"
        )

def _numpy_to_metal_buffer(arr: np.ndarray, options=0):
    """Create a Metal buffer from numpy array."""
    import Metal
    
    _ensure_metal_attn_initialized()
    arr = np.ascontiguousarray(arr)
    buffer = _metal_device.newBufferWithBytes_length_options_(
        arr.tobytes(), arr.nbytes, options
    )
    return buffer

def _metal_buffer_to_numpy(buffer, dtype, shape):
    """Read Metal buffer to numpy."""
    import ctypes
    
    # Get buffer length and contents pointer
    length = buffer.length()
    contents = buffer.contents()
    size = int(np.prod(shape))
    
    if dtype == np.float32:
        ptr_type = ctypes.c_float
    elif dtype == np.int32:
        ptr_type = ctypes.c_int32
    else:
        ptr_type = ctypes.c_float
    
    # contents can be:
    # 1. An integer (raw pointer address) 
    # 2. A PyObjC pointer wrapper object
    # We need to get the raw integer address
    
    if isinstance(contents, int):
        # Already an integer address
        ptr_int = contents
    elif hasattr(contents, '__int__'):
        # Can be converted to int
        ptr_int = int(contents)
    else:
        # Try treating it directly as a pointer address
        # On some PyObjC versions, contents() returns an opaque pointer
        try:
            ptr_int = contents.__pointer__ if hasattr(contents, '__pointer__') else id(contents)
        except:
            # Last resort: read via memmove
            raw_bytes = bytearray(length)
            ctypes.memmove((ctypes.c_char * length).from_buffer(raw_bytes), contents, length)
            arr = np.frombuffer(raw_bytes, dtype=dtype).copy()
            return arr.reshape(shape)
    
    # Create ctypes pointer from address
    c_ptr = ctypes.cast(ptr_int, ctypes.POINTER(ptr_type))
    
    # Create numpy array from buffer 
    arr = np.ctypeslib.as_array(c_ptr, shape=(size,)).copy()
    
    return arr.reshape(shape).astype(dtype)

# ============================================================================
# FLASH ATTENTION IMPLEMENTATION
# ============================================================================

def metal_flash_attention_block_diag(
    q: np.ndarray,  # [total_q, num_heads, head_dim]
    k: np.ndarray,  # [total_kv, num_heads, head_dim]
    v: np.ndarray,  # [total_kv, num_heads, head_dim]
    cu_seqlens_q: np.ndarray,  # [batch+1]
    cu_seqlens_kv: np.ndarray,  # [batch+1]
    softmax_scale: float = None,
) -> np.ndarray:
    """Execute block-diagonal flash attention using Metal."""
    import Metal
    
    _ensure_metal_attn_initialized()
    
    total_q, num_heads, head_dim = q.shape
    batch_size = len(cu_seqlens_q) - 1
    
    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(head_dim)
    
    # Create buffers
    q_buffer = _numpy_to_metal_buffer(q.astype(np.float32))
    k_buffer = _numpy_to_metal_buffer(k.astype(np.float32))
    v_buffer = _numpy_to_metal_buffer(v.astype(np.float32))
    cu_q_buffer = _numpy_to_metal_buffer(cu_seqlens_q.astype(np.int32))
    cu_kv_buffer = _numpy_to_metal_buffer(cu_seqlens_kv.astype(np.int32))
    
    output = np.zeros_like(q, dtype=np.float32)
    output_buffer = _numpy_to_metal_buffer(output)
    
    batch_buffer = _numpy_to_metal_buffer(np.array([batch_size], dtype=np.int32))
    heads_buffer = _numpy_to_metal_buffer(np.array([num_heads], dtype=np.int32))
    dim_buffer = _numpy_to_metal_buffer(np.array([head_dim], dtype=np.int32))
    scale_buffer = _numpy_to_metal_buffer(np.array([softmax_scale], dtype=np.float32))
    
    # Max sequence length for grid sizing
    max_q_len = max(cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(batch_size))
    
    cmd_buffer = _metal_command_queue.commandBuffer()
    encoder = cmd_buffer.computeCommandEncoder()
    
    encoder.setComputePipelineState_(_metal_attn_functions["flash_attention_block_diag"])
    encoder.setBuffer_offset_atIndex_(q_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(k_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(v_buffer, 0, 2)
    encoder.setBuffer_offset_atIndex_(cu_q_buffer, 0, 3)
    encoder.setBuffer_offset_atIndex_(cu_kv_buffer, 0, 4)
    encoder.setBuffer_offset_atIndex_(output_buffer, 0, 5)
    encoder.setBuffer_offset_atIndex_(batch_buffer, 0, 6)
    encoder.setBuffer_offset_atIndex_(heads_buffer, 0, 7)
    encoder.setBuffer_offset_atIndex_(dim_buffer, 0, 8)
    encoder.setBuffer_offset_atIndex_(scale_buffer, 0, 9)
    
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSize(max_q_len, num_heads, batch_size),
        Metal.MTLSize(1, 1, 1)
    )
    
    encoder.endEncoding()
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()
    
    return _metal_buffer_to_numpy(output_buffer, np.float32, q.shape)

# ============================================================================
# PYTORCH INTERFACE
# ============================================================================

def metal_masked_sdpa(
    q: torch.Tensor,  # [1, total_q, num_heads, head_dim]
    k: torch.Tensor,  # [1, total_kv, num_heads, head_dim]
    v: torch.Tensor,  # [1, total_kv, num_heads, head_dim]
    q_seqlen: List[int],
    kv_seqlen: List[int],
) -> torch.Tensor:
    """
    Metal-accelerated masked SDPA for block-diagonal attention.
    
    Drop-in replacement for masked_sdpa in SAM-3D.
    Falls back to PyTorch SDPA if Metal is unavailable.
    """
    device = q.device
    dtype = q.dtype
    
    if not is_metal_attention_available():
        # Fall back to existing implementation
        from .masked_sdpa import masked_sdpa
        return masked_sdpa(q, k, v, q_seqlen, kv_seqlen)
    
    # Remove batch dimension and prepare for Metal
    q_np = q.squeeze(0).cpu().numpy().astype(np.float32)  # [total_q, H, D]
    k_np = k.squeeze(0).cpu().numpy().astype(np.float32)
    v_np = v.squeeze(0).cpu().numpy().astype(np.float32)
    
    # Build cumulative sequence lengths
    cu_seqlens_q = np.array([0] + list(np.cumsum(q_seqlen)), dtype=np.int32)
    cu_seqlens_kv = np.array([0] + list(np.cumsum(kv_seqlen)), dtype=np.int32)
    
    # Execute Metal kernel
    output_np = metal_flash_attention_block_diag(
        q_np, k_np, v_np, cu_seqlens_q, cu_seqlens_kv
    )
    
    # Convert back to torch
    output = torch.from_numpy(output_np).to(device=device, dtype=dtype)
    
    return output

def metal_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Simple flash attention interface for dense tensors.
    
    Args:
        q, k, v: [batch, seq, num_heads, head_dim]
        is_causal: Not yet supported
    
    Returns:
        output: [batch, seq, num_heads, head_dim]
    """
    if is_causal:
        # Fall back to PyTorch for causal attention
        q_t = q.permute(0, 2, 1, 3)  # [B, H, S, D]
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        return out.permute(0, 2, 1, 3)
    
    if not is_metal_attention_available():
        # PyTorch fallback
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.permute(0, 2, 1, 3)
    
    batch, seq_q, num_heads, head_dim = q.shape
    seq_kv = k.shape[1]
    
    # Use block-diagonal with single sequence per batch
    q_seqlen = [seq_q] * batch
    kv_seqlen = [seq_kv] * batch
    
    # Flatten batch dimension
    q_flat = q.reshape(-1, num_heads, head_dim)  # [B*S, H, D]
    k_flat = k.reshape(-1, num_heads, head_dim)
    v_flat = v.reshape(-1, num_heads, head_dim)
    
    q_np = q_flat.cpu().numpy().astype(np.float32)
    k_np = k_flat.cpu().numpy().astype(np.float32)
    v_np = v_flat.cpu().numpy().astype(np.float32)
    
    cu_q = np.array([i * seq_q for i in range(batch + 1)], dtype=np.int32)
    cu_kv = np.array([i * seq_kv for i in range(batch + 1)], dtype=np.int32)
    
    output_np = metal_flash_attention_block_diag(q_np, k_np, v_np, cu_q, cu_kv)
    
    output = torch.from_numpy(output_np).to(device=q.device, dtype=q.dtype)
    return output.reshape(batch, seq_q, num_heads, head_dim)


# ============================================================================
# TESTING
# ============================================================================

def test_metal_flash_attention():
    """Test Metal Flash Attention."""
    print("Testing Metal Flash Attention...")
    
    if not is_metal_attention_available():
        print("Metal attention not available, testing fallback...")
    
    batch = 2
    seq = 128
    heads = 8
    dim = 64
    
    q = torch.randn(batch, seq, heads, dim)
    k = torch.randn(batch, seq, heads, dim)
    v = torch.randn(batch, seq, heads, dim)
    
    import time
    
    # Warmup
    _ = metal_flash_attention(q, k, v)
    
    start = time.time()
    for _ in range(10):
        output = metal_flash_attention(q, k, v)
    elapsed = (time.time() - start) / 10
    
    print(f"Input: batch={batch}, seq={seq}, heads={heads}, dim={dim}")
    print(f"Output shape: {output.shape}")
    print(f"Time: {elapsed*1000:.2f} ms")
    
    # Compare with PyTorch
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    expected = F.scaled_dot_product_attention(q_t, k_t, v_t).permute(0, 2, 1, 3)
    
    diff = (output - expected).abs().max().item()
    print(f"Max diff from PyTorch SDPA: {diff:.6f}")
    
    return output


if __name__ == "__main__":
    test_metal_flash_attention()
