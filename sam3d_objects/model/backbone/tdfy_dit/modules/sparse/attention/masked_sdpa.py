# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F


def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    """
    Create an additive attention mask for block-diagonal attention.
    The result is shape [sum_q, sum_kv], with 0.0 in the valid
    region(s) and -inf elsewhere.
    """
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    # Start with everything "masked out"
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len

    return attn_mask


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """
    Mimic xFormers' memory_efficient_attention using PyTorch 2.0 scaled_dot_product_attention.
    On MPS, avoids global N x N masks by processing sequences in a loop.
    """
    device = q.device
    
    # MPS Optimization: avoid global N x N matrix if sequence is too long
    # sum(q_seqlen) is typically 25,000+. 25,000^2 fits in 48GB but NOT in 30GB MTLBuffer.
    if device.type == "mps":
        # Process each sequence independently to avoid both N x N mask memory explosion
        # and the MPSNDArrayMatrixMultiplication crash.
        outs = []
        q_start = 0
        kv_start = 0
        
        # Permute headers for SDPA: [B, H, L, C]
        q = q.permute(0, 2, 1, 3) # [1, H, total_L, C]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Stability: Run the core attention on CPU to avoid MPS NDArray assertion failures
        # Small windowed sequences (e.g. 512) are very fast on CPU.
        q_cpu = q.to("cpu", dtype=torch.float32)
        k_cpu = k.to("cpu", dtype=torch.float32)
        v_cpu = v.to("cpu", dtype=torch.float32)
        
        for q_len, kv_len in zip(q_seqlen, kv_seqlen):
            q_part = q_cpu[:, :, q_start : q_start + q_len, :]
            k_part = k_cpu[:, :, kv_start : kv_start + kv_len, :]
            v_part = v_cpu[:, :, kv_start : kv_start + kv_len, :]
            
            # CPU handles float32 perfectly and reliably
            out_part = F.scaled_dot_product_attention(
                q_part, k_part, v_part, attn_mask=None, is_causal=False
            )
            outs.append(out_part.to(device))
            
            q_start += q_len
            kv_start += kv_len
        
        out = torch.cat(outs, dim=2) # Concatenate along sequence dimension
        out = out.permute(0, 2, 1, 3) # [1, total_L, H, C]
        return out[0]

    # Original CUDA/CPU path (using global mask for speed if optimized kernel is available)
    # Build the block-diagonal additive mask
    attn_mask_2d = block_diag_attn_mask(
        q_seqlen, kv_seqlen, device=q.device, dtype=q.dtype
    )

    # PyTorch’s scaled_dot_product_attention expects a mask broadcastable to
    # [batch_size, n_heads, q_len, kv_len].
    attn_mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0)
    q = q.permute(0, 2, 1, 3)  # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)  # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)  # [N, H, L, C]

    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask_4d,
        dropout_p=0.0,
        is_causal=False,
    )
    out = out.permute(0, 2, 1, 3)
    return out[0]
