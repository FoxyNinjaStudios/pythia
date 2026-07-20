/*
 * Metal Flash Attention Kernels for SAM-3D
 *
 * Optimized GPU kernels for attention on Apple Silicon.
 * Implements memory-efficient attention with tiling to avoid O(N²) memory.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant float SOFTMAX_SCALE_DEFAULT = 0.125f;  // 1/sqrt(64), typical head dim

// ============================================================================
// KERNEL 1: Fused Block-Diagonal Flash Attention
// ============================================================================
/*
 * Block-diagonal attention: each sequence only attends to itself.
 * 
 * Input:
 *   - Q: [total_q, num_heads, head_dim]
 *   - K: [total_kv, num_heads, head_dim]
 *   - V: [total_kv, num_heads, head_dim]
 *   - cu_seqlens_q: [batch+1] cumulative sequence lengths for Q
 *   - cu_seqlens_kv: [batch+1] cumulative sequence lengths for KV
 *
 * Output:
 *   - O: [total_q, num_heads, head_dim]
 */
kernel void flash_attention_block_diag(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const int* cu_seqlens_q [[buffer(3)]],
    device const int* cu_seqlens_kv [[buffer(4)]],
    device float* O [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& num_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant float& softmax_scale [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // gid.x = query token index within batch
    // gid.y = head index
    // gid.z = batch index
    
    uint batch_idx = gid.z;
    uint head_idx = gid.y;
    
    if (batch_idx >= uint(batch_size) || head_idx >= uint(num_heads)) return;
    
    // Get sequence boundaries
    int q_start = cu_seqlens_q[batch_idx];
    int q_end = cu_seqlens_q[batch_idx + 1];
    int kv_start = cu_seqlens_kv[batch_idx];
    int kv_end = cu_seqlens_kv[batch_idx + 1];
    
    int q_len = q_end - q_start;
    int kv_len = kv_end - kv_start;
    
    uint local_q_idx = gid.x;
    if (local_q_idx >= uint(q_len)) return;
    
    int global_q_idx = q_start + int(local_q_idx);
    
    // Load query vector for this position
    float q_vec[128];  // Max head_dim = 128
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[(global_q_idx * num_heads + head_idx) * head_dim + d];
    }
    
    // Compute attention scores and accumulate output
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc[128] = {0.0f};
    
    // First pass: find max score (for numerical stability)
    for (int kv_idx = kv_start; kv_idx < kv_end; kv_idx++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float k_val = K[(kv_idx * num_heads + head_idx) * head_dim + d];
            score += q_vec[d] * k_val;
        }
        score *= softmax_scale;
        max_score = max(max_score, score);
    }
    
    // Second pass: compute softmax and accumulate
    for (int kv_idx = kv_start; kv_idx < kv_end; kv_idx++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float k_val = K[(kv_idx * num_heads + head_idx) * head_dim + d];
            score += q_vec[d] * k_val;
        }
        score *= softmax_scale;
        
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        // Accumulate weighted value
        for (int d = 0; d < head_dim; d++) {
            float v_val = V[(kv_idx * num_heads + head_idx) * head_dim + d];
            output_acc[d] += exp_score * v_val;
        }
    }
    
    // Normalize and write output
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        O[(global_q_idx * num_heads + head_idx) * head_dim + d] = output_acc[d] * inv_sum;
    }
}


// ============================================================================
// KERNEL 2: Standard Multi-Head Attention (dense)
// ============================================================================
/*
 * Standard dense multi-head attention with optional mask.
 * 
 * Input:
 *   - Q: [batch, seq_q, num_heads, head_dim]
 *   - K: [batch, seq_kv, num_heads, head_dim]
 *   - V: [batch, seq_kv, num_heads, head_dim]
 *
 * Output:
 *   - O: [batch, seq_q, num_heads, head_dim]
 */
kernel void flash_attention_dense(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& seq_q [[buffer(5)]],
    constant int& seq_kv [[buffer(6)]],
    constant int& num_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant float& softmax_scale [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid.z;
    uint head_idx = gid.y;
    uint q_idx = gid.x;
    
    if (batch_idx >= uint(batch_size) || head_idx >= uint(num_heads) || q_idx >= uint(seq_q)) return;
    
    // Compute base indices
    int q_base = ((batch_idx * seq_q + q_idx) * num_heads + head_idx) * head_dim;
    
    // Load query vector
    float q_vec[128];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_base + d];
    }
    
    // Compute attention
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc[128] = {0.0f};
    
    // First pass: find max
    for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
        int k_base = ((batch_idx * seq_kv + kv_idx) * num_heads + head_idx) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_base + d];
        }
        score *= softmax_scale;
        max_score = max(max_score, score);
    }
    
    // Second pass: softmax and accumulate
    for (int kv_idx = 0; kv_idx < seq_kv; kv_idx++) {
        int k_base = ((batch_idx * seq_kv + kv_idx) * num_heads + head_idx) * head_dim;
        int v_base = k_base;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_base + d];
        }
        score *= softmax_scale;
        
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;
        
        for (int d = 0; d < head_dim; d++) {
            output_acc[d] += exp_score * V[v_base + d];
        }
    }
    
    // Write output
    int o_base = q_base;
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        O[o_base + d] = output_acc[d] * inv_sum;
    }
}


// ============================================================================
// KERNEL 3: Tiled Flash Attention (for large sequences)
// ============================================================================
/*
 * Memory-efficient tiled attention for very large sequences.
 * Uses online softmax algorithm to avoid materializing full attention matrix.
 * 
 * Tile size = 64 for balance between memory and parallelism.
 */
constant int TILE_SIZE = 64;

kernel void flash_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* L [[buffer(4)]],  // [batch, num_heads, seq_q] log-sum-exp
    device float* M [[buffer(5)]],  // [batch, num_heads, seq_q] max scores
    constant int& batch_size [[buffer(6)]],
    constant int& seq_q [[buffer(7)]],
    constant int& seq_kv [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& head_dim [[buffer(10)]],
    constant float& softmax_scale [[buffer(11)]],
    constant int& tile_idx [[buffer(12)]],  // Which KV tile we're processing
    uint3 gid [[thread_position_in_grid]],
    threadgroup float* shared_k [[threadgroup(0)]],
    threadgroup float* shared_v [[threadgroup(1)]]
) {
    uint batch_idx = gid.z;
    uint head_idx = gid.y;
    uint q_idx = gid.x;
    
    if (batch_idx >= uint(batch_size) || head_idx >= uint(num_heads) || q_idx >= uint(seq_q)) return;
    
    // Compute KV tile boundaries
    int kv_tile_start = tile_idx * TILE_SIZE;
    int kv_tile_end = min(kv_tile_start + TILE_SIZE, seq_kv);
    
    // Load query (same for all KV tiles)
    int q_base = ((batch_idx * seq_q + q_idx) * num_heads + head_idx) * head_dim;
    float q_vec[128];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = Q[q_base + d];
    }
    
    // Load previous running max and log-sum-exp
    int lm_idx = (batch_idx * num_heads + head_idx) * seq_q + q_idx;
    float prev_max = (tile_idx == 0) ? -INFINITY : M[lm_idx];
    float prev_sum = (tile_idx == 0) ? 0.0f : L[lm_idx];
    
    // Load previous output accumulator
    int o_base = q_base;
    float output_acc[128];
    for (int d = 0; d < head_dim; d++) {
        output_acc[d] = (tile_idx == 0) ? 0.0f : O[o_base + d] * prev_sum;
    }
    
    // Compute attention for this tile
    float tile_max = -INFINITY;
    
    // First pass: find max in tile
    for (int kv_idx = kv_tile_start; kv_idx < kv_tile_end; kv_idx++) {
        int k_base = ((batch_idx * seq_kv + kv_idx) * num_heads + head_idx) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_base + d];
        }
        score *= softmax_scale;
        tile_max = max(tile_max, score);
    }
    
    // Update global max
    float new_max = max(prev_max, tile_max);
    float scale_prev = (tile_idx == 0) ? 0.0f : exp(prev_max - new_max);
    
    // Rescale previous accumulator
    for (int d = 0; d < head_dim; d++) {
        output_acc[d] *= scale_prev;
    }
    float new_sum = prev_sum * scale_prev;
    
    // Second pass: accumulate with new max
    for (int kv_idx = kv_tile_start; kv_idx < kv_tile_end; kv_idx++) {
        int k_base = ((batch_idx * seq_kv + kv_idx) * num_heads + head_idx) * head_dim;
        int v_base = k_base;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_vec[d] * K[k_base + d];
        }
        score *= softmax_scale;
        
        float exp_score = exp(score - new_max);
        new_sum += exp_score;
        
        for (int d = 0; d < head_dim; d++) {
            output_acc[d] += exp_score * V[v_base + d];
        }
    }
    
    // Store updated M, L, and O
    M[lm_idx] = new_max;
    L[lm_idx] = new_sum;
    
    // Normalize output (final tile only, or store un-normalized for multi-tile)
    int num_tiles = (seq_kv + TILE_SIZE - 1) / TILE_SIZE;
    if (tile_idx == num_tiles - 1) {
        // Final tile: normalize
        float inv_sum = 1.0f / new_sum;
        for (int d = 0; d < head_dim; d++) {
            O[o_base + d] = output_acc[d] * inv_sum;
        }
    } else {
        // Intermediate tile: store unnormalized (will be rescaled)
        float inv_sum = 1.0f / new_sum;
        for (int d = 0; d < head_dim; d++) {
            O[o_base + d] = output_acc[d] * inv_sum;
        }
    }
}
