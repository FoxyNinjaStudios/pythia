/*
 * Metal Sparse 3D Convolution Kernels for SAM-3D
 * 
 * Optimized GPU kernels for sparse convolution on Apple Silicon.
 * 
 * Key operations:
 * 1. build_hash_table - Create spatial hash for O(1) neighbor lookup
 * 2. sparse_conv3x3x3 - Fused gather-compute-scatter sparse convolution
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// CONSTANTS
// ============================================================================

constant int KERNEL_SIZE = 3;
constant int KERNEL_VOLUME = 27;  // 3 * 3 * 3
constant int HASH_EMPTY = -1;

// 3x3x3 kernel offsets (pre-computed for efficiency)
constant int3 KERNEL_OFFSETS[27] = {
    int3(-1, -1, -1), int3(-1, -1, 0), int3(-1, -1, 1),
    int3(-1, 0, -1),  int3(-1, 0, 0),  int3(-1, 0, 1),
    int3(-1, 1, -1),  int3(-1, 1, 0),  int3(-1, 1, 1),
    int3(0, -1, -1),  int3(0, -1, 0),  int3(0, -1, 1),
    int3(0, 0, -1),   int3(0, 0, 0),   int3(0, 0, 1),
    int3(0, 1, -1),   int3(0, 1, 0),   int3(0, 1, 1),
    int3(1, -1, -1),  int3(1, -1, 0),  int3(1, -1, 1),
    int3(1, 0, -1),   int3(1, 0, 0),   int3(1, 0, 1),
    int3(1, 1, -1),   int3(1, 1, 0),   int3(1, 1, 1)
};

// ============================================================================
// UTILITY FUNCTIONS  
// ============================================================================

// Compute spatial hash for a 4D coordinate (batch, z, y, x)
inline uint coord_to_hash(int4 coord, int3 spatial_shape) {
    // hash = batch * D*H*W + z * H*W + y * W + x
    int D = spatial_shape.x;
    int H = spatial_shape.y;
    int W = spatial_shape.z;
    return uint(coord.x * D * H * W + coord.y * H * W + coord.z * W + coord.w);
}

// ============================================================================
// KERNEL 1: Build Hash Table
// ============================================================================
/*
 * Builds a spatial hash table for O(1) neighbor lookup.
 * 
 * Input:
 *   - coords: [N, 4] int tensor (batch, z, y, x)
 *   - N: number of active voxels
 *   - spatial_shape: (D, H, W)
 * 
 * Output:
 *   - hash_table: [D*H*W * batch_size] -> index or HASH_EMPTY
 */
kernel void build_hash_table(
    device const int4* coords [[buffer(0)]],
    device int* hash_table [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int3& spatial_shape [[buffer(3)]],
    constant int& table_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(N)) return;
    
    int4 coord = coords[gid];
    uint hash = coord_to_hash(coord, spatial_shape);
    
    // Clamp to table size (should not happen if table_size is correct)
    if (hash < uint(table_size)) {
        hash_table[hash] = int(gid);
    }
}

// ============================================================================
// KERNEL 2: Sparse 3x3x3 SubManifold Convolution
// ============================================================================
/*
 * Performs a 3x3x3 submanifold sparse convolution.
 * 
 * For each active voxel:
 *   1. Look up its 27 neighbors in the hash table
 *   2. Gather neighbor features
 *   3. Apply convolution weights
 *   4. Accumulate to output
 * 
 * Input:
 *   - features: [N, C_in] float tensor
 *   - coords: [N, 4] int tensor (batch, z, y, x)
 *   - weights: [27, C_in, C_out] convolution weights (K, C_in, C_out)
 *   - bias: [C_out] or nullptr
 *   - hash_table: [table_size] spatial hash table
 *   - N, C_in, C_out, spatial_shape
 * 
 * Output:
 *   - output: [N, C_out] float tensor
 */
kernel void sparse_conv3x3x3_subm(
    device const float* features [[buffer(0)]],
    device const int4* coords [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const int* hash_table [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& C_in [[buffer(7)]],
    constant int& C_out [[buffer(8)]],
    constant int3& spatial_shape [[buffer(9)]],
    constant int& table_size [[buffer(10)]],
    constant int& has_bias [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = voxel index, gid.y = output channel (tiled)
    uint voxel_idx = gid.x;
    uint out_ch_start = gid.y * 8;  // Process 8 output channels per thread
    
    if (voxel_idx >= uint(N)) return;
    
    int4 center_coord = coords[voxel_idx];
    int D = spatial_shape.x;
    int H = spatial_shape.y;
    int W = spatial_shape.z;
    
    // Accumulate output for this voxel
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Loop over 27 kernel positions
    for (int k = 0; k < KERNEL_VOLUME; k++) {
        int3 offset = KERNEL_OFFSETS[k];
        int4 neighbor_coord = center_coord + int4(0, offset.x, offset.y, offset.z);
        
        // Boundary check
        if (neighbor_coord.y < 0 || neighbor_coord.y >= D ||
            neighbor_coord.z < 0 || neighbor_coord.z >= H ||
            neighbor_coord.w < 0 || neighbor_coord.w >= W) {
            continue;
        }
        
        // Hash lookup
        uint hash = coord_to_hash(neighbor_coord, spatial_shape);
        if (hash >= uint(table_size)) continue;
        
        int neighbor_idx = hash_table[hash];
        if (neighbor_idx == HASH_EMPTY) continue;
        
        // Gather neighbor features and apply weights
        // weights layout: [K=27, C_in, C_out]
        for (uint c_out = 0; c_out < 8 && (out_ch_start + c_out) < uint(C_out); c_out++) {
            float sum = 0.0f;
            for (int c_in = 0; c_in < C_in; c_in++) {
                float feat = features[neighbor_idx * C_in + c_in];
                float w = weights[k * C_in * C_out + c_in * C_out + (out_ch_start + c_out)];
                sum += feat * w;
            }
            accum[c_out] += sum;
        }
    }
    
    // Write output with bias
    for (uint c_out = 0; c_out < 8 && (out_ch_start + c_out) < uint(C_out); c_out++) {
        float result = accum[c_out];
        if (has_bias) {
            result += bias[out_ch_start + c_out];
        }
        output[voxel_idx * C_out + (out_ch_start + c_out)] = result;
    }
}

// ============================================================================
// KERNEL 3: Sparse Strided Convolution (for downsampling)
// ============================================================================
/*
 * Strided sparse convolution for spatial downsampling.
 * Output coordinates are input_coords // stride.
 * Uses atomic adds due to potential collisions.
 */
kernel void sparse_conv3x3x3_strided(
    device const float* features [[buffer(0)]],
    device const int4* in_coords [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device atomic_float* output [[buffer(4)]],  // Atomic for thread safety
    device const int* out_coord_mapping [[buffer(5)]],  // input_idx -> output_idx
    constant int& N [[buffer(6)]],
    constant int& C_in [[buffer(7)]],
    constant int& C_out [[buffer(8)]],
    constant int& has_bias [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint voxel_idx = gid.x;
    uint out_ch = gid.y;
    
    if (voxel_idx >= uint(N) || out_ch >= uint(C_out)) return;
    
    int out_idx = out_coord_mapping[voxel_idx];
    if (out_idx < 0) return;
    
    // Simplified: apply center weight only (k=13 is center of 3x3x3)
    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        float feat = features[voxel_idx * C_in + c_in];
        float w = weights[13 * C_in * C_out + c_in * C_out + out_ch];  // Center weight
        sum += feat * w;
    }
    
    if (has_bias && voxel_idx == 0) {
        sum += bias[out_ch];
    }
    
    // Atomic add to handle collisions
    atomic_fetch_add_explicit(&output[out_idx * C_out + out_ch], sum, memory_order_relaxed);
}

// ============================================================================
// KERNEL 4: Vector-Matrix Multiply (for dense layers in sparse tensor)
// ============================================================================
/*
 * Batched vector-matrix multiply: out[i] = features[i] @ weight + bias
 * Used for linear layers on sparse tensors.
 */
kernel void sparse_linear(
    device const float* features [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& C_in [[buffer(5)]],
    constant int& C_out [[buffer(6)]],
    constant int& has_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint voxel_idx = gid.x;
    uint out_ch = gid.y;
    
    if (voxel_idx >= uint(N) || out_ch >= uint(C_out)) return;
    
    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; c_in++) {
        sum += features[voxel_idx * C_in + c_in] * weight[c_in * C_out + out_ch];
    }
    
    if (has_bias) {
        sum += bias[out_ch];
    }
    
    output[voxel_idx * C_out + out_ch] = sum;
}
