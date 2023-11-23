const MULTI_PREFETCH_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

constexpr constant uint n_tiles = 2;

kernel void multi_prefetch(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* tiles_A [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    if (global_pos.y >= M || global_pos.x >= N) return;
    float sum = 0.0f;
    uint square_block_size = block_size.x * block_size.x;
    uint two_tile_size = square_block_size * 2;
    threadgroup float* tiles_B = tiles_A + two_tile_size * n_tiles;
    threadgroup float* temp;

    uint local_y_block_size = local_pos.y * block_size.x;
    uint a_addr = local_y_block_size + local_pos.x;
    uint b_addr = local_pos.y * block_size.x + local_pos.x + square_block_size;
    uint a_ind = global_pos.y * K + local_pos.x;

    // Fetch tilesA
    for (uint i = 0; i < n_tiles; ++i) {
        uint m = i * block_size.x;
        uint tileInd = i * two_tile_size;
        tiles_A[tileInd + a_addr] = A[a_ind + m];
        tiles_A[tileInd + b_addr] = B[(m + local_pos.y) * N + global_pos.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = n_tiles; i < (K / block_size.x); i += n_tiles) {
        // Prefetch tilesB
        for (uint tile = 0; tile < n_tiles; ++tile) {
            uint m = (i + tile) * block_size.x;
            uint tileInd = tile * two_tile_size;
            tiles_B[tileInd + a_addr] = A[a_ind + m];
            tiles_B[tileInd + b_addr] = B[(m + local_pos.y) * N + global_pos.x];
        }
        // Compute tilesA
        for (uint tile = 0; tile < n_tiles; ++tile) {
            uint tileInd = tile * two_tile_size;
            for (uint e = 0; e < block_size.x; ++e) {
                sum = fast::fma(tiles_A[tileInd + local_y_block_size + e], tiles_A[tileInd + e * block_size.x + local_pos.x + square_block_size], sum);
            }
        }
        // Swap tilesA and tilesB
        temp = tiles_A;
        tiles_A = tiles_B;
        tiles_B = temp;

        // Wait
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute tilesA
    for (uint tile = 0; tile < n_tiles; ++tile) {
        uint tileInd = tile * two_tile_size;
        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(tiles_A[tileInd + local_y_block_size + e], tiles_A[tileInd + e * block_size.x + local_pos.x + square_block_size], sum);
        }
    }

    C[global_pos.y * N + global_pos.x] = sum;
}";
