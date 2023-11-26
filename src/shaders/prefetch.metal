#include <metal_stdlib>
using namespace metal;

constexpr constant uint threadgroup_size = 8;

kernel void prefetch(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* tile0 [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    float sum = 0.0f;
    uint square_block_size = block_size.x * block_size.x;
    threadgroup float* tile1 = tile0 + square_block_size * 2;
    threadgroup float* temp;

    uint local_x_block_size = local_pos.x * block_size.x;
    uint local_y_square_block_size = local_pos.y + square_block_size;
    uint a_addr = local_x_block_size + local_pos.y;
    uint b_addr = a_addr + square_block_size;
    uint a_ind = global_pos.x * K + local_pos.y;

    // First tile prefetch
    tile0[a_addr] = local_pos.y < K ? A[a_ind] : 0.0f;
    tile0[b_addr] = local_pos.x < K ? B[local_pos.x * N + global_pos.y] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint m = block_size.x; m < K; m += block_size.x) {
        // Prefetch next block
        tile1[a_addr] = local_pos.y + m < K ? A[a_ind + m] : 0.0f;
        tile1[b_addr] = local_pos.x < K ? B[(m + local_pos.x) * N + global_pos.y] : 0.0f;

        // Compute current block
        #pragma unroll(threadgroup_size)
        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(tile0[local_x_block_size + e], tile0[e * block_size.x + local_y_square_block_size], sum);
        }

        // Swap pointers
        temp = tile0;
        tile0 = tile1;
        tile1 = temp;

        // Wait
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute final block
    #pragma unroll(threadgroup_size)
    for (uint e = 0; e < block_size.x; ++e) {
        sum = fast::fma(tile0[local_x_block_size + e], tile0[e * block_size.x + local_y_square_block_size], sum);
    }

    C[global_pos.x * N + global_pos.y] = sum;
}