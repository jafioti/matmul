#include <metal_stdlib>
using namespace metal;

constexpr constant uint threadgroup_size = 8;

kernel void tiled(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    if (global_pos.y >= M || global_pos.x >= N) return;
    float sum = 0.0f;

    threadgroup float* b_start = shared_memory + block_size.x * block_size.x;
    uint local_y_block_size = local_pos.y * block_size.x;
    uint a_addr = local_y_block_size + local_pos.x;
    uint b_addr = local_pos.y * block_size.x + local_pos.x;
    uint a_ind = global_pos.y * K + local_pos.x;
    for (uint m = 0; m < K; m += block_size.x) {
        shared_memory[a_addr] = A[a_ind + m];
        b_start[b_addr] = B[(m + local_pos.y) * N + global_pos.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(threadgroup_size)
        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(shared_memory[local_y_block_size + e], b_start[e * block_size.x + local_pos.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[global_pos.y * N + global_pos.x] = sum;
}