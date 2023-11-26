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
    if (global_pos.x >= M || global_pos.y >= N) return;
    float sum = 0.0f;

    threadgroup float* b_start = shared_memory + block_size.x * block_size.x;
    uint shared_mem_addr = local_pos.x * block_size.x + local_pos.y;
    for (uint m = 0; m < K; m += block_size.x) {
        shared_memory[shared_mem_addr] = local_pos.y + m < K ? A[global_pos.x * K + local_pos.y + m] : 0.0f;
        b_start[shared_mem_addr] = m + local_pos.x < K ? B[(m + local_pos.x) * N + global_pos.y] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(threadgroup_size)
        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(shared_memory[local_pos.x * block_size.x + e], b_start[e * block_size.x + local_pos.y], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[global_pos.x * N + global_pos.y] = sum;
}