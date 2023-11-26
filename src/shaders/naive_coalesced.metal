#include <metal_stdlib>
using namespace metal;

kernel void naive_coalesced(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    uint x = block_pos.x * block_size.x + (local_pos.x / block_size.x);
    uint y = block_pos.y * block_size.x + (local_pos.x % block_size.x);
    if (x < M && y < N) {
        float value = 0.0f;
        for (uint i = 0; i < K; ++i) {
            value = fast::fma(A[x * K + i], B[i * N + y], value);
        }
        C[x * N + y] = value;
    }
}