#include <metal_stdlib>
using namespace metal;

kernel void naive(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    if (global_pos.y < M || global_pos.x < N) {
        float value = 0.0f;
        uint pos_k = global_pos.y * K;
        for (uint i = 0; i < K; ++i) {
            value = fast::fma(A[pos_k + i], B[i * N + global_pos.x], value);
        }
        C[global_pos.y * N + global_pos.x] = value;
    }
}