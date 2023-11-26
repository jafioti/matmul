#include <metal_stdlib>
using namespace metal;

// Second kernel is much faster. Why? Coalesced memory access?
// First is general, second only works with square matrixes

kernel void naive(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    if (global_pos.x < M || global_pos.y < N) {
        float value = 0.0f;
        uint pos_x = global_pos.x * K;
        for (uint i = 0; i < K; ++i) {
            value = fast::fma(A[pos_x + i], B[i * N + global_pos.y], value);
        }
        C[global_pos.x * N + global_pos.y] = value;
    }
}

/*kernel void naive(
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
        uint pos_y = global_pos.y * K;
        for (uint i = 0; i < K; ++i) {
            value = fast::fma(A[pos_y + i], B[i * N + global_pos.x], value);
        }
        C[global_pos.y * N + global_pos.x] = value;
    }
}*/