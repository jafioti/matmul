// Access memory such that each warp accesses contiguous memory
// Each warp writes to a row in C

#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    device uint& BLOCK_SIZE [[buffer(6)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint row = threadgroup_position_in_grid.x * BLOCK_SIZE + (thread_position_in_threadgroup.x / BLOCK_SIZE);
    uint column = threadgroup_position_in_grid.y * BLOCK_SIZE + (thread_position_in_threadgroup.x % BLOCK_SIZE);

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = value;
    }
}
