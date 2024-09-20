// Super straightforward matmul

#include <metal_stdlib>
using namespace metal;

constant uint M[[function_constant(0)]];
constant uint K[[function_constant(1)]];
constant uint N[[function_constant(2)]];

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.x;
    uint column = tid.y;

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = value;
    }
}
