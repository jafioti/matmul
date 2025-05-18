// Load tiles of memory into shared memory to reduce access latency in inner loop

#include <metal_stdlib>
using namespace metal;

constant uint M[[function_constant(0)]];
constant uint K[[function_constant(1)]];
constant uint N[[function_constant(2)]];

kernel void matmul(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device uint& BLOCK_SIZE [[buffer(3)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint thread_row = thread_position_in_threadgroup.x / BLOCK_SIZE;
    uint thread_col = thread_position_in_threadgroup.x % BLOCK_SIZE;
    uint row = threadgroup_position_in_grid.x;
    uint column = threadgroup_position_in_grid.y;

    // Advance pointers to starting points
    A += row * BLOCK_SIZE * K;
    B += column * BLOCK_SIZE;
    C += row * BLOCK_SIZE * N + column * BLOCK_SIZE;

    float tmp[1] = {0.0};

    // Setup shared memory pointers
    threadgroup float* As = shared_memory;
    threadgroup float* Bs = shared_memory + (BLOCK_SIZE * BLOCK_SIZE);

    // Tile loop
    for (int tileIndex = 0; tileIndex < K; tileIndex += BLOCK_SIZE) {
        // Load tile into shared memory
        As[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        Bs[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Advance pointers to next tile
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // Do matmul on SMEM block
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            tmp[0] += As[thread_row * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + thread_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[thread_row * N + thread_col] = tmp[0];
}
