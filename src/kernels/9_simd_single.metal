// We load an 8x8 input matrix for each input.
// Accumulate into an 8x8 accumulator.
// This utilizes metal's special simdgroup_multiply_accumulate hardware.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size [[threads_per_threadgroup]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    // Step pointers (8 is the size of a simdgroup matrix)
    A += block_pos.x * 8 * K;
    B += global_pos.y * 8;
    C += block_pos.x * 8 * N + global_pos.y * 8;

    // Initialize accumulator
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    // Initialize simdgroup source matricies
    simdgroup_float8x8 simdA;
    simdgroup_float8x8 simdB;

    // K loop
    for (uint k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

        // Load sources into simdgroup matricies
        simdgroup_load(simdA, A + k, K);
        simdgroup_load(simdB, B + k * N, N);

        // Tile matmul
        simdgroup_multiply_accumulate(acc, simdA, simdB, acc);
    }

    // Save results
    simdgroup_store(acc, C, N);
}
