// We load an 8x8 input matrix for each input.
// Accumulate into an 8x8 accumulator.
// This utilizes metal's special simdgroup_multiply_accumulate hardware.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

constant uint M[[function_constant(0)]];
constant uint K[[function_constant(1)]];
constant uint N[[function_constant(2)]];

kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size [[threads_per_threadgroup]],
    uint3 thread_pos [[thread_position_in_threadgroup]]
) {
    // Initialize accumulating simdgroup matricies
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    // K loop
    for (uint k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

        // Load sources into simdgroup matricies
        simdgroup_float8x8 simdA;
        simdgroup_load(simdA, A + block_pos.x * 8 * K + k, K);
        simdgroup_float8x8 simdB;
        simdgroup_load(simdB, B + block_pos.y * 8 + (k * N), N);

        // Do matmul by looping through the result matricies and multiply-accumulating them with the appropriate input mats
        simdgroup_multiply_accumulate(acc, simdA, simdB, acc);
    }

    // Save results
    simdgroup_store(acc, C + block_pos.x * 8 * N + block_pos.y * 8, N);
}
