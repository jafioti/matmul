// We load a vector of 4 8x8 input matricies for each input.
// Accumulate into a 4x4 matrix of 8x8 accumulators.
// This utilizes metal's special simdgroup_multiply_accumulate hardware.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_simdgroup>
using namespace metal;

#define LENGTH 8

constant uint M[[function_constant(0)]];
constant uint K[[function_constant(1)]];
constant uint N[[function_constant(2)]];

kernel void matmul(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size [[threads_per_threadgroup]],
    uint3 global_pos [[thread_position_in_grid]]
) {
    // Step pointers
    A += block_pos.x * LENGTH * 8 * K;
    B += global_pos.y * LENGTH * 8;
    C += block_pos.x * LENGTH * 8 * N + global_pos.y * LENGTH * 8;

    // Initialize accumulating simdgroup matricies
    simdgroup_float8x8 acc[LENGTH][LENGTH];
    #pragma unroll(LENGTH)
    for (uint i = 0; i < LENGTH; ++i) {
        #pragma unroll(LENGTH)
        for (uint j = 0; j < LENGTH; ++j) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    // Initialize simdgroup source matricies
    simdgroup_float8x8 simdA[LENGTH];
    simdgroup_float8x8 simdB[LENGTH];

    // K loop
    for (uint k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

        // Load sources into simdgroup matricies
        #pragma unroll(LENGTH)
        for (int i = 0; i < LENGTH; ++i) {
            simdgroup_load(simdA[i], A + (i * 8 * K) + k, K); // K is row width, loop down
            simdgroup_load(simdB[i], B + (k * N) + (i * 8), N); // N is row width, loop right
        }

        // Do matmul by looping through the result matricies and multiply-accumulating them with the appropriate input mats
        #pragma unroll(LENGTH)
        for (int i = 0; i < LENGTH; ++i) {
            #pragma unroll(LENGTH)
            for (int j = 0; j < LENGTH; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], simdA[j], simdB[i], acc[i][j]);
            }
        }
    }

    // Save results
    #pragma unroll(LENGTH)
    for (int i = 0; i < LENGTH; ++i) {
        #pragma unroll(LENGTH)
        for (int j = 0; j < LENGTH; ++j) {
            simdgroup_store(acc[j][i], C + (i * 8 * N) + (j * 8), N);
        }
    }
}
