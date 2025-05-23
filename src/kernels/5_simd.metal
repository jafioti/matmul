// We load a vector of 4 8x8 input matricies for each input.
// Accumulate into a 4x4 matrix of 8x8 accumulators.
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
    simdgroup_float8x8 acc[16] = {simdgroup_float8x8(0)};

    // K loop
    for (uint k = 0; k < K; k+=8) {
        // threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

        // Do matmul by looping through the result matricies and multiply-accumulating them with the appropriate input mats
        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            #pragma unroll(4)
            for (int j = 0; j < 4; ++j) {
            // simdgroup_float8x8 acc = simdgroup_float8x8(0.0);

	            simdgroup_float8x8 simdA[1];
	           	simdgroup_load(simdA[0], A + block_pos.x * 32 * K + j * 8 * K + k, K);
	            simdgroup_float8x8 simdB[1];
	        	simdgroup_load(simdB[0], B + (block_pos.y * 8 + thread_pos.y) * 32 + (k * N) + (i * 8), N);
                simdgroup_multiply_accumulate(acc[i * 4 + j], simdA[0], simdB[0], acc[i * 4 + j]);
            }
        }
    }

    // Save results
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        #pragma unroll(4)
        for (int j = 0; j < 4; ++j) {
        	simdgroup_store(acc[i * 4 + j], C + block_pos.x * 32 * N + (block_pos.y * 8 + thread_pos.y) * 32 + (j * 8 * N) + (i * 8), N);
        }
    }
}
