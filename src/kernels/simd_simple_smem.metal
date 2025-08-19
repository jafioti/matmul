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
    uint3 thread_pos [[thread_position_in_threadgroup]],
    threadgroup float* smem [[threadgroup(0)]]
){

    simdgroup_float8x8 acc = simdgroup_float8x8(0);
    threadgroup float* a_tile = smem;
    threadgroup float* b_tile = smem + (8 * 8);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        // Stage A and B elements in SMEM
        a_tile[thread_pos.x * 8 + (thread_pos.y * 2)] = A[(block_pos.x * 8 + thread_pos.x) * K + ((thread_pos.y * 2) + k0)];
        a_tile[thread_pos.x * 8 + ((thread_pos.y * 2) + 1)] = A[(block_pos.x * 8 + thread_pos.x) * K + ((thread_pos.y * 2) + 1 + k0)];
        b_tile[thread_pos.x * 8 + (thread_pos.y * 2)] = B[(thread_pos.x) * N + (block_pos.y * 8 + (thread_pos.y * 2)) + k0 * N];
        b_tile[thread_pos.x * 8 + ((thread_pos.y * 2) + 1)] = B[(thread_pos.x) * N + (block_pos.y * 8 + (thread_pos.y * 2) + 1) + k0 * N];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 simdA, simdB;
        simdgroup_load(simdA, a_tile, 8);
        simdgroup_load(simdB, b_tile, 8);
        simdgroup_multiply_accumulate(acc, simdA, simdB, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, C + (block_pos.x * 8) * N + (block_pos.y * 8), N);
}