// Same as SIMD, but we prefetch into shared memory one loop before we load from shared memory into simdgroup registers

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
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size [[threads_per_threadgroup]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint simdgroup_pos [[simdgroup_index_in_threadgroup]],
    uint local_simdgroup_pos [[thread_index_in_simdgroup]]
) {
    // Step pointers
    A += block_pos.x * block_size.x * K;
    B += global_pos.y * 32;
    C += block_pos.x * 32 * N + global_pos.y * 32;
    threadgroup float* sharedA = shared_memory + 8 * 8 * 4 * 2 * simdgroup_pos; // width x height x num tiles x num inputs x simdgroup_pos
    threadgroup float* sharedB = sharedA + 8 * 8 * 4; // height x width x num tiles

    // Initialize accumulating simdgroup matricies
    simdgroup_float8x8 acc[4][4];
    #pragma unroll(4)
    for (uint i = 0; i < 4; ++i) {
        #pragma unroll(4)
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    // Initialize simdgroup source matricies
    simdgroup_float8x8 simdA[4];
    simdgroup_float8x8 simdB[4];

    // Load initial sources into shared memory
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        // Load A into shared memory
        uint shared_mem_loading_index = local_simdgroup_pos * 2;
        sharedA[shared_mem_loading_index + 8 * 8 * i] = A[(i * 8 * K) + (shared_mem_loading_index / 8) * K + (shared_mem_loading_index % 8)];
        sharedA[shared_mem_loading_index + 8 * 8 * i + 1] = A[(i * 8 * K) + ((shared_mem_loading_index + 1) / 8) * K + ((shared_mem_loading_index + 1) % 8)];
        // Load B into shared memory
        sharedB[shared_mem_loading_index + 8 * 8 * i] = B[(i * 8) + (shared_mem_loading_index / 8) * N + (shared_mem_loading_index % 8)];
        sharedB[shared_mem_loading_index + 8 * 8 * i + 1] = B[(i * 8) + ((shared_mem_loading_index + 1) / 8) * N + ((shared_mem_loading_index + 1) % 8)];
    }

    // K loop
    for (uint k = 0; k < K; k+=8) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

        // Load sources into simdgroup matricies
        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            simdgroup_load(simdA[i], sharedA + 8 * 8 * i, 8); // K is row width, loop down
            simdgroup_load(simdB[i], sharedB + 8 * 8 * i, 8); // N is row width, loop right
        }

        // Load sources into shared memory
        if (k < K - 8) {
            A += 8;
            B += 8 * N;
            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                // Load A into shared memory
                uint shared_mem_loading_index = local_simdgroup_pos * 2;
                sharedA[shared_mem_loading_index + 8 * 8 * i] = A[(i * 8 * K) + (shared_mem_loading_index / 8) * K + (shared_mem_loading_index % 8)];
                sharedA[shared_mem_loading_index + 8 * 8 * i + 1] = A[(i * 8 * K) + ((shared_mem_loading_index + 1) / 8) * K + ((shared_mem_loading_index + 1) % 8)];
                // Load B into shared memory
                sharedB[shared_mem_loading_index + 8 * 8 * i] = B[(i * 8) + (shared_mem_loading_index / 8) * N + (shared_mem_loading_index % 8)];
                sharedB[shared_mem_loading_index + 8 * 8 * i + 1] = B[(i * 8) + ((shared_mem_loading_index + 1) / 8) * N + ((shared_mem_loading_index + 1) % 8)];
            }
        }

        // Do matmul by looping through the result matricies and multiply-accumulating them with the appropriate input mats
        #pragma unroll(4)
        for (int i = 0; i < 4; ++i) {
            #pragma unroll(4)
            for (int j = 0; j < 4; ++j) {
                simdgroup_multiply_accumulate(acc[i][j], simdA[j], simdB[i], acc[i][j]);
            }
        }
    }

    // Save results
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        #pragma unroll(4)
        for (int j = 0; j < 4; ++j) {
            simdgroup_store(acc[j][i], C + (i * 8 * N) + (j * 8), N);
        }
    }
}
