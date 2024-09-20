// Output a 2D matrix of TMxTN elements per thread

#include <metal_stdlib>
using namespace metal;

#define TM 8
#define TN 8

constant uint M[[function_constant(0)]];
constant uint K[[function_constant(1)]];
constant uint N[[function_constant(2)]];

kernel void matmul(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device uint& BM [[buffer(3)]],
    device uint& BN [[buffer(4)]],
    device uint& BK [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threads_per_threadgroup [[threads_per_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint row = threadgroup_position_in_grid.y;
    uint column = threadgroup_position_in_grid.x;

    uint thread_row = thread_position_in_threadgroup.x / (BN / TN);
    uint thread_col = thread_position_in_threadgroup.x % (BN / TN);

    // Setup shared memory pointers
    threadgroup float* As = shared_memory;
    threadgroup float* Bs = shared_memory + (BM * BK);

    // Advance pointers to starting points
    A += row * BM * K;
    B += column * BN;
    C += row * BM * N + column * BN;

    // warp-level GMEM coalescing
    uint innerColA = thread_position_in_threadgroup.x % BK;
    uint innerRowA = thread_position_in_threadgroup.x / BK;
    uint strideA = threads_per_threadgroup.x / BK;
    uint innerColB = thread_position_in_threadgroup.x % BN;
    uint innerRowB = thread_position_in_threadgroup.x / BN;
    uint strideB = threads_per_threadgroup.x / BN;

    float tmp[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // Tile loop
    for (int tileIndex = 0; tileIndex < K; tileIndex += BK) {
        // Load tile into shared memory
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Advance pointers to next tile
        A += BK;
        B += BK * N;

        // Do matmul on SMEM block
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            #pragma unroll(TM)
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(thread_row * TM + i) * BK + dotIdx];
            }
            #pragma unroll(TN)
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + thread_col * TN + i];
            }
            // dot products
            #pragma unroll(TM)
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll(TN)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    tmp[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll(TM)
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll(TN)
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(thread_row * TM + resIdxM) * N + thread_col * TN + resIdxN] = tmp[resIdxM * TN + resIdxN];
        }
    }
}
