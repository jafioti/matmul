// Output a 1D vector of TM elements per thread

#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    device uint& BM [[buffer(6)]],
    device uint& BN [[buffer(7)]],
    device uint& BK [[buffer(8)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint row = threadgroup_position_in_grid.y;
    uint column = threadgroup_position_in_grid.x;
    uint thread_row = thread_position_in_threadgroup.x / BN;
    uint thread_col = thread_position_in_threadgroup.x % BN;

    // Advance pointers to starting points
    A += row * BM * K;
    B += column * BN;
    C += row * BM * N + column * BN;

    const uint TM = 8;
    float tmp[TM] = {0.0};

    // Setup shared memory pointers
    threadgroup float* As = shared_memory;
    threadgroup float* Bs = shared_memory + (BM * BK);

    // warp-level GMEM coalescing
    uint innerColA = thread_position_in_threadgroup.x % BK;
    uint innerRowA = thread_position_in_threadgroup.x / BK;
    uint innerColB = thread_col;
    uint innerRowB = thread_row;

    // Tile loop
    for (int tileIndex = 0; tileIndex < K; tileIndex += BK) {
        // Load tile into shared memory
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Advance pointers to next tile
        A += BK;
        B += BK * N;

        // Do matmul on SMEM block
        for (int i = 0; i < BK; ++i) {
            float tmpB = Bs[i * BN + thread_col];
            #pragma unroll(TM)
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                tmp[resIdx] += As[(thread_row * TM + resIdx) * BK + i] * tmpB;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < TM; ++i) {
        C[(thread_row * TM + i) * N + thread_col] = tmp[i];
    }
}
