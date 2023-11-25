#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
using namespace metal;

kernel void simple_simd(
     device float *a [[buffer(2)]],
     device const float *data1 [[buffer(0)]],
     device const float *data2 [[buffer(1)]],
     device uint& N [[buffer(4)]],
     uint3 gid [[threadgroup_position_in_grid]],
     uint3 global_id [[thread_position_in_grid]],
     uint3 block_size [[threads_per_threadgroup]]) {

  a += gid.x * 32 * N + global_id.y * 32;
  data1 += gid.x * 32 * N;
  data2 += global_id.y * 32;

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      acc[i][j] = simdgroup_float8x8(0);
    }
  }

  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  for (uint k = 0; k < N; k+=8) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float *d1 = data1+k;
    uint n8 = 8*N;
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        simdgroup_load(A[i], d1 + i * n8, N);
    }
    device const float *d2 = data2+k*N;
    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        simdgroup_load(B[i], 8 * i + d2, N);
    }

    #pragma unroll(4)
    for (int i = 0; i < 4; ++i) {
        #pragma unroll(4)
        for (int j = 0; j < 4; ++j) {
            simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
        }
    }
  }

  #pragma unroll(4)
  for (int i = 0; i < 4; ++i) {
    #pragma unroll(4)
    for (int j = 0; j < 4; ++j) {
        simdgroup_store(acc[j][i], a+(8*j+8*i*N), N);
    }
  }
}