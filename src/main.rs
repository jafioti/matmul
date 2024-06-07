use std::mem::size_of;

use metal::{
    objc::rc::autoreleasepool, Buffer, CommandQueue, ComputePassDescriptor, ComputePipelineState,
    Device, MTLResourceOptions, MTLSize,
};
use rand::Rng;

mod utils;

const M: u64 = 4096;
const N: u64 = 4096;
const K: u64 = 4096;

const TRIALS: usize = 5;

fn main() {
    autoreleasepool(|| {
        let mut rng = rand::thread_rng();
        let device = Device::system_default().unwrap();
        let a: Vec<f32> = (0..M * K).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let b: Vec<f32> = (0..K * N).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let c = utils::cpu_matmul(&a, &b, M as usize, N as usize, K as usize);
        let a_buf = utils::new_buffer(&a, &device);
        let b_buf = utils::new_buffer(&b, &device);

        println!(
            "Naive: {} ms",
            time_kernel(
                NAIVE,
                (M / 32, N / 32, 1),
                (32, 32, 1),
                &a_buf,
                &b_buf,
                &c,
                &[],
                0
            )
        );
        println!(
            "Warp Coalesced: {} ms",
            time_kernel(
                WARP_COALESCED,
                (M / 32, N / 32, 1),
                (32 * 32, 1, 1),
                &a_buf,
                &b_buf,
                &c,
                &[32],
                0
            )
        );
        println!(
            "SMEM Tiled: {} ms",
            time_kernel(
                SMEM_TILED,
                (M / 32, N / 32, 1),
                (32 * 32, 1, 1),
                &a_buf,
                &b_buf,
                &c,
                &[32],
                32 * 32 * 2 * size_of::<f32>() as u64
            )
        );
        println!(
            "Register 1D Tiled: {} ms",
            time_kernel(
                REGISTER_1D_TILE,
                (N / 64, M / 64, 1),
                ((64 * 64) / 8, 1, 1),
                &a_buf,
                &b_buf,
                &c,
                &[64, 64, 8],
                ((64 * 8) + (8 * 64)) * size_of::<f32>() as u64
            )
        );
        println!(
            "Register 2D Tiled: {} ms",
            time_kernel(
                REGISTER_2D_TILE,
                (N / 64, M / 64, 1),
                ((64 * 64) / (8 * 8), 1, 1),
                &a_buf,
                &b_buf,
                &c,
                &[64, 64, 8],
                ((64 * 8) + (8 * 64)) * size_of::<f32>() as u64
            )
        );
    })
}

#[allow(clippy::too_many_arguments)]
fn time_kernel(
    kernel: &str,
    grid_size: (u64, u64, u64),
    block_size: (u64, u64, u64),
    a_buf: &Buffer,
    b_buf: &Buffer,
    c: &[f32],
    other_inps: &[u32],
    shared_mem: u64,
) -> u128 {
    let device = Device::system_default().unwrap();
    let kernel = utils::compile_function("matmul", kernel, &device);
    let command_queue = device.new_command_queue();
    let c_buffer = device.new_buffer(M * N * 4, MTLResourceOptions::StorageModeShared);
    let start = std::time::Instant::now();
    for _ in 0..TRIALS {
        run_kernel(
            &command_queue,
            &kernel,
            a_buf,
            b_buf,
            &c_buffer,
            grid_size,
            block_size,
            other_inps,
            shared_mem,
        );
    }
    let total_time = start.elapsed().as_micros() / TRIALS as u128;
    let mut c_readback_data = vec![0.; (M * N) as usize];
    let ptr = c_buffer.contents() as *mut f32;
    for (i, d) in c_readback_data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }

    for (a, b) in c.iter().zip(c_readback_data.iter()) {
        if (*a - *b).abs() > 1e-3 {
            println!("{a} ne {b}");
        }
    }
    total_time / 1_000
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    queue: &CommandQueue,
    kernel: &ComputePipelineState,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    grid_size: (u64, u64, u64),
    block_size: (u64, u64, u64),
    other_inps: &[u32],
    shared_mem: u64,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder =
        command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_buffer(0, Some(a_buf), 0);
    encoder.set_buffer(1, Some(b_buf), 0);
    encoder.set_buffer(2, Some(c_buf), 0);
    encoder.set_bytes(3, 4, &(M as u32) as *const u32 as *const _);
    encoder.set_bytes(4, 4, &(K as u32) as *const u32 as *const _);
    encoder.set_bytes(5, 4, &(N as u32) as *const u32 as *const _);
    for (i, inp) in other_inps.iter().enumerate() {
        encoder.set_bytes(6 + i as u64, 4, inp as *const u32 as *const _);
    }
    encoder.set_threadgroup_memory_length(0, shared_mem);

    encoder.dispatch_thread_groups(
        MTLSize::new(grid_size.0, grid_size.1, grid_size.2),
        MTLSize::new(block_size.0, block_size.1, block_size.2),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
}

const NAIVE: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.x;
    uint column = tid.y;

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = value;
    }
}
";

// Access memory such that each warp accesses contiguous memory (each warp writes to a row in C)
const WARP_COALESCED: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    device uint& BLOCK_SIZE [[buffer(6)]],
    uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    uint row = threadgroup_position_in_grid.x * BLOCK_SIZE + (thread_position_in_threadgroup.x / BLOCK_SIZE);
    uint column = threadgroup_position_in_grid.y * BLOCK_SIZE + (thread_position_in_threadgroup.x % BLOCK_SIZE);

    if(row < M && column < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = value;
    }
}
";

// Load tiles of memory into shared memory to reduce access latency in inner loop
const SMEM_TILED: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& K [[buffer(4)]],
    device uint& N [[buffer(5)]],
    device uint& BLOCK_SIZE [[buffer(6)]],
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

    float tmp = 0.0;

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
            tmp += As[thread_row * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + thread_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    C[thread_row * N + thread_col] = tmp;
}
";

const REGISTER_1D_TILE: &str = "
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
    for (int i = 0; i < BK; ++i) {
        C[(thread_row * TM + i) * N + thread_col] = tmp[i];
    }
}
";

const REGISTER_2D_TILE: &str = "
#include <metal_stdlib>
using namespace metal;

#define TM 8
#define TN 8

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
";
