use std::mem::size_of;

use metal::{
    objc::rc::autoreleasepool, Buffer, CommandQueue, ComputePassDescriptor, ComputePipelineState,
    Device, MTLResourceOptions, MTLSize,
};
use rand::Rng;

mod utils;
use crate::utils::SetInt;

const M: u64 = 4096;
const N: u64 = 4096;
const K: u64 = 4096;

const TRIALS: usize = 1;

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
                include_str!("kernels/0_naive.metal"),
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
                include_str!("kernels/1_warp_coalesced.metal"),
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
                include_str!("kernels/2_smem_tiled.metal"),
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
            "1D Register Tiled: {} ms",
            time_kernel(
                include_str!("kernels/3_1D_register_tile.metal"),
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
            "2D Register Tiled: {} ms",
            time_kernel(
                include_str!("kernels/4_2D_register_tile.metal"),
                (N / 64, M / 64, 1),
                ((64 * 64) / (8 * 8), 1, 1),
                &a_buf,
                &b_buf,
                &c,
                &[64, 64, 8],
                ((64 * 8) + (8 * 64)) * size_of::<f32>() as u64
            )
        );
        println!(
            "SIMD: {} ms",
            time_kernel(
                include_str!("kernels/5_simd.metal"),
                (N / 32, N / 256, 1),
                (32, 8, 1),
                &a_buf,
                &b_buf,
                &c,
                &[],
                0
            )
        );
        println!(
            "2D SIMD: {} ms",
            time_kernel(
                include_str!("kernels/6_2D_simd.metal"),
                (N / 32, N / 256, 1),
                (32, 8, 1),
                &a_buf,
                &b_buf,
                &c,
                &[],
                0
            )
        );
        println!(
            "SIMD Prefetch: {} ms",
            time_kernel(
                include_str!("kernels/7_simd_prefetch.metal"),
                (N / 32, N / 256, 1),
                (32, 8, 1),
                &a_buf,
                &b_buf,
                &c,
                &[],
                8 * 8 * 4 * 2 * 8 * 4 // height x width x num tiles x 2 inputs x 8 simdgroups per threadgroup
            )
        );
        println!(
            "MLX: {} ms",
            time_kernel(
                include_str!("kernels/8_mlx.metal"),
                (N / 32, M / 32, 1),
                (32, 2, 2),
                &a_buf,
                &b_buf,
                &c,
                &[0, 0, 0, 0],
                0,
            )
        );
        println!(
            "Simple SIMD: {} ms",
            time_kernel(
                include_str!("kernels/9_simd_single.metal"),
                (N / 8, N / 64, 1),
                (32, 8, 1),
                &a_buf,
                &b_buf,
                &c,
                &[],
                0
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
    encoder.set_u32(3, M as u32);
    encoder.set_u32(4, K as u32);
    encoder.set_u32(5, N as u32);
    for (i, inp) in other_inps.iter().enumerate() {
        encoder.set_u32(6 + i, *inp);
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
