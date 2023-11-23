use metal::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, CompileOptions,
    ComputeCommandEncoderRef, ComputePassDescriptor, ComputePipelineDescriptor,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const NAIVE_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void naieve(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 block_size[[threads_per_threadgroup]]
) {
    if (global_pos.x < N && global_pos.y < M) {
        float value = 0.0f;
        uint pos_k = global_pos.y * K;
        for (int i = 0; i < K; ++i) {
            value = fast::fma(A[pos_k + i], B[i * N + global_pos.x], value);
        }
        C[global_pos.y * N + global_pos.x] = value;
    }
}";

const TILED_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void tiled(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    if (global_pos.y >= M || global_pos.x >= N) return;
    float sum = 0.0f;

    threadgroup float* b_start = shared_memory + block_size.x * block_size.x;
    uint local_y_block_size = local_pos.y * block_size.x;
    uint a_addr = local_y_block_size + local_pos.x;
    uint b_addr = local_pos.y * block_size.x + local_pos.x;
    uint a_ind = global_pos.y * K + local_pos.x;
    for (uint m = 0; m < K; m += block_size.x) {
        shared_memory[a_addr] = A[a_ind + m];
        b_start[b_addr] = B[(m + local_pos.y) * N + global_pos.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(shared_memory[local_y_block_size + e], b_start[e * block_size.x + local_pos.x], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[global_pos.y * N + global_pos.x] = sum;
}";

const PREFETCH_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void prefetch(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* tile0 [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_size [[threads_per_threadgroup]]
) {
    if (global_pos.y >= M || global_pos.x >= N) return;
    float sum = 0.0f;
    uint square_block_size = block_size.x * block_size.x;
    threadgroup float* tile1 = tile0 + square_block_size * 2;
    threadgroup float* temp;

    uint local_y_block_size = local_pos.y * block_size.x;
    uint a_addr = local_y_block_size + local_pos.x;
    uint b_addr = local_pos.y * block_size.x + local_pos.x + square_block_size;
    uint a_ind = global_pos.y * K + local_pos.x;

    // First tile prefetch
    tile0[a_addr] = A[a_ind];
    tile0[b_addr] = B[local_pos.y * N + global_pos.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint m = block_size.x; m < K; m += block_size.x) {
        // Prefetch next block
        tile1[a_addr] = A[a_ind + m];
        tile1[b_addr] = B[(m + local_pos.y) * N + global_pos.x];

        // Compute current block
        for (uint e = 0; e < block_size.x; ++e) {
            sum = fast::fma(tile0[local_y_block_size + e], tile0[e * block_size.x + local_pos.x + square_block_size], sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Swap pointers
        temp = tile0;
        tile0 = tile1;
        tile1 = temp;        
    }

    // Compute final block
    for (uint e = 0; e < block_size.x; ++e) {
        sum = fast::fma(tile0[local_y_block_size + e], tile0[e * block_size.x + local_pos.x + square_block_size], sum);
    }

    C[global_pos.y * N + global_pos.x] = sum;
}";

fn setup_command_encoder(
    command_buffer: &CommandBufferRef,
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    c_buffer: &Buffer,
    shader: &ComputePipelineState,
    mat_size: usize,
) {
    let encoder =
        command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());

    encoder.set_compute_pipeline_state(shader);
    encoder.set_buffer(0, Some(a_buffer), 0);
    encoder.set_buffer(1, Some(b_buffer), 0);
    encoder.set_buffer(2, Some(c_buffer), 0);
    set_input_u32(encoder, mat_size as u32, 3);
    set_input_u32(encoder, mat_size as u32, 4);
    set_input_u32(encoder, mat_size as u32, 5);
    let thread_block_size = 32;
    encoder.set_threadgroup_memory_length(
        0,
        4 * thread_block_size * thread_block_size * std::mem::size_of::<f32>() as u64,
    );
    encoder.dispatch_threads(
        MTLSize {
            width: mat_size as u64,
            height: mat_size as u64,
            depth: 1,
        },
        MTLSize {
            width: thread_block_size,
            height: thread_block_size,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

fn run_n_times(
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    mat_size: usize,
    shader: &ComputePipelineState,
    dev: &Device,
    queue: &CommandQueue,
    n: usize,
) -> Option<(Vec<f32>, f32)> {
    autoreleasepool(|| {
        let c_buffer = copy_to_buffer(&vec![0.; mat_size * mat_size], dev);
        let command_buffer = queue.new_command_buffer();

        for _ in 0..n {
            setup_command_encoder(
                command_buffer,
                a_buffer,
                b_buffer,
                &c_buffer,
                shader,
                mat_size,
            );
        }

        command_buffer.commit();
        let now = std::time::Instant::now();
        command_buffer.wait_until_completed();
        let micros = now.elapsed().as_micros();
        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => {
                Some((copy_from_buffer(&c_buffer), micros as f32 / 1_000.))
            }
            _ => None,
        }
    })
}

fn main() {
    autoreleasepool(|| {
        let mat_size = 4096;
        let trials = 10;
        let mut rng = StdRng::seed_from_u64(0);
        let a_data: Vec<f32> = (0..(mat_size * mat_size))
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let b_data: Vec<f32> = (0..(mat_size * mat_size))
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();

        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let a_buffer = copy_to_buffer(&a_data, &dev);
        let b_buffer = copy_to_buffer(&b_data, &dev);

        let shaders = [
            (NAIVE_SHADER, "naieve"),
            (TILED_SHADER, "tiled"),
            (PREFETCH_SHADER, "prefetch"),
        ];
        let shader = compile_function("naieve", NAIVE_SHADER, &dev);
        let reference = run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 1)
            .unwrap()
            .0;
        for (shader, name) in shaders {
            println!("{name}");
            let shader = compile_function(name, shader, &dev);
            let res = run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 1)
                .unwrap()
                .0;
            println!("Res: {:?}", &res[..10]);
            assert_close(&res, &reference);
            println!(
                "1: {}ms",
                (0..trials)
                    .map(
                        |_| run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 1)
                            .unwrap()
                            .1
                    )
                    .sum::<f32>()
                    / trials as f32
            );
            println!(
                "2: {}ms",
                (0..trials)
                    .map(
                        |_| run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 2)
                            .unwrap()
                            .1
                    )
                    .sum::<f32>()
                    / trials as f32
            );
            println!(
                "5: {}ms",
                (0..trials)
                    .map(
                        |_| run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 5)
                            .unwrap()
                            .1
                    )
                    .sum::<f32>()
                    / trials as f32
            );
            // println!(
            //     "10: {}ms",
            //     (0..trials)
            //         .map(
            //             |_| run_n_times(&a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 10)
            //                 .unwrap()
            //                 .1
            //         )
            //         .sum::<f32>()
            //         / trials as f32
            // );
            // println!(
            //     "100: {}ms",
            //     (0..trials)
            //         .map(|_| run_n_times(
            //             &a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 100
            //         )
            //         .unwrap()
            //         .1)
            //         .sum::<f32>() / trials as f32
            // );
            // println!(
            //     "1000: {}ms",
            //     (0..trials)
            //         .map(|_| run_n_times(
            //             &a_buffer, &b_buffer, mat_size, &shader, &dev, &queue, 200
            //         )
            //         .unwrap()
            //         .1)
            //         .sum::<f32>() / trials as f32
            // );
        }
    })
}

fn set_input_u32(encoder: &ComputeCommandEncoderRef, num: u32, index: u64) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &(num) as *const u32 as *const _,
    );
}

fn copy_to_buffer(v: &[f32], dev: &Device) -> Buffer {
    dev.new_buffer_with_data(
        unsafe { std::mem::transmute(v.as_ptr()) },
        std::mem::size_of_val(v) as u64,
        MTLResourceOptions::StorageModeManaged,
    )
}

fn copy_from_buffer(buffer: &Buffer) -> Vec<f32> {
    let mut data = vec![0.0; buffer.length() as usize / std::mem::size_of::<f32>()];
    let ptr = buffer.contents() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_source(code, &CompileOptions::new())
        .unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, None).unwrap()));
    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

fn assert_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        assert!((a - b).abs() < 1e-3);
    }
}
