use metal::{
    objc::rc::autoreleasepool, Buffer, CommandBufferRef, CommandQueue, CompileOptions,
    ComputeCommandEncoderRef, ComputePassDescriptor, ComputePipelineDescriptor,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{fs, mem::size_of};

fn read_shader_from_file(file_path: &str) -> String {
    fs::read_to_string(file_path)
        .unwrap_or_else(|_| panic!("Failed to read shader file {}", file_path))
}

#[allow(clippy::too_many_arguments)]
fn setup_command_encoder(
    command_buffer: &CommandBufferRef,
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    c_buffer: &Buffer,
    shader: &ComputePipelineState,
    threadgroups_per_grid: MTLSize,
    threads_per_threadgroup: MTLSize,
    mat_size: usize,
    threadgroup_memory: u64,
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
    encoder.set_threadgroup_memory_length(
        0,
        threadgroup_memory,
        // 4 * thread_block_size * thread_block_size * std::mem::size_of::<f32>() as u64,
    );
    encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    encoder.end_encoding();
}

#[allow(clippy::too_many_arguments)]
fn run_n_times(
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    mat_size: usize,
    shader: &ComputePipelineState,
    dev: &Device,
    queue: &CommandQueue,
    n: usize,
    threadgroups_per_grid: MTLSize,
    threads_per_threadgroup: MTLSize,
    threadgroup_memory: u64,
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
                threadgroups_per_grid,
                threads_per_threadgroup,
                mat_size,
                threadgroup_memory,
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
        let thread_block_size = 32;
        // let mat_size = 16;
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
            (
                "./src/shaders/naive.metal",
                "naive",
                MTLSize {
                    width: mat_size as u64 / thread_block_size,
                    height: mat_size as u64 / thread_block_size,
                    depth: 1,
                },
                MTLSize {
                    width: thread_block_size,
                    height: thread_block_size,
                    depth: 1,
                },
                0,
            ),
            (
                "./src/shaders/tiled.metal",
                "tiled",
                MTLSize {
                    width: mat_size as u64 / thread_block_size,
                    height: mat_size as u64 / thread_block_size,
                    depth: 1,
                },
                MTLSize {
                    width: thread_block_size,
                    height: thread_block_size,
                    depth: 1,
                },
                thread_block_size * thread_block_size * 4 * size_of::<f32>() as u64,
            ),
            (
                "./src/shaders/prefetch.metal",
                "prefetch",
                MTLSize {
                    width: mat_size as u64 / thread_block_size,
                    height: mat_size as u64 / thread_block_size,
                    depth: 1,
                },
                MTLSize {
                    width: thread_block_size,
                    height: thread_block_size,
                    depth: 1,
                },
                thread_block_size * thread_block_size * 4 * size_of::<f32>() as u64,
            ),
            (
                "./src/shaders/simd.metal",
                "simple_simd",
                MTLSize {
                    width: mat_size as u64 / thread_block_size,
                    height: mat_size as u64 / (32 * 8),
                    depth: 1,
                },
                MTLSize {
                    width: thread_block_size,
                    height: 8,
                    depth: 1,
                },
                0,
            ),
        ];

        let shader_source = read_shader_from_file("./src/shaders/naive.metal");
        let shader = compile_function("naive", &shader_source, &dev);
        let reference = run_n_times(
            &a_buffer,
            &b_buffer,
            mat_size,
            &shader,
            &dev,
            &queue,
            1,
            MTLSize {
                width: mat_size as u64 / thread_block_size,
                height: mat_size as u64 / thread_block_size,
                depth: 1,
            },
            MTLSize {
                width: thread_block_size,
                height: thread_block_size,
                depth: 1,
            },
            0,
        )
        .unwrap()
        .0;
        for (
            shader_path,
            name,
            threadgroups_per_grid,
            threads_per_threadgroup,
            threadgroup_memory,
        ) in shaders
        {
            println!("{name}");
            let shader_source = read_shader_from_file(shader_path);
            let compiled_shader = compile_function(name, &shader_source, &dev);
            let res = run_n_times(
                &a_buffer,
                &b_buffer,
                mat_size,
                &compiled_shader,
                &dev,
                &queue,
                1,
                threadgroups_per_grid,
                threads_per_threadgroup,
                threadgroup_memory,
            )
            .unwrap()
            .0;
            println!("Res: {:?}", &res[res.len() - 10..]);

            assert_close(&res, &reference);
            println!(
                "1: {}ms",
                (0..trials)
                    .map(|_| run_n_times(
                        &a_buffer,
                        &b_buffer,
                        mat_size,
                        &compiled_shader,
                        &dev,
                        &queue,
                        1,
                        threadgroups_per_grid,
                        threads_per_threadgroup,
                        threadgroup_memory,
                    )
                    .unwrap()
                    .1)
                    .sum::<f32>()
                    / trials as f32
            );
            // println!(
            //     "2: {}ms",
            //     (0..trials)
            //         .map(|_| run_n_times(
            //             &a_buffer,
            //             &b_buffer,
            //             mat_size,
            //             &compiled_shader,
            //             &dev,
            //             &queue,
            //             2,
            //             threads_per_grid,
            //             threads_per_threadgroup,
            //             threadgroup_memory,
            //         )
            //         .unwrap()
            //         .1)
            //         .sum::<f32>()
            //         / trials as f32
            // );
            // println!(
            //     "5: {}ms",
            //     (0..trials)
            //         .map(|_| run_n_times(
            //             &a_buffer,
            //             &b_buffer,
            //             mat_size,
            //             &compiled_shader,
            //             &dev,
            //             &queue,
            //             5,
            //             simd_size,
            //         )
            //         .unwrap()
            //         .1)
            //         .sum::<f32>()
            //         / trials as f32
            // );
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
    for (i, (a, b)) in a.iter().zip(b.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "{i}");
    }
}
