use metal::{
    objc::rc::autoreleasepool, Buffer, BufferRef, CommandBufferRef, CompileOptions,
    ComputeCommandEncoderRef, ComputePassDescriptor, ComputePassDescriptorRef,
    ComputePipelineDescriptor, ComputePipelineState, CounterSampleBuffer, CounterSampleBufferRef,
    Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const NUM_SAMPLES: u64 = 20;

const NAIEVE_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    if(tid.y < M && tid.x < N) {
        float value = 0.0f;
        for(int i = 0; i < K; ++i) {
            value = fast::fma(A[tid.y * K + i], B[i * N + tid.x], value);
        }
        C[tid.y * N + tid.x] = value;
    }
}";

const TILED_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void tiled_matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size[[threads_per_threadgroup]]
) {
    int row = block_pos.y * block_size.y + local_pos.y;
    int col = block_pos.x * block_size.x + local_pos.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int m = 0; m < (K + block_size.x - 1) / block_size.x; ++m) {
        if (m * block_size.x + local_pos.x < K) {
            shared_memory[local_pos.y * block_size.x + local_pos.x] = A[row * K + m * block_size.x + local_pos.x];
        } else {
            shared_memory[local_pos.y * block_size.x + local_pos.x] = 0.0f;
        }

        if (m * block_size.y + local_pos.y < K) {
            shared_memory[(block_size.y + local_pos.y) * block_size.x + local_pos.x] = B[(m * block_size.y + local_pos.y) * N + col];
        } else {
            shared_memory[(block_size.y + local_pos.y) * block_size.x + local_pos.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int e = 0; e < block_size.x; ++e) {
            sum = fast::fma(shared_memory[local_pos.y * block_size.x + e], shared_memory[(block_size.y + e) * block_size.x + local_pos.x], sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[row * N + col] = sum;
}";

const PREFETCH_SHADER: &str = "
#include <metal_stdlib>
using namespace metal;

kernel void prefetch_matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    device uint& M [[buffer(3)]],
    device uint& N [[buffer(4)]],
    device uint& K [[buffer(5)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint3 global_pos [[thread_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint3 block_pos [[threadgroup_position_in_grid]],
    uint3 block_size[[threads_per_threadgroup]]
) {
    int row = block_pos.y * block_size.y + local_pos.y;
    int col = block_pos.x * block_size.x + local_pos.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int numTiles = (K + block_size.x - 1) / block_size.x;

    threadgroup float* tile0 = shared_memory;
    threadgroup float* tile1 = shared_memory + block_size.x * block_size.y * 2;

    if (local_pos.x < K) {
        tile0[local_pos.y * block_size.x + local_pos.x] = A[row * K + local_pos.x];
    } else {
        tile0[local_pos.y * block_size.x + local_pos.x] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int m = 1; m < numTiles; ++m) {
        if (m * block_size.x + local_pos.x < K) {
            tile1[local_pos.y * block_size.x + local_pos.x] = A[row * K + m * block_size.x + local_pos.x];
        } else {
            tile1[local_pos.y * block_size.x + local_pos.x] = 0.0f;
        }

        for (int e = 0; e < block_size.x; ++e) {
            sum = fast::fma(tile0[local_pos.y * block_size.x + e], B[(m - 1) * block_size.y * N + e * N + col], sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* temp = tile0;
        tile0 = tile1;
        tile1 = temp;
    }

    for (int e = 0; e < block_size.x; ++e) {
        sum = fast::fma(tile0[local_pos.y * block_size.x + e], B[(numTiles - 1) * block_size.y * N + e * N + col], sum);
    }

    C[row * N + col] = sum;
}";

fn run(
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    shader: &ComputePipelineState,
    dev: &Device,
    mat_size: usize,
) -> Option<(Vec<f32>, f32)> {
    autoreleasepool(|| {
        let mut cpu_start = 0;
        let mut gpu_start = 0;
        dev.sample_timestamps(&mut cpu_start, &mut gpu_start);

        let counter_sample_buffer = create_counter_sample_buffer(dev);
        let destination_buffer = dev.new_buffer(
            (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = dev.new_buffer(
            (mat_size * mat_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();

        let compute_pass_descriptor = ComputePassDescriptor::new();
        handle_compute_pass_sample_buffer_attachment(
            compute_pass_descriptor,
            &counter_sample_buffer,
        );

        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor);

        encoder.set_compute_pipeline_state(shader);
        encoder.set_buffer(0, Some(a_buffer), 0);
        encoder.set_buffer(1, Some(b_buffer), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        set_input_u32(encoder, 3, mat_size as u64);
        set_input_u32(encoder, 4, mat_size as u64);
        set_input_u32(encoder, 5, mat_size as u64);
        let thread_block_size = 32;
        encoder.set_threadgroup_memory_length(
            0,
            thread_block_size * thread_block_size * 2 * 2 * std::mem::size_of::<f32>() as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (mat_size as u64 + thread_block_size - 1) / thread_block_size,
                height: (mat_size as u64 + thread_block_size - 1) / thread_block_size,
                depth: 1,
            },
            MTLSize {
                width: thread_block_size,
                height: thread_block_size,
                depth: 1,
            },
        );
        encoder.end_encoding();
        resolve_samples_into_buffer(command_buffer, &counter_sample_buffer, &destination_buffer);
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let mut cpu_end = 0;
        let mut gpu_end = 0;
        dev.sample_timestamps(&mut cpu_end, &mut gpu_end);
        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => Some((
                copy_from_buffer(&c_buffer),
                handle_timestamps(&destination_buffer, cpu_start, cpu_end, gpu_start, gpu_end),
            )),
            _ => None,
        }
    })
}

fn main() {
    autoreleasepool(|| {
        let mat_size = 4096 * 2;
        let iters = 100;
        let mut rng = StdRng::seed_from_u64(0);
        let a_data: Vec<f32> = (0..(mat_size * mat_size))
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let b_data: Vec<f32> = (0..(mat_size * mat_size))
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();

        let dev = Device::system_default().unwrap();
        let a_buffer = copy_to_buffer(&a_data, &dev);
        let b_buffer = copy_to_buffer(&b_data, &dev);

        let shader = compile_function("matmul", NAIEVE_SHADER, &dev);
        let mut data: Option<Vec<f32>> = None;
        let mut successes = 0;
        let mut total_time = 0.0;
        for _ in 0..iters {
            let curr_data = run(&a_buffer, &b_buffer, &shader, &dev, mat_size);
            if let Some((curr_data, time)) = curr_data {
                match &mut data {
                    Some(d) => {
                        successes += 1;
                        total_time += time;
                        for (i, (a, b)) in d.iter().zip(curr_data.iter()).enumerate() {
                            if (*a - *b).abs() > 1e-5 {
                                println!("Index {i} A: {a} B: {b}");
                            }
                        }
                    }
                    None => {
                        data = Some(curr_data);
                    }
                }
            }
        }
        println!("Naieve Time: {}ms", total_time / successes as f32);

        let shader = compile_function("tiled_matmul", TILED_SHADER, &dev);
        let mut successes = 0;
        let mut total_time = 0.0;
        for _ in 0..iters {
            let curr_data = run(&a_buffer, &b_buffer, &shader, &dev, mat_size);
            if let Some((curr_data, time)) = curr_data {
                match &mut data {
                    Some(d) => {
                        successes += 1;
                        total_time += time;
                        for (i, (a, b)) in d.iter().zip(curr_data.iter()).enumerate() {
                            if (*a - *b).abs() > 1e-5 {
                                println!("Index {i} A: {a} B: {b}");
                            }
                        }
                    }
                    None => {
                        data = Some(curr_data);
                    }
                }
            }
        }

        println!("Tiled Time: {}ms", total_time / successes as f32);

        let shader = compile_function("prefetch_matmul", PREFETCH_SHADER, &dev);
        let mut successes = 0;
        let mut total_time = 0.0;
        for _ in 0..iters {
            let curr_data = run(&a_buffer, &b_buffer, &shader, &dev, mat_size);
            if let Some((curr_data, time)) = curr_data {
                match &mut data {
                    Some(d) => {
                        successes += 1;
                        total_time += time;
                        for (i, (a, b)) in d.iter().zip(curr_data.iter()).enumerate() {
                            if (*a - *b).abs() > 1e-5 {
                                println!("Index {i} A: {a} B: {b}");
                            }
                        }
                    }
                    None => {
                        data = Some(curr_data);
                    }
                }
            }
        }

        println!("Prefetch Time: {}ms", total_time / successes as f32);
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
        MTLResourceOptions::StorageModeShared,
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
    let opts = CompileOptions::new();
    opts.set_preserve_invariance(true);
    let library = device.new_library_with_source(code, &opts).unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, None).unwrap()));
    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

fn handle_compute_pass_sample_buffer_attachment(
    compute_pass_descriptor: &ComputePassDescriptorRef,
    counter_sample_buffer: &CounterSampleBufferRef,
) {
    let sample_buffer_attachment_descriptor = compute_pass_descriptor
        .sample_buffer_attachments()
        .object_at(0)
        .unwrap();

    sample_buffer_attachment_descriptor.set_sample_buffer(counter_sample_buffer);
    sample_buffer_attachment_descriptor.set_start_of_encoder_sample_index(0);
    sample_buffer_attachment_descriptor.set_end_of_encoder_sample_index(1);
}

fn resolve_samples_into_buffer(
    command_buffer: &CommandBufferRef,
    counter_sample_buffer: &CounterSampleBufferRef,
    destination_buffer: &BufferRef,
) {
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.resolve_counters(
        counter_sample_buffer,
        NSRange::new(0_u64, NUM_SAMPLES),
        destination_buffer,
        0_u64,
    );
    blit_encoder.end_encoding();
}

fn handle_timestamps(
    resolved_sample_buffer: &BufferRef,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> f32 {
    let samples = unsafe {
        std::slice::from_raw_parts(
            resolved_sample_buffer.contents() as *const u64,
            NUM_SAMPLES as usize,
        )
    };
    let pass_start = samples[0];
    let pass_end = samples[1];

    let cpu_time_span = cpu_end - cpu_start;
    let gpu_time_span = gpu_end - gpu_start;

    let millis = milliseconds_between_begin(pass_start, pass_end, gpu_time_span, cpu_time_span);
    millis as f32
}

fn milliseconds_between_begin(begin: u64, end: u64, gpu_time_span: u64, cpu_time_span: u64) -> f64 {
    let time_span = (end as f64) - (begin as f64);
    let nanoseconds = time_span / (gpu_time_span as f64) * (cpu_time_span as f64);
    nanoseconds / 1_000_000.0
}

fn create_counter_sample_buffer(device: &Device) -> CounterSampleBuffer {
    let counter_sample_buffer_desc = metal::CounterSampleBufferDescriptor::new();
    counter_sample_buffer_desc.set_storage_mode(metal::MTLStorageMode::Shared);
    counter_sample_buffer_desc.set_sample_count(NUM_SAMPLES);
    let counter_sets = device.counter_sets();

    let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp");

    counter_sample_buffer_desc
        .set_counter_set(timestamp_counter.expect("No timestamp counter found"));

    device
        .new_counter_sample_buffer_with_descriptor(&counter_sample_buffer_desc)
        .unwrap()
}
