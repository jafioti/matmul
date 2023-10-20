use std::ffi::c_void;

use metal::{
    objc::rc::autoreleasepool, Buffer, CompileOptions, ComputePassDescriptor,
    ComputePipelineDescriptor, ComputePipelineState, Device, FunctionConstantValues, MTLDataType,
    MTLResourceOptions, MTLSize,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn run_tiled_prefetch(
    a: &Buffer,
    b: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    dev: &Device,
) -> (Vec<f32>, f32) {
    autoreleasepool(|| {
        let shader = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
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
    int tx = local_pos.x;
    int ty = local_pos.y;
    int bx = block_pos.x;
    int by = block_pos.y;
    int bx_size = block_size.x;
    int by_size = block_size.y;

    threadgroup float* shared_memory_0 = shared_memory;
    threadgroup float* shared_memory_1 = shared_memory + (block_size.x * block_size.y * 2);
    int row = by * by_size + ty;
    int col = bx * bx_size + tx;

    float sum = 0.0f;

    int m = 0;
    if (m * bx_size + tx < K && row < M) {
        shared_memory_0[ty * bx_size + tx] = A[row * K + m * bx_size + tx];
    } else {
        shared_memory_0[ty * bx_size + tx] = 0.0f;
    }

    if (m * by_size + ty < K && col < N) {
        shared_memory_0[(by_size + ty) * bx_size + tx] = B[(m * by_size + ty) * N + col];
    } else {
        shared_memory_0[(by_size + ty) * bx_size + tx] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int m = 0; m < (K + bx_size - 1) / bx_size; m++) {
        if ((m + 1) * bx_size + tx < K && row < M) {
            shared_memory_1[ty * bx_size + tx] = A[row * K + (m + 1) * bx_size + tx];
        } else {
            shared_memory_1[ty * bx_size + tx] = 0.0f;
        }

        if ((m + 1) * by_size + ty < K && col < N) {
            shared_memory_1[(by_size + ty) * bx_size + tx] = B[((m + 1) * by_size + ty) * N + col];
        } else {
            shared_memory_1[(by_size + ty) * bx_size + tx] = 0.0f;
        }

        for (int e = 0; e < bx_size; e++) {
            sum += shared_memory_0[ty * bx_size + e] * shared_memory_0[(by_size + e) * bx_size + tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* tmp = shared_memory_0;
        shared_memory_0 = shared_memory_1;
        shared_memory_1 = tmp;
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
    ";
        let c_buffer = dev.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let shader = compile_function("matmul", shader, dev, FunctionConstantValues::new());

        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        encoder.set_compute_pipeline_state(&shader);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(m as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(k as u32) as *const u32 as *const _,
        );
        let thread_block_size = 32;
        encoder.set_threadgroup_memory_length(
            0,
            thread_block_size * thread_block_size * 2 * 2 * std::mem::size_of::<f32>() as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64 + thread_block_size - 1) / thread_block_size,
                height: (n as u64 + thread_block_size - 1) / thread_block_size,
                depth: 1,
            },
            MTLSize {
                width: thread_block_size,
                height: thread_block_size,
                depth: 1,
            },
        );
        let now = std::time::Instant::now();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let millis = now.elapsed().as_millis();

        let mut c_data = vec![0.0; c_buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = c_buffer.contents() as *mut f32;
        for (i, d) in c_data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        (c_data, millis as f32)
    })
}

fn run_tiled(
    a: &Buffer,
    b: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    dev: &Device,
    stored_shader: &mut Option<ComputePipelineState>,
) -> (Vec<f32>, f32) {
    autoreleasepool(|| {
        let shader = "
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
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
    int tx = local_pos.x;
    int ty = local_pos.y;
    int bx = block_pos.x;
    int by = block_pos.y;
    int bx_size = block_size.x;
    int by_size = block_size.y;

    int row = by * by_size + ty;
    int col = bx * bx_size + tx;

    float sum = 0.0f;

    for (int m = 0; m < (K + bx_size - 1) / bx_size; m++) {
        if (m * bx_size + tx < K && row < M) {
            shared_memory[ty * bx_size + tx] = A[row * K + m * bx_size + tx];
        } else {
            shared_memory[ty * bx_size + tx] = 0.0f;
        }

        if (m * by_size + ty < K && col < N) {
            shared_memory[(by_size + ty) * bx_size + tx] = B[(m * by_size + ty) * N + col];
        } else {
            shared_memory[(by_size + ty) * bx_size + tx] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int e = 0; e < bx_size; e++) {
            sum += shared_memory[ty * bx_size + e] * shared_memory[(by_size + e) * bx_size + tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
    ";
        let c_buffer = dev.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        if stored_shader.is_none() {
            *stored_shader = Some(compile_function(
                "matmul",
                shader,
                dev,
                FunctionConstantValues::new(),
            ));
        }
        encoder.set_compute_pipeline_state(stored_shader.as_ref().unwrap());
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(m as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(k as u32) as *const u32 as *const _,
        );
        let thread_block_size = 32;
        encoder.set_threadgroup_memory_length(
            0,
            thread_block_size * thread_block_size * 2 * std::mem::size_of::<f32>() as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64 + thread_block_size - 1) / thread_block_size,
                height: (n as u64 + thread_block_size - 1) / thread_block_size,
                depth: 1,
            },
            MTLSize {
                width: thread_block_size,
                height: thread_block_size,
                depth: 1,
            },
        );
        let now = std::time::Instant::now();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let millis = now.elapsed().as_millis();

        let mut c_data = vec![0.0; c_buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = c_buffer.contents() as *mut f32;
        for (i, d) in c_data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        (c_data, millis as f32)
    })
}

fn run_naive(
    a: &Buffer,
    b: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    dev: &Device,
    stored_shader: &mut Option<ComputePipelineState>,
) -> (Vec<f32>, f32) {
    autoreleasepool(|| {
        let shader = "#include <metal_stdlib>
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
        uint row = tid.y;
        uint column = tid.x;
    
        if(row < M && column < N) {
            float value = 0.0f;
            for(int i = 0; i < K; ++i) {
                uint A_index = row * K + i;
                uint B_index = i * N + column;
                value += A[A_index] * B[B_index];
            }
            C[row * N + column] = value;
        }
    }
    ";
        let c_buffer = dev.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        let command_queue = dev.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder =
            command_buffer.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
        if stored_shader.is_none() {
            *stored_shader = Some(compile_function(
                "matmul",
                shader,
                dev,
                FunctionConstantValues::new(),
            ));
        }
        encoder.set_compute_pipeline_state(stored_shader.as_ref().unwrap());
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(&c_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(m as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(n as u32) as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &(k as u32) as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (m as u64 + 15) / 16,
                height: (n as u64 + 15) / 16,
                depth: 1,
            },
            MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            },
        );
        let now = std::time::Instant::now();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let millis = now.elapsed().as_millis();

        let mut c_data = vec![0.0; c_buffer.length() as usize / std::mem::size_of::<f32>()];
        let ptr = c_buffer.contents() as *mut f32;
        for (i, d) in c_data.iter_mut().enumerate() {
            *d = unsafe { *ptr.add(i) };
        }
        (c_data, millis as f32)
    })
}

fn main() {
    let mat_size = 2048;
    let iters = 500;
    let mut rng = StdRng::seed_from_u64(0);
    let a_data: Vec<f32> = (0..(mat_size * mat_size))
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    let b_data: Vec<f32> = (0..(mat_size * mat_size))
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();
    let dev = Device::system_default().unwrap();
    let a_buffer = dev.new_buffer_with_data(
        unsafe { std::mem::transmute(a_data.as_ptr()) },
        (a_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let b_buffer = dev.new_buffer_with_data(
        unsafe { std::mem::transmute(b_data.as_ptr()) },
        (b_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let (mut naieve, mut tiled, mut prefetch) = (0., 0., 0.);
    let (mut naieve_shader, mut tiled_shader) = (None, None);
    for _ in 0..iters {
        let (a, a_time) = autoreleasepool(|| {
            run_naive(
                &a_buffer,
                &b_buffer,
                mat_size,
                mat_size,
                mat_size,
                &dev,
                &mut naieve_shader,
            )
        });
        naieve += a_time;
        let (b, b_time) = autoreleasepool(|| {
            run_tiled(
                &a_buffer,
                &b_buffer,
                mat_size,
                mat_size,
                mat_size,
                &dev,
                &mut tiled_shader,
            )
        });
        tiled += b_time;
        assert_eq!(a, b, "A not equal to B");
        // let (c, c_time) = autoreleasepool(|| {
        //     run_tiled_prefetch(&a_buffer, &b_buffer, mat_size, mat_size, mat_size, &dev)
        // });
        // prefetch += c_time;
        // assert_eq!(b, c, "B not equal to C");
    }
    println!(
        "Naieve: {}ms, Tiled: {}ms, Prefetch: {}ms",
        naieve / iters as f32,
        tiled / iters as f32,
        prefetch / iters as f32,
    );
}

trait SetConstant {
    fn constant<T>(&self, val: T, index: u64, dtype: MTLDataType);
}

impl SetConstant for FunctionConstantValues {
    fn constant<T>(&self, val: T, index: u64, dtype: MTLDataType) {
        self.set_constant_value_at_index(&val as *const T as *const c_void, dtype, index);
    }
}

fn compile_function(
    name: &str,
    code: &str,
    device: &Device,
    constants: FunctionConstantValues,
) -> ComputePipelineState {
    let library = device
        .new_library_with_source(code, &CompileOptions::new())
        .unwrap();
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&library.get_function(name, Some(constants)).unwrap()));

    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}
