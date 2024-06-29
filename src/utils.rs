#![allow(unused)]

use metal::*;

pub fn compile_lib(device: &Device, source: &str) -> Library {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    device.new_library_with_source(source, &options).unwrap()
}

pub fn select_function_from_lib(
    lib: &Library,
    function: &str,
    device: &Device,
) -> ComputePipelineState {
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&lib.get_function(function, None).unwrap()));
    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

pub fn new_buffer(data: &[f32], device: &Device) -> Buffer {
    device.new_buffer_with_bytes_no_copy(
        data.as_ptr() as _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    )
}

pub fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let library = compile_lib(device, code);
    select_function_from_lib(&library, name, device)
}

pub fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    c
}

pub fn cpu_transpose(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            transposed[j * m + i] = a[i * n + j];
        }
    }

    transposed
}

#[allow(dead_code)]
pub trait SetInt {
    fn set_i32(&self, index: usize, value: i32);
    fn set_u32(&self, index: usize, value: u32);
    fn set_f32(&self, index: usize, value: f32);
    fn set_i64(&self, index: usize, value: i64);
    fn set_u64(&self, index: usize, value: u64);
    fn set_f64(&self, index: usize, value: f64);
}

impl SetInt for &ComputeCommandEncoderRef {
    fn set_i32(&self, index: usize, value: i32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i32>() as u64,
            &value as *const i32 as *const _,
        );
    }
    fn set_u32(&self, index: usize, value: u32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
    fn set_f32(&self, index: usize, value: f32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
    fn set_i64(&self, index: usize, value: i64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i64>() as u64,
            &value as *const i64 as *const _,
        );
    }
    fn set_u64(&self, index: usize, value: u64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u64>() as u64,
            &value as *const u64 as *const _,
        );
    }
    fn set_f64(&self, index: usize, value: f64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f64>() as u64,
            &value as *const f64 as *const _,
        );
    }
}

#[allow(dead_code)]
pub trait SetConstant {
    fn set_i32(&self, index: usize, value: i32);
    fn set_u32(&self, index: usize, value: u32);
    fn set_f32(&self, index: usize, value: f32);
    fn set_bool(&self, index: usize, value: bool);
}

impl SetConstant for FunctionConstantValues {
    fn set_i32(&self, index: usize, value: i32) {
        self.set_constant_value_at_index(
            &value as *const i32 as *const _,
            MTLDataType::Int,
            index as u64,
        );
    }
    fn set_u32(&self, index: usize, value: u32) {
        self.set_constant_value_at_index(
            &value as *const u32 as *const _,
            MTLDataType::UInt,
            index as u64,
        );
    }
    fn set_f32(&self, index: usize, value: f32) {
        self.set_constant_value_at_index(
            &value as *const f32 as *const _,
            MTLDataType::Float,
            index as u64,
        );
    }
    fn set_bool(&self, index: usize, value: bool) {
        self.set_constant_value_at_index(
            &value as *const bool as *const _,
            MTLDataType::Bool,
            index as u64,
        );
    }
}
