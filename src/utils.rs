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
