// START
// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

// Invoking the generation of LLVM bitcode for async copies.
//
//   %struct._simdgroup_event_t = type opaque
//
struct _simdgroup_event_t;

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t *
    __metal_simdgroup_async_copy_1d(
        ulong, ulong, threadgroup void * ,
        const device void * , ulong)
__asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t *
    __metal_simdgroup_async_copy_1d(
        ulong, ulong, device void * ,
        const threadgroup void * , ulong)
__asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p3i8.p1i8(
//       i64, i64,
//       i8 addrspace(3)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(1)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t *
    __metal_simdgroup_async_copy_2d(
        ulong, ulong,
        threadgroup void * , ulong, ulong, ulong2,
        const device void * , ulong, ulong, ulong2,
            long2, int)
__asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p1i8.p3i8(
//       i64, i64,
//       i8 addrspace(1)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(3)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t *
    __metal_simdgroup_async_copy_2d(
        ulong, ulong,
        device void * , ulong, ulong, ulong2,
        const threadgroup void * , ulong, ulong, ulong2,
            long2, int)
__asm("air.simdgroup_async_copy_2d.p1i8.p3i8");

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: convergent nounwind
//   declare void
//     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
//     local_unnamed_addr #3
//
void __metal_wait_simdgroup_events(
    int, thread _simdgroup_event_t ** )
__asm("air.wait_simdgroup_events");

#pragma METAL internals: enable
namespace metal {
    enum class simdgroup_async_copy_clamp_mode {
        clamp_to_zero = 0,
            clamp_to_edge = 1
    };

    struct simdgroup_event {
        METAL_FUNC simdgroup_event() thread {}

        template < typename T >
            METAL_FUNC void async_copy(
                threadgroup T * dst,
                const device T * src,
                    ulong n_elements
            ) thread {
                event = __metal_simdgroup_async_copy_1d(
                    // Description of the data type.
                    sizeof(T),
                    alignof(T),

                    // Description of the arguments.
                    reinterpret_cast < threadgroup void * > (dst),
                    reinterpret_cast <
                    const device void * > (src),
                        n_elements);
            }

        template < typename T >
            METAL_FUNC void async_copy(
                device T * dst,
                const threadgroup T * src,
                    ulong n_elements
            ) thread {
                event = __metal_simdgroup_async_copy_1d(
                    // Description of the data type.
                    sizeof(T),
                    alignof(T),

                    // Description of the arguments.
                    reinterpret_cast < device void * > (dst),
                    reinterpret_cast <
                    const threadgroup void * > (src),
                        n_elements);
            }

        template < typename T >
            METAL_FUNC void async_copy(
                // Description of the destination.
                threadgroup T * dst,
                ushort dst_elements_per_row,
                ushort2 dst_tile_dimensions,

                // Description of the source.
                const device T * src,
                    uint src_elements_per_row,
                    ushort2 src_tile_dimensions,

                    // Other arguments.
                    bool transpose_matrix = false,
                    simdgroup_async_copy_clamp_mode clamp_mode =
                    simdgroup_async_copy_clamp_mode::clamp_to_zero
            ) thread {
                if (transpose_matrix) {
                    src_tile_dimensions = src_tile_dimensions.yx;
                    dst_tile_dimensions = dst_tile_dimensions.yx;
                }
                event = __metal_simdgroup_async_copy_2d(
                    // Description of the data type.
                    sizeof(T),
                    alignof(T),

                    // Description of the destination.
                    reinterpret_cast < threadgroup void * > (dst),
                    ushort(dst_elements_per_row),
                    1,
                    ulong2(dst_tile_dimensions),

                    // Description of the source.
                    reinterpret_cast <
                    const device void * > (src),
                        uint(src_elements_per_row),
                        1,
                        ulong2(src_tile_dimensions),

                        // Other arguments.
                        long2(0),
                        static_cast < int > (clamp_mode));
            }

        template < typename T >
            METAL_FUNC void async_copy(
                // Description of the destination.
                device T * dst,
                uint dst_elements_per_row,
                ushort2 dst_tile_dimensions,

                // Description of the source.
                const threadgroup T * src,
                    ushort src_elements_per_row,
                    ushort2 src_tile_dimensions,

                    // Other arguments.
                    bool transpose_matrix = false
            ) thread {
                if (transpose_matrix) {
                    src_tile_dimensions = src_tile_dimensions.yx;
                    dst_tile_dimensions = dst_tile_dimensions.yx;
                }
                event = __metal_simdgroup_async_copy_2d(
                    // Description of the data type.
                    sizeof(T),
                    alignof(T),

                    // Description of the destination.
                    reinterpret_cast < device void * > (dst),
                    uint(dst_elements_per_row),
                    1,
                    ulong2(dst_tile_dimensions),

                    // Description of the source.
                    reinterpret_cast <
                    const threadgroup void * > (src),
                        ushort(src_elements_per_row),
                        1,
                        ulong2(src_tile_dimensions),

                        // Other arguments.
                        long2(0),
                        0);
            }

        METAL_FUNC static void wait(int count, thread simdgroup_event * events) {
            __metal_wait_simdgroup_events(
                count, reinterpret_cast < thread _simdgroup_event_t ** > (events));
        }

        private:
            // Invoking the generation of LLVM bitcode for async copies.
            //
            //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
            //
            thread _simdgroup_event_t * event;
    };
} // namespace metal
#pragma METAL internals: disable

#endif // __METAL_SIMDGROUP_EVENT
// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
    ushort lane_id = thread_index_in_simdgroup;
    ushort quad_id = lane_id / 4;

    constexpr ushort QUADRANT_SPAN_M = 4;
    constexpr ushort THREADS_PER_QUADRANT = 8;
    ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
    ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
    ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;

    ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
    ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
    ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;

    return ushort2(N_in_simd, M_in_simd);
}

#pragma METAL internals: enable
namespace metal {
    template < typename T >
        struct simdgroup_matrix_storage {
            typedef vec < T, 64 > storage_type;

            storage_type t;

            METAL_FUNC thread vec < T, 2 > * thread_elements() thread {
                return reinterpret_cast < thread vec < T, 2 > * > ( & t);
            }

            METAL_FUNC simdgroup_matrix_storage() thread =
                default;

            METAL_FUNC simdgroup_matrix_storage(vec < T, 2 > thread_elements) thread {
                *(this -> thread_elements()) = thread_elements;
            }

            METAL_FUNC static device T * apply_offset(device T * src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
                } else {
                    return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
                }
            }

            METAL_FUNC static threadgroup T * apply_offset(threadgroup T * src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    return src + matrix_origin.x * elements_per_row + matrix_origin.y;
                } else {
                    return src + matrix_origin.y * elements_per_row + matrix_origin.x;
                }
            }
            template < typename U >
                METAL_FUNC void load(const device U * src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                    if (transpose_matrix) {
                        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
                        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
                        U memoryForm0 = src[address0];
                        U memoryForm1 = src[address1];
                        ((thread T * ) thread_elements())[0] = T(memoryForm0);
                        ((thread T * ) thread_elements())[1] = T(memoryForm1);
                    } else if (elements_per_row % 2 != 0) {
                        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
                        U memoryForm0 = src[address0];
                        U memoryForm1 = src[address1];
                        ((thread T * ) thread_elements())[0] = T(memoryForm0);
                        ((thread T * ) thread_elements())[1] = T(memoryForm1);
                    } else {
                        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                        vec < U, 2 > memoryForm = * (const device vec < U, 2 > * )(src + combinedAddress);
                        *(thread_elements()) = vec < T, 2 > (memoryForm);
                    }
                }

            // WARNING: 'T' must be 'float'.
            METAL_FUNC void load_bfloat(const device bfloat * src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
                    uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
                    bfloat memoryForm0 = src[address0];
                    bfloat memoryForm1 = src[address1];

                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[1] = memoryForm0;
                    registerForm[3] = memoryForm1;
                    ((thread bfloat4 * ) thread_elements())[0] = registerForm;
                } else {
                    auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                    bfloat2 memoryForm = * (const device packed_bfloat2 * )(src + combinedAddress);

                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    ((thread float * ) & registerForm)[1] = * (thread float * )( & memoryForm);
                    ((thread bfloat * ) & registerForm)[1] = memoryForm[0];
                    ((thread bfloat4 * ) thread_elements())[0] = registerForm;
                }
            }

            template < typename U >
                METAL_FUNC void load(const threadgroup U * src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                    if (transpose_matrix) {
                        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
                        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
                        U memoryForm0 = src[address0];
                        U memoryForm1 = src[address1];
                        ((thread T * ) thread_elements())[0] = T(memoryForm0);
                        ((thread T * ) thread_elements())[1] = T(memoryForm1);
                    } else if (elements_per_row % 2 != 0) {
                        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
                        U memoryForm0 = src[address0];
                        U memoryForm1 = src[address1];
                        ((thread T * ) thread_elements())[0] = T(memoryForm0);
                        ((thread T * ) thread_elements())[1] = T(memoryForm1);
                    } else {
                        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                        vec < U, 2 > memoryForm = * (const threadgroup vec < U, 2 > * )(src + combinedAddress);
                        *(thread_elements()) = vec < T, 2 > (memoryForm);
                    }
                }

            // WARNING: 'T' must be 'float'.
            METAL_FUNC void load_bfloat(const threadgroup bfloat * src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
                    ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
                    bfloat memoryForm0 = src[address0];
                    bfloat memoryForm1 = src[address1];

                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[1] = memoryForm0;
                    registerForm[3] = memoryForm1;
                    ((thread bfloat4 * ) thread_elements())[0] = registerForm;
                } else {
                    auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                    bfloat2 memoryForm = * (const threadgroup packed_bfloat2 * )(src + combinedAddress);

                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    ((thread float * ) & registerForm)[1] = * (thread float * )( & memoryForm);
                    ((thread bfloat * ) & registerForm)[1] = memoryForm[0];
                    ((thread bfloat4 * ) thread_elements())[0] = registerForm;
                }
            }

            template < typename U >
                METAL_FUNC void store(device U * dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                    if (transpose_matrix) {
                        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
                        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
                        T registerForm0 = ((thread T * ) thread_elements())[0];
                        T registerForm1 = ((thread T * ) thread_elements())[1];
                        dst[address0] = U(registerForm0);
                        dst[address1] = U(registerForm1);
                    } else if (elements_per_row % 2 != 0) {
                        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
                        T registerForm0 = ((thread T * ) thread_elements())[0];
                        T registerForm1 = ((thread T * ) thread_elements())[1];
                        dst[address0] = U(registerForm0);
                        dst[address1] = U(registerForm1);
                    } else {
                        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                        vec < T, 2 > registerForm = * (thread_elements());
                        *(device vec < U, 2 > * )(dst + combinedAddress) = vec < U, 2 > (registerForm);
                    }
                }

            // WARNING: 'T' must be 'float'.
            METAL_FUNC void store_bfloat(device bfloat * dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
                    uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[2] = registerForm[1];
                    dst[address0] = registerForm[2];
                    dst[address1] = registerForm[3];
                } else {
                    uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
                    uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[2] = registerForm[1];
                    dst[address0] = registerForm[2];
                    dst[address1] = registerForm[3];
                }
            }

            template < typename U >
                METAL_FUNC void store(threadgroup U * dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                    if (transpose_matrix) {
                        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
                        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
                        T registerForm0 = ((thread T * ) thread_elements())[0];
                        T registerForm1 = ((thread T * ) thread_elements())[1];
                        dst[address0] = U(registerForm0);
                        dst[address1] = U(registerForm1);
                    } else if (elements_per_row % 2 != 0) {
                        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
                        T registerForm0 = ((thread T * ) thread_elements())[0];
                        T registerForm1 = ((thread T * ) thread_elements())[1];
                        dst[address0] = U(registerForm0);
                        dst[address1] = U(registerForm1);
                    } else {
                        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                        vec < T, 2 > registerForm = * (thread_elements());
                        *(threadgroup vec < U, 2 > * )(dst + combinedAddress) = vec < U, 2 > (registerForm);
                    }
                }

            // WARNING: 'T' must be 'float'.
            METAL_FUNC void store_bfloat(threadgroup bfloat * dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
                if (transpose_matrix) {
                    ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
                    ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[2] = registerForm[1];
                    dst[address0] = registerForm[2];
                    dst[address1] = registerForm[3];
                } else {
                    ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
                    ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
                    bfloat4 registerForm = * (thread bfloat4 * )(thread_elements());
                    registerForm[2] = registerForm[1];
                    dst[address0] = registerForm[2];
                    dst[address1] = registerForm[3];
                }
            }

            template < typename U, typename V >
                METAL_FUNC void multiply(simdgroup_matrix_storage < U > a, simdgroup_matrix_storage < V > b, bool accumulate = true) {
                    if (!accumulate) {
                        *(thread_elements()) = vec < T, 2 > (0);
                    }
                    t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage < T > ::storage_type());
                }
        };
} // namespace metal
#pragma METAL internals: disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE

using namespace metal;

// Dimensions of each matrix.
// - Limitations to matrix size:
//   - 2^32 in each dimension (M/N/K).
//   - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//     good chance this will significantly degrade performance, and require
//     changing the data type of several variables that process addresses. The
//     client is responsible for ensuring correctness and performance with
//     matrices spanning several billion elements in one direction.
//   - The matrix dimensions must be known at compile time, via function
//     constants. Dynamic matrix shapes are beyond the scope of this reference
//     implementation. Dynamic shapes cause a non-negligible regression to
//     shader execution speed. However, they could minimize a compilation
//     latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M[[function_constant(0)]];
constant uint N[[function_constant(1)]];
constant uint K[[function_constant(2)]];

// Specify the leading dimensions at PSO creation time.
constant uint A_leading_dimension[[function_constant(5)]];
constant uint B_leading_dimension[[function_constant(6)]];
constant uint C_leading_dimension[[function_constant(7)]];

// Whether to load the previous value of C, and add it to the accumulator.
constant bool load_previous_C[[function_constant(10)]];

// Whether each matrix is transposed.
constant bool A_trans = false;
constant bool B_trans = true;

// Define the memory layout of the matrix block.
constant ushort M_group = 32;
constant ushort N_group = 32;
constant ushort K_group = 32;

// Thresholds that mark the matrix edge.
constant uint M_edge = M - (M % M_group);
constant uint N_edge = N - (N % N_group);

// Find the number of elements in the final block. If the matrix
// dimensions are perfectly divisibly by block dimensions, we don't want
// this value to be zero. The final block is a full block.
constant ushort M_remainder = (M % 16 == 0) ?
    16 : M % 16;
constant ushort N_remainder = (N % 16 == 0) ?
    16 : N % 16;
constant ushort K_remainder = (K % K_group == 0) ?
    K_group : K % K_group;
constant ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

// Shift the final block, so it doesn't access out-of-bounds memory.
constant ushort M_shift = (M < M_group) ? 0 : 16 - M_remainder;
constant ushort N_shift = (N < N_group) ? 0 : 16 - N_remainder;

// Indexes into an array of registers.
//
// Calls to this function are expected to be evaluated at compile time. The
// array indices transform into register offsets, which are embedded into the
// assembly code.
template < typename T >
    METAL_FUNC thread simdgroup_matrix_storage < T > * get_sram(
        thread simdgroup_matrix_storage < T > * sram,
        ushort sram_leading_dim,
        ushort2 matrix_origin
    ) {
        return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
    }
// One multiply-accumulate loop iteration, or 8 dot products.
METAL_FUNC void multiply_accumulate(
    const device float * A_src,
        const device float * B_src,
            thread simdgroup_matrix_storage < float > * A_sram,
            thread simdgroup_matrix_storage < float > * B_sram,
            thread simdgroup_matrix_storage < float > * C_sram,
            ushort k
) {
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < 16; m += 8) {
        ushort2 origin(0, m);
        auto A = get_sram(A_sram, 8, origin);
        A -> load(A_src, A_leading_dimension, ushort2(k, m), A_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < 16; n += 8) {
        ushort2 origin(n, 0);
        auto B = get_sram(B_sram, 16, origin);
        B -> load(B_src, B_leading_dimension, ushort2(n, k), B_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < 16; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < 16; n += 8) {
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            auto B = get_sram(B_sram, 16, ushort2(n, 0));
            auto C = get_sram(C_sram, 16, ushort2(n, m));
            C -> multiply( * A, * B);
        }
    }
}

// One multiply-accumulate loop iteration, or 8 dot products.
METAL_FUNC void multiply_accumulate(
    const threadgroup float * A_src,
        const threadgroup float * B_src,
            thread simdgroup_matrix_storage < float > * A_sram,
            thread simdgroup_matrix_storage < float > * B_sram,
            thread simdgroup_matrix_storage < float > * C_sram,
            ushort k
) {
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < 16; m += 8) {
        ushort2 origin(0, m);
        auto A = get_sram(A_sram, 8, origin);
        A -> load(A_src, 32, ushort2(k, m), A_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < 16; n += 8) {
        ushort2 origin(n, 0);
        auto B = get_sram(B_sram, 16, origin);
        B -> load(B_src, 32, ushort2(n, k), B_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < 16; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < 16; n += 8) {
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            auto B = get_sram(B_sram, 16, ushort2(n, 0));
            auto C = get_sram(C_sram, 16, ushort2(n, m));
            C -> multiply( * A, * B);
        }
    }
}

// Metal function arguments.
//
// A: the left-hand side matrix
// - dimensions: M x K
//               K x M (transposed)
// - memory precision: memA
// - register precision: regA
//
// B: the right-hand side matrix
// - dimensions: K x N
//               N x K (transposed)
// - memory precision: memB
// - register precision: regB
//
// C: the output matrix, alternatively the dot product accumulator
// - dimensions: M x N
// - memory precision: memC
// - register precision: regC
//
// threadgroup_block: the chunk of threadgroup memory allocated at runtime
// - ideally 10 KB or less
// - precision: void/8-bit integer to make the pointer arithmetic more legible

kernel void matmul(device float * A[[buffer(0)]],
    device float * B[[buffer(1)]],
    device float * C[[buffer(2)]],
    threadgroup uchar * threadgroup_block[[threadgroup(0)]],

    uint3 gid[[threadgroup_position_in_grid]],
    ushort sidx[[simdgroup_index_in_threadgroup]],
    ushort lane_id[[thread_index_in_simdgroup]]) {
    ushort2 sid(sidx % 2, sidx / 2);
    ushort2 morton_offset = morton_order(lane_id);

    // Return early if the SIMD is out of bounds.
    //
    // There could be some threadgroups where the matrix edge cuts straight
    // through the middle of the block. SIMDs on the right or bottom of the
    // dividing line must be stopped from causing out-of-bounds accesses. This is
    // the reason for the early exit.
    uint M_offset = gid.y * M_group;
    uint N_offset = gid.x * N_group;
    if (M_offset + sid.y * 16 >= M ||
        N_offset + sid.x * 16 >= N) {
        return;
    }
    ushort2 offset_in_group(sid.x * 16 + morton_offset.x,
        sid.y * 16 + morton_offset.y);

    // Shift the matrix block within bounds, if possible.
    if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
        M_offset -= M_shift;
    }
    if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
        N_offset -= N_shift;
    }

    simdgroup_matrix_storage < float > C_sram[
        4];

    if (load_previous_C) {

        if (false) {
            // Fast path for matrices that qualify.
            uint2 C_offset(N_offset + offset_in_group.x,
                M_offset + offset_in_group.y);
            auto C_dst = simdgroup_matrix_storage < float > ::apply_offset(
                C, C_leading_dimension, C_offset);

            // Write the accumulator to device memory.
            #pragma clang loop unroll(full)
            for (ushort m = 0; m < 16; m += 8) {
                #pragma clang loop unroll(full)
                for (ushort n = 0; n < 16; n += 8) {
                    ushort2 origin(n, m);
                    auto C = get_sram(C_sram, 16, origin);
                    C -> load(C_dst, C_leading_dimension, origin);
                }
            }
        } else {
            // Slow path for when memory must be handled more carefully.
            auto C_block = (threadgroup float * )(threadgroup_block);
            auto C_block_dst =
                simdgroup_matrix_storage < float > ::apply_offset(
                    C_block, 32, offset_in_group);

            // Launch the async copy from threadgroup to device memory.
            if (sidx == 0) {
                uint2 C_offset(N_offset, M_offset);
                ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                    min(uint(M_group), M - C_offset.y));
                auto C_dst = simdgroup_matrix_storage < float > ::apply_offset(
                    C, C_leading_dimension, C_offset);

                simdgroup_event event;
                event.async_copy(
                    C_block, 32, C_tile,
                    C_dst, C_leading_dimension, C_tile);
                simdgroup_event::wait(1, & event);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Read the accumulator from threadgroup memory.
            #pragma clang loop unroll(full)
            for (ushort m = 0; m < 16; m += 8) {
                #pragma clang loop unroll(full)
                for (ushort n = 0; n < 16; n += 8) {
                    ushort2 origin(n, m);
                    auto C = get_sram(C_sram, 16, origin);
                    C -> load(
                        C_block_dst, 32, origin);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    } else {
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < 16; m += 8) {
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < 16; n += 8) {
                ushort2 origin(n, m);
                auto C = get_sram(C_sram, 16, origin);
                * C = simdgroup_matrix_storage < float > (0);
            }
        }
    }

    // Perform the iterations where async copy is avoided.
    for (uint k = 0; k < 0; k += 8) {
        uint2 A_offset(k, M_offset);
        uint2 B_offset(N_offset, k);
        A_offset += uint2(morton_offset.x, offset_in_group.y);
        B_offset += uint2(offset_in_group.x, morton_offset.y);

        auto A_src = simdgroup_matrix_storage < float > ::apply_offset(
            A, A_leading_dimension, A_offset, A_trans);
        auto B_src = simdgroup_matrix_storage < float > ::apply_offset(
            B, B_leading_dimension, B_offset, B_trans);

        simdgroup_matrix_storage < float > A_sram[
            2 * (8 / 8)];
        simdgroup_matrix_storage < float > B_sram[
            (8 / 8) * 2];
        multiply_accumulate(A_src, B_src,
            A_sram, B_sram, C_sram, 0);
    }

    // Perform the iterations where async copy is used.
    for (uint k = 0; k < K; k += K_group) {
        auto A_block = (threadgroup float * )(
            threadgroup_block);
        auto B_block = (threadgroup float * )(
            threadgroup_block + 4096);

        // Launch an async copy from device to threadgroup memory.
        if (sidx == 0) {
            uint2 A_offset(k, M_offset);
            uint2 B_offset(N_offset, k);
            auto A_src = simdgroup_matrix_storage < float > ::apply_offset(
                A, A_leading_dimension, A_offset, A_trans);
            auto B_src = simdgroup_matrix_storage < float > ::apply_offset(
                B, B_leading_dimension, B_offset, B_trans);

            ushort M_tile_dimension = min(uint(M_group), M - M_offset);
            ushort N_tile_dimension = min(uint(N_group), N - N_offset);
            ushort K_tile_dimension = min(uint(K_group), K - k);
            ushort K_tile_padded = min(uint(K_group), (K + K_remainder_padded - K_remainder) - k);

            ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
            ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
            ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
            ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

            simdgroup_event events[2];
            events[0].async_copy(
                A_block, 32, A_tile_dst,
                A_src, A_leading_dimension, A_tile_src, A_trans);
            events[1].async_copy(
                B_block, 32, B_tile_dst,
                B_src, B_leading_dimension, B_tile_src, B_trans);
            simdgroup_event::wait(2, events);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
        ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
        auto A_block_src = A_block;
        auto B_block_src = B_block;
        A_block_src = simdgroup_matrix_storage < float > ::apply_offset(
            A_block_src, 32, A_block_offset, A_trans);
        B_block_src = simdgroup_matrix_storage < float > ::apply_offset(
            B_block_src, 32, B_block_offset, B_trans);

        simdgroup_matrix_storage < float > A_sram[
            2 * (K_group / 8)];
        simdgroup_matrix_storage < float > B_sram[
            (K_group / 8) * 2];
        #pragma clang loop unroll(full)
        for (ushort k = 0; k < K_remainder_padded; k += 8) {
            multiply_accumulate(A_block_src, B_block_src,
                A_sram, B_sram, C_sram, k);
        }

        // Will there be any iterations after this one?
        if (k + K_group < K) {
            // If so, we haven't reached the edge of either input matrix yet.
            #pragma clang loop unroll(full)
            for (ushort k = K_remainder_padded; k < K_group; k += 8) {
                multiply_accumulate(A_block_src, B_block_src,
                    A_sram, B_sram, C_sram, k);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (false) {
        // Fast path for matrices that qualify.
        uint2 C_offset(N_offset + offset_in_group.x,
            M_offset + offset_in_group.y);
        auto C_dst = simdgroup_matrix_storage < float > ::apply_offset(
            C, C_leading_dimension, C_offset);

        // Write the accumulator to device memory.
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < 16; m += 8) {
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < 16; n += 8) {
                ushort2 origin(n, m);
                auto C = get_sram(C_sram, 16, origin);
                C -> store(C_dst, C_leading_dimension, origin);
            }
        }
    } else {
        // Slow path for when memory must be handled more carefully.
        auto C_block = (threadgroup float * )(threadgroup_block);
        auto C_block_dst =
            simdgroup_matrix_storage < float > ::apply_offset(
                C_block, 32, offset_in_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write the accumulator to threadgroup memory.
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < 16; m += 8) {
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < 16; n += 8) {
                ushort2 origin(n, m);
                auto C = get_sram(C_sram, 16, origin);
                C -> store(
                    C_block_dst, 32, origin);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Launch the async copy from threadgroup to device memory.
        if (sidx == 0) {
            uint2 C_offset(gid.x * N_group, gid.y * M_group);
            ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                min(uint(M_group), M - C_offset.y));
            auto C_dst = simdgroup_matrix_storage < float > ::apply_offset(
                C, C_leading_dimension, C_offset);

            // If we shift successfully, the garbage zone moves from the bottom right
            // to the top left.
            if ((M_shift != 0) || (N_shift != 0)) {
                ushort2 C_block_shift(0, 0);
                if ((M_shift != 0) && (C_offset.y >= M_edge)) {
                    C_block_shift.y = M_shift;
                }
                if ((N_shift != 0) && (C_offset.x >= N_edge)) {
                    C_block_shift.x = N_shift;
                }
                C_block = simdgroup_matrix_storage < float > ::apply_offset(
                    C_block, 32, C_block_shift);
            }

            simdgroup_event event;
            event.async_copy(
                C_dst, C_leading_dimension, C_tile,
                C_block, 32, C_tile);
        }
    }
}
