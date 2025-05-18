#include <metal_stdlib>
using namespace metal;

constexpr uint  HEAD_DIM = 64;      // d
constexpr uint  BLOCK_M  = 32;      // queries per thread‑group
constexpr uint  BLOCK_N  = 128;     // keys/values loaded per pass
constexpr float SCALE    = 1.0f / sqrt((float)HEAD_DIM);

struct Params {
    uint seq_len;   // tokens in this (head, batch)
    uint stride_q;  // elements, not bytes:  seq_len * HEAD_DIM
    uint stride_k;
    uint stride_v;
    uint stride_o;
};

kernel void flash_attn_forward(
        device const float*  restrict Q [[buffer(0)]],
        device const float*  restrict K [[buffer(1)]],
        device const float*  restrict V [[buffer(2)]],
        device       float*        O   [[buffer(3)]],
        constant     Params&       p   [[buffer(4)]],
        uint3 tg_id  [[threadgroup_position_in_grid]],
        uint  tid    [[thread_index_in_threadgroup]])
{
    /* ── shared memory for one K/V tile ─────────────────────────────── */
    threadgroup float shK[BLOCK_N * HEAD_DIM];
    threadgroup float shV[BLOCK_N * HEAD_DIM];

    /* ── this thread’s query row ────────────────────────────────────── */
    const uint q_idx = tg_id.x * BLOCK_M + tid;
    if (q_idx >= p.seq_len) return;

    float q[HEAD_DIM];
    device const float* q_ptr = Q + q_idx * p.stride_q;
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM; ++d) {
       	q[d] = q_ptr[d];
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.f;
    float out[HEAD_DIM] = {0};

    /* ── iterate over K/V tiles ─────────────────────────────────────── */
    for (uint k0 = 0; k0 < p.seq_len; k0 += BLOCK_N) {
        /* load BLOCK_N keys & values into threadgroup memory */
        for (uint l = tid; l < BLOCK_N * HEAD_DIM; l += BLOCK_M) {
            uint tok   = k0 + l / HEAD_DIM;
            uint dim   = l % HEAD_DIM;
            float kval = (tok < p.seq_len) ? K[tok * p.stride_k + dim] : 0.f;
            float vval = (tok < p.seq_len) ? V[tok * p.stride_v + dim] : 0.f;
            shK[l] = kval;
            shV[l] = vval;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* dot(Q, K_tile) and running soft‑max */
        float scores[BLOCK_N];
        #pragma unroll
        for (uint j = 0; j < BLOCK_N; ++j) {
            float s = 0.f;
            const uint base = j * HEAD_DIM;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM; ++d) {
                s = mad(q[d], shK[base + d], s);
            }
            scores[j] = s * SCALE;
        }

        float blk_max = -FLT_MAX;
        #pragma unroll
        for (uint j = 0; j < BLOCK_N; ++j) {
        	blk_max = max(blk_max, scores[j]);
        }

        float new_row_max = max(row_max, blk_max);
        float row_scale   = exp(row_max - new_row_max);
        row_sum          *= row_scale;
        row_max           = new_row_max;

        float exp_s[BLOCK_N];
        #pragma unroll
        for (uint j = 0; j < BLOCK_N; ++j) {
            exp_s[j] = exp(scores[j] - row_max);
            row_sum += exp_s[j];
        }

        /* accumulate O = Σ softmax(Q·K) · V */
        #pragma unroll
        for (uint j = 0; j < BLOCK_N; ++j) {
            const float w = exp_s[j] / row_sum;   // soft‑max weight
            const uint  base = j * HEAD_DIM;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM; ++d) {
                out[d] = mad(w, shV[base + d], out[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* ── write result ──────────────────────────────────────────────── */
    device float* o_ptr = O + q_idx * p.stride_o;
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM; ++d) {
    	o_ptr[d] = out[d];
    }
}
