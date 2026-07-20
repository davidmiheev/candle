// Fused kernels for decode-path launch-overhead reduction.
//
// 1) fused_add_rmsnorm: out[0] = x + residual (new residual stream),
//    out[1] = rmsnorm(x + residual) * (w [+1]) — replaces an add kernel, a
//    reduction kernel and a scale kernel (and their HBM round-trips) with a
//    single launch. `plus_one` selects the gemma (1 + w) convention.
// 2) fused_swiglu: given a packed [rows, 2*d] gate|up tensor, computes
//    act(gate) * up in one launch. act_mode: 0 = silu, 1 = gelu(tanh).
//
// Both are row-per-block kernels sized for decode/prefill activation shapes
// (d up to a few tens of thousands).

#include "cuda_utils.cuh"
#include <cmath>
#include <stdint.h>

#define WARP_SZ 32

static __device__ __forceinline__ float fused_warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SZ);
    }
    return x;
}

template <typename T>
__device__ __forceinline__ float to_f32(T v) { return static_cast<float>(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v) { return static_cast<T>(v); }

template <typename T>
__device__ void fused_add_rmsnorm(
    const T* __restrict__ x,        // [rows, d]
    const T* __restrict__ residual, // [rows, d]
    const T* __restrict__ weight,   // [d]
    T* __restrict__ out,            // [2, rows, d]: [0]=sum, [1]=normed
    const int rows,
    const int d,
    const float eps,
    const int plus_one
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const T* xr = x + (size_t)row * d;
    const T* rr = residual + (size_t)row * d;
    T* sum_out = out + (size_t)row * d;
    T* norm_out = out + (size_t)rows * d + (size_t)row * d;

    extern __shared__ float shm[];
    // Pass 1: sum of squares of (x + residual), stash sums in shm-free style:
    // recompute in pass 2 (activations are tiny; recompute beats a shared
    // buffer of size d for large d).
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float s = to_f32<T>(xr[i]) + to_f32<T>(rr[i]);
        ss += s * s;
    }
    // block reduction
    ss = fused_warp_reduce_sum(ss);
    const int lane = threadIdx.x % WARP_SZ;
    const int wid = threadIdx.x / WARP_SZ;
    if (lane == 0) shm[wid] = ss;
    __syncthreads();
    ss = (threadIdx.x < blockDim.x / WARP_SZ) ? shm[threadIdx.x] : 0.0f;
    if (wid == 0) ss = fused_warp_reduce_sum(ss);
    if (threadIdx.x == 0) shm[0] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf(shm[0] / d + eps);

    // Pass 2: write both outputs.
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float s = to_f32<T>(xr[i]) + to_f32<T>(rr[i]);
        float w = to_f32<T>(weight[i]);
        if (plus_one) w += 1.0f;
        sum_out[i] = from_f32<T>(s);
        norm_out[i] = from_f32<T>(s * inv_rms * w);
    }
}

__device__ __forceinline__ float act_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float act_gelu_tanh(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template <typename T>
__device__ void fused_swiglu(
    const T* __restrict__ gate_up, // [rows, 2*d] (gate = first d, up = last d)
    T* __restrict__ out,           // [rows, d]
    const int rows,
    const int d,
    const int act_mode // 0 = silu, 1 = gelu(tanh)
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)rows * d;
    if (idx >= total) return;
    const size_t row = idx / d;
    const size_t col = idx % d;
    const float g = to_f32<T>(gate_up[row * 2 * d + col]);
    const float u = to_f32<T>(gate_up[row * 2 * d + d + col]);
    const float a = act_mode == 0 ? act_silu(g) : act_gelu_tanh(g);
    out[idx] = from_f32<T>(a * u);
}

#define FUSED_ADD_RMSNORM_OP(TY, FN)                                          \
extern "C" __global__ void FN(                                                \
    const TY* x, const TY* residual, const TY* weight, TY* out,               \
    const int rows, const int d, const float eps, const int plus_one) {       \
    fused_add_rmsnorm<TY>(x, residual, weight, out, rows, d, eps, plus_one);  \
}

#define FUSED_SWIGLU_OP(TY, FN)                                               \
extern "C" __global__ void FN(                                                \
    const TY* gate_up, TY* out, const int rows, const int d,                  \
    const int act_mode) {                                                     \
    fused_swiglu<TY>(gate_up, out, rows, d, act_mode);                        \
}

FUSED_ADD_RMSNORM_OP(float, fused_add_rmsnorm_f32)
FUSED_SWIGLU_OP(float, fused_swiglu_f32)

#if __CUDA_ARCH__ >= 800
FUSED_ADD_RMSNORM_OP(__nv_bfloat16, fused_add_rmsnorm_bf16)
FUSED_SWIGLU_OP(__nv_bfloat16, fused_swiglu_bf16)
#endif

// ── Static-decode kernels (CUDA-graph-replayable decode step) ──────────────
// The decode position lives in a device u32 scalar; these kernels index off
// it so a captured graph replays correctly as the position advances.

// Write one new K/V vector into preallocated [heads, max_seq, dim] buffers
// at the position read from pos.
template <typename T>
__device__ void kv_write(
    T* __restrict__ kbuf,          // [heads, max_seq, dim]
    T* __restrict__ vbuf,
    const T* __restrict__ knew,    // [heads, dim]
    const T* __restrict__ vnew,
    const unsigned int* __restrict__ pos,
    const int heads,
    const int max_seq,
    const int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = heads * dim;
    if (idx >= total) return;
    const int h = idx / dim;
    const int d = idx % dim;
    const unsigned int p = *pos;
    const size_t off = ((size_t)h * max_seq + p) * dim + d;
    kbuf[off] = knew[idx];
    vbuf[off] = vnew[idx];
}

// Additive causal mask row for attention over the full static buffer:
// mask[j] = 0 where the key at j is visible from position *pos, else -inf.
// window == 0 => global causal; window > 0 => sliding (j > pos - window).
template <typename T>
__device__ void mask_from_pos(
    T* __restrict__ mask,          // [max_seq]
    const unsigned int* __restrict__ pos,
    const int max_seq,
    const int window
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= max_seq) return;
    const int p = (int)(*pos);
    bool vis = j <= p;
    if (window > 0) vis = vis && (j > p - window);
    mask[j] = vis ? (T)0.0f : (T)(-1e30f);
}

extern "C" __global__ void set_u32(unsigned int* dst, unsigned int v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *dst = v;
    }
}

extern "C" __global__ void incr_u32(unsigned int* pos) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *pos = *pos + 1u;
    }
}

#define KV_WRITE_OP(TY, FN)                                                   \
extern "C" __global__ void FN(                                                \
    TY* kbuf, TY* vbuf, const TY* knew, const TY* vnew,                       \
    const unsigned int* pos, const int heads, const int max_seq,              \
    const int dim) {                                                          \
    kv_write<TY>(kbuf, vbuf, knew, vnew, pos, heads, max_seq, dim);           \
}

#define MASK_FROM_POS_OP(TY, FN)                                              \
extern "C" __global__ void FN(                                                \
    TY* mask, const unsigned int* pos, const int max_seq, const int window) { \
    mask_from_pos<TY>(mask, pos, max_seq, window);                            \
}

KV_WRITE_OP(float, kv_write_f32)
MASK_FROM_POS_OP(float, mask_from_pos_f32)

#if __CUDA_ARCH__ >= 800
KV_WRITE_OP(__nv_bfloat16, kv_write_bf16)
MASK_FROM_POS_OP(__nv_bfloat16, mask_from_pos_bf16)
#endif

// ── Gated-DeltaNet sequential scan ──────────────────────────────────────────
// Runs the whole delta-rule recurrence for one layer in a single launch:
//   S <- decay_t * S;  err = (v_t - S k_t) * beta_t;  S += err (x) k_t;
//   o_t = S q_t
// State rows (d_v) are independent, so blocks tile (v_head, d_v-chunk) with
// the chunk's state rows resident in shared memory; k/q for the step are
// staged once per block. Grid: n_v * ceil(d_v/GDN_ROWS). Block: GDN_ROWS.
#define GDN_ROWS 64
extern "C" __global__ void gdn_scan_f32(
    const float* __restrict__ q,     // [T, n_k, d_k]
    const float* __restrict__ k,     // [T, n_k, d_k]
    const float* __restrict__ v,     // [T, n_v, d_v]
    const float* __restrict__ beta,  // [T, n_v]
    const float* __restrict__ decay, // [T, n_v]
    float* __restrict__ s,           // [n_v, d_v, d_k] (in/out)
    float* __restrict__ o,           // [T, n_v, d_v]
    const int T, const int n_k, const int n_v, const int d_k, const int d_v) {
    const int chunks = (d_v + GDN_ROWS - 1) / GDN_ROWS;
    const int h = blockIdx.x / chunks;      // v-head
    const int chunk = blockIdx.x % chunks;  // d_v row chunk
    const int row0 = chunk * GDN_ROWS;
    const int kh = (h * n_k) / n_v;         // grouped k-head
    const int tid = threadIdx.x;
    const int row = row0 + tid;             // this thread's d_v row

    extern __shared__ float smem[];
    float* S = smem;               // [GDN_ROWS, d_k]
    float* kk = smem + GDN_ROWS * d_k; // [d_k]
    float* qq = kk + d_k;              // [d_k]

    // Load this chunk's state rows.
    if (row < d_v) {
        const float* src = s + ((size_t)h * d_v + row) * d_k;
        float* dst = S + (size_t)tid * d_k;
        for (int j = 0; j < d_k; ++j) dst[j] = src[j];
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        for (int j = tid; j < d_k; j += blockDim.x) {
            kk[j] = k[((size_t)t * n_k + kh) * d_k + j];
            qq[j] = q[((size_t)t * n_k + kh) * d_k + j];
        }
        __syncthreads();
        if (row < d_v) {
            const float dcy = decay[(size_t)t * n_v + h];
            const float bt = beta[(size_t)t * n_v + h];
            float* Sr = S + (size_t)tid * d_k;
            float pred = 0.f;
            #pragma unroll 4
            for (int j = 0; j < d_k; ++j) {
                Sr[j] *= dcy;
                pred += Sr[j] * kk[j];
            }
            const float err = (v[((size_t)t * n_v + h) * d_v + row] - pred) * bt;
            float out = 0.f;
            #pragma unroll 4
            for (int j = 0; j < d_k; ++j) {
                Sr[j] += err * kk[j];
                out += Sr[j] * qq[j];
            }
            o[((size_t)t * n_v + h) * d_v + row] = out;
        }
        __syncthreads();
    }

    if (row < d_v) {
        float* dst = s + ((size_t)h * d_v + row) * d_k;
        const float* src = S + (size_t)tid * d_k;
        for (int j = 0; j < d_k; ++j) dst[j] = src[j];
    }
}
