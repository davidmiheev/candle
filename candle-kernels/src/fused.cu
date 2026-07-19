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
