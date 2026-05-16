/*
 * cpu_decoder.c — CPU DAC decoder with multi-level SIMD dispatch.
 *
 * ISA level selection at runtime via CPUID:
 *   scalar  → all x86-64 (no SIMD required)
 *   sse4.2  → Nehalem 2008+
 *   avx     → Sandy Bridge / Bulldozer 2011+    ← amd64 baseline
 *   avx2    → Haswell 2013+
 *   avx512  → Skylake-SP 2017+, Zen 4 2022+
 *
 * Each kernel exists in scalar and SIMD variants.
 * The ops dispatch table selects the best at init.
 */

#include "dac_model.h"
#include "model_loader.h"
#include "../include/tsac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#define X86_64 1
#elif defined(__riscv)
#include "arch/riscv/cpu_riscv.h"
#define RISCV 1
#endif

#ifdef __aarch64__
#include "arch/arm/cpu_arm.h"
#endif

/* ================================================================ */
/*  CPU feature detection                                           */
/* ================================================================ */

typedef enum { SIMD_SCALAR, SIMD_SSE42, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;

static int cpu_has(unsigned int leaf, unsigned int reg, unsigned int bit) {
#if X86_64
    unsigned int a, b, c, d;
    if (!__get_cpuid(leaf, &a, &b, &c, &d)) return 0;
    switch (reg) {
        case 0: return (a >> bit) & 1;
        case 1: return (b >> bit) & 1;
        case 2: return (c >> bit) & 1;
        case 3: return (d >> bit) & 1;
    }
#endif
    (void)leaf; (void)reg; (void)bit; return 0;
}
#define HAS(leaf, reg, bit) cpu_has(leaf, reg, bit)

const char *cpu_simd_name(void) {
    if (HAS(7, 1, 16)) return "AVX-512F";
    if (HAS(7, 1, 5))  return "AVX2";
    if (HAS(1, 2, 28)) return "AVX+FMA";
    if (HAS(1, 2, 20)) return "SSE4.2";
    return "scalar";
}

/* ================================================================ */
/*  Tensor finder                                                   */
/* ================================================================ */

DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

/* ================================================================ */
/*  Scalar kernels (baseline, works on ALL x86-64)                  */
/* ================================================================ */

void conv1d_s(float *o, const float *x, const float *w, const float *b,
              int T, int K, int Ci, int Co) {
    int P = K/2;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) { int ii = oi + j - P;
                    if (ii >= 0 && ii < T) s += x[ic*T+ii] * w[oc*Ci*K + ic*K + j]; }
            o[oc*T+oi] = s;
        }
}

void convt1d_s(float *o, const float *x, const float *w,
               int Ti, int To, int K, int Ci, int Co) {
    int P = K/2; memset(o, 0, Co*To*sizeof(float));
    for (int ic = 0; ic < Ci; ic++) for (int ii = 0; ii < Ti; ii++) {
        float v = x[ic*Ti+ii]; if (v == 0) continue;
        for (int oc = 0; oc < Co; oc++) for (int j = 0; j < K; j++) {
            int oi = ii*2 + j - P;
            if (oi >= 0 && oi < To) o[oc*To+oi] += v * w[oc*Ci*K + ic*K + j];
        }
    }
}

void gn_s(float *o, const float *x, const float *w, const float *b,
          int N, int G, float eps) {
    int E = N/G;
    for (int g = 0; g < G; g++) {
        float s = 0, sq = 0;
        for (int i = 0; i < E; i++) { float v = x[g*E+i]; s += v; sq += v*v; }
        float mn = s/E, vr = sq/E - mn*mn, is = 1.0f/sqrtf(fmaxf(vr+eps, 1e-10f));
        for (int i = 0; i < E; i++) {
            int idx = g*E+i; o[idx] = (x[idx]-mn)*is*(w?w[g]:1)+(b?b[g]:0);
        }
    }
}

void snake_s(float *o, const float *x, const float *a, int n, int C) {
    for (int i = 0; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

/* ================================================================ */
/*  SIMD helper functions                                           */
/* ================================================================ */

#ifdef __AVX__

/* Horizontal sum of 8 floats in __m256 */
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    vlow  = _mm_hadd_ps(vlow, vlow);
    vlow  = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}

#ifdef __AVX512F__
/* Horizontal sum of 16 floats in __m512 using AVX512F */
static inline float hsum512_ps(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

/* Gather 8 non-contiguous floats from input tensor
 * x: input tensor [Ci, T] layout
 * ic_base: starting input channel (0, 8, 16, ...)
 * ii: time index
 * T: time dimension size
 * Returns: __m256 with x[ic_base* T + ii], x[(ic_base+1)*T + ii], ... */
static inline __m256 gather8(const float *x, int ic_base, int ii, int T, int Ci) {
    float tmp[8];
    for (int k = 0; k < 8 && (ic_base + k) < Ci; k++) {
        tmp[k] = x[(ic_base + k) * T + ii];
    }
    for (int k = (Ci - ic_base); k < 8; k++) {
        tmp[k] = 0.0f;
    }
    return _mm256_loadu_ps(tmp);
}

#ifdef __AVX512F__
/* Gather 16 non-contiguous floats from input tensor for AVX-512 */
static inline __m512 gather16(const float *x, int ic_base, int ii, int T, int Ci) {
    float tmp[16];
    for (int k = 0; k < 16 && (ic_base + k) < Ci; k++) {
        tmp[k] = x[(ic_base + k) * T + ii];
    }
    for (int k = (Ci - ic_base); k < 16; k++) {
        tmp[k] = 0.0f;
    }
    return _mm512_loadu_ps(tmp);
}
#endif

/* ================================================================ */
/*  AVX kernels (8-wide, requires AVX + FMA)                        */
/* ================================================================ */

__attribute__((target("avx,fma")))
void conv1d_avx(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co) {
    int P = K/2;
    int Ci_block = Ci & ~7;  /* Round down to multiple of 8 */
    
    for (int oc = 0; oc < Co; oc++) {
        __m256 bias_vec = b ? _mm256_set1_ps(b[oc]) : _mm256_setzero_ps();
        
        for (int oi = 0; oi < T; oi++) {
            __m256 sum8 = bias_vec;
            
            /* Vectorized loop over input channels (8 at a time) */
            for (int icb = 0; icb < Ci_block; icb += 8) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        /* Load 8 input values (non-contiguous across channels) */
                        __m256 x8 = gather8(x, icb, ii, T, Ci);
                        /* Load 8 weights (contiguous) */
                        int w_idx = oc * Ci * K + icb * K + j;
                        __m256 w8 = _mm256_loadu_ps(&w[w_idx]);
                        /* FMA: sum8 += x8 * w8 */
                        sum8 = _mm256_fmadd_ps(x8, w8, sum8);
                    }
                }
            }
            
            /* Horizontal reduce sum8 to scalar */
            float sum = hsum256_ps(sum8);
            
            /* Handle remaining channels (Ci % 8) with scalar fallback */
            for (int ic = Ci_block; ic < Ci; ic++) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        sum += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
            
            o[oc*T+oi] = sum;
        }
    }
}

__attribute__((target("avx,fma")))
void convt1d_avx(float *o, const float *x, const float *w,
                 int Ti, int To, int K, int Ci, int Co) {
    int P = K/2;
    int Co_block = Co & ~7;
    memset(o, 0, Co*To*sizeof(float));
    
    for (int ic = 0; ic < Ci; ic++) {
        for (int ii = 0; ii < Ti; ii++) {
            float v = x[ic*Ti+ii];
            if (v == 0) continue;
            
            __m256 vb = _mm256_set1_ps(v);
            
            /* Vectorized loop over output channels (8 at a time) */
            for (int ocb = 0; ocb < Co_block; ocb += 8) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*2 + j - P;
                    if (oi >= 0 && oi < To) {
                        /* Load 8 weights: w[ocb:ocb+8, ic, j] */
                        int w_idx = ocb * Ci * K + ic * K + j;
                        __m256 w8 = _mm256_loadu_ps(&w[w_idx]);
                        /* Load current output values */
                        __m256 ov = _mm256_loadu_ps(&o[ocb*To + oi]);
                        /* FMA: ov += vb * w8 */
                        ov = _mm256_fmadd_ps(vb, w8, ov);
                        /* Store back */
                        _mm256_storeu_ps(&o[ocb*To + oi], ov);
                    }
                }
            }
            
            /* Handle remaining channels with scalar fallback */
            for (int oc = Co_block; oc < Co; oc++) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*2 + j - P;
                    if (oi >= 0 && oi < To) {
                        o[oc*To+oi] += v * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
        }
    }
}

__attribute__((target("avx,fma")))
void snake_avx(float *o, const float *x, const float *a, int n, int C) {
    int n_block = n & ~7;
    
    for (int i = 0; i < n_block; i += 8) {
        /* Load 8 values from x */
        __m256 xv = _mm256_loadu_ps(x + i);
        
        /* Load corresponding alphas (need to handle modulo) */
        float alpha_vals[8];
        for (int k = 0; k < 8; k++) {
            float al = a[(i+k)%C];
            alpha_vals[k] = (al < 1e-6f) ? 1e-6f : al;
        }
        __m256 av = _mm256_loadu_ps(alpha_vals);
        
        /* Compute av * xv for sin input */
        __m256 av_xv = _mm256_mul_ps(av, xv);
        
        /* sinf has no SIMD intrinsic - use scalar fallback */
        float tmp[8];
        _mm256_storeu_ps(tmp, av_xv);
        for (int k = 0; k < 8; k++) {
            float s = sinf(tmp[k]);
            tmp[k] = tmp[k] + s*s / alpha_vals[k];
        }
        __m256 res = _mm256_loadu_ps(tmp);
        
        _mm256_storeu_ps(o + i, res);
    }
    
    /* Handle remaining elements with scalar fallback */
    for (int i = n_block; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

__attribute__((target("avx,fma")))
void add_avx(float *o, const float *x, const float *y, int n) {
    int n_block = n & ~7;
    
    for (int i = 0; i < n_block; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(o + i, _mm256_add_ps(xv, yv));
    }
    
    /* Handle remaining elements */
    for (int i = n_block; i < n; i++) {
        o[i] = x[i] + y[i];
    }
}

/* ================================================================ */
/*  AVX2 kernels (8-wide, requires AVX2 + FMA)                      */
/* ================================================================ */

/* AVX2 kernels - same implementation as AVX but with AVX2 target for better
 * compiler optimizations. The main difference is AVX2 has better horizontal
 * operations and permute support. */

__attribute__((target("avx2,fma")))
void conv1d_avx2(float *o, const float *x, const float *w, const float *b,
                 int T, int K, int Ci, int Co) {
    /* Same as AVX version - AVX2 mainly adds integer ops, 
     * but we use the target attribute for potential compiler optimizations */
    conv1d_avx(o, x, w, b, T, K, Ci, Co);
}

__attribute__((target("avx2,fma")))
void convt1d_avx2(float *o, const float *x, const float *w,
                  int Ti, int To, int K, int Ci, int Co) {
    convt1d_avx(o, x, w, Ti, To, K, Ci, Co);
}

__attribute__((target("avx2,fma")))
void snake_avx2(float *o, const float *x, const float *a, int n, int C) {
    snake_avx(o, x, a, n, C);
}

__attribute__((target("avx2,fma")))
void add_avx2(float *o, const float *x, const float *y, int n) {
    add_avx(o, x, y, n);
}

/* ================================================================ */
/*  AVX-512 kernels (16-wide, requires AVX-512F)                    */
/* ================================================================ */

#ifdef __AVX512F__

__attribute__((target("avx512f")))
void conv1d_avx512(float *o, const float *x, const float *w, const float *b,
                   int T, int K, int Ci, int Co) {
    int P = K/2;
    int Ci_block = Ci & ~15;  /* Round down to multiple of 16 */
    
    for (int oc = 0; oc < Co; oc++) {
        __m512 bias_vec = b ? _mm512_set1_ps(b[oc]) : _mm512_setzero_ps();
        
        for (int oi = 0; oi < T; oi++) {
            __m512 sum16 = bias_vec;
            
            /* Vectorized loop over input channels (16 at a time) */
            for (int icb = 0; icb < Ci_block; icb += 16) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        /* Load 16 input values */
                        __m512 x16 = gather16(x, icb, ii, T, Ci);
                        /* Load 16 weights (contiguous) */
                        int w_idx = oc * Ci * K + icb * K + j;
                        __m512 w16 = _mm512_loadu_ps(&w[w_idx]);
                        /* FMA: sum16 += x16 * w16 */
                        sum16 = _mm512_fmadd_ps(x16, w16, sum16);
                    }
                }
            }
            
            /* Horizontal reduce sum16 to scalar */
            float sum = hsum512_ps(sum16);
            
            /* Handle remaining channels (Ci % 16) with scalar fallback */
            for (int ic = Ci_block; ic < Ci; ic++) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        sum += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
            
            o[oc*T+oi] = sum;
        }
    }
}

__attribute__((target("avx512f")))
void convt1d_avx512(float *o, const float *x, const float *w,
                    int Ti, int To, int K, int Ci, int Co) {
    int P = K/2;
    int Co_block = Co & ~15;
    memset(o, 0, Co*To*sizeof(float));
    
    for (int ic = 0; ic < Ci; ic++) {
        for (int ii = 0; ii < Ti; ii++) {
            float v = x[ic*Ti+ii];
            if (v == 0) continue;
            
            __m512 vb = _mm512_set1_ps(v);
            
            /* Vectorized loop over output channels (16 at a time) */
            for (int ocb = 0; ocb < Co_block; ocb += 16) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*2 + j - P;
                    if (oi >= 0 && oi < To) {
                        /* Load 16 weights */
                        int w_idx = ocb * Ci * K + ic * K + j;
                        __m512 w16 = _mm512_loadu_ps(&w[w_idx]);
                        /* Load current output values */
                        __m512 ov = _mm512_loadu_ps(&o[ocb*To + oi]);
                        /* FMA: ov += vb * w16 */
                        ov = _mm512_fmadd_ps(vb, w16, ov);
                        /* Store back */
                        _mm512_storeu_ps(&o[ocb*To + oi], ov);
                    }
                }
            }
            
            /* Handle remaining channels with scalar fallback */
            for (int oc = Co_block; oc < Co; oc++) {
                for (int j = 0; j < K; j++) {
                    int oi = ii*2 + j - P;
                    if (oi >= 0 && oi < To) {
                        o[oc*To+oi] += v * w[oc*Ci*K + ic*K + j];
                    }
                }
            }
        }
    }
}

__attribute__((target("avx512f")))
void snake_avx512(float *o, const float *x, const float *a, int n, int C) {
    int n_block = n & ~15;
    
    for (int i = 0; i < n_block; i += 16) {
        /* Load 16 values from x */
        __m512 xv = _mm512_loadu_ps(x + i);
        
        /* Load corresponding alphas */
        float alpha_vals[16];
        for (int k = 0; k < 16; k++) {
            float al = a[(i+k)%C];
            alpha_vals[k] = (al < 1e-6f) ? 1e-6f : al;
        }
        __m512 av = _mm512_loadu_ps(alpha_vals);
        
        /* Compute av * xv for sin input */
        __m512 av_xv = _mm512_mul_ps(av, xv);
        
        /* sinf has no SIMD intrinsic - use scalar fallback */
        float tmp[16];
        _mm512_storeu_ps(tmp, av_xv);
        for (int k = 0; k < 16; k++) {
            float s = sinf(tmp[k]);
            tmp[k] = tmp[k] + s*s / alpha_vals[k];
        }
        __m512 res = _mm512_loadu_ps(tmp);
        
        _mm512_storeu_ps(o + i, res);
    }
    
    /* Handle remaining elements with scalar fallback */
    for (int i = n_block; i < n; i++) {
        float v = x[i], al = a[i%C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al*v);
        o[i] = v + sa*sa/al;
    }
}

__attribute__((target("avx512f")))
void add_avx512(float *o, const float *x, const float *y, int n) {
    int n_block = n & ~15;
    
    for (int i = 0; i < n_block; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 yv = _mm512_loadu_ps(y + i);
        _mm512_storeu_ps(o + i, _mm512_add_ps(xv, yv));
    }
    
    /* Handle remaining elements */
    for (int i = n_block; i < n; i++) {
        o[i] = x[i] + y[i];
    }
}

#endif /* __AVX512F__ */

#endif /* __x86_64__ */

/* ================================================================ */
/*  Dispatch table                                                  */
/* ================================================================ */

typedef struct {
    void (*conv1d)(float*,const float*,const float*,const float*,int,int,int,int);
    void (*conv_transpose1d)(float*,const float*,const float*,int,int,int,int,int);
    void (*group_norm)(float*,const float*,const float*,const float*,int,int,float);
    void (*snake)(float*,const float*,const float*,int,int);
    void (*add)(float*,const float*,const float*,int);
} CPUOps;

/* Dispatch function — picks the best SIMD level at runtime via CPUID */
static CPUOps get_ops(void) {
    CPUOps ops = { conv1d_s, convt1d_s, gn_s, snake_s, NULL };
    
#ifdef __x86_64__
#ifdef __AVX__
    if (HAS(7, 1, 16)) {
        /* AVX-512F available */
#ifdef __AVX512F__
        ops.conv1d = conv1d_avx512;
        ops.conv_transpose1d = convt1d_avx512;
        ops.snake = snake_avx512;
        ops.add = add_avx512;
#else
        /* AVX-512 detected but not compiled in - fall back to AVX2 */
        ops.conv1d = conv1d_avx2;
        ops.conv_transpose1d = convt1d_avx2;
        ops.snake = snake_avx2;
        ops.add = add_avx2;
#endif
    } else if (HAS(7, 1, 5)) {
        /* AVX2 available */
        ops.conv1d = conv1d_avx2;
        ops.conv_transpose1d = convt1d_avx2;
        ops.snake = snake_avx2;
        ops.add = add_avx2;
    } else if (HAS(1, 2, 28)) {
        /* AVX + FMA available */
        ops.conv1d = conv1d_avx;
        ops.conv_transpose1d = convt1d_avx;
        ops.snake = snake_avx;
        ops.add = add_avx;
    }
    /* Otherwise keep scalar defaults */
#endif
#endif

#ifdef __aarch64__
    if (cpu_arch_init() == 0) {
        if (cpu_arch_has_sve()) {
#ifdef __ARM_FEATURE_SVE
            ops.conv1d = conv1d_sve;
            ops.snake = snake_sve;
            ops.add = add_sve;
#else
            ops.conv1d = conv1d_neon;
            ops.snake = snake_neon;
            ops.add = add_neon;
#endif
        } else {
            ops.conv1d = conv1d_neon;
            ops.snake = snake_neon;
            ops.add = add_neon;
        }
        ops.group_norm = group_norm_neon;
    }
#endif

#if RISCV
    cpu_arch_init();
    if (riscv_rvv_available()) {
        ops.conv1d = conv1d_riscv;
        ops.snake = snake_riscv;
        ops.add = add_riscv;
    }
#endif

    return ops;
}

/* ================================================================ */
/*  BF8 Dequantization                                              */
/* ================================================================ */

float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
                              const DACTensor *bias,
                              int *out_Ci, int *out_K, int *out_Co, int *is_conv_transpose) {
    if (!weight_v) return NULL;

    int nd = weight_v->ndims;
    if (nd != 3) return NULL;

    int d0 = weight_v->dims[0];
    int d1 = weight_v->dims[1];
    int d2 = weight_v->dims[2];

    int Co = bias ? bias->dims[0] : d2;
    int K = d1;
    int Ci = (d0 == Co) ? d2 : d0;

    if (is_conv_transpose) *is_conv_transpose = (d0 == Co) ? 1 : 0;

    *out_Ci = Ci;
    *out_K = K;
    *out_Co = Co;

    int total_size = Ci * K * Co;
    float *w_f32 = (float *)malloc(total_size * sizeof(float));
    if (!w_f32) return NULL;

    if (weight_v->elem_size == 4) {
        const float *src = (const float *)weight_v->data;
        int per_input = weight_g ? (Ci == (int)weight_g->dims[2]) : 0;
        for (int ci = 0; ci < Ci; ci++) {
            for (int k = 0; k < K; k++) {
                for (int co = 0; co < Co; co++) {
                    int src_idx = per_input
                        ? co * K * Ci + k * Ci + ci
                        : ci * K * Co + k * Co + co;
                    int dst_idx = co * Ci * K + ci * K + k;
                    w_f32[dst_idx] = src[src_idx];
                }
            }
        }
        return w_f32;
    }

    if (!weight_g) { free(w_f32); return NULL; }

    int per_input = (Ci == (int)weight_g->dims[2]);

    const float *g_scales = (const float *)weight_g->data;
    const uint8_t *v_data = weight_v->data;

    for (int ci = 0; ci < Ci; ci++) {
        float g = g_scales[per_input ? ci : (ci * (int)weight_g->dims[2] / Ci)];
        for (int k = 0; k < K; k++) {
            for (int co = 0; co < Co; co++) {
                int src_idx;
                if (per_input) {
                    src_idx = co * K * Ci + k * Ci + ci;
                } else {
                    src_idx = ci * K * Co + k * Co + co;
                }
                int8_t v_val = (int8_t)v_data[src_idx];
                if (!per_input) g = g_scales[co];
                int dst_idx = co * Ci * K + ci * K + k;
                w_f32[dst_idx] = g * ((float)v_val - 128.0f) / 127.0f;
            }
        }
    }

    return w_f32;
}

/* ================================================================ */
/*  DAC decoder entry point                                         */
/* ================================================================ */

int cpu_decoder_run(DACTensor *ts, int nt,
                     const int *codes, int n_frames, int n_cb,
                     float *pcm, int n_samples, int ch)
{
    (void)n_samples;
    fprintf(stderr, "[cpu_dac] SIMD=%s frames=%d cb=%d ch=%d\n",
            cpu_simd_name(), n_frames, n_cb, ch);

    CPUOps ops = get_ops();
    
    /* Step 1: RVQ Lookup - aggregate codebook entries into features */
    /* Output: [1024, n_frames] features */
    float *rvq_out = (float *)calloc(1024 * n_frames, sizeof(float));
    if (!rvq_out) return TSAC_ERR_MEMORY;
    
    int rvq_dim = 1024;

    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        char cb_name[128];
        snprintf(cb_name, sizeof(cb_name),
                 "quantizer.quantizers.%d.codebook.weight", cb);
        DACTensor *codebook = tf(ts, nt, cb_name);
        if (!codebook) {
            fprintf(stderr, "[cpu_dac] WARNING: codebook %d not found\n", cb);
            continue;
        }

        int entries = codebook->dims[0];
        int dim     = codebook->dims[1];

        /* Codebook is raw float32 (elem_size=4) */
        const float *cb_data = (const float *)codebook->data;

        for (int f = 0; f < n_frames; f++) {
            int code_idx = f * n_cb + cb;
            int entry = codes[code_idx];
            if (entry < 0 || entry >= entries) entry = entry % entries;
            if (entry < 0) entry = 0;

            for (int d = 0; d < dim && d < rvq_dim; d++) {
                rvq_out[d * n_frames + f] += cb_data[entry * dim + d];
            }
        }
    }

    /* Step 2: Decoder model.0 - Conv1d(1024->1536, K=7) */
    DACTensor *m0_wv = tf(ts, nt, "decoder.model.0.weight_v");
    DACTensor *m0_wg = tf(ts, nt, "decoder.model.0.weight_g");
    DACTensor *m0_b = tf(ts, nt, "decoder.model.0.bias");
    
    int m0_Ci = 1024, m0_K = 7, m0_Co = 1536;
    float *m0_w = dequant_weights(m0_wv, m0_wg, m0_b, &m0_Ci, &m0_K, &m0_Co, NULL);
    const float *m0_b_data = m0_b ? (const float *)m0_b->data : NULL;
    
    /* Allocate intermediate buffers */
    float *buf0 = (float *)malloc(m0_Co * n_frames * sizeof(float));
    if (!buf0) { free(rvq_out); free(m0_w); return TSAC_ERR_MEMORY; }
    memset(buf0, 0, m0_Co * n_frames * sizeof(float));
    
    if (m0_w) {
        ops.conv1d(buf0, rvq_out, m0_w, m0_b_data, n_frames, m0_K, m0_Ci, m0_Co);
    }
    
    free(rvq_out);
    free(m0_w);
    
    /* Current feature map: buf0 [m0_Co, n_frames] = [1536, n_frames] */
    float *current = buf0;
    int current_C = m0_Co;
    
    /* Step 3: Decoder blocks 1-4 with Snake + ConvTranspose + GroupNorm */
    for (int block = 1; block <= 4; block++) {
        int target_C;
        switch (block) {
            case 1: target_C = 768; break;
            case 2: target_C = 384; break;
            case 3: target_C = 192; break;
            case 4: target_C = 96; break;
            default: target_C = current_C / 2;
        }
        
        /* Snake activation before conv */
        char snake_name[128];
        snprintf(snake_name, sizeof(snake_name), "decoder.model.%d.block.0.alpha", block);
        DACTensor *snake_alpha = tf(ts, nt, snake_name);
        
        if (snake_alpha) {
            const float *alpha = (const float *)snake_alpha->data;
            int alpha_C = snake_alpha->dims[0];
            ops.snake(current, current, alpha, current_C * n_frames, alpha_C);
        }
        
        /* ConvTranspose1d layer (upsampling factor 2) */
        char wv_name[128], wg_name[128], b_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
        snprintf(wg_name, sizeof(wg_name), "decoder.model.%d.block.1.weight_g", block);
        snprintf(b_name, sizeof(b_name), "decoder.model.%d.block.1.bias", block);
        
        DACTensor *wv = tf(ts, nt, wv_name);
        DACTensor *wg = tf(ts, nt, wg_name);
        DACTensor *b = tf(ts, nt, b_name);
        
        int conv_Ci, conv_K, conv_Co;
        int is_convt;
        float *w = dequant_weights(wv, wg, b, &conv_Ci, &conv_K, &conv_Co, &is_convt);
        const float *b_data = b ? (const float *)b->data : NULL;
        
        /* Output frames after upsampling */
        int n_frames_out = n_frames * (1 << block); /* 2x, 4x, 8x, 16x */
        
        float *next_buf = (float *)malloc(target_C * n_frames_out * sizeof(float));
        if (!next_buf) {
            free(current);
            free(w);
            return TSAC_ERR_MEMORY;
        }
        
        if (w) {
            /* Use conv_transpose1d with stride 2 for upsampling */
                ops.conv_transpose1d(next_buf, current, w, n_frames, n_frames_out, 
                                     conv_K, conv_Ci, conv_Co);
                /* Add bias */
                if (b_data) {
                for (int c = 0; c < conv_Co; c++) {
                    for (int t = 0; t < n_frames_out; t++) {
                        next_buf[c * n_frames_out + t] += b_data[c];
                    }
                }
            }
        }
        
        free(current);
        free(w);
        current = next_buf;
        current_C = target_C;
        
        /* Inner residual blocks (3x) */
        for (int inner = 2; inner <= 4; inner++) {
            char inner_snake[128], inner_wv[128], inner_wg[128], inner_b[128];
            snprintf(inner_snake, sizeof(inner_snake), 
                     "decoder.model.%d.block.%d.block.0.alpha", block, inner);
            snprintf(inner_wv, sizeof(inner_wv),
                     "decoder.model.%d.block.%d.block.1.weight_v", block, inner);
            snprintf(inner_wg, sizeof(inner_wg),
                     "decoder.model.%d.block.%d.block.1.weight_g", block, inner);
            snprintf(inner_b, sizeof(inner_b),
                     "decoder.model.%d.block.%d.block.1.bias", block, inner);
            
            DACTensor *is_alpha = tf(ts, nt, inner_snake);
            DACTensor *iwv = tf(ts, nt, inner_wv);
            DACTensor *iwg = tf(ts, nt, inner_wg);
            DACTensor *ib = tf(ts, nt, inner_b);
            
            float *residual = (float *)malloc(current_C * n_frames_out * sizeof(float));
            if (!residual) continue;
            memcpy(residual, current, current_C * n_frames_out * sizeof(float));
            
            /* Snake activation */
            if (is_alpha) {
                const float *alpha = (const float *)is_alpha->data;
                int alpha_C = is_alpha->dims[0];
                ops.snake(current, current, alpha, current_C * n_frames_out, alpha_C);
            }
            
            /* Conv1d */
            int inner_Ci, inner_K, inner_Co;
            float *iw = dequant_weights(iwv, iwg, ib, &inner_Ci, &inner_K, &inner_Co, NULL);
            const float *ib_data = ib ? (const float *)ib->data : NULL;
            
            if (iw) {
                float *conv_out = (float *)malloc(inner_Co * n_frames_out * sizeof(float));
                if (conv_out) {
                    ops.conv1d(conv_out, current, iw, ib_data, n_frames_out, inner_K, inner_Ci, inner_Co);
                    
                    /* Group norm */
                    char gn_w_name[128], gn_b_name[128];
                    snprintf(gn_w_name, sizeof(gn_w_name),
                             "decoder.model.%d.block.%d.block.2.weight", block, inner);
                    snprintf(gn_b_name, sizeof(gn_b_name),
                             "decoder.model.%d.block.%d.block.2.bias", block, inner);
                    DACTensor *gn_w = tf(ts, nt, gn_w_name);
                    DACTensor *gn_b = tf(ts, nt, gn_b_name);
                    
                    if (gn_w || gn_b) {
                        const float *gn_w_data = gn_w ? (const float *)gn_w->data : NULL;
                        const float *gn_b_data = gn_b ? (const float *)gn_b->data : NULL;
                        /* Assume 32 groups or fewer */
                        int G = 32;
                        while (inner_Co % G != 0 && G > 1) G /= 2;
                        ops.group_norm(conv_out, conv_out, gn_w_data, gn_b_data, 
                                       inner_Co * n_frames_out, G, 1e-5f);
                    }
                    
                    /* Add residual */
                    if (ops.add) {
                        ops.add(current, conv_out, residual, current_C * n_frames_out);
                    } else {
                        for (int i = 0; i < current_C * n_frames_out && i < inner_Co * n_frames_out; i++) {
                            current[i] = conv_out[i] + residual[i];
                        }
                    }
                    
                    free(conv_out);
                }
                free(iw);
            }
            
            free(residual);
        }
    }
    
    /* Step 4: Decoder model.5 - Snake activation */
    DACTensor *m5_alpha = tf(ts, nt, "decoder.model.5.alpha");
    if (m5_alpha && current_C == 96) {
        const float *alpha = (const float *)m5_alpha->data;
        int n_frames_16x = n_frames * 16;
        int alpha_C = m5_alpha->dims[0];
        ops.snake(current, current, alpha, current_C * n_frames_16x, alpha_C);
    }
    
    /* Step 5: Decoder model.6 - Conv1d(96->2, K=7) - output layer */
    DACTensor *m6_wv = tf(ts, nt, "decoder.model.6.weight_v");
    DACTensor *m6_wg = tf(ts, nt, "decoder.model.6.weight_g");
    DACTensor *m6_b = tf(ts, nt, "decoder.model.6.bias");
    
    int m6_Ci, m6_K, m6_Co;
    float *m6_w = dequant_weights(m6_wv, m6_wg, m6_b, &m6_Ci, &m6_K, &m6_Co, NULL);
    
    const float *m6_b_data = m6_b ? (const float *)m6_b->data : NULL;
    
    int n_frames_final = n_frames * 16;
    float *output = (float *)malloc(2 * n_frames_final * sizeof(float));
    
    if (m6_w && output) {
        ops.conv1d(output, current, m6_w, m6_b_data, n_frames_final, m6_K, m6_Ci, m6_Co);
        
        /* Copy to output PCM buffer */
        /* Output format: [ch, n_samples] interleaved or planar */
        int samples_to_copy = n_frames_final;
        if (samples_to_copy > n_samples) samples_to_copy = n_samples;
        
        for (int c = 0; c < ch && c < m6_Co; c++) {
            for (int s = 0; s < samples_to_copy; s++) {
                float val = output[c * n_frames_final + s];
                /* Apply tanh soft clipping */
                if (val > 1.0f) val = tanhf(val);
                if (val < -1.0f) val = tanhf(val);
                pcm[c * n_samples + s] = val;
            }
        }
        
        /* Fill remaining samples with zeros */
        for (int c = 0; c < ch; c++) {
            for (int s = samples_to_copy; s < n_samples; s++) {
                pcm[c * n_samples + s] = 0.0f;
            }
        }
    }
    
    free(current);
    free(m6_w);
    free(output);
    
    return TSAC_OK;
}
