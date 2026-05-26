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

#define BATCH_FRAMES    16
#define CONTEXT_PAD     10
#define DEBUG_DECODER    0

#if DEBUG_DECODER
#define DBG(...) fprintf(stderr, __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

extern void conv1d_s(float*, const float*, const float*, const float*, int, int, int, int);
extern void convt1d_s(float*, const float*, const float*, int, int, int, int, int, int);
extern void snake_s(float*, const float*, const float*, int, int);

typedef struct {
    int oc_start, oc_end;
    float *o;
    const float *x, *w, *b;
    int T, K, Ci, Co;
    void (*kernel)(float*,const float*,const float*,const float*,int,int,int,int);
} Conv1dJob;

static void *conv1d_thread(void *arg) {
    Conv1dJob *j = (Conv1dJob *)arg;
    int Co_sub = j->oc_end - j->oc_start;
    if (Co_sub <= 0) return NULL;
    j->kernel(
        j->o + j->oc_start * j->T,
        j->x,
        j->w + j->oc_start * j->Ci * j->K,
        j->b ? j->b + j->oc_start : NULL,
        j->T, j->K, j->Ci, Co_sub
    );
    return NULL;
}

/* Parallel conv1d: split output channels across threads */
static void conv1d_parallel(void (*kern)(float*,const float*,const float*,const float*,int,int,int,int),
                            float *o, const float *x, const float *w, const float *b,
                            int T, int K, int Ci, int Co, int n_threads)
{
    if (n_threads <= 1 || Co < n_threads * 2) {
        kern(o, x, w, b, T, K, Ci, Co);
        return;
    }
    int nt = (n_threads < Co) ? n_threads : Co;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    Conv1dJob jobs[nt];
    int per = (Co + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (Conv1dJob){
            .oc_start = t * per,
            .oc_end   = (t + 1) * per < Co ? (t + 1) * per : Co,
            .o = o, .x = x, .w = w, .b = b,
            .T = T, .K = K, .Ci = Ci, .Co = Co,
            .kernel = kern
        };
        pthread_create(&tid[t], NULL, conv1d_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: conv_transpose parallel over output channels */
typedef struct {
    int oc_start, oc_end;
    float *o;
    const float *x, *w;
    int Ti, To, K, Ci, Co, stride;
    void (*kernel)(float*,const float*,const float*,int,int,int,int,int,int);
} Convt1dJob;

static void *convt1d_thread(void *arg) {
    Convt1dJob *j = (Convt1dJob *)arg;
    int Co_sub = j->oc_end - j->oc_start;
    if (Co_sub <= 0) return NULL;
    j->kernel(
        j->o + j->oc_start * j->To,
        j->x,
        j->w + j->oc_start * j->Ci * j->K,
        j->Ti, j->To, j->K, j->Ci, Co_sub, j->stride
    );
    return NULL;
}

static void convt1d_parallel(void (*kern)(float*,const float*,const float*,int,int,int,int,int,int),
                             float *o, const float *x, const float *w,
                             int Ti, int To, int K, int Ci, int Co, int stride, int n_threads)
{
    memset(o, 0, (size_t)Co * To * sizeof(float));
    if (n_threads <= 1 || Co < n_threads * 2) {
        kern(o, x, w, Ti, To, K, Ci, Co, stride);
        return;
    }
    int nt = (n_threads < Co) ? n_threads : Co;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    Convt1dJob jobs[nt];
    int per = (Co + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (Convt1dJob){
            .oc_start = t * per,
            .oc_end   = (t + 1) * per < Co ? (t + 1) * per : Co,
            .o = o, .x = x, .w = w,
            .Ti = Ti, .To = To, .K = K, .Ci = Ci, .Co = Co, .stride = stride,
            .kernel = kern
        };
        pthread_create(&tid[t], NULL, convt1d_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: snake parallel over elements */
typedef struct {
    int start, end;
    float *o;
    const float *x, *a;
    int C;
} SnakeJob;

static void *snake_thread(void *arg) {
    SnakeJob *j = (SnakeJob *)arg;
    for (int i = j->start; i < j->end; i++) {
        float v = j->x[i], al = j->a[i % j->C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al * v);
        j->o[i] = v + sa * sa / al;
    }
    return NULL;
}

static void snake_parallel(float *o, const float *x, const float *a, int n, int C, int n_threads) {
    if (n_threads <= 1 || n < n_threads * 1024) {
        snake_s(o, x, a, n, C);
        return;
    }
    int nt = n_threads;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    SnakeJob jobs[nt];
    int per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (SnakeJob){
            .start = t * per,
            .end   = (t + 1) * per < n ? (t + 1) * per : n,
            .o = o, .x = x, .a = a, .C = C
        };
        pthread_create(&tid[t], NULL, snake_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* Thread worker: add parallel over elements */
typedef struct {
    int start, end;
    float *o;
    const float *x, *y;
} AddJob;

static void *add_thread(void *arg) {
    AddJob *j = (AddJob *)arg;
    for (int i = j->start; i < j->end; i++)
        j->o[i] = j->x[i] + j->y[i];
    return NULL;
}

static void add_parallel(float *o, const float *x, const float *y, int n, int n_threads) {
    if (n_threads <= 1 || n < n_threads * 2048) {
        for (int i = 0; i < n; i++) o[i] = x[i] + y[i];
        return;
    }
    int nt = n_threads;
    pthread_t *tid = (pthread_t *)alloca(nt * sizeof(pthread_t));
    AddJob jobs[nt];
    int per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        jobs[t] = (AddJob){
            .start = t * per,
            .end   = (t + 1) * per < n ? (t + 1) * per : n,
            .o = o, .x = x, .y = y
        };
        pthread_create(&tid[t], NULL, add_thread, &jobs[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

/* ================================================================ */
/*  CPU feature detection                                           */
/* ================================================================ */
/*  CPU feature detection                                           */
/* ================================================================ */

typedef enum { SIMD_SCALAR, SIMD_SSE42, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;

static int cpu_has(unsigned int leaf, unsigned int reg, unsigned int bit) {
#if X86_64
    unsigned int a, b, c, d;
    int ok;
    if (leaf == 7) {
        ok = __get_cpuid_count(7, 0, &a, &b, &c, &d);
    } else {
        ok = __get_cpuid(leaf, &a, &b, &c, &d);
    }
    if (!ok) return 0;
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
#if defined(__x86_64__) || defined(__i386__)
    if (HAS(7, 1, 16)) return "AVX-512F";
    if (HAS(7, 1, 5))  return "AVX2";
    if (HAS(1, 2, 28)) return "AVX+FMA";
    if (HAS(1, 2, 20)) return "SSE4.2";
    return "scalar";
#elif defined(__aarch64__)
    extern int cpu_arch_has_sve(void);
    extern const char *cpu_arch_name(void);
    (void)cpu_arch_init();
    return cpu_arch_name();
#elif defined(__riscv)
    extern const char *cpu_arch_name(void);
    (void)cpu_arch_init();
    return cpu_arch_name();
#else
    return "scalar";
#endif
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

static void conv1d_dilated_s(float *o, const float *x, const float *w, const float *b,
                             int T, int K, int Ci, int Co, int dilation) {
    int P = (K/2) * dilation;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) {
                    int ii = oi + j*dilation - P;
                    if (ii >= 0 && ii < T) s += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                }
            o[oc*T+oi] = s;
        }
}

static void conv1d_strided_s(float *o, const float *x, const float *w, const float *b,
                              int T_in, int T_out, int K, int Ci, int Co, int stride) {
    int P = K/2; memset(o, 0, (size_t)Co * T_out * sizeof(float));
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T_out; oi++) {
            float s = b ? b[oc] : 0;
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j++) {
                    int ii = oi * stride + j - P;
                    if (ii >= 0 && ii < T_in)
                        s += x[ic * T_in + ii] * w[oc * Ci * K + ic * K + j];
                }
            o[oc * T_out + oi] = s;
        }
}

void convt1d_s(float *o, const float *x, const float *w,
               int Ti, int To, int K, int Ci, int Co, int stride) {
    int P = K/2; memset(o, 0, (size_t)Co*To*sizeof(float));
    for (int ic = 0; ic < Ci; ic++) for (int ii = 0; ii < Ti; ii++) {
        float v = x[ic*Ti+ii]; if (v == 0) continue;
        for (int oc = 0; oc < Co; oc++) for (int j = 0; j < K; j++) {
            int oi = ii*stride + j - P;
            if (oi >= 0 && oi < To) o[oc*To+oi] += v * w[oc*Ci*K + ic*K + j];
        }
    }
}

void gn_s(float *o, const float *x, const float *w, const float *b,
          int C, int T, int G, float eps) {
    int Cg = C/G;
    int E = Cg*T;
    for (int g = 0; g < G; g++) {
        float s = 0, sq = 0;
        for (int i = 0; i < E; i++) { float v = x[g*E+i]; s += v; sq += v*v; }
        float mn = s/E, vr = sq/E - mn*mn, is = 1.0f/sqrtf(fmaxf(vr+eps, 1e-10f));
        for (int i = 0; i < E; i++) {
            int idx = g*E+i;
            int channel = g*Cg + i/T;
            o[idx] = (x[idx]-mn)*is*(w?w[channel]:1)+(b?b[channel]:0);
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
                 int Ti, int To, int K, int Ci, int Co, int stride) {
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
                    int oi = ii*stride + j - P;
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
                    int oi = ii*stride + j - P;
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
                  int Ti, int To, int K, int Ci, int Co, int stride) {
    convt1d_avx(o, x, w, Ti, To, K, Ci, Co, stride);
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
                    int Ti, int To, int K, int Ci, int Co, int stride) {
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
                    int oi = ii*stride + j - P;
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
                    int oi = ii*stride + j - P;
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
    void (*conv_transpose1d)(float*,const float*,const float*,int,int,int,int,int,int);
    void (*group_norm)(float*,const float*,const float*,const float*,int,int,int,float);
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

    /* Determine layer type: conv_transpose has bias->dims[0] == weight_v->dims[0],
     * meaning the stored layout is [Co, K, Ci] rather than [Ci, K, Co].
     * K-based heuristic fails for encoder convs with K=4/8/16. */
    int is_ct = (bias && bias->dims[0] == d0 && d0 != d2) ? 1 : 0;
    int Ci = is_ct ? d2 : d0;

    if (is_conv_transpose) *is_conv_transpose = is_ct;

    *out_Ci = Ci;
    *out_K = K;
    *out_Co = Co;

    int total_size = Ci * K * Co;
    float *w_f32 = (float *)malloc(total_size * sizeof(float));
    if (!w_f32) return NULL;

    int src_size = d0 * d1 * d2;
    float *src_f32 = (float *)malloc((size_t)src_size * sizeof(float));
    if (!src_f32) { free(w_f32); return NULL; }

    if (weight_v->elem_size == 4) {
        memcpy(src_f32, weight_v->data, (size_t)src_size * sizeof(float));
    } else if (weight_v->data_size == src_size) {
        const uint8_t *v_data = weight_v->data;
        for (int i = 0; i < src_size; i++) src_f32[i] = ((float)v_data[i] - 128.0f) / 127.0f;
    } else {
        /* LibNC q8/BF8 tensors store grouped value bytes plus one scale byte per group.
         * The scale byte is a linear block scale centered around 127, so common bytes
         * 127/129 keep the group near unit scale while preserving group alignment. */
        int n_groups = weight_v->data_size - src_size;
        if (n_groups <= 0 || src_size % n_groups != 0) {
            free(src_f32); free(w_f32); return NULL;
        }
        int group_size = src_size / n_groups;
        const uint8_t *p = weight_v->data;
        int out = 0;
        for (int g = 0; g < n_groups; g++) {
            float group_scale = (float)(p[group_size] ? p[group_size] : 1) / 127.0f;
            for (int i = 0; i < group_size; i++) {
                src_f32[out++] = ((float)*p++ - 128.0f) * group_scale;
            }
            p++; /* grouped BF8 scale byte */
        }
    }

    const float *g_scales = weight_g ? (const float *)weight_g->data : NULL;
    int norm_channels = weight_g ? (int)weight_g->dims[2] : d2;
    float *norms = (float *)calloc((size_t)norm_channels, sizeof(float));
    if (!norms) { free(src_f32); free(w_f32); return NULL; }

    for (int i0 = 0; i0 < d0; i0++)
        for (int k = 0; k < K; k++)
            for (int i2 = 0; i2 < d2; i2++) {
                int src_idx = i0 * K * d2 + k * d2 + i2;
                float v = src_f32[src_idx];
                if (i2 < norm_channels) norms[i2] += v * v;
            }
    for (int i = 0; i < norm_channels; i++) norms[i] = sqrtf(norms[i] + 1e-12f);

    for (int ci = 0; ci < Ci; ci++) {
        for (int k = 0; k < K; k++) {
            for (int co = 0; co < Co; co++) {
                int src_idx = is_ct
                    ? co * K * Ci + k * Ci + ci
                    : ci * K * Co + k * Co + co;
                int norm_idx = is_ct ? ci : co;
                float g = weight_g ? g_scales[norm_idx] : 1.0f;
                float n = (norm_idx < norm_channels) ? norms[norm_idx] : 1.0f;
                int dst_idx = co * Ci * K + ci * K + k;
                w_f32[dst_idx] = src_f32[src_idx] * g / n;
            }
        }
    }

    free(norms);
    free(src_f32);

    return w_f32;
}

/* ================================================================ */
/*  DAC decoder entry point                                         */
/* ================================================================ */

#if DEBUG_DECODER
static int count_nan(const float *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(data[i])) count++;
    }
    return count;
}
#endif

static int decode_batch(DACTensor *ts, int nt,
                        const int *codes, int n_cb, int code_offset,
                        int ctx_frames, int n_threads,
                        float *pcm, int n_samples, int ch,
                        int batch_start, int batch_frames,
                        int total_upscale, CPUOps ops)
{
    int rvq_dim = 1024;

    /* RVQ lookup for ctx_frames starting at code_offset */
    float *rvq_out = (float *)calloc(1024 * ctx_frames, sizeof(float));
    if (!rvq_out) return TSAC_ERR_MEMORY;

    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        char ip_name[128], op_name[128];
        snprintf(ip_name, sizeof(ip_name),
                 "quantizer.quantizers.%d.in_proj.weight_v", cb);
        snprintf(op_name, sizeof(op_name),
                 "quantizer.quantizers.%d.out_proj.weight_v", cb);

        DACTensor *ip_wv = tf(ts, nt, ip_name);
        DACTensor *op_wv = tf(ts, nt, op_name);
        DACTensor *ip_wg = NULL;
        if (ip_wv && op_wv) {
            ip_name[strlen(ip_name)-1] = 'g';
            ip_wg = tf(ts, nt, ip_name);
        }
        if (!ip_wv || !op_wv) continue;

        int ip_Ci, ip_K, ip_Co, op_Ci, op_K, op_Co, dummy;
        float *ip_f32 = dequant_weights(ip_wv, ip_wg, NULL, &ip_Ci, &ip_K, &ip_Co, &dummy);
        DACTensor *op_bias = tf(ts, nt, "dummy");
        float *op_f32 = dequant_weights(op_wv, NULL, op_bias, &op_Ci, &op_K, &op_Co, &dummy);
        if (!ip_f32 || !op_f32) { free(ip_f32); free(op_f32); continue; }

        /* in_proj: [1024, 1, 8] → 1024 rows of 8 values
         * out_proj: [8, 1, 1024] → 8 rows of 1024 values */
        for (int f = 0; f < ctx_frames; f++) {
            int code_idx = (code_offset + f) * n_cb + cb;
            int raw = codes[code_idx];
            if (raw < 0) raw = 0;
            if (raw >= ip_Ci) raw = ip_Ci - 1;

            /* in_proj lookup: row 'raw' gives 8 intermediate values */
            float ip_vec[8];
            for (int o = 0; o < 8 && o < ip_Co; o++)
                ip_vec[o] = ip_f32[raw * ip_Co + o];

            /* out_proj: 8×1024 matrix multiply → 1024-dim feature */
            for (int d = 0; d < op_Co && d < rvq_dim; d++) {
                float sum = 0;
                for (int o = 0; o < 8 && o < op_Ci; o++)
                    sum += ip_vec[o] * op_f32[o * op_Co + d];
                rvq_out[d * ctx_frames + f] += sum;
            }
        }
        free(ip_f32);
        free(op_f32);
    }

    DBG("[DEBUG] RVQ NaN: %d/%d\n", count_nan(rvq_out, 1024*ctx_frames), 1024*ctx_frames);

    /* model.0 conv1d */
    DACTensor *m0_wv = tf(ts, nt, "decoder.model.0.weight_v");
    DACTensor *m0_wg = tf(ts, nt, "decoder.model.0.weight_g");
    DACTensor *m0_b  = tf(ts, nt, "decoder.model.0.bias");

    int m0_Ci = 1024, m0_K = 7, m0_Co = 1536;
    float *m0_w = dequant_weights(m0_wv, m0_wg, m0_b, &m0_Ci, &m0_K, &m0_Co, NULL);
    const float *m0_b_data = m0_b ? (const float *)m0_b->data : NULL;

    float *buf0 = (float *)malloc((size_t)m0_Co * ctx_frames * sizeof(float));
    if (!buf0) { free(rvq_out); free(m0_w); return TSAC_ERR_MEMORY; }
    memset(buf0, 0, (size_t)m0_Co * ctx_frames * sizeof(float));

    if (m0_w) {
        conv1d_parallel(ops.conv1d, buf0, rvq_out, m0_w, m0_b_data,
                        ctx_frames, m0_K, m0_Ci, m0_Co, n_threads);
    }
    free(rvq_out);
    free(m0_w);

    DBG("[DEBUG] After m0 conv1d NaN: %d/%d\n", count_nan(buf0, m0_Co*ctx_frames), m0_Co*ctx_frames);

    float *current = buf0;
    int current_C = m0_Co;
    int cur_frames = ctx_frames;

    /* decoder blocks 1-4 */
    for (int block = 1; block <= 4; block++) {
        int target_C;
        switch (block) {
            case 1: target_C = 768; break;
            case 2: target_C = 384; break;
            case 3: target_C = 192; break;
            case 4: target_C = 96; break;
            default: target_C = current_C / 2;
        }

        /* snake before conv */
        char snake_name[128];
        snprintf(snake_name, sizeof(snake_name), "decoder.model.%d.block.0.alpha", block);
        DACTensor *snake_alpha = tf(ts, nt, snake_name);
        if (snake_alpha) {
            const float *alpha = (const float *)snake_alpha->data;
            int alpha_C = snake_alpha->dims[0];
            snake_parallel(current, current, alpha,
                           current_C * cur_frames, alpha_C, n_threads);
        }

        DBG("[DEBUG] After block%d snake NaN: %d/%d\n", block, count_nan(current, current_C*cur_frames), current_C*cur_frames);

        /* conv_transpose upsampling */
        char wv_name[128], wg_name[128], b_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
        snprintf(wg_name, sizeof(wg_name), "decoder.model.%d.block.1.weight_g", block);
        snprintf(b_name, sizeof(b_name), "decoder.model.%d.block.1.bias", block);

        DACTensor *wv = tf(ts, nt, wv_name);
        DACTensor *wg = tf(ts, nt, wg_name);
        DACTensor *b  = tf(ts, nt, b_name);

        int conv_Ci, conv_K, conv_Co, is_convt;
        float *w = dequant_weights(wv, wg, b, &conv_Ci, &conv_K, &conv_Co, &is_convt);
        const float *b_data = b ? (const float *)b->data : NULL;

        int conv_stride = conv_K / 2;
        int n_frames_out = cur_frames * conv_stride;

        float *next_buf = (float *)malloc((size_t)target_C * n_frames_out * sizeof(float));
        if (!next_buf) { free(current); free(w); return TSAC_ERR_MEMORY; }

        if (w) {
            convt1d_parallel(ops.conv_transpose1d, next_buf, current, w,
                             cur_frames, n_frames_out,
                             conv_K, conv_Ci, conv_Co, conv_stride, n_threads);
            if (b_data) {
                for (int c = 0; c < conv_Co; c++) {
                    for (int t = 0; t < n_frames_out; t++) {
                        next_buf[c * n_frames_out + t] += b_data[c];
                    }
                }
            }
        }

        DBG("[DEBUG] After block%d convt NaN: %d/%d\n", block, count_nan(next_buf, target_C*n_frames_out), target_C*n_frames_out);

        free(current);
        free(w);
        current = next_buf;
        current_C = target_C;
        cur_frames = n_frames_out;

        /* DAC residual units: Snake -> dilated Conv1d(K=7) -> Snake -> Conv1d(K=1) -> skip */
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
            DACTensor *ib  = tf(ts, nt, inner_b);

            float *residual = (float *)malloc((size_t)current_C * cur_frames * sizeof(float));
            if (!residual) continue;
            memcpy(residual, current, (size_t)current_C * cur_frames * sizeof(float));

            if (is_alpha) {
                const float *alpha = (const float *)is_alpha->data;
                int alpha_C = is_alpha->dims[0];
                snake_parallel(current, current, alpha,
                               current_C * cur_frames, alpha_C, n_threads);
            }

            int ic_Ci, ic_K, ic_Co;
            float *iw = dequant_weights(iwv, iwg, ib, &ic_Ci, &ic_K, &ic_Co, NULL);
            const float *ib_data = ib ? (const float *)ib->data : NULL;

            if (iw) {
                float *conv_out = (float *)malloc((size_t)ic_Co * cur_frames * sizeof(float));
                if (conv_out) {
                    int dilation = (inner == 2) ? 1 : (inner == 3 ? 3 : 9);
                    if (dilation == 1)
                        conv1d_parallel(ops.conv1d, conv_out, current, iw, ib_data,
                                        cur_frames, ic_K, ic_Ci, ic_Co, n_threads);
                    else
                        conv1d_dilated_s(conv_out, current, iw, ib_data,
                                         cur_frames, ic_K, ic_Ci, ic_Co, dilation);

                    char snake2_name[128], wv2_name[128], wg2_name[128], b2_name[128];
                    snprintf(snake2_name, sizeof(snake2_name),
                             "decoder.model.%d.block.%d.block.2.alpha", block, inner);
                    snprintf(wv2_name, sizeof(wv2_name),
                             "decoder.model.%d.block.%d.block.3.weight_v", block, inner);
                    snprintf(wg2_name, sizeof(wg2_name),
                             "decoder.model.%d.block.%d.block.3.weight_g", block, inner);
                    snprintf(b2_name, sizeof(b2_name),
                             "decoder.model.%d.block.%d.block.3.bias", block, inner);
                    DACTensor *snake2 = tf(ts, nt, snake2_name);
                    if (snake2) {
                        snake_parallel(conv_out, conv_out, (const float *)snake2->data,
                                       ic_Co * cur_frames, snake2->dims[0], n_threads);
                    }

                    DACTensor *wv2 = tf(ts, nt, wv2_name);
                    DACTensor *wg2 = tf(ts, nt, wg2_name);
                    DACTensor *b2 = tf(ts, nt, b2_name);
                    int c2_Ci, c2_K, c2_Co;
                    float *w2 = dequant_weights(wv2, wg2, b2, &c2_Ci, &c2_K, &c2_Co, NULL);
                    float *unit_out = conv_out;
                    if (w2) {
                        unit_out = (float *)malloc((size_t)c2_Co * cur_frames * sizeof(float));
                        if (unit_out) {
                            conv1d_parallel(ops.conv1d, unit_out, conv_out, w2,
                                            b2 ? (const float *)b2->data : NULL,
                                            cur_frames, c2_K, c2_Ci, c2_Co, n_threads);
                        } else {
                            unit_out = conv_out;
                        }
                        free(w2);
                    }

                    int n = current_C * cur_frames;
                    if (ops.add) add_parallel(current, unit_out, residual, n, n_threads);
                    else for (int i = 0; i < n; i++) current[i] = unit_out[i] + residual[i];

                    if (unit_out != conv_out) free(unit_out);
                    free(conv_out);
                }
                free(iw);
            }
            free(residual);
        }
    }

    /* model.5 snake */
    DACTensor *m5_alpha = tf(ts, nt, "decoder.model.5.alpha");
    if (m5_alpha && current_C == 96) {
        const float *alpha = (const float *)m5_alpha->data;
        int alpha_C = m5_alpha->dims[0];
        snake_parallel(current, current, alpha,
                       current_C * cur_frames, alpha_C, n_threads);
    }

    /* model.6 output conv1d */
    DACTensor *m6_wv = tf(ts, nt, "decoder.model.6.weight_v");
    DACTensor *m6_wg = tf(ts, nt, "decoder.model.6.weight_g");
    DACTensor *m6_b  = tf(ts, nt, "decoder.model.6.bias");

    int m6_Ci, m6_K, m6_Co;
    float *m6_w = dequant_weights(m6_wv, m6_wg, m6_b, &m6_Ci, &m6_K, &m6_Co, NULL);
    const float *m6_b_data = m6_b ? (const float *)m6_b->data : NULL;

    DBG("[DEBUG] m6: Ci=%d, K=%d, Co=%d, cur_frames=%d\n", m6_Ci, m6_K, m6_Co, cur_frames);

    float *output = (float *)malloc((size_t)m6_Co * cur_frames * sizeof(float));

    if (m6_w && output) {
        ops.conv1d(output, current, m6_w, m6_b_data,
                   cur_frames, m6_K, m6_Ci, m6_Co);

        DBG("[DEBUG] After m6 conv1d NaN: %d/%d\n", count_nan(output, m6_Co*cur_frames), m6_Co*cur_frames);

        /* Compute how many output samples to discard from the beginning
         * due to context padding. code_offset is the start of the context
         * window, batch_start is the start of the actual batch data. */
        int discard_frames = batch_start - code_offset;
        if (discard_frames < 0) discard_frames = 0;
        int discard_samples = discard_frames * total_upscale;

        int out_start = batch_start * total_upscale;
        int out_count = batch_frames * total_upscale;

        /* Bounds check: ensure we don't read past output buffer */
        int max_s = cur_frames - discard_samples;
        if (max_s > out_count) max_s = out_count;
        if (max_s < 0) max_s = 0;

        int clipped = 0;
        for (int c = 0; c < ch && c < m6_Co; c++) {
            for (int s = 0; s < max_s; s++) {
                int final_s = out_start + s;
                if (final_s >= n_samples) break;
                float val = tanhf(output[c * cur_frames + discard_samples + s]);
                if (val > 1.0f) { val = 1.0f; clipped++; }
                if (val < -1.0f) { val = -1.0f; clipped++; }
                pcm[c * n_samples + final_s] = val;
            }
        }
        DBG("[DEBUG] Clipped samples: %d/%d\n", clipped, max_s * ch);
    }

    free(current);
    free(m6_w);
    free(output);
    return TSAC_OK;
}

int cpu_decoder_run(DACTensor *ts, int nt,
                     const int *codes, int n_frames, int n_cb,
                     float *pcm, int n_samples, int ch,
                     int n_threads)
{
    (void)n_samples;
    int total_batches = (n_frames + BATCH_FRAMES - 1) / BATCH_FRAMES;
    fprintf(stderr, "[cpu_dac] SIMD=%s frames=%d cb=%d ch=%d threads=%d\n",
            cpu_simd_name(), n_frames, n_cb, ch, n_threads);
    fprintf(stderr, "batch_size=%d bs=%d n_cb=%d block_len=%d\n",
            BATCH_FRAMES, 1, n_cb, total_batches);

    CPUOps ops = get_ops();

    int total_upscale = 1;
    for (int block = 1; block <= 4; block++) {
        char wv_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
        DACTensor *wv = tf(ts, nt, wv_name);
        if (wv && wv->ndims >= 2) {
            int K = wv->dims[1];
            total_upscale *= (K / 2);
        }
    }

    int processed_frames = 0;
    int batch_idx = 0;

    for (int batch_start = 0; batch_start < n_frames; batch_start += BATCH_FRAMES) {
        int batch_end = batch_start + BATCH_FRAMES;
        if (batch_end > n_frames) batch_end = n_frames;
        int batch_frames = batch_end - batch_start;
        batch_idx++;

        int ctx_start = batch_start - CONTEXT_PAD;
        if (ctx_start < 0) ctx_start = 0;
        int ctx_end = batch_end + CONTEXT_PAD;
        if (ctx_end > n_frames) ctx_end = n_frames;
        int ctx_frames = ctx_end - ctx_start;

        int ret = decode_batch(ts, nt, codes, n_cb, ctx_start,
                               ctx_frames, n_threads,
                               pcm, n_samples, ch,
                               batch_start, batch_frames, total_upscale, ops);
        if (ret != TSAC_OK) return ret;

        processed_frames += batch_frames;
        int pct = (int)((float)batch_idx / (float)total_batches * 100.0f);
        if (pct > 100) pct = 100;
        fprintf(stderr, "%3u%%\r", pct);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    return TSAC_OK;
}

/* ================================================================ */
/*  CPU Encoder                                                      */
/* ================================================================ */

int cpu_encoder_run(DACTensor *ts, int nt,
                    const float *pcm, int n_samples, int channels,
                    int n_codebooks, int block_len,
                    int **codebook_indices, int *n_frames) {
    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    *n_frames = nf;

    CPUOps ops = get_ops();
    int rvq_dim = 1024;

    int padded_samples = nf * block_len;
    float *padded = (float *)calloc((size_t)padded_samples * channels, sizeof(float));
    if (!padded) return TSAC_ERR_MEMORY;
    memcpy(padded, pcm, (size_t)n_samples * channels * sizeof(float));

    fprintf(stderr, "[cpu_enc] frames=%d cb=%d ch=%d blk=%d\n", nf, n_codebooks, channels, block_len);
    fprintf(stderr, "batch_size=%d bs=%d n_cb=%d block_len=%d\n", 1, 1, n_codebooks, nf);

    /* Encoder tensor naming: encoder.block.X (NOT encoder.model.X).
     * Internal block mapping differs from decoder:
     *   block.3.alpha = snake before conv  (we call block.0)
     *   block.4.*     = conv_transpose      (we call block.1)
     *   block.{0,1,2}.block.* = inner residual units (we call inner={2,3,4}) */
    DACTensor *e6_wv = tf(ts, nt, "encoder.block.0.weight_v");
    DACTensor *e6_wg = tf(ts, nt, "encoder.block.0.weight_g");
    DACTensor *e6_b  = tf(ts, nt, "encoder.block.0.bias");
    int e6_Ci, e6_K, e6_Co;
    float *e6_w = dequant_weights(e6_wv, e6_wg, e6_b, &e6_Ci, &e6_K, &e6_Co, NULL);
    const float *e6_b_data = e6_b ? (const float *)e6_b->data : NULL;
    int blk_T = block_len; /* temporal dimension: process all block samples through encoder */

    /* Allocate per-block feature buffer and per-frame output */
    float *features = (float *)calloc((size_t)rvq_dim * nf, sizeof(float));
    if (!features) { free(padded); free(e6_w); return TSAC_ERR_MEMORY; }

    if (!e6_w) { free(padded); free(e6_w); free(features); return TSAC_ERR_MEMORY; }

    /* Process each block independently through full encoder, keeping temporal
     * context (block_len time steps) throughout all layers. Only reduce to
     * a single feature vector at the output via center-frame selection. */
    float *blk_in  = (float *)malloc((size_t)e6_Ci * blk_T * sizeof(float));
    if (!blk_in) { free(padded); free(e6_w); free(features); return TSAC_ERR_MEMORY; }

    for (int f = 0; f < nf; f++) {
        int start = f * block_len;
        for (int t = 0; t < blk_T; t++) {
            for (int c = 0; c < e6_Ci; c++) {
                int si = start + t;
                int src_c = (c < channels) ? c : 0;
                blk_in[c * blk_T + t] = (si < padded_samples)
                    ? padded[si * channels + src_c] : 0.0f;
            }
        }

        /* model.6: Conv1d(ch→64, K=7) over block_len samples */
        float *cur = (float *)malloc((size_t)e6_Co * blk_T * sizeof(float));
        conv1d_s(cur, blk_in, e6_w, e6_b_data, blk_T, e6_K, e6_Ci, e6_Co);
        int cur_C = e6_Co;
        int cur_T = blk_T;

        /* Blocks 4→1: strided convs for temporal reduction */
        int enc_c_out[4] = {128, 256, 512, 1024};
        int strides[4] = {8, 4, 4, 2}; /* K/2-based strides: 320→40→10→2→1 */
        for (int blk = 4; blk >= 1; blk--) {
            int idx = 4 - blk;
            int stride = strides[idx];
            int nxt_T = (cur_T + stride - 1) / stride;

            /* Snake before conv */
            char sn[128]; snprintf(sn, sizeof(sn), "encoder.block.%d.block.3.alpha", blk);
            DACTensor *sa = tf(ts, nt, sn);
            if (sa) snake_s(cur, cur, (const float *)sa->data, cur_C * cur_T, sa->dims[0]);

            /* Strided conv1d for channel expansion + temporal reduction */
            char wvn[128]; snprintf(wvn, sizeof(wvn), "encoder.block.%d.block.4.weight_v", blk);
            char wgn[128]; snprintf(wgn, sizeof(wgn), "encoder.block.%d.block.4.weight_g", blk);
            char bn[128];  snprintf(bn, sizeof(bn), "encoder.block.%d.block.4.bias", blk);
            DACTensor *wv = tf(ts, nt, wvn), *wg = tf(ts, nt, wgn), *b = tf(ts, nt, bn);
            int c_Ci, c_K, c_Co;
            float *cw = dequant_weights(wv, wg, b, &c_Ci, &c_K, &c_Co, NULL);
            float *nxt = (float *)malloc((size_t)enc_c_out[idx] * nxt_T * sizeof(float));
            if (cw && c_Ci <= cur_C)
                conv1d_strided_s(nxt, cur, cw, b?(const float*)b->data:NULL, cur_T, nxt_T, c_K, c_Ci, c_Co, stride);
            free(cur); free(cw); cur = nxt; cur_C = enc_c_out[idx]; cur_T = nxt_T;

            /* Inner residual units on reduced temporal dimension */
            for (int inner = 2; inner <= 4; inner++) {
                int io = inner - 2;
                char isn[128]; snprintf(isn, sizeof(isn), "encoder.block.%d.block.%d.block.0.alpha", blk, io);
                DACTensor *isa = tf(ts, nt, isn);
                float *res = (float *)malloc((size_t)cur_C * cur_T * sizeof(float));
                if (!res) continue;
                memcpy(res, cur, (size_t)cur_C * cur_T * sizeof(float));
                if (isa) snake_s(cur, cur, (const float *)isa->data, cur_C * cur_T, isa->dims[0]);
                char iwn[128]; snprintf(iwn, sizeof(iwn), "encoder.block.%d.block.%d.block.1.weight_v", blk, io);
                char igw[128]; snprintf(igw, sizeof(igw), "encoder.block.%d.block.%d.block.1.weight_g", blk, io);
                char ibn[128]; snprintf(ibn, sizeof(ibn), "encoder.block.%d.block.%d.block.1.bias", blk, io);
                DACTensor *iwv = tf(ts, nt, iwn), *iwg = tf(ts, nt, igw), *ib = tf(ts, nt, ibn);
                int ic_Ci, ic_K, ic_Co;
                float *iw = dequant_weights(iwv, iwg, ib, &ic_Ci, &ic_K, &ic_Co, NULL);
                if (!iw) { free(res); continue; }
                float *co = (float *)malloc((size_t)ic_Co * cur_T * sizeof(float));
                if (co) {
                    conv1d_s(co, cur, iw, ib?(const float*)ib->data:NULL, cur_T, ic_K, ic_Ci, ic_Co);
                    char s2n[128]; snprintf(s2n, sizeof(s2n), "encoder.block.%d.block.%d.block.2.alpha", blk, io);
                    DACTensor *s2a = tf(ts, nt, s2n);
                    if (s2a) snake_s(co, co, (const float *)s2a->data, ic_Co * cur_T, s2a->dims[0]);
                    char w2n[128]; snprintf(w2n, sizeof(w2n), "encoder.block.%d.block.%d.block.3.weight_v", blk, io);
                    char g2n[128]; snprintf(g2n, sizeof(g2n), "encoder.block.%d.block.%d.block.3.weight_g", blk, io);
                    char b2n[128]; snprintf(b2n, sizeof(b2n), "encoder.block.%d.block.%d.block.3.bias", blk, io);
                    DACTensor *w2v=tf(ts,nt,w2n),*w2g=tf(ts,nt,g2n),*b2=tf(ts,nt,b2n);
                    int c2_Ci,c2_K,c2_Co; float *w2=dequant_weights(w2v,w2g,b2,&c2_Ci,&c2_K,&c2_Co,NULL);
                    float *uo=co;
                    if(w2){uo=(float*)malloc((size_t)c2_Co*cur_T*sizeof(float));
                     if(uo)conv1d_s(uo,co,w2,b2?(const float*)b2->data:NULL,cur_T,c2_K,c2_Ci,c2_Co);
                     else uo=co; free(w2);}
                    int nn=cur_C*cur_T;
                    for(int i=0;i<nn;i++)cur[i]=uo[i]+res[i];
                    if(uo!=co)free(uo); free(co);
                }
                free(iw); free(res);
            }
        }

        /* model.5 snake + feature extraction (cur_T should be ~1 after strides) */
        DACTensor *e5_a = tf(ts, nt, "encoder.block.5.alpha");
        if (e5_a) snake_s(cur, cur, (const float *)e5_a->data, cur_C * cur_T, e5_a->dims[0]);

        /* Take center frame as single feature vector per block */
        int center = cur_T / 2;
         for (int d = 0; d < rvq_dim && d < cur_C; d++)
            features[d * nf + f] = cur[d * cur_T + center];
        free(cur);
        int pct = (int)((float)(f + 1) / (float)nf * 100.0f);
        if (pct > 100) pct = 100;
        fprintf(stderr, "%3u%%\r", pct);
        fflush(stderr);
    }
    fprintf(stderr, "\n");
    free(blk_in); free(padded); free(e6_w);

    /* RVQ quantization on [rvq_dim, nf] features */

    /* RVQ quantization */
    int *indices = (int *)calloc((size_t)nf * n_codebooks, sizeof(int));
    if (!indices) { free(features); return TSAC_ERR_MEMORY; }
    float *residual_buf = (float *)malloc((size_t)rvq_dim * nf * sizeof(float));
    if (!residual_buf) { free(indices); free(features); return TSAC_ERR_MEMORY; }
    memcpy(residual_buf, features, (size_t)rvq_dim * nf * sizeof(float));

    for (int cb = 0; cb < n_codebooks && cb < 12; cb++) {
        char cb_name[128];
        snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
        DACTensor *codebook = tf(ts, nt, cb_name);
        if (!codebook) continue;
        int entries = codebook->dims[0];
        const float *cb_data = (const float *)codebook->data;
        for (int f = 0; f < nf; f++) {
            float best = 1e30f; int best_e = 0;
            for (int e = 0; e < entries; e++) {
                float d = 0;
                const float *c = cb_data + e * rvq_dim;
                for (int i = 0; i < rvq_dim; i++) {
                    float df = residual_buf[i * nf + f] - c[i];
                    d += df * df;
                }
                if (d < best) { best = d; best_e = e; }
            }
            indices[f * n_codebooks + cb] = best_e;
            const float *ch = cb_data + best_e * rvq_dim;
            for (int i = 0; i < rvq_dim; i++)
                residual_buf[i * nf + f] -= ch[i];
        }
    }
    free(features); free(residual_buf);
    *codebook_indices = indices;
    return TSAC_OK;
}
