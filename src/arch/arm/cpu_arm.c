/*
 * cpu_arm.c — ARM64 NEON/SVE CPU acceleration for TSAC.
 * Compile with: -O3 -mfpu=neon (or default for ARM64)
 * Runtime detection via getauxval(AT_HWCAP).
 */

#include "cpu_arm.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __aarch64__
#include <sys/auxv.h>
#include <arm_neon.h>

#define HWCAP_ASIMD   (1 << 1)
#define HWCAP_SVE     (1 << 22)

static int arm_has_sve = 0;

int cpu_arch_init(void) {
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_SVE) arm_has_sve = 1;
    return (hwcap & HWCAP_ASIMD) ? 0 : -1;
}

const char *cpu_arch_name(void) {
    if (arm_has_sve) return "ARM64 SVE";
    return "ARM64 NEON";
}

int cpu_arch_has_sve(void) {
    return arm_has_sve;
}

/* NEON-optimized elementwise add */
void add_neon(float *o, const float *a, const float *b, int n) {
    int i; for (i = 0; i <= n - 4; i += 4)
        vst1q_f32(o + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) o[i] = a[i] + b[i];
}

/* NEON conv1d — inner loop uses FMLA */
void conv1d_neon(float *out, const float *in, const float *w,
                  const float *bias, int T, int K, int Ci, int Co)
{
    int P = K / 2;
    for (int oc = 0; oc < Co; oc++)
        for (int oi = 0; oi < T; oi++) {
            float32x4_t sum4 = vdupq_n_f32(bias ? bias[oc] : 0.0f);
            for (int ic = 0; ic < Ci; ic++)
                for (int j = 0; j < K; j += 4) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        float32x4_t x4 = vld1q_f32(in + ic * T + ii);
                        float32x4_t w4 = vld1q_f32(w + oc * Ci * K + ic * K + j);
                        sum4 = vfmaq_f32(sum4, x4, w4);
                    }
                }
            float sum = vaddvq_f32(sum4);
            out[oc * T + oi] = sum;
        }
}

/* Snake activation (NEON-assisted — sinf has no NEON intrinsic, use scalar) */
void snake_neon(float *out, const float *in, const float *alpha,
                int n, int C)
{
    for (int i = 0; i < n; i++) {
        float x = in[i], a = alpha[i % C];
        if (a < 1e-6f) a = 1e-6f;
        float sa = sinf(a * x);
        out[i] = x + sa * sa / a;
    }
}

/* NEON group normalization */
void group_norm_neon(float *o, const float *x, const float *w,
                      const float *b, int N, int G, float eps)
{
    int E = N / G;
    for (int g = 0; g < G; g++) {
        float32x4_t sum4 = vdupq_n_f32(0), sum_sq4 = vdupq_n_f32(0);
        int i = 0;
        for (; i <= E - 4; i += 4) {
            float32x4_t v4 = vld1q_f32(x + g*E + i);
            sum4 = vaddq_f32(sum4, v4);
            sum_sq4 = vfmaq_f32(sum_sq4, v4, v4);
        }
        float s = vaddvq_f32(sum4), sq = vaddvq_f32(sum_sq4);
        for (; i < E; i++) { float v = x[g*E+i]; s += v; sq += v*v; }
        float mn = s / E, vr = sq / E - mn * mn;
        float is = 1.0f / sqrtf(fmaxf(vr + eps, 1e-10f));
        float32x4_t mn4 = vdupq_n_f32(mn), is4 = vdupq_n_f32(is);
        float wg = w ? w[g] : 1.0f, bg = b ? b[g] : 0.0f;
        float32x4_t wg4 = vdupq_n_f32(wg), bg4 = vdupq_n_f32(bg);
        for (i = 0; i <= E - 4; i += 4) {
            float32x4_t v4 = vld1q_f32(x + g*E + i);
            vst1q_f32(o + g*E + i,
                vfmaq_f32(bg4, vmulq_f32(vsubq_f32(v4, mn4), is4), wg4));
        }
        for (; i < E; i++) {
            int idx = g*E + i; o[idx] = (x[idx]-mn) * is * wg + bg;
        }
    }
}

/* SVE-optimized kernels (Scalable Vector Extension) */
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

void add_sve(float *o, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t av = svld1(pg, a + i);
        svfloat32_t bv = svld1(pg, b + i);
        svst1(pg, o + i, svadd_f32_m(pg, av, bv));
    }
}

void conv1d_sve(float *o, const float *x, const float *w, const float *b,
                int T, int K, int Ci, int Co)
{
    int P = K / 2;
    for (int oc = 0; oc < Co; oc++) {
        float bias = b ? b[oc] : 0.0f;
        for (int oi = 0; oi < T; oi++) {
            svfloat32_t sum = svdup_f32(bias);
            for (int ic = 0; ic < Ci; ic += svcntw()) {
                svbool_t pg = svwhilelt_b32(ic, Ci);
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T) {
                        svfloat32_t xv = svld1(pg, &x[ic*T + ii]);
                        svfloat32_t wv = svld1(pg, &w[oc*Ci*K + ic*K + j]);
                        sum = svmla_f32_m(pg, sum, xv, wv);
                    }
                }
            }
            o[oc*T+oi] = svaddv(svptrue_b32(), sum);
        }
    }
}

void snake_sve(float *o, const float *x, const float *a, int n, int C)
{
    for (int i = 0; i < n; i++) {
        float xv = x[i], av = a[i % C];
        if (av < 1e-6f) av = 1e-6f;
        float sa = sinf(av * xv);
        o[i] = xv + sa * sa / av;
    }
}
#endif /* __ARM_FEATURE_SVE */

#else /* not ARM64 — stub */
int cpu_arch_init(void) { return -1; }
const char *cpu_arch_name(void) { return "not ARM64"; }
int cpu_arch_has_sve(void) { return 0; }
void add_neon(float *o, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) o[i] = a[i] + b[i];
}
void conv1d_neon(float *o, const float *x, const float *w, const float *b,
                  int T, int K, int Ci, int Co) { (void)o;(void)x;(void)w;(void)b;(void)T;(void)K;(void)Ci;(void)Co; }
void snake_neon(float *o, const float *x, const float *a, int n, int C) { (void)o;(void)x;(void)a;(void)n;(void)C; }
void group_norm_neon(float *o, const float *x, const float *w, const float *b,
                      int N, int G, float eps) { (void)o;(void)x;(void)w;(void)b;(void)N;(void)G;(void)eps; }
#endif
