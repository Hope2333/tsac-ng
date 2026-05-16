#include <hip/hip_runtime.h>
#include "../src/dac_model.h"
#include "../include/tsac.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BLK 256

__global__ void add_k(float *o, const float *a, const float *b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

__global__ void add_bias_k(float *x, const float *b, int T, int C) {
    int c = blockIdx.x, t = blockIdx.y * BLK + threadIdx.y;
    if (c >= C || t >= T) return;
    x[c * T + t] += b[c];
}

__global__ void mul_k(float *o, const float *a, const float *b, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i];
}

__global__ void i8tof32_k(float *o, const int8_t *x, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) o[i] = (float)x[i] / 127.0f;
}

__global__ void snake_k(float *o, const float *x, const float *a,
                         int n, int C) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float al = a[i % C];
    o[i] = v + __sinf(al * v) * __sinf(al * v) / fmaxf(al, 1e-6f);
}

__global__ void conv1d_k(float *o, const float *x, const float *w,
    const float *b, int T, int K, int Ci, int Co, int S) {
    int oc = blockIdx.x, oi = blockIdx.y * BLK + threadIdx.y;
    if (oc >= Co || oi >= T) return;
    float s = b ? b[oc] : 0;
    int P = K/2;
    for (int ic = 0; ic < Ci; ic++)
        for (int j = 0; j < K; j++) {
            int ii = oi * S + j - P;
            if (ii >= 0 && ii < T)
                s += x[ic*T + ii] * w[oc*Ci*K + ic*K + j];
        }
    o[oc*T + oi] = s;
}

__global__ void convt_k(float *o, const float *x, const float *w,
    int T_in, int T_out, int K, int Ci, int Co, int S) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Co * T_out;
    if (tid >= total) return;

    int oi = tid % T_out;
    int oc = tid / T_out;
    int P = K / 2;

    float sum = 0.0f;
    for (int ic = 0; ic < Ci; ic++) {
        for (int j = 0; j < K; j++) {
            int tmp = oi + P - j;
            if (tmp >= 0 && tmp % S == 0) {
                int ii = tmp / S;
                if (ii < T_in)
                    sum += x[ic * T_in + ii] * w[oc * Ci * K + ic * K + j];
            }
        }
    }
    o[tid] = sum;
}

__global__ void gn_k(float *o, const float *x, const float *w,
    const float *b, int E, int G, float eps) {
    extern __shared__ float sh[];
    int t = threadIdx.x;
    float s = 0, sq = 0;
    for (int i = t; i < E; i += blockDim.x) {
        float v = x[blockIdx.x*E + i];
        s += v;
        sq += v*v;
    }
    sh[t] = s;
    sh[blockDim.x + t] = sq;
    __syncthreads();
    for (int r = blockDim.x/2; r > 0; r >>= 1) {
        if (t < r) { sh[t] += sh[t+r]; sh[blockDim.x+t] += sh[blockDim.x+t+r]; }
        __syncthreads();
    }
    float mn = sh[0]/E, vr = sh[blockDim.x]/E - mn*mn;
    float is = rsqrtf(fmaxf(vr + eps, 1e-10f));
    for (int i = t; i < E; i += blockDim.x) {
        int idx = blockIdx.x*E + i;
        o[idx] = (x[idx] - mn) * is * (w ? w[blockIdx.x] : 1) + (b ? b[blockIdx.x] : 0);
    }
}

__global__ void us2x_k(float *o, const float *x, int C, int T) {
    int c = blockIdx.x, t = threadIdx.x;
    if (c >= C || t >= T) return;
    o[c*T*2 + t*2] = o[c*T*2 + t*2+1] = x[c*T + t];
}

__global__ void rvq_lookup_k(float *features, const int *codes,
    const float **codebooks, int n_frames, int n_cb, int rvq_dim) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    
    for (int cb = 0; cb < n_cb; cb++) {
        int entry = codes[f * n_cb + cb];
        const float *cb_data = codebooks[cb];
        if (!cb_data) continue;
        
        for (int d = 0; d < rvq_dim; d++) {
            atomicAdd(&features[d * n_frames + f], cb_data[entry * rvq_dim + d]);
        }
    }
}

__global__ void rvq_lookup_simple_k(float *features, const int *codes,
    const float *cb_data, int cb_idx, int entries, int n_frames, int n_cb, int rvq_dim) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    
    int entry = codes[f * n_cb + cb_idx];
    if (entry < 0) entry = 0;
    if (entry >= entries) entry = entry % entries;
    if (entry < 0) entry = 0;
    
    for (int d = 0; d < rvq_dim; d++) {
        features[d * n_frames + f] += cb_data[entry * rvq_dim + d];
    }
}

__global__ void tanh_clip_k(float *x, int n) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v > 1.0f) v = tanhf(v);
    if (v < -1.0f) v = tanhf(v);
    x[i] = v;
}

/* RVQ quantization kernels for encoder */
__global__ void rvq_quantize_k(float *indices, const float *features,
    const float *codebook, int n_frames, int rvq_dim, int entries) {
    int f = blockIdx.x * BLK + threadIdx.x;
    if (f >= n_frames) return;
    float best = 1e30f; int best_e = 0;
    for (int e = 0; e < entries; e++) {
        float d = 0;
        for (int dim = 0; dim < rvq_dim; dim++) {
            float df = features[f * rvq_dim + dim] - codebook[e * rvq_dim + dim];
            d += df * df;
        }
        if (d < best) { best = d; best_e = e; }
    }
    indices[f] = (float)best_e;
}

__global__ void rvq_subtract_k(float *features, const float *codebook,
    const float *indices, int n_frames, int rvq_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_frames * rvq_dim) return;
    int f = tid / rvq_dim, d = tid % rvq_dim;
    features[tid] -= codebook[(int)indices[f] * rvq_dim + d];
}

/* ================================================================ */
/*  CPU-side tensor finder and weight upload                         */
/* ================================================================ */

static DACTensor *F(DACTensor *ts, int n, const char *s) {
    for (int i = 0; i < n; i++) if (!strcmp(ts[i].name, s)) return &ts[i];
    return NULL;
}

static float *upload_f32(DACTensor *t, hipStream_t st, DACTensor *ts, int nt, DACTensor *bias_t) {
    if (!t || !t->data) return NULL;
    if (t->dev_f32) return t->dev_f32;

    if (t->elem_size == 4) {
        hipMalloc(&t->dev_f32, t->data_size);
        hipMemcpyAsync(t->dev_f32, t->data, t->data_size, hipMemcpyHostToDevice, st);
    } else if (t->elem_size == 1) {
        char gn[256];
        strncpy(gn, t->name, 255);
        gn[255] = 0;
        char *vp = strstr(gn, "weight_v");
        if (!vp) return NULL;
        memcpy(vp, "weight_g", 8);
        DACTensor *gt = F(ts, nt, gn);
        if (!gt || gt->elem_size != 4) return NULL;

        /* Weight dims: [Ci, K, Co] or [Co, K, Ci] - figure out which */
        int d0 = t->dims[0];
        int d1 = t->dims[1];
        int d2 = t->dims[2];

        /* Use bias size for Co */
        int Co = bias_t ? bias_t->dims[0] : d2;
        int K = d1;
        int Ci = (d0 == Co) ? d2 : d0;

        /* Determine if scales are per-input-channel or per-output-channel */
        int per_input = (Ci == (int)gt->dims[2]);

        int nel_elements = Ci * K * Co;

        float *cpu_buf = (float*)malloc(nel_elements * sizeof(float));
        const float *g_scales = (const float*)gt->data;
        const uint8_t *v_data = (const uint8_t*)t->data;

        /* Dequantize with transposition to [Co, Ci, K] format */
        for (int co = 0; co < Co; co++) {
            for (int ci = 0; ci < Ci; ci++) {
                float g_val = per_input ? g_scales[ci] : g_scales[co];
                for (int k = 0; k < K; k++) {
                    int src_idx;
                    if (per_input) {
                        /* [Co, K, Ci] layout: src[co][k][ci] */
                        src_idx = co * K * Ci + k * Ci + ci;
                    } else {
                        /* [Ci, K, Co] layout: src[ci][k][co] */
                        src_idx = ci * K * Co + k * Co + co;
                    }
                    int8_t v_val = (int8_t)v_data[src_idx];
                    int dst_idx = co * Ci * K + ci * K + k;  /* [Co, Ci, K] layout */
                    cpu_buf[dst_idx] = g_val * ((float)v_val - 128.0f) / 127.0f;
                }
            }
        }
        hipMalloc(&t->dev_f32, nel_elements * sizeof(float));
        hipMemcpyAsync(t->dev_f32, cpu_buf, nel_elements * sizeof(float), hipMemcpyHostToDevice, st);
        free(cpu_buf);
    }
    return t->dev_f32;
}

/* ================================================================ */
/*  GPU weight cache (ported from CUDA backend)                      */
/* ================================================================ */

#define MAX_GPU_WEIGHTS 128

typedef struct {
    char     name[128];
    float   *d_data;
    int      Ci, K, Co;
    int      is_convt;
} GpuWeight;

typedef struct {
    int          initialized;
    int          weights_uploaded;
    GpuWeight    gpu_weights[MAX_GPU_WEIGHTS];
    int          n_gpu_weights;
    float       *d_cb_data;
    int         *d_cb_offsets;
    int          cb_offsets[13];
    int          n_cb;
    int          cb_dim;
    int          cb_entries;
    float       *d_buf[8];
    int          buf_sizes[8];
    int          n_bufs;
    int         *d_codes;
    float       *d_features;
    hipStream_t  stream;
} HipBackend;

static void hip_backend_free_bufs(HipBackend *b) {
    for (int i = 0; i < b->n_bufs; i++) {
        if (b->d_buf[i]) hipFree(b->d_buf[i]);
        b->d_buf[i] = NULL;
        b->buf_sizes[i] = 0;
    }
    b->n_bufs = 0;
}

static float *hip_backend_get_buf(HipBackend *b, int idx, size_t needed) {
    if (idx >= 8) { fprintf(stderr, "[hip] buf index %d out of range\n", idx); return NULL; }
    while (b->n_bufs <= idx) {
        b->d_buf[b->n_bufs] = NULL;
        b->buf_sizes[b->n_bufs] = 0;
        b->n_bufs++;
    }
    if (b->buf_sizes[idx] < (int)needed) {
        if (b->d_buf[idx]) hipFree(b->d_buf[idx]);
        hipError_t e = hipMalloc(&b->d_buf[idx], needed);
        if (e != hipSuccess) return NULL;
        b->buf_sizes[idx] = (int)needed;
    }
    return b->d_buf[idx];
}

static GpuWeight *hip_find_weight(HipBackend *b, const char *name) {
    for (int i = 0; i < b->n_gpu_weights; i++)
        if (!strcmp(b->gpu_weights[i].name, name))
            return &b->gpu_weights[i];
    return NULL;
}

static GpuWeight *hip_find_enc_weight(HipBackend *b, const char *name) {
    char fullname[256];
    snprintf(fullname, sizeof(fullname), "enc:%s", name);
    return hip_find_weight(b, fullname);
}

/* CPU-side dequantization helper for weight cache */
static float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
                               const DACTensor *bias,
                               int *out_Ci, int *out_K, int *out_Co, int *is_conv_transpose) {
    if (!weight_v || !weight_g) return NULL;

    int d0 = weight_v->dims[0];
    int d1 = weight_v->dims[1];
    int d2 = weight_v->dims[2];

    int Co = bias ? bias->dims[0] : d2;
    int K = d1;
    int Ci = (d0 == Co) ? d2 : d0;

    *out_Ci = Ci;
    *out_K = K;
    *out_Co = Co;
    *is_conv_transpose = (d0 == Co) ? 1 : 0;

    int per_input = (Ci == (int)weight_g->dims[2]);
    int nel = Ci * K * Co;

    float *cpu_buf = (float*)malloc(nel * sizeof(float));
    if (!cpu_buf) return NULL;

    const float *g_scales = (const float*)weight_g->data;
    const uint8_t *v_data = (const uint8_t*)weight_v->data;

    for (int co = 0; co < Co; co++) {
        for (int ci = 0; ci < Ci; ci++) {
            float g_val = per_input ? g_scales[ci] : g_scales[co];
            for (int k = 0; k < K; k++) {
                int src_idx;
                if (per_input) {
                    src_idx = co * K * Ci + k * Ci + ci;
                } else {
                    src_idx = ci * K * Co + k * Co + co;
                }
                int8_t v_val = (int8_t)v_data[src_idx];
                int dst_idx = co * Ci * K + ci * K + k;
                cpu_buf[dst_idx] = g_val * ((float)v_val - 128.0f) / 127.0f;
            }
        }
    }

    return cpu_buf;
}

static int hip_upload_weights(HipBackend *b, DACTensor *ts, int nt) {
    if (!ts) return TSAC_ERR_PARAM;

    /* Upload dequantized conv weights */
    const char *weight_names[] = {
        "decoder.model.0",
        "decoder.model.1.block.1",
        "decoder.model.2.block.1",
        "decoder.model.3.block.1",
        "decoder.model.4.block.1",
        "decoder.model.6",
        /* Inner blocks (model.1-4, inner 2-4) */
        "decoder.model.1.block.2.block.1",
        "decoder.model.1.block.3.block.1",
        "decoder.model.1.block.4.block.1",
        "decoder.model.2.block.2.block.1",
        "decoder.model.2.block.3.block.1",
        "decoder.model.2.block.4.block.1",
        "decoder.model.3.block.2.block.1",
        "decoder.model.3.block.3.block.1",
        "decoder.model.3.block.4.block.1",
        "decoder.model.4.block.2.block.1",
        "decoder.model.4.block.3.block.1",
        "decoder.model.4.block.4.block.1",
    };
    int n_wl = sizeof(weight_names) / sizeof(weight_names[0]);

    b->n_gpu_weights = 0;
    for (int i = 0; i < n_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      weight_names[i]);

        DACTensor *twv = F(ts, nt, wv);
        DACTensor *twg = F(ts, nt, wg);
        DACTensor *tbi = F(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "%s", weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        hipMalloc(&gw->d_data, w_bytes);
        hipMemcpy(gw->d_data, w_f32, w_bytes, hipMemcpyHostToDevice);
        free(w_f32);

        /* Upload bias tensor separately */
        if (tbi && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "%s.bias", weight_names[i]);
            gbias->Ci = tbi->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)tbi->dims[0] * sizeof(float);
            hipMalloc(&gbias->d_data, bias_bytes);
            hipMemcpy(gbias->d_data, tbi->data, bias_bytes, hipMemcpyHostToDevice);
        }
    }

    /* Upload encoder weights with "enc:" prefix */
    const char *enc_weight_names[] = {
        "encoder.model.0",
        "encoder.model.1.block.1",
        "encoder.model.2.block.1",
        "encoder.model.3.block.1",
        "encoder.model.4.block.1",
        "encoder.model.6",
        /* Inner blocks */
        "encoder.model.1.block.2.block.1",
        "encoder.model.1.block.3.block.1",
        "encoder.model.1.block.4.block.1",
        "encoder.model.2.block.2.block.1",
        "encoder.model.2.block.3.block.1",
        "encoder.model.2.block.4.block.1",
        "encoder.model.3.block.2.block.1",
        "encoder.model.3.block.3.block.1",
        "encoder.model.3.block.4.block.1",
        "encoder.model.4.block.2.block.1",
        "encoder.model.4.block.3.block.1",
        "encoder.model.4.block.4.block.1",
    };
    int n_enc_wl = sizeof(enc_weight_names) / sizeof(enc_weight_names[0]);
    for (int i = 0; i < n_enc_wl && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        char wv[160], wg[160], bi[160];
        snprintf(wv, sizeof(wv), "%s.weight_v", enc_weight_names[i]);
        snprintf(wg, sizeof(wg), "%s.weight_g", enc_weight_names[i]);
        snprintf(bi, sizeof(bi), "%s.bias",      enc_weight_names[i]);

        DACTensor *twv = F(ts, nt, wv);
        DACTensor *twg = F(ts, nt, wg);
        DACTensor *tbi = F(ts, nt, bi);

        if (!twv) continue;

        int Ci, K, Co, is_convt;
        float *w_f32 = dequant_weights(twv, twg, tbi, &Ci, &K, &Co, &is_convt);
        if (!w_f32) continue;

        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_weight_names[i]);
        gw->Ci = Ci;
        gw->K  = K;
        gw->Co = Co;
        gw->is_convt = is_convt;

        size_t w_bytes = (size_t)Ci * K * Co * sizeof(float);
        hipMalloc(&gw->d_data, w_bytes);
        hipMemcpy(gw->d_data, w_f32, w_bytes, hipMemcpyHostToDevice);
        free(w_f32);

        if (tbi && b->n_gpu_weights < MAX_GPU_WEIGHTS) {
            GpuWeight *gbias = &b->gpu_weights[b->n_gpu_weights++];
            snprintf(gbias->name, sizeof(gbias->name), "enc:%s.bias", enc_weight_names[i]);
            gbias->Ci = tbi->dims[0];
            gbias->K = 1;
            gbias->Co = 1;
            gbias->is_convt = 0;
            size_t bias_bytes = (size_t)tbi->dims[0] * sizeof(float);
            hipMalloc(&gbias->d_data, bias_bytes);
            hipMemcpy(gbias->d_data, tbi->data, bias_bytes, hipMemcpyHostToDevice);
        }
    }

    /* Upload snake alpha tensors */
    const char *snake_names[] = {
        "decoder.model.1.block.0.alpha",
        "decoder.model.2.block.0.alpha",
        "decoder.model.3.block.0.alpha",
        "decoder.model.4.block.0.alpha",
        "decoder.model.5.alpha",
        "decoder.model.1.block.2.block.0.alpha",
        "decoder.model.1.block.3.block.0.alpha",
        "decoder.model.1.block.4.block.0.alpha",
        "decoder.model.2.block.2.block.0.alpha",
        "decoder.model.2.block.3.block.0.alpha",
        "decoder.model.2.block.4.block.0.alpha",
        "decoder.model.3.block.2.block.0.alpha",
        "decoder.model.3.block.3.block.0.alpha",
        "decoder.model.3.block.4.block.0.alpha",
        "decoder.model.4.block.2.block.0.alpha",
        "decoder.model.4.block.3.block.0.alpha",
        "decoder.model.4.block.4.block.0.alpha",
    };
    for (int i = 0; i < (int)(sizeof(snake_names)/sizeof(snake_names[0])) && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        DACTensor *ta = F(ts, nt, snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "%s", snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        hipMalloc(&gw->d_data, n_bytes);
        hipMemcpy(gw->d_data, ta->data, n_bytes, hipMemcpyHostToDevice);
    }

    /* Upload encoder snake alphas with "enc:" prefix */
    const char *enc_snake_names[] = {
        "encoder.model.1.block.0.alpha",
        "encoder.model.2.block.0.alpha",
        "encoder.model.3.block.0.alpha",
        "encoder.model.4.block.0.alpha",
        "encoder.model.5.alpha",
        "encoder.model.1.block.2.block.0.alpha",
        "encoder.model.1.block.3.block.0.alpha",
        "encoder.model.1.block.4.block.0.alpha",
        "encoder.model.2.block.2.block.0.alpha",
        "encoder.model.2.block.3.block.0.alpha",
        "encoder.model.2.block.4.block.0.alpha",
        "encoder.model.3.block.2.block.0.alpha",
        "encoder.model.3.block.3.block.0.alpha",
        "encoder.model.3.block.4.block.0.alpha",
        "encoder.model.4.block.2.block.0.alpha",
        "encoder.model.4.block.3.block.0.alpha",
        "encoder.model.4.block.4.block.0.alpha",
    };
    for (int i = 0; i < (int)(sizeof(enc_snake_names)/sizeof(enc_snake_names[0])) && b->n_gpu_weights < MAX_GPU_WEIGHTS; i++) {
        DACTensor *ta = F(ts, nt, enc_snake_names[i]);
        if (!ta) continue;
        size_t n_bytes = (size_t)ta->dims[0] * sizeof(float);
        GpuWeight *gw = &b->gpu_weights[b->n_gpu_weights++];
        snprintf(gw->name, sizeof(gw->name), "enc:%s", enc_snake_names[i]);
        gw->Ci = ta->dims[0]; gw->K = 1; gw->Co = 1; gw->is_convt = 0;
        hipMalloc(&gw->d_data, n_bytes);
        hipMemcpy(gw->d_data, ta->data, n_bytes, hipMemcpyHostToDevice);
    }

    /* Upload all codebooks */
    {
        int total_entries = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = F(ts, nt, cb_name);
            if (!cb_t) break;
            b->cb_offsets[cb] = total_entries;
            total_entries += cb_t->dims[0] * cb_t->dims[1];
        }
        b->n_cb = 12;
        b->cb_dim = 1024;
        b->cb_entries = 8;

        size_t cb_total = (size_t)total_entries * sizeof(float);
        hipMalloc(&b->d_cb_data, cb_total);
        hipMalloc(&b->d_cb_offsets, 13 * sizeof(int));

        size_t copied = 0;
        for (int cb = 0; cb < 12; cb++) {
            char cb_name[160];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *cb_t = F(ts, nt, cb_name);
            if (!cb_t) { b->cb_offsets[cb] = (int)copied / (int)sizeof(float); continue; }
            size_t sz = (size_t)cb_t->dims[0] * cb_t->dims[1] * sizeof(float);
            hipMemcpy((char *)b->d_cb_data + copied, cb_t->data, sz, hipMemcpyHostToDevice);
            b->cb_offsets[cb] = (int)(copied / sizeof(float));
            copied += sz;
        }
        b->cb_offsets[12] = (int)(copied / sizeof(float));
        hipMemcpy(b->d_cb_offsets, b->cb_offsets, 13 * sizeof(int), hipMemcpyHostToDevice);
    }

    b->weights_uploaded = 1;
    fprintf(stderr, "[hip] uploaded %d weights + %d snake alphas + codebooks\n",
            n_wl, (int)(sizeof(snake_names)/sizeof(snake_names[0])));

    return TSAC_OK;
}

/* ================================================================ */
/*  Main decoder run function (ported from CUDA backend)             */
/* ================================================================ */

extern "C" int dac_decoder_run(DACTensor *ts, int nt,
    const int *codes, int n_frames, int n_cb,
    float *pcm, int n_samples, int ch)
{
    fprintf(stderr, "[hip] dac_decoder_run: nt=%d frames=%d cb=%d samples=%d ch=%d\n",
            nt, n_frames, n_cb, n_samples, ch);

    /* Static backend storage for weight caching across calls */
    static HipBackend *b = NULL;
    if (!b) {
        b = (HipBackend *)calloc(1, sizeof(HipBackend));
        if (!b) return TSAC_ERR_MEMORY;
        hipSetDevice(0);
        hipStreamCreate(&b->stream);
        b->initialized = 1;
    }

    hipStream_t s = b->stream;

    /* Lazy weight upload on first call */
    if (!b->weights_uploaded) {
        int ret = hip_upload_weights(b, ts, nt);
        if (ret != TSAC_OK) return ret;
        /* Weight sanity check */
        GpuWeight *w0 = hip_find_weight(b, "decoder.model.0");
        if (w0) {
            float test[4];
            hipMemcpy(test, w0->d_data, 4*sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "[hip] w0[0..3]=%f %f %f %f\n", test[0], test[1], test[2], test[3]);
        } else {
            fprintf(stderr, "[hip] WARNING: decoder.model.0 weight not found!\n");
        }
    }

    if (n_frames < 1) return TSAC_OK;

    int rvq_dim = 1024;

    /* Upload codebook indices */
    size_t codes_bytes = (size_t)n_frames * n_cb * sizeof(int);
    hipMalloc(&b->d_codes, codes_bytes);
    hipMemcpy(b->d_codes, codes, codes_bytes, hipMemcpyHostToDevice);

    /* RVQ lookup -> features [1024, n_frames] */
    size_t feat_bytes = (size_t)rvq_dim * n_frames * sizeof(float);
    float *d_feat = hip_backend_get_buf(b, 0, feat_bytes);
    if (!d_feat) return TSAC_ERR_MEMORY;
    hipMemsetAsync(d_feat, 0, feat_bytes, s);

    /* Use rvq_lookup_simple_k for each codebook */
    for (int cb = 0; cb < n_cb && cb < 12; cb++) {
        char cb_name[160];
        snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
        DACTensor *cb_t = F(ts, nt, cb_name);
        if (!cb_t) continue;

        int entries = cb_t->dims[0];
        int dim = cb_t->dims[1];

        float *d_cb = NULL;
        if (cb_t->dev_f32) {
            d_cb = cb_t->dev_f32;
        } else {
            hipMalloc(&cb_t->dev_f32, cb_t->data_size);
            hipMemcpyAsync(cb_t->dev_f32, cb_t->data, cb_t->data_size, hipMemcpyHostToDevice, s);
            d_cb = cb_t->dev_f32;
        }

        rvq_lookup_simple_k<<<(n_frames + BLK - 1) / BLK, BLK, 0, s>>>(
            d_feat, b->d_codes, d_cb, cb, entries, n_frames, n_cb, dim);
    }
    hipStreamSynchronize(s);

    /* model.0: Conv1d(1024->1536, K=7) */
    GpuWeight *w0 = hip_find_weight(b, "decoder.model.0");
    int T0 = n_frames;
    int C0 = w0 ? w0->Co : 1536;
    size_t b0_bytes = (size_t)C0 * T0 * sizeof(float);
    float *d_buf0 = hip_backend_get_buf(b, 1, b0_bytes);
    hipMemsetAsync(d_buf0, 0, b0_bytes, s);
    if (w0) {
        GpuWeight *b0_bias = hip_find_weight(b, "decoder.model.0.bias");
        conv1d_k<<<dim3(C0, (T0+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_buf0, d_feat, w0->d_data, b0_bias ? b0_bias->d_data : NULL,
            T0, w0->K, w0->Ci, w0->Co, 1);
    }
    hipStreamSynchronize(s);

    float *d_cur = d_buf0;
    int cur_C = C0;
    int cur_T = T0;

    /* Blocks 1-4 */
    int c_out[4] = {768, 384, 192, 96};
    int convt_K[4] = {16, 16, 8, 4};

    for (int blk = 1; blk <= 4; blk++) {
        char wname[160], sname[160], bname[160];
        snprintf(wname, sizeof(wname), "decoder.model.%d.block.1", blk);
        snprintf(sname, sizeof(sname), "decoder.model.%d.block.0.alpha", blk);
        snprintf(bname, sizeof(bname), "decoder.model.%d.block.1.bias", blk);

        /* Snake before conv */
        GpuWeight *gs = hip_find_weight(b, sname);
        if (gs) {
            snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                d_cur, d_cur, gs->d_data, cur_C * cur_T, gs->Ci);
        }

        /* ConvTranspose1d - 2x upsampling */
        GpuWeight *gw = hip_find_weight(b, wname);
        int next_C = c_out[blk - 1];
        int next_T = cur_T * 2;
        int K = convt_K[blk - 1];

        size_t nb = (size_t)next_C * next_T * sizeof(float);
        float *d_next = hip_backend_get_buf(b, blk + 1, nb);
        hipMemsetAsync(d_next, 0, nb, s);

        if (gw) {
            /* Use convt_k with grid config: dim3(Ci, (Ti+BLK-1)/BLK) */
            convt_k<<<dim3(gw->Ci, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_next, d_cur, gw->d_data, cur_T, next_T, K, gw->Ci, gw->Co, 2);

            /* Add bias */
            GpuWeight *gbias = hip_find_weight(b, bname);
            if (gbias) {
                add_bias_k<<<dim3(next_C, (next_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_next, gbias->d_data, next_T, next_C);
            }
        }

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner blocks (3x per residual block) */
        for (int inner = 2; inner <= 4; inner++) {
            char iname[160], isname[160], ibname[160];
            snprintf(iname, sizeof(iname), "decoder.model.%d.block.%d.block.1", blk, inner);
            snprintf(isname, sizeof(isname), "decoder.model.%d.block.%d.block.0.alpha", blk, inner);
            snprintf(ibname, sizeof(ibname), "decoder.model.%d.block.%d.block.1.bias", blk, inner);

            GpuWeight *gis = hip_find_weight(b, isname);
            if (gis) {
                snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                    d_cur, d_cur, gis->d_data, cur_C * cur_T, gis->Ci);
            }

            /* Use temp buffer for conv1d */
            GpuWeight *giw = hip_find_weight(b, iname);
            GpuWeight *gbias = hip_find_weight(b, ibname);
            if (giw) {
                float *d_tmp = hip_backend_get_buf(b, 6, (size_t)cur_C * cur_T * sizeof(float));
                hipMemsetAsync(d_tmp, 0, (size_t)cur_C * cur_T * sizeof(float), s);
                conv1d_k<<<dim3(cur_C, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_tmp, d_cur, giw->d_data, gbias ? gbias->d_data : NULL,
                    cur_T, giw->K, giw->Ci, giw->Co, 1);
                d_cur = d_tmp;
            }
        }
        hipStreamSynchronize(s);
    }

    /* model.5: Snake(96) */
    GpuWeight *gm5 = hip_find_weight(b, "decoder.model.5.alpha");
    if (gm5) {
        snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
            d_cur, d_cur, gm5->d_data, cur_C * cur_T, gm5->Ci);
    }
    hipStreamSynchronize(s);

    /* model.6: Conv1d(96->2, K=7) */
    GpuWeight *gm6 = hip_find_weight(b, "decoder.model.6");
    if (gm6) {
        int out_C = gm6->Co;
        int out_T = cur_T;
        size_t ob = (size_t)out_C * out_T * sizeof(float);
        float *d_out = hip_backend_get_buf(b, 7, ob);
        hipMemsetAsync(d_out, 0, ob, s);
        GpuWeight *b6 = hip_find_weight(b, "decoder.model.6.bias");
        conv1d_k<<<dim3(out_C, (out_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_out, d_cur, gm6->d_data, b6 ? b6->d_data : NULL,
            out_T, gm6->K, gm6->Ci, gm6->Co, 1);
        tanh_clip_k<<<(out_C * out_T + BLK - 1) / BLK, BLK, 0, s>>>(d_out, out_C * out_T);

        /* Copy to host PCM */
        int to_copy = out_T;
        if (to_copy > n_samples) to_copy = n_samples;
        if (to_copy > 0) {
            hipMemcpyAsync(pcm, d_out, (size_t)to_copy * sizeof(float), hipMemcpyDeviceToHost, s);
        }
    }

    hipStreamSynchronize(s);
    hipFree(b->d_codes);
    b->d_codes = NULL;

    fprintf(stderr, "[hip] completed: output samples=%d\n", cur_T);
    return TSAC_OK;
}

/* ================================================================ */
/*  Encoder run function (mirrors CUDA encoder)                      */
/* ================================================================ */

extern "C" int dac_encoder_run(DACTensor *ts, int nt,
    const float *pcm, int n_samples, int channels,
    int n_codebooks, int *codes, int *n_frames)
{
    fprintf(stderr, "[hip] dac_encoder_run: nt=%d samples=%d ch=%d cb=%d\n",
            nt, n_samples, channels, n_codebooks);

    /* Static backend storage for weight caching across calls */
    static HipBackend *b = NULL;
    if (!b) {
        b = (HipBackend *)calloc(1, sizeof(HipBackend));
        if (!b) return TSAC_ERR_MEMORY;
        hipSetDevice(0);
        hipStreamCreate(&b->stream);
        b->initialized = 1;
    }

    hipStream_t s = b->stream;

    /* Lazy weight upload on first call */
    if (!b->weights_uploaded) {
        int ret = hip_upload_weights(b, ts, nt);
        if (ret != TSAC_OK) return ret;
    }

    if (n_samples < 1) return TSAC_OK;

    /* Calculate frame count */
    int block_len = 512;  /* DAC default block length */
    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;
    *n_frames = nf;

    /* Upload PCM to GPU */
    size_t pcm_bytes = (size_t)n_samples * channels * sizeof(float);
    float *d_pcm;
    hipMalloc(&d_pcm, pcm_bytes);
    hipMemcpy(d_pcm, pcm, pcm_bytes, hipMemcpyHostToDevice);

    /* encoder.model.6: Conv1d(channels->96, K=7) */
    GpuWeight *e6 = hip_find_enc_weight(b, "encoder.model.6");
    float *d_cur = hip_backend_get_buf(b, 0, 96 * nf * sizeof(float));
    if (!d_cur) { hipFree(d_pcm); return TSAC_ERR_MEMORY; }
    hipMemsetAsync(d_cur, 0, 96 * nf * sizeof(float), s);
    if (e6) {
        GpuWeight *b6 = hip_find_enc_weight(b, "encoder.model.6.bias");
        conv1d_k<<<dim3(96, (nf+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_cur, d_pcm, e6->d_data, b6 ? b6->d_data : NULL,
            nf, e6->K, e6->Ci, e6->Co, 1);
    }
    hipFree(d_pcm);

    /* encoder.model.5: Snake(96) */
    GpuWeight *e5 = hip_find_enc_weight(b, "encoder.model.5.alpha");
    if (e5) {
        snake_k<<<(96 * nf + BLK - 1) / BLK, BLK, 0, s>>>(
            d_cur, d_cur, e5->d_data, 96 * nf, 96);
    }

    /* Blocks 4->3->2->1 (reverse order: 96->192->384->768->1536) */
    int enc_c_out[4] = {192, 384, 768, 1536};
    int cur_C = 96;
    int cur_T = nf;

    for (int blk = 4; blk >= 1; blk--) {
        int idx = 4 - blk;

        /* Snake before conv */
        char sname[128];
        snprintf(sname, sizeof(sname), "encoder.model.%d.block.0.alpha", blk);
        GpuWeight *sa = hip_find_enc_weight(b, sname);
        if (sa) {
            snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                d_cur, d_cur, sa->d_data, cur_C * cur_T, sa->Ci);
        }

        /* Conv1d (no downsampling in DAC encoder, stride=1) */
        char wname[128];
        snprintf(wname, sizeof(wname), "encoder.model.%d.block.1", blk);
        GpuWeight *cw = hip_find_enc_weight(b, wname);
        int next_C = enc_c_out[idx];
        int next_T = cur_T;

        float *d_next = hip_backend_get_buf(b, idx + 1, next_C * next_T * sizeof(float));
        if (!d_next) return TSAC_ERR_MEMORY;
        hipMemsetAsync(d_next, 0, next_C * next_T * sizeof(float), s);

        if (cw) {
            conv1d_k<<<dim3(next_C, (next_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_next, d_cur, cw->d_data, NULL,
                next_T, cw->K, cw->Ci, cw->Co, 1);
            char bname[128];
            snprintf(bname, sizeof(bname), "encoder.model.%d.block.1.bias", blk);
            GpuWeight *cb = hip_find_enc_weight(b, bname);
            if (cb) {
                add_bias_k<<<dim3(next_T, (next_C+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_next, cb->d_data, next_T, next_C);
            }
        }

        d_cur = d_next;
        cur_C = next_C;
        cur_T = next_T;

        /* Inner blocks (snake->conv1d K=7->snake->conv1d K=1) x 3 */
        for (int inner = 2; inner <= 4; inner++) {
            /* Snake */
            char isname[160];
            snprintf(isname, sizeof(isname), "encoder.model.%d.block.%d.block.0.alpha", blk, inner);
            GpuWeight *gis = hip_find_enc_weight(b, isname);
            if (gis) {
                snake_k<<<(cur_C * cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
                    d_cur, d_cur, gis->d_data, cur_C * cur_T, gis->Ci);
            }

            /* Conv1d K=7 */
            char iwname[160], ibname[160];
            snprintf(iwname, sizeof(iwname), "encoder.model.%d.block.%d.block.1", blk, inner);
            snprintf(ibname, sizeof(ibname), "encoder.model.%d.block.%d.block.1.bias", blk, inner);
            GpuWeight *giw = hip_find_enc_weight(b, iwname);
            GpuWeight *gib = hip_find_enc_weight(b, ibname);
            if (giw) {
                float *d_tmp = hip_backend_get_buf(b, 6, cur_C * cur_T * sizeof(float));
                if (!d_tmp) return TSAC_ERR_MEMORY;
                hipMemsetAsync(d_tmp, 0, cur_C * cur_T * sizeof(float), s);
                conv1d_k<<<dim3(cur_C, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                    d_tmp, d_cur, giw->d_data, NULL,
                    cur_T, giw->K, giw->Ci, giw->Co, 1);
                if (gib) {
                    add_bias_k<<<dim3(cur_T, (cur_C+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                        d_tmp, gib->d_data, cur_T, cur_C);
                }
                d_cur = d_tmp;
            }
        }
    }

    /* encoder.model.0: Conv1d(1536->1024, K=7) */
    GpuWeight *e0 = hip_find_enc_weight(b, "encoder.model.0");
    float *d_features = hip_backend_get_buf(b, 5, 1024 * cur_T * sizeof(float));
    if (!d_features) return TSAC_ERR_MEMORY;
    hipMemsetAsync(d_features, 0, 1024 * cur_T * sizeof(float), s);
    if (e0) {
        conv1d_k<<<dim3(1024, (cur_T+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
            d_features, d_cur, e0->d_data, NULL,
            cur_T, e0->K, e0->Ci, e0->Co, 1);
        GpuWeight *b0 = hip_find_enc_weight(b, "encoder.model.0.bias");
        if (b0) {
            add_bias_k<<<dim3(cur_T, (1024+BLK-1)/BLK), dim3(1, BLK), 0, s>>>(
                d_features, b0->d_data, cur_T, 1024);
        }
    }

    hipStreamSynchronize(s);

    /* RVQ quantization */
    float *d_residual;
    hipMalloc(&d_residual, 1024 * cur_T * sizeof(float));
    hipMemcpy(d_residual, d_features, 1024 * cur_T * sizeof(float), hipMemcpyDeviceToDevice);

    float *d_indices;
    hipMalloc(&d_indices, cur_T * sizeof(float));

    for (int cb = 0; cb < n_codebooks && cb < b->n_cb; cb++) {
        /* Quantize against codebook cb */
        rvq_quantize_k<<<(cur_T + BLK - 1) / BLK, BLK, 0, s>>>(
            d_indices, d_residual,
            b->d_cb_data + b->cb_offsets[cb],
            cur_T, 1024, b->cb_entries);
        hipGetLastError();

        /* Copy indices to host */
        float *h_indices_f = (float *)malloc(cur_T * sizeof(float));
        hipMemcpy(h_indices_f, d_indices, cur_T * sizeof(float), hipMemcpyDeviceToHost);
        for (int i = 0; i < cur_T; i++) {
            codes[cb * cur_T + i] = (int)h_indices_f[i];
        }
        free(h_indices_f);

        /* Subtract codebook entry */
        rvq_subtract_k<<<(cur_T * 1024 + BLK - 1) / BLK, BLK, 0, s>>>(
            d_residual, b->d_cb_data + b->cb_offsets[cb],
            d_indices, cur_T, 1024);
        hipGetLastError();
    }

    hipFree(d_residual);
    hipFree(d_indices);

    hipStreamSynchronize(s);

    fprintf(stderr, "[hip] encoder completed: frames=%d\n", cur_T);
    return TSAC_OK;
}
