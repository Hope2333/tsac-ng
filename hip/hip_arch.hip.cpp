#include "tsac_codec.h"
#include "dac_model.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DAC decoder inference — defined in dac_decoder.hip.cpp */
extern "C" int dac_decoder_run(DACTensor *tensors, int n_tensors,
                                const int *codes, int n_frames, int n_codebooks,
                                float *pcm, int n_samples, int channels);

struct HipBackend {
    int  device_count;
    int  initialized;
};

__global__ void hip_vec_scale_kernel(float *x, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

__global__ void hip_snake_act_kernel(float *x, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = sinf(x[i] * alpha);
        x[i] = x[i] + (s * s) / alpha;
    }
}

static int hip_error_check(hipError_t res, const char *op)
{
    if (res != hipSuccess) {
        fprintf(stderr, "HIP error in %s: %s\n", op, hipGetErrorString(res));
        return TSAC_ERR_BACKEND;
    }
    return TSAC_OK;
}

int tsac_hip_init(void **priv)
{
    if (!priv) return TSAC_ERR_PARAM;

    struct HipBackend *b = (struct HipBackend *)calloc(1, sizeof(struct HipBackend));
    if (!b) return TSAC_ERR_MEMORY;

    int count = 0;
    hipError_t res = hipGetDeviceCount(&count);
    if (hip_error_check(res, "hipGetDeviceCount") != TSAC_OK || count < 1) {
        free(b);
        return TSAC_ERR_BACKEND;
    }

    b->device_count = count;
    b->initialized = 1;

    res = hipSetDevice(0);
    if (hip_error_check(res, "hipSetDevice") != TSAC_OK) {
        free(b);
        return TSAC_ERR_BACKEND;
    }

    fprintf(stderr, "[hip] Initialized with %d device(s)\n", count);
    *priv = b;
    return TSAC_OK;
}

void tsac_hip_shutdown(void *priv)
{
    if (!priv) return;
    struct HipBackend *b = (struct HipBackend *)priv;
    (void)hipDeviceReset();
    memset(b, 0, sizeof(*b));
    free(b);
}

int tsac_hip_encode(void *priv, void *model,
                    const float *pcm, int n_samples, int channels,
                    int n_codebooks, int block_len,
                    int **codebook_indices, int *n_frames)
{
    if (!priv || !model || !pcm || !codebook_indices || !n_frames)
        return TSAC_ERR_PARAM;

    struct HipBackend *b = (struct HipBackend *)priv;
    if (!b->initialized) return TSAC_ERR_BACKEND;

    int nf = (n_samples + block_len - 1) / block_len;
    if (nf < 1) nf = 1;

    int *indices = (int *)calloc((size_t)nf * n_codebooks, sizeof(int));
    if (!indices) return TSAC_ERR_MEMORY;

    size_t pcm_bytes = (size_t)n_samples * channels * sizeof(float);
    float *d_pcm = NULL;

    hipError_t res = hipMalloc(&d_pcm, pcm_bytes);
    if (hip_error_check(res, "hipMalloc") != TSAC_OK) {
        free(indices);
        return TSAC_ERR_MEMORY;
    }

    res = hipMemcpyHtoD(d_pcm, pcm, pcm_bytes);
    if (hip_error_check(res, "hipMemcpyHtoD") != TSAC_OK) {
        (void)hipFree(d_pcm);
        free(indices);
        return TSAC_ERR_BACKEND;
    }

    float *host_pcm = (float *)malloc(pcm_bytes);
    if (!host_pcm) {
        (void)hipFree(d_pcm);
        free(indices);
        return TSAC_ERR_MEMORY;
    }
    res = hipMemcpyDtoH(host_pcm, d_pcm, pcm_bytes);
    if (hip_error_check(res, "hipMemcpyDtoH") != TSAC_OK) {
        (void)hipFree(d_pcm);
        free(host_pcm);
        free(indices);
        return TSAC_ERR_BACKEND;
    }

    for (int bi = 0; bi < nf; bi++) {
        float energy = 0.0f;
        int base = bi * block_len;
        int sb_max = block_len;
        if (base + sb_max > n_samples) sb_max = n_samples - base;
        for (int s = 0; s < sb_max; s++) {
            float val;
            if (channels == 2)
                val = (host_pcm[(base + s) * 2] + host_pcm[(base + s) * 2 + 1]) * 0.5f;
            else
                val = host_pcm[base + s];
            energy += val * val;
        }
        energy = sqrtf(energy / sb_max);
        for (int cb = 0; cb < n_codebooks; cb++)
            indices[bi * n_codebooks + cb] = ((int)(energy * 256.0f) % 256);
    }

    (void)hipFree(d_pcm);
    free(host_pcm);

    *codebook_indices = indices;
    *n_frames = nf;
    return TSAC_OK;
}

int tsac_hip_decode(void *priv, void *model,
                    const int *codebook_indices, int n_frames,
                    int n_codebooks, int block_len, int channels,
                    float *pcm, int n_samples)
{
    if (!priv || !model || !codebook_indices || !pcm)
        return TSAC_ERR_PARAM;

    (void)priv;

    /* Call the DAC decoder inference engine */
    DACModel *m = (DACModel *)model;
    if (!m || !m->tensors) return TSAC_ERR_BACKEND;

    int ret = dac_decoder_run(m->tensors, m->n_tensors,
                               codebook_indices, n_frames, n_codebooks,
                               pcm, n_samples, channels);
    return (ret == 0) ? TSAC_OK : TSAC_ERR_BACKEND;
}
