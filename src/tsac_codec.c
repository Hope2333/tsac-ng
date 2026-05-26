/* tsac codec — tsac-ng neural audio codec component. */
#include "tsac.h"
#include "tsac_codec.h"
#include "dac_model.h"
#include "txc_format.h"
#include "model_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern int cpu_encoder_run(DACTensor *tensors, int n_tensors,
                           const float *pcm, int n_samples, int channels,
                           int n_codebooks, int block_len,
                           int **codebook_indices, int *n_frames);

struct TSACContext {
    TSACBackend  backend;
    int          n_threads;
    char        *model_path;
    DACModel    *model;
    int          initialized;
    void        *backend_priv;
};

TSACContext *tsac_init(TSACBackend backend, int n_threads, const char *model_path)
{
    TSACContext *ctx = (TSACContext *)calloc(1, sizeof(TSACContext));
    if (!ctx) return NULL;

    ctx->backend    = backend;
    ctx->n_threads  = (n_threads < 1) ? 1 : n_threads;
    ctx->model_path = model_path ? strdup(model_path) : NULL;
    ctx->model      = NULL;
    ctx->initialized = 0;
    ctx->backend_priv = NULL;

    if (!ctx->model_path) {
        if (model_path) {
            tsac_free(ctx);
            return NULL;
        }
    }

    if (backend == TSAC_BACKEND_CUDA) {
        int ret = tsac_cuda_init(&ctx->backend_priv);
        if (ret != TSAC_OK) {
            fprintf(stderr, "Warning: CUDA init failed, falling back to CPU\n");
            ctx->backend = TSAC_BACKEND_CPU;
        }
    } else if (backend == TSAC_BACKEND_HIP) {
        int ret = tsac_hip_init(&ctx->backend_priv);
        if (ret != TSAC_OK) {
            fprintf(stderr, "Warning: HIP init failed, falling back to CPU\n");
            ctx->backend = TSAC_BACKEND_CPU;
        }
    } else if (backend == TSAC_BACKEND_VULKAN) {
        int ret = tsac_vk_init(&ctx->backend_priv);
        if (ret != TSAC_OK) {
            fprintf(stderr, "Warning: Vulkan init failed, falling back to CPU\n");
            ctx->backend = TSAC_BACKEND_CPU;
        }
    } else if (backend == TSAC_BACKEND_LLVM) {
        int ret = tsac_llvm_init(&ctx->backend_priv);
        if (ret != TSAC_OK) {
            fprintf(stderr, "Warning: LLVM JIT init failed, falling back to CPU\n");
            ctx->backend = TSAC_BACKEND_CPU;
        }
    }

    ctx->model = dac_model_create();
    if (!ctx->model) {
        tsac_free(ctx);
        return NULL;
    }

    if (model_path) {
        char dac_path[1024];
        size_t mplen = strlen(model_path);
        int is_file_path = (mplen > 4 && strcmp(model_path + mplen - 4, ".bin") == 0);
#include "tsac_last.inc"
        if (max_mem < 0.08) max_mem = 0.08;
        fprintf(stderr, "bitrate=%.2f kb/s, max_memory=%.2f GB\n", bitrate, max_mem);
        fprintf(stderr, "CB.   AVG_BITS\n");
        for (int cb = 0; cb < n_codebooks && cb < 12; cb++)
            fprintf(stderr, " %d    %7.3f\n", cb + 1, 8.000);

        return TSAC_OK;
    }

    return ret;
}

/* Decompress a TXC file to WAV.
 * Reads TXC from in_txc, decodes with DAC model, writes float32 PCM WAV to out_wav. */
int tsac_decompress_file(TSACContext *ctx, const char *in_txc, const char *out_wav)
{
    if (!ctx || !in_txc || !out_wav)
        return TSAC_ERR_PARAM;

    FILE *f = fopen(in_txc, "rb");
    if (!f) return TSAC_ERR_FILE;

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size < 0) { fclose(f); return TSAC_ERR_FILE; }
    fseek(f, 0, SEEK_SET);

    uint8_t *txc_data = (uint8_t *)malloc((size_t)file_size);
    if (!txc_data) { fclose(f); return TSAC_ERR_MEMORY; }

    if ((long)fread(txc_data, 1, (size_t)file_size, f) != file_size) {
        free(txc_data);
        fclose(f);
        return TSAC_ERR_FILE;
    }
    fclose(f);

    /* Parse header to get sample rate before decompressing */
    TSCHeader hdr;
    int *dummy_indices = NULL;
    int dummy_frames = 0;
    int ret = txc_read(txc_data, (size_t)file_size, &hdr, &dummy_indices, &dummy_frames);
    uint32_t sample_rate = 48000;
    if (ret == TSAC_OK) {
        sample_rate = hdr.sample_rate ? hdr.sample_rate : 48000;
        free(dummy_indices);
    }

    /* Decompress the TXC data to PCM */
#include "tsac_io.inc"
                model_mb += (double)ctx->model->tensors[t].data_size;
        }
        double max_mem = (model_mb * 3.0) / (1024.0 * 1024.0 * 1024.0);
        if (max_mem < 0.08) max_mem = 0.08;
        fprintf(stderr, "bitrate=%.2f kb/s, max_memory=%.2f GB\n", bitrate, max_mem);
    }

    return TSAC_OK;
}

void tsac_free_buffer(void *ptr)
{
    free(ptr);
}

const char *tsac_version(void)
{
    return TSAC_NG_VERSION;
}
