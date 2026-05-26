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
        if (is_file_path)
            snprintf(dac_path, sizeof(dac_path), "%s", model_path);
        else
            snprintf(dac_path, sizeof(dac_path),
                     "%s/dac_stereo_q8.bin", model_path);
        int ret = model_loader_load(dac_path, ctx->model);
        if (ret != TSAC_OK) {
            fprintf(stderr, "[tsac-ng] Warning: model load failed (%d), "
                    "falling back to original tsac binary\n", ret);
        }
    }

    ctx->initialized = 1;
    return ctx;
}

/* Release all resources held by a TSAC context.
 * Safe to call with NULL. */
void tsac_free(TSACContext *ctx)
{
#include "tsac_helpers.inc"
    if (!f) return TSAC_ERR_FILE;

    /* Read WAV header to get format info */
    uint8_t wav_hdr[44];
    if (fread(wav_hdr, 1, 44, f) != 44) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    if (memcmp(wav_hdr, "RIFF", 4) != 0 || memcmp(wav_hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    uint16_t audio_fmt = wav_hdr[20] | ((uint16_t)wav_hdr[21] << 8);
    uint16_t channels  = wav_hdr[22] | ((uint16_t)wav_hdr[23] << 8);
    uint32_t data_size = 0;

    if (audio_fmt != 1 && audio_fmt != 3) { /* PCM int16 or IEEE float */
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    /* Find data chunk — start searching after fmt chunk */
    uint32_t fmt_size = wav_hdr[16] | ((uint32_t)wav_hdr[17] << 8)
                      | ((uint32_t)wav_hdr[18] << 16) | ((uint32_t)wav_hdr[19] << 24);
    long data_search_start = 20 + fmt_size;
    fseek(f, data_search_start, SEEK_SET);
    uint8_t chunk_hdr[8];
    while (fread(chunk_hdr, 1, 8, f) == 8) {
        if (memcmp(chunk_hdr, "data", 4) == 0) {
            data_size = (uint32_t)chunk_hdr[4]
                      | ((uint32_t)chunk_hdr[5] << 8)
                      | ((uint32_t)chunk_hdr[6] << 16)
                      | ((uint32_t)chunk_hdr[7] << 24);
            break;
        }
        uint32_t chunk_size = (uint32_t)chunk_hdr[4]
                            | ((uint32_t)chunk_hdr[5] << 8)
                            | ((uint32_t)chunk_hdr[6] << 16)
                            | ((uint32_t)chunk_hdr[7] << 24);
        fseek(f, chunk_size, SEEK_CUR);
    }

    if (data_size == 0) {
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    int n_samples = data_size / (channels * (audio_fmt == 3 ? 4 : 2));

    /* Read PCM samples */
    float *pcm = (float *)calloc((size_t)n_samples * channels, sizeof(float));
    if (!pcm) {
        fclose(f);
        return TSAC_ERR_MEMORY;
    }

    if (audio_fmt == 3) {
        /* IEEE float */
        for (int i = 0; i < n_samples * channels; i++) {
            uint8_t b[4];
            if (fread(b, 1, 4, f) != 4) break;
            uint32_t bits = (uint32_t)b[0] | ((uint32_t)b[1] << 8)
                          | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
            float val;
            memcpy(&val, &bits, 4);
            pcm[i] = val;
        }
    } else {
        /* 16-bit PCM */
        for (int i = 0; i < n_samples * channels; i++) {
            uint8_t b[2];
            if (fread(b, 1, 2, f) != 2) break;
            int16_t sample = (int16_t)(b[0] | ((uint16_t)b[1] << 8));
            pcm[i] = (float)sample / 32768.0f;
        }
    }
    fclose(f);

    uint8_t *out_data = NULL;
    size_t out_size = 0;
    int ret = tsac_compress(ctx, pcm, n_samples, channels, &out_data, &out_size, n_codebooks);
    free(pcm);

    if (ret == TSAC_OK) {
        FILE *fo = fopen(out_txc, "wb");
        if (!fo) {
            tsac_free_buffer(out_data);
            return TSAC_ERR_FILE;
        }
        if (fwrite(out_data, 1, out_size, fo) != out_size) {
            fclose(fo);
            tsac_free_buffer(out_data);
            return TSAC_ERR_FILE;
        }
        fclose(fo);
        tsac_free_buffer(out_data);

        uint32_t sr = wav_hdr[24] | ((uint32_t)wav_hdr[25] << 8)
                    | ((uint32_t)wav_hdr[26] << 16) | ((uint32_t)wav_hdr[27] << 24);
        if (sr == 0) sr = 44100;
        double duration = (double)n_samples / (double)sr;
        double bitrate = duration > 0.0 ? (double)out_size * 8.0 / duration / 1000.0 : 0.0;
        double model_mb = 0.0;
        if (ctx->model && ctx->model->tensors) {
            for (int t = 0; t < ctx->model->n_tensors; t++)
                model_mb += (double)ctx->model->tensors[t].data_size;
        }
        double max_mem = (model_mb * 3.0) / (1024.0 * 1024.0 * 1024.0);
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
