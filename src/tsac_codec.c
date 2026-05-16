#include "tsac.h"
#include "tsac_codec.h"
#include "dac_model.h"
#include "txc_format.h"
#include "model_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
        snprintf(dac_path, sizeof(dac_path),
                 "%s/dac_stereo_q8.bin", model_path);
        /* Try to load model; failure is non-fatal, we fall back to tsac binary */
        int ret = model_loader_load(dac_path, ctx->model);
        if (ret != TSAC_OK) {
            fprintf(stderr, "[tsac-ng] Warning: model load failed (%d), "
                    "falling back to original tsac binary\n", ret);
        }
    }

    ctx->initialized = 1;
    return ctx;
}

void tsac_free(TSACContext *ctx)
{
    if (!ctx) return;

    if (ctx->backend == TSAC_BACKEND_CUDA)
        tsac_cuda_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_HIP)
        tsac_hip_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_VULKAN)
        tsac_vk_shutdown(ctx->backend_priv);
    else if (ctx->backend == TSAC_BACKEND_LLVM)
        tsac_llvm_shutdown(ctx->backend_priv);

    dac_model_destroy(ctx->model);
    free(ctx->model_path);
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}

int tsac_compress(TSACContext *ctx,
                  const float *pcm, int n_samples, int channels,
                  uint8_t **out_data, size_t *out_size,
                  int n_codebooks)
{
    if (!ctx || !ctx->initialized || !pcm || !out_data || !out_size)
        return TSAC_ERR_PARAM;

    if (n_codebooks < 1 || n_codebooks > 12)
        return TSAC_ERR_PARAM;

    if (channels != 1 && channels != 2)
        return TSAC_ERR_PARAM;

    if (n_samples < 1)
        return TSAC_ERR_PARAM;

    TSCHeader hdr;
    txc_header_init(&hdr, channels == 2 ? 1 : 0, n_codebooks, 44100);

    hdr.block_len  = 320;
    hdr.n_blocks   = (n_samples + hdr.block_len - 1) / hdr.block_len;

    int total_samples = hdr.n_blocks * hdr.block_len;
    int n_frames = hdr.n_blocks;

    int *codebook_indices = NULL;

    if (ctx->backend == TSAC_BACKEND_CUDA) {
        int ret = tsac_cuda_encode(ctx->backend_priv, ctx->model,
                                   pcm, n_samples, channels,
                                   n_codebooks, hdr.block_len,
                                   &codebook_indices, &n_frames);
        if (ret != TSAC_OK) return ret;
    } else if (ctx->backend == TSAC_BACKEND_HIP) {
        int ret = tsac_hip_encode(ctx->backend_priv, ctx->model,
                                  pcm, n_samples, channels,
                                  n_codebooks, hdr.block_len,
                                  &codebook_indices, &n_frames);
        if (ret != TSAC_OK) return ret;
    } else {
        /* CPU encode not yet implemented - fall through to compress_file fallback */
        return TSAC_ERR_BACKEND;
    }

    int ret = txc_write(&hdr, codebook_indices, n_frames,
                        out_data, out_size);
    free(codebook_indices);

    return ret;
}

int tsac_decompress(TSACContext *ctx,
                    const uint8_t *txc_data, size_t txc_size,
                    float **out_pcm, int *out_samples, int *out_channels)
{
    if (!ctx || !ctx->initialized || !txc_data || !out_pcm || !out_samples || !out_channels)
        return TSAC_ERR_PARAM;

    TSCHeader hdr;
    int *codebook_indices = NULL;
    int n_frames = 0;

    int ret = txc_read(txc_data, txc_size, &hdr, &codebook_indices, &n_frames);
    if (ret != TSAC_OK) return ret;

    *out_channels = hdr.flags ? 2 : 1;
    *out_samples  = hdr.n_blocks * hdr.block_len;

    size_t total_samples = (size_t)(*out_samples) * (*out_channels);
    float *pcm = (float *)calloc(total_samples, sizeof(float));
    if (!pcm) {
        free(codebook_indices);
        return TSAC_ERR_MEMORY;
    }

    if (ctx->backend == TSAC_BACKEND_CUDA) {
        ret = tsac_cuda_decode(ctx->backend_priv, ctx->model,
                               codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_HIP) {
        ret = tsac_hip_decode(ctx->backend_priv, ctx->model,
                               codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_VULKAN) {
        ret = tsac_vk_decode(ctx->backend_priv, ctx->model,
                              codebook_indices, n_frames,
                              hdr.n_codebooks, hdr.block_len,
                              *out_channels,
                              pcm, *out_samples);
    } else if (ctx->backend == TSAC_BACKEND_LLVM) {
        ret = tsac_llvm_decode(ctx->backend_priv, ctx->model,
                                codebook_indices, n_frames,
                                hdr.n_codebooks, hdr.block_len,
                                *out_channels,
                                pcm, *out_samples);
    } else {
        ret = dac_model_decode(ctx->model, codebook_indices, n_frames,
                               hdr.n_codebooks, hdr.block_len,
                               *out_channels,
                               pcm, *out_samples,
                               ctx->n_threads);
    }

    if (ret != TSAC_OK) {
        free(pcm);
        free(codebook_indices);
        return ret;
    }

    free(codebook_indices);
    *out_pcm = pcm;
    return TSAC_OK;
}

int tsac_compress_file(TSACContext *ctx, const char *in_wav, const char *out_txc, int n_codebooks)
{
    if (!ctx || !in_wav || !out_txc)
        return TSAC_ERR_PARAM;

    FILE *f = fopen(in_wav, "rb");
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

    if (audio_fmt != 1) { /* PCM */
        fclose(f);
        return TSAC_ERR_FORMAT;
    }

    /* Find data chunk */
    fseek(f, 44, SEEK_SET);
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
        return TSAC_OK;
    }

    return ret;
}

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
    float *pcm = NULL;
    int out_samples = 0;
    int out_channels = 0;
    ret = tsac_decompress(ctx, txc_data, (size_t)file_size, &pcm, &out_samples, &out_channels);
    free(txc_data);
    txc_data = NULL;

    if (ret != TSAC_OK) {
        return ret;
    }

    /* Write WAV file with float PCM data (format 3 = IEEE float) */
    FILE *fo = fopen(out_wav, "wb");
    if (!fo) {
        free(pcm);
        return TSAC_ERR_FILE;
    }

    uint32_t pcm_data_size = (uint32_t)(out_samples * out_channels * sizeof(float));
    uint32_t riff_size = 36 + pcm_data_size;

    uint8_t wav_hdr[44];
    /* RIFF chunk */
    memcpy(wav_hdr + 0, "RIFF", 4);
    wav_hdr[4] = (uint8_t)(riff_size);
    wav_hdr[5] = (uint8_t)(riff_size >> 8);
    wav_hdr[6] = (uint8_t)(riff_size >> 16);
    wav_hdr[7] = (uint8_t)(riff_size >> 24);
    memcpy(wav_hdr + 8, "WAVE", 4);
    /* fmt chunk */
    memcpy(wav_hdr + 12, "fmt ", 4);
    wav_hdr[16] = 16; /* fmt chunk size (little-endian) */
    wav_hdr[17] = 0;
    wav_hdr[18] = 0;
    wav_hdr[19] = 0;
    wav_hdr[20] = 3; /* format = IEEE float */
    wav_hdr[21] = 0;
    wav_hdr[22] = (uint8_t)out_channels;
    wav_hdr[23] = 0;
    wav_hdr[24] = (uint8_t)(sample_rate);
    wav_hdr[25] = (uint8_t)(sample_rate >> 8);
    wav_hdr[26] = (uint8_t)(sample_rate >> 16);
    wav_hdr[27] = (uint8_t)(sample_rate >> 24);
    uint32_t byte_rate = sample_rate * out_channels * sizeof(float);
    wav_hdr[28] = (uint8_t)(byte_rate);
    wav_hdr[29] = (uint8_t)(byte_rate >> 8);
    wav_hdr[30] = (uint8_t)(byte_rate >> 16);
    wav_hdr[31] = (uint8_t)(byte_rate >> 24);
    uint16_t block_align = out_channels * sizeof(float);
    wav_hdr[32] = (uint8_t)(block_align);
    wav_hdr[33] = (uint8_t)(block_align >> 8);
    wav_hdr[34] = 32; /* bits per sample */
    wav_hdr[35] = 0;
    /* data chunk */
    memcpy(wav_hdr + 36, "data", 4);
    wav_hdr[40] = (uint8_t)(pcm_data_size);
    wav_hdr[41] = (uint8_t)(pcm_data_size >> 8);
    wav_hdr[42] = (uint8_t)(pcm_data_size >> 16);
    wav_hdr[43] = (uint8_t)(pcm_data_size >> 24);

    if (fwrite(wav_hdr, 1, 44, fo) != 44) {
        fclose(fo);
        free(pcm);
        return TSAC_ERR_FILE;
    }

    /* Write interleaved float PCM samples */
    for (int i = 0; i < out_samples * out_channels; i++) {
        uint32_t bits;
        memcpy(&bits, &pcm[i], 4);
        uint8_t b[4];
        b[0] = (uint8_t)(bits);
        b[1] = (uint8_t)(bits >> 8);
        b[2] = (uint8_t)(bits >> 16);
        b[3] = (uint8_t)(bits >> 24);
        if (fwrite(b, 1, 4, fo) != 4) {
            fclose(fo);
            free(pcm);
            return TSAC_ERR_FILE;
        }
    }

    fclose(fo);
    free(pcm);
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
