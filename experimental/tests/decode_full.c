/*
 * decode_full.c — 完整解码 .txc 文件并生成 WAV，用于与原版 tsac 对比
 * 支持调试模式：导出每层中间结果
 */
#include "src/model_loader.h"
#include "src/dac_model.h"
#include "src/txc_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern void conv1d_s(float*, const float*, const float*, const float*, int, int, int, int);
extern void convt1d_s(float*, const float*, const float*, int, int, int, int, int, int);

static DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

static int count_nan(const float *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) if (isnan(data[i])) count++;
    return count;
}

static void print_stats(const char *name, const float *data, int n) {
    float mn = 1e30, mx = -1e30, sum = 0;
    int nan_count = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(data[i])) nan_count++;
        else {
            if (data[i] < mn) mn = data[i];
            if (data[i] > mx) mx = data[i];
            sum += data[i];
        }
    }
    fprintf(stderr, "%-30s: min=%12.4f max=%12.4f avg=%12.4f nan=%d/%d\n",
            name, mn, mx, sum/(n-nan_count), nan_count, n);
}

static float *dequant_weights(const DACTensor *weight_v, const DACTensor *weight_g,
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
    int is_ct = (K != 7) ? 1 : 0;
    int Ci = is_ct ? d2 : d0;

    if (is_conv_transpose) *is_conv_transpose = is_ct;
    *out_Ci = Ci; *out_K = K; *out_Co = Co;

    int total_size = Ci * K * Co;
    float *w_f32 = (float *)malloc(total_size * sizeof(float));
    if (!w_f32) return NULL;

    if (weight_v->elem_size == 4) {
        const float *src = (const float *)weight_v->data;
        int per_input = weight_g ? (Ci == (int)weight_g->dims[2]) : 0;
        for (int ci = 0; ci < Ci; ci++)
            for (int k = 0; k < K; k++)
                for (int co = 0; co < Co; co++) {
                    int src_idx = is_ct
                        ? (per_input ? co * K * Ci + k * Ci + ci : co * K * Ci + k * Ci + ci)
                        : (per_input ? co * K * Ci + k * Ci + ci : ci * K * Co + k * Co + co);
                    int dst_idx = co * Ci * K + ci * K + k;
                    w_f32[dst_idx] = src[src_idx];
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
                int src_idx = is_ct
                    ? (per_input ? co * K * Ci + k * Ci + ci : co * K * Ci + k * Ci + ci)
                    : (per_input ? co * K * Ci + k * Ci + ci : ci * K * Co + k * Co + co);
                uint8_t v_val = v_data[src_idx];
                if (!per_input) g = g_scales[co];
                int dst_idx = co * Ci * K + ci * K + k;
                w_f32[dst_idx] = g * ((float)v_val - 128.0f) / 127.0f;
            }
        }
    }
    return w_f32;
}

static void apply_snake(float *data, int C, int T, const float *alpha) {
    for (int i = 0; i < C * T; i++) {
        float v = data[i], al = alpha[i % C];
        if (al < 1e-6f) al = 1e-6f;
        float sa = sinf(al * v);
        data[i] = v + sa * sa / al;
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "用法: %s <model.bin> <input.txc> <output.wav> [max_frames]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *txc_path = argv[2];
    const char *output_path = argv[3];
    int max_frames = (argc > 4) ? atoi(argv[4]) : 0;

    /* Load model */
    DACModel model = {0};
    int ret = model_loader_load(model_path, &model);
    if (ret != 0) { fprintf(stderr, "模型加载失败: %d\n", ret); return 1; }
    fprintf(stderr, "加载 %d 个张量\n", model.n_tensors);

    /* Read TXC */
    FILE *f = fopen(txc_path, "rb");
    if (!f) { fprintf(stderr, "无法打开 TXC\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *txc_data = (uint8_t *)malloc(fsize);
    fread(txc_data, 1, fsize, f);
    fclose(f);

    TSCHeader hdr;
    int *codes = NULL;
    int n_frames = 0;
    ret = txc_read(txc_data, fsize, &hdr, &codes, &n_frames);
    if (ret != 0) { fprintf(stderr, "TXC 解析失败: %d\n", ret); return 1; }
    fprintf(stderr, "TXC: %d 帧, %d codebooks, %s\n",
            n_frames, hdr.n_codebooks, (hdr.flags & 1) ? "立体声" : "单声道");

    int decode_frames = max_frames > 0 ? max_frames : n_frames;
    if (decode_frames > n_frames) decode_frames = n_frames;

    /* Decode parameters */
    int n_cb = hdr.n_codebooks;
    int n_ch = (hdr.flags & 1) ? 2 : 1;
    int rvq_dim = 1024;
    int sample_rate = hdr.sample_rate;
    int stride = 512;  // DAC hop length

    /* Output buffer */
    int total_output_samples = decode_frames * stride;
    float *output_buffer = (float *)calloc(total_output_samples * n_ch, sizeof(float));

    /* Batch processing */
    int batch_frames = 5000;
    int context_pad = 10;

    for (int batch_start = 0; batch_start < decode_frames; batch_start += batch_frames) {
        int batch_end = batch_start + batch_frames;
        if (batch_end > decode_frames) batch_end = decode_frames;
        int batch_size = batch_end - batch_start;

        int ctx_start = batch_start > context_pad ? batch_start - context_pad : 0;
        int ctx_end = batch_end + context_pad < n_frames ? batch_end + context_pad : n_frames;
        int ctx_frames = ctx_end - ctx_start;

        fprintf(stderr, "\n处理批次 %d-%d (ctx: %d-%d)\n", batch_start, batch_end, ctx_start, ctx_end);

        /* RVQ lookup */
        float *rvq_out = (float *)calloc(rvq_dim * ctx_frames, sizeof(float));
        for (int cb = 0; cb < n_cb && cb < 12; cb++) {
            char cb_name[128];
            snprintf(cb_name, sizeof(cb_name), "quantizer.quantizers.%d.codebook.weight", cb);
            DACTensor *codebook = tf(model.tensors, model.n_tensors, cb_name);
            if (!codebook) continue;
            int entries = codebook->dims[0];
            int dim = codebook->dims[1];
            const float *cb_data = (const float *)codebook->data;
            for (int f = 0; f < ctx_frames; f++) {
                int code_idx = (ctx_start + f) * n_cb + cb;
                int entry = codes[code_idx];
                if (entry < 0 || entry >= entries) entry = entry % entries;
                if (entry < 0) entry = 0;
                for (int d = 0; d < dim && d < rvq_dim; d++) {
                    rvq_out[d * ctx_frames + f] += cb_data[entry * dim + d];
                }
            }
        }
        print_stats("RVQ", rvq_out, rvq_dim * ctx_frames);

        /* model.0 conv1d: 1024 -> 1536 */
        DACTensor *m0_wv = tf(model.tensors, model.n_tensors, "decoder.model.0.weight_v");
        DACTensor *m0_wg = tf(model.tensors, model.n_tensors, "decoder.model.0.weight_g");
        DACTensor *m0_b  = tf(model.tensors, model.n_tensors, "decoder.model.0.bias");
        int m0_Ci=1024, m0_K=7, m0_Co=1536;
        float *m0_w = dequant_weights(m0_wv, m0_wg, m0_b, &m0_Ci, &m0_K, &m0_Co, NULL);
        const float *m0_b_data = m0_b ? (const float *)m0_b->data : NULL;

        float *buf0 = (float *)calloc(m0_Co * ctx_frames, sizeof(float));
        if (m0_w) {
            conv1d_s(buf0, rvq_out, m0_w, m0_b_data, ctx_frames, m0_K, m0_Ci, m0_Co);
        }
        print_stats("m0 conv1d", buf0, m0_Co * ctx_frames);
        free(m0_w);
        free(rvq_out);

        /* Process blocks 1-4 */
        float *current = buf0;
        int current_C = m0_Co;
        int cur_frames = ctx_frames;

        for (int block = 1; block <= 4; block++) {
            int target_C = (block==1)?768:(block==2)?384:(block==3)?192:96;

            /* snake */
            char snake_name[128];
            snprintf(snake_name, sizeof(snake_name), "decoder.model.%d.block.0.alpha", block);
            DACTensor *snake_alpha = tf(model.tensors, model.n_tensors, snake_name);
            if (snake_alpha) {
                apply_snake(current, current_C, cur_frames, (const float *)snake_alpha->data);
            }
            char stage_name[64];
            snprintf(stage_name, sizeof(stage_name), "block%d snake", block);
            print_stats(stage_name, current, current_C * cur_frames);

            /* conv_transpose */
            char wv_name[128], wg_name[128], b_name[128];
            snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.1.weight_v", block);
            snprintf(wg_name, sizeof(wg_name), "decoder.model.%d.block.1.weight_g", block);
            snprintf(b_name, sizeof(b_name), "decoder.model.%d.block.1.bias", block);

            DACTensor *wv = tf(model.tensors, model.n_tensors, wv_name);
            DACTensor *wg = tf(model.tensors, model.n_tensors, wg_name);
            DACTensor *b  = tf(model.tensors, model.n_tensors, b_name);

            int conv_Ci, conv_K, conv_Co, is_convt;
            float *w = dequant_weights(wv, wg, b, &conv_Ci, &conv_K, &conv_Co, &is_convt);
            const float *b_data = b ? (const float *)b->data : NULL;

            int conv_stride = conv_K / 2;
            int n_frames_out = cur_frames * conv_stride - conv_K / 2;

            fprintf(stderr, "  block%d convt: Ci=%d Co=%d K=%d stride=%d -> frames %d->%d\n",
                    block, conv_Ci, conv_Co, conv_K, conv_stride, cur_frames, n_frames_out);

            float *next_buf = (float *)calloc(target_C * n_frames_out, sizeof(float));
            if (w && is_convt) {
                convt1d_s(next_buf, current, w, cur_frames, n_frames_out,
                          conv_K, conv_Ci, conv_Co, conv_stride);
                if (b_data) {
                    for (int c = 0; c < conv_Co; c++)
                        for (int t = 0; t < n_frames_out; t++)
                            next_buf[c * n_frames_out + t] += b_data[c];
                }
            }
            snprintf(stage_name, sizeof(stage_name), "block%d convt", block);
            print_stats(stage_name, next_buf, target_C * n_frames_out);

            free(current);
            free(w);
            current = next_buf;
            current_C = target_C;
            cur_frames = n_frames_out;

            /* Inner residual blocks - currently skipped due to ISS-012 */
            for (int inner = 2; inner <= 4; inner++) {
                char inner_wv[128];
                snprintf(inner_wv, sizeof(inner_wv), "decoder.model.%d.block.%d.block.1.weight_v", block, inner);
                DACTensor *iwv = tf(model.tensors, model.n_tensors, inner_wv);
                if (iwv) {
                    /* TODO: Implement inner block processing once ISS-012 is fixed */
                }
            }
        }

        /* model.5 snake */
        DACTensor *m5_alpha = tf(model.tensors, model.n_tensors, "decoder.model.5.alpha");
        if (m5_alpha && current_C == 96) {
            apply_snake(current, current_C, cur_frames, (const float *)m5_alpha->data);
        }
        print_stats("m5 snake", current, current_C * cur_frames);

        /* model.6 output conv1d */
        DACTensor *m6_wv = tf(model.tensors, model.n_tensors, "decoder.model.6.weight_v");
        DACTensor *m6_wg = tf(model.tensors, model.n_tensors, "decoder.model.6.weight_g");
        DACTensor *m6_b  = tf(model.tensors, model.n_tensors, "decoder.model.6.bias");

        int m6_Ci, m6_K, m6_Co;
        float *m6_w = dequant_weights(m6_wv, m6_wg, m6_b, &m6_Ci, &m6_K, &m6_Co, NULL);
        const float *m6_b_data = m6_b ? (const float *)m6_b->data : NULL;

        fprintf(stderr, "m6: Ci=%d K=%d Co=%d cur_frames=%d\n", m6_Ci, m6_K, m6_Co, cur_frames);

        float *output = (float *)calloc(m6_Co * cur_frames, sizeof(float));
        if (m6_w && output) {
            conv1d_s(output, current, m6_w, m6_b_data, cur_frames, m6_K, m6_Ci, m6_Co);
        }
        print_stats("m6 output", output, m6_Co * cur_frames);

        /* Copy to output buffer (handle context padding) */
        int code_offset = ctx_start;
        int discard_samples = (batch_start - code_offset) * stride;
        int copy_samples = batch_size * stride;

        if (copy_samples > cur_frames * stride) copy_samples = cur_frames * stride;

        for (int ch = 0; ch < n_ch && ch < m6_Co; ch++) {
            for (int s = 0; s < copy_samples; s++) {
                int out_idx = (batch_start * stride + s) * n_ch + ch;
                int src_idx = ch * cur_frames + discard_samples / stride + s / stride;
                if (src_idx < cur_frames && out_idx < total_output_samples * n_ch) {
                    output_buffer[out_idx] = output[src_idx];
                }
            }
        }

        free(m6_w);
        free(output);
        free(current);
    }

    /* Write WAV */
    {
        int sr = sample_rate, bits = 16;
        int num_samples = total_output_samples;
        float max_abs = 0;
        for (int i = 0; i < num_samples * n_ch; i++) {
            float a = fabsf(output_buffer[i]);
            if (a > max_abs) max_abs = a;
        }
        float scale = (max_abs > 1.0f) ? 1.0f / max_abs : 1.0f;

        FILE *wav = fopen(output_path, "wb");
        if (wav) {
            int data_size = num_samples * n_ch * bits / 8;
            int chunk_size = 36 + data_size;
            fwrite("RIFF", 1, 4, wav);
            fwrite(&chunk_size, 4, 1, wav);
            fwrite("WAVE", 1, 4, wav);
            fwrite("fmt ", 1, 4, wav);
            int fmt_size = 16; fwrite(&fmt_size, 4, 1, wav);
            short af = 1; fwrite(&af, 2, 1, wav);
            fwrite(&n_ch, 2, 1, wav);
            fwrite(&sr, 4, 1, wav);
            int br = sr * n_ch * bits / 8; fwrite(&br, 4, 1, wav);
            short ba = n_ch * bits / 8; fwrite(&ba, 2, 1, wav);
            fwrite(&bits, 2, 1, wav);
            fwrite("data", 1, 4, wav);
            fwrite(&data_size, 4, 1, wav);
            for (int i = 0; i < num_samples; i++)
                for (int c = 0; c < n_ch; c++) {
                    float v = output_buffer[c * num_samples + i] * scale;
                    if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
                    short s = (short)(v * 32767.0f);
                    fwrite(&s, 2, 1, wav);
                }
            fclose(wav);
            fprintf(stderr, "\n写入 %s (%d samples, scale=%.6f, max_abs=%.2f)\n",
                    output_path, num_samples, scale, max_abs);
        }
    }

    free(output_buffer);
    free(codes);
    free(txc_data);
    model_loader_free(&model);

    return 0;
}
