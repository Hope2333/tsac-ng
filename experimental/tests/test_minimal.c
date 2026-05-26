/*
 * test_minimal.c — Minimal test: load model, dump tensor dims, decode 10 frames
 */
#include "src/model_loader.h"
#include "src/dac_model.h"
#include "src/txc_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Extern scalar kernels for direct testing */
extern void conv1d_s(float*, const float*, const float*, const float*, int, int, int, int);
extern void convt1d_s(float*, const float*, const float*, int, int, int, int, int, int);

/* Tensor finder */
static DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

static int count_nan(const float *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) if (isnan(data[i])) count++;
    return count;
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

int main(int argc, char **argv) {
    const char *model_path = "/usr/share/tsac/dac_stereo_q8.bin";
    const char *txc_path = "test-simples/P丸様。-自分後回し@A.txc";

    /* Load model */
    DACModel model = {0};
    int ret = model_loader_load(model_path, &model);
    if (ret != 0) { fprintf(stderr, "Model load failed: %d\n", ret); return 1; }
    fprintf(stderr, "Loaded %d tensors\n", model.n_tensors);

    /* Dump key tensor dimensions */
    const char *key_names[] = {
        "decoder.model.0.weight_v", "decoder.model.0.weight_g", "decoder.model.0.bias",
        "decoder.model.1.block.1.weight_v", "decoder.model.1.block.1.weight_g", "decoder.model.1.block.1.bias",
        "decoder.model.2.block.1.weight_v", "decoder.model.2.block.1.weight_g", "decoder.model.2.block.1.bias",
        "decoder.model.3.block.1.weight_v", "decoder.model.3.block.1.weight_g", "decoder.model.3.block.1.bias",
        "decoder.model.4.block.1.weight_v", "decoder.model.4.block.1.weight_g", "decoder.model.4.block.1.bias",
        "decoder.model.6.weight_v", "decoder.model.6.weight_g", "decoder.model.6.bias",
        "quantizer.quantizers.0.codebook.weight",
        NULL
    };
    for (int i = 0; key_names[i]; i++) {
        DACTensor *t = tf(model.tensors, model.n_tensors, key_names[i]);
        if (t) {
            fprintf(stderr, "  %s: dims=[%d,%d,%d,%d] ndims=%d elem_size=%d data_size=%d\n",
                    t->name, t->dims[0], t->dims[1], t->dims[2], t->dims[3],
                    t->ndims, t->elem_size, t->data_size);
        } else {
            fprintf(stderr, "  %s: NOT FOUND\n", key_names[i]);
        }
    }

    /* Read TXC */
    FILE *f = fopen(txc_path, "rb");
    if (!f) { fprintf(stderr, "TXC open failed\n"); return 1; }
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
    if (ret != 0) { fprintf(stderr, "TXC read failed: %d\n", ret); return 1; }
    fprintf(stderr, "TXC: %d frames, %d codebooks, stereo=%d, block_len=%d\n",
            n_frames, hdr.n_codebooks, hdr.flags & 1, hdr.block_len);

    /* Decode just 20 frames with context */
    int test_frames = 20;
    int ctx_pad = 10;
    int ctx_start = 0;
    int ctx_frames = test_frames + 2 * ctx_pad;
    int n_cb = hdr.n_codebooks;
    int ch = hdr.flags ? 2 : 1;
    int rvq_dim = 1024;

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
    fprintf(stderr, "RVQ: NaN=%d/%d\n", count_nan(rvq_out, rvq_dim*ctx_frames), rvq_dim*ctx_frames);
    /* Check RVQ stats */
    { float mn=1e30,mx=-1e30; for(int i=0;i<rvq_dim*ctx_frames;i++){if(rvq_out[i]<mn)mn=rvq_out[i];if(rvq_out[i]>mx)mx=rvq_out[i];}
      fprintf(stderr,"  RVQ range: [%.4f, %.4f]\n",mn,mx); }

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
    fprintf(stderr, "After m0 conv1d: Ci=%d Co=%d K=%d NaN=%d/%d\n",
            m0_Ci, m0_Co, m0_K, count_nan(buf0, m0_Co*ctx_frames), m0_Co*ctx_frames);

    /* Check weight stats */
    if (m0_w) {
        float w_min=1e30, w_max=-1e30, w_sum=0;
        int w_nan=0;
        int total = m0_Ci * m0_K * m0_Co;
        for (int i = 0; i < total; i++) {
            if (isnan(m0_w[i])) w_nan++;
            else { if (m0_w[i] < w_min) w_min = m0_w[i]; if (m0_w[i] > w_max) w_max = m0_w[i]; }
            w_sum += m0_w[i];
        }
        fprintf(stderr, "  m0 weights: min=%.4f max=%.4f avg=%.4f nan=%d/%d\n",
                w_min, w_max, w_sum/total, w_nan, total);
    }
    free(m0_w);
    free(rvq_out);

    /* Now run through blocks 1-4 quickly */
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
            const float *alpha = (const float *)snake_alpha->data;
            int alpha_C = snake_alpha->dims[0];
            for (int i = 0; i < current_C * cur_frames; i++) {
                float v = current[i], al = alpha[i % alpha_C];
                if (al < 1e-6f) al = 1e-6f;
                float sa = sinf(al * v);
                current[i] = v + sa * sa / al;
            }
        }
        fprintf(stderr, "After block%d snake: NaN=%d/%d\n", block, count_nan(current, current_C*cur_frames), current_C*cur_frames);

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

        fprintf(stderr, "  block%d convt: Ci=%d Co=%d K=%d stride=%d is_convt=%d\n",
                block, conv_Ci, conv_Co, conv_K, conv_K/2, is_convt);

        /* Check weight stats */
        if (w) {
            float w_min=1e30, w_max=-1e30; int w_nan=0;
            int total = conv_Ci * conv_K * conv_Co;
            for (int i = 0; i < total; i++) {
                if (isnan(w[i])) w_nan++;
                else { if (w[i] < w_min) w_min = w[i]; if (w[i] > w_max) w_max = w[i]; }
            }
            fprintf(stderr, "  weights: min=%.4f max=%.4f nan=%d/%d\n", w_min, w_max, w_nan, total);
        }

        int conv_stride = conv_K / 2;
        int n_frames_out = cur_frames * conv_stride - conv_K / 2;

        float *next_buf = (float *)calloc(target_C * n_frames_out, sizeof(float));
        if (w && is_convt) {
            /* Use scalar convt1d_s kernel (already has 1.0f/K normalization) */
            convt1d_s(next_buf, current, w, cur_frames, n_frames_out,
                      conv_K, conv_Ci, conv_Co, conv_stride);
            if (b_data) {
                for (int c = 0; c < conv_Co; c++)
                    for (int t = 0; t < n_frames_out; t++)
                        next_buf[c * n_frames_out + t] += b_data[c];
            }
        }

        fprintf(stderr, "After block%d convt: NaN=%d/%d\n", block, count_nan(next_buf, target_C*n_frames_out), target_C*n_frames_out);
        { float mn=1e30,mx=-1e30; for(int i=0;i<target_C*n_frames_out;i++){if(next_buf[i]<mn)mn=next_buf[i];if(next_buf[i]>mx)mx=next_buf[i];} fprintf(stderr,"  range: [%.4f, %.4f]\n",mn,mx); }

        free(current);
        free(w);
        current = next_buf;
        current_C = target_C;
        cur_frames = n_frames_out;

        /* Inner residual blocks - DISABLED: too slow, investigate separately */
        for (int inner = 2; inner <= 4; inner++) {
            char inner_wv[128];
            snprintf(inner_wv, sizeof(inner_wv), "decoder.model.%d.block.%d.block.1.weight_v", block, inner);
            DACTensor *iwv = tf(model.tensors, model.n_tensors, inner_wv);
            if (iwv) fprintf(stderr, "  (inner block %d exists but skipped)\n", inner);
        }
        fprintf(stderr, "After block%d inner: NaN=%d/%d\n", block, count_nan(current, current_C*cur_frames), current_C*cur_frames);
        { float mn=1e30,mx=-1e30; for(int i=0;i<current_C*cur_frames;i++){if(current[i]<mn)mn=current[i];if(current[i]>mx)mx=current[i];} fprintf(stderr,"  range: [%.4f, %.4f]\n",mn,mx); }
    }

    /* model.5 snake */
    DACTensor *m5_alpha = tf(model.tensors, model.n_tensors, "decoder.model.5.alpha");
    if (m5_alpha && current_C == 96) {
        const float *alpha = (const float *)m5_alpha->data;
        int alpha_C = m5_alpha->dims[0];
        for (int i = 0; i < current_C * cur_frames; i++) {
            float v = current[i], al = alpha[i % alpha_C];
            if (al < 1e-6f) al = 1e-6f;
            float sa = sinf(al * v);
            current[i] = v + sa * sa / al;
        }
    }
    fprintf(stderr, "After m5 snake: NaN=%d/%d\n", count_nan(current, current_C*cur_frames), current_C*cur_frames);

    /* model.6 output conv1d */
    DACTensor *m6_wv = tf(model.tensors, model.n_tensors, "decoder.model.6.weight_v");
    DACTensor *m6_wg = tf(model.tensors, model.n_tensors, "decoder.model.6.weight_g");
    DACTensor *m6_b  = tf(model.tensors, model.n_tensors, "decoder.model.6.bias");

    int m6_Ci, m6_K, m6_Co;
    float *m6_w = dequant_weights(m6_wv, m6_wg, m6_b, &m6_Ci, &m6_K, &m6_Co, NULL);
    const float *m6_b_data = m6_b ? (const float *)m6_b->data : NULL;

    fprintf(stderr, "m6: Ci=%d K=%d Co=%d cur_frames=%d\n", m6_Ci, m6_K, m6_Co, cur_frames);

    /* Check m6 weight stats */
    if (m6_w) {
        float w_min=1e30, w_max=-1e30, w_sum=0; int w_nan=0;
        int total = m6_Ci * m6_K * m6_Co;
        for (int i = 0; i < total; i++) {
            if (isnan(m6_w[i])) w_nan++;
            else { if (m6_w[i] < w_min) w_min = m6_w[i]; if (m6_w[i] > w_max) w_max = m6_w[i]; }
            w_sum += m6_w[i];
        }
        fprintf(stderr, "  m6 weights: min=%.4f max=%.4f avg=%.4f nan=%d/%d\n", w_min, w_max, w_sum/total, w_nan, total);

        /* Check input stats */
        float x_min=1e30, x_max=-1e30; int x_nan=0;
        int x_total = current_C * cur_frames;
        for (int i = 0; i < x_total; i++) {
            if (isnan(current[i])) x_nan++;
            else { if (current[i] < x_min) x_min = current[i]; if (current[i] > x_max) x_max = current[i]; }
        }
        fprintf(stderr, "  m6 input: min=%.4f max=%.4f nan=%d/%d\n", x_min, x_max, x_nan, x_total);
    }

    float *output = (float *)calloc(m6_Co * cur_frames, sizeof(float));
    if (m6_w && output) {
        conv1d_s(output, current, m6_w, m6_b_data, cur_frames, m6_K, m6_Ci, m6_Co);
        fprintf(stderr, "After m6 conv1d: NaN=%d/%d\n", count_nan(output, m6_Co*cur_frames), m6_Co*cur_frames);

        /* Check output stats (excluding NaN) */
        float o_min=1e30, o_max=-1e30; int o_nan=0;
        int o_total = m6_Co * cur_frames;
        for (int i = 0; i < o_total; i++) {
            if (isnan(output[i])) o_nan++;
            else { if (output[i] < o_min) o_min = output[i]; if (output[i] > o_max) o_max = output[i]; }
        }
        fprintf(stderr, "  m6 output: min=%.4f max=%.4f nan=%d/%d\n", o_min, o_max, o_nan, o_total);

        /* Write WAV output for comparison */
        {
            int sr = 44100, bits = 16, num_ch = 2;
            int num_samples = o_total / num_ch;
            float max_abs = fmaxf(-o_min, o_max);
            float scale = (max_abs > 1.0f) ? 1.0f / max_abs : 1.0f;
            FILE *wav = fopen("test_minimal_output.wav", "wb");
            if (wav) {
                int data_size = num_samples * num_ch * bits / 8;
                int chunk_size = 36 + data_size;
                fwrite("RIFF", 1, 4, wav);
                fwrite(&chunk_size, 4, 1, wav);
                fwrite("WAVEfmt ", 1, 8, wav);
                int fmt_size = 16; fwrite(&fmt_size, 4, 1, wav);
                short af = 1; fwrite(&af, 2, 1, wav);
                fwrite(&num_ch, 2, 1, wav);
                fwrite(&sr, 4, 1, wav);
                int br = sr * num_ch * bits / 8; fwrite(&br, 4, 1, wav);
                short ba = num_ch * bits / 8; fwrite(&ba, 2, 1, wav);
                fwrite(&bits, 2, 1, wav);
                fwrite("data", 1, 4, wav);
                fwrite(&data_size, 4, 1, wav);
                for (int i = 0; i < num_samples; i++)
                    for (int c = 0; c < num_ch; c++) {
                        float v = output[c * num_samples + i] * scale;
                        if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
                        short s = (short)(v * 32767.0f);
                        fwrite(&s, 2, 1, wav);
                    }
                fclose(wav);
                fprintf(stderr, "  Wrote test_minimal_output.wav (%d samples, scale=%.6f, max_abs=%.2f)\n", num_samples, scale, max_abs);
            }
        }
    }

    free(m6_w);
    free(output);
    free(current);
    free(codes);
    free(txc_data);
    model_loader_free(&model);

    fprintf(stderr, "DONE\n");
    return 0;
}
