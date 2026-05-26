/*
 * test_layout.c — Verify weight layout and dequant for m0 layer
 */
#include "src/model_loader.h"
#include "src/dac_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_path = "/usr/share/tsac/dac_stereo_q8.bin";
    DACModel model = {0};
    int ret = model_loader_load(model_path, &model);
    if (ret != 0) { fprintf(stderr, "Model load failed: %d\n", ret); return 1; }

    /* Examine m0 weight_v raw data distribution */
    DACTensor *m0_wv = tf(model.tensors, model.n_tensors, "decoder.model.0.weight_v");
    DACTensor *m0_wg = tf(model.tensors, model.n_tensors, "decoder.model.0.weight_g");
    DACTensor *m0_b  = tf(model.tensors, model.n_tensors, "decoder.model.0.bias");

    fprintf(stderr, "m0.weight_v: dims=[%d,%d,%d] elem=%d size=%d\n",
            m0_wv->dims[0], m0_wv->dims[1], m0_wv->dims[2], m0_wv->elem_size, m0_wv->data_size);
    fprintf(stderr, "m0.weight_g: dims=[%d,%d,%d] elem=%d size=%d\n",
            m0_wg->dims[0], m0_wg->dims[1], m0_wg->dims[2], m0_wg->elem_size, m0_wg->data_size);
    fprintf(stderr, "m0.bias: dims=[%d] elem=%d size=%d\n",
            m0_b->dims[0], m0_b->elem_size, m0_b->data_size);

    /* Raw weight_v byte distribution */
    const uint8_t *v = m0_wv->data;
    int total = m0_wv->data_size;
    int hist[256] = {0};
    for (int i = 0; i < total && i < 1000000; i++) hist[v[i]]++;

    fprintf(stderr, "\nRaw weight_v byte distribution (first 1M bytes):\n");
    for (int bucket = 0; bucket < 16; bucket++) {
        int count = 0;
        for (int i = bucket*16; i < (bucket+1)*16; i++) count += hist[i];
        fprintf(stderr, "  [%3d-%3d]: %d\n", bucket*16, (bucket+1)*16-1, count);
    }

    /* weight_g scales distribution */
    const float *g_scales = (const float *)m0_wg->data;
    fprintf(stderr, "\nweight_g scales (first 20):\n");
    float g_min = 1e30, g_max = -1e30;
    int ng = m0_wg->data_size / 4;
    for (int i = 0; i < ng; i++) {
        if (i < 20) fprintf(stderr, "  g[%d] = %.6f\n", i, g_scales[i]);
        if (g_scales[i] < g_min) g_min = g_scales[i];
        if (g_scales[i] > g_max) g_max = g_scales[i];
    }
    fprintf(stderr, "  g range: [%.6f, %.6f] count=%d\n", g_min, g_max, ng);

    /* bias distribution */
    const float *bias = (const float *)m0_b->data;
    fprintf(stderr, "\nbias (first 20):\n");
    for (int i = 0; i < 20 && i < m0_b->dims[0]; i++)
        fprintf(stderr, "  b[%d] = %.6f\n", i, bias[i]);

    /* Test dequant with BOTH layout interpretations */
    int d0 = m0_wv->dims[0], d1 = m0_wv->dims[1], d2 = m0_wv->dims[2];

    /* Interpretation A: [Ci, K, Co] = [1024, 7, 1536] */
    fprintf(stderr, "\n=== Layout A: [Ci=%d, K=%d, Co=%d] ===\n", d0, d1, d2);
    {
        int Ci=d0, K=d1, Co=d2;
        float w_min=1e30, w_max=-1e30, w_sum=0;
        int w_neg=0, w_pos=0, w_zero=0;
        for (int ci = 0; ci < Ci; ci++) {
            for (int k = 0; k < K; k++) {
                for (int co = 0; co < Co; co++) {
                    int src_idx = ci * K * Co + k * Co + co;
                    uint8_t v_val = v[src_idx];
                    float g = g_scales[co];
                    float w = g * ((float)v_val - 128.0f) / 127.0f;
                    if (w < w_min) w_min = w;
                    if (w > w_max) w_max = w;
                    w_sum += w;
                    if (w < -0.001f) w_neg++;
                    else if (w > 0.001f) w_pos++;
                    else w_zero++;
                }
            }
        }
        int total_w = Ci*K*Co;
        fprintf(stderr, "  range: [%.4f, %.4f] avg=%.4f neg=%d pos=%d zero=%d\n",
                w_min, w_max, w_sum/total_w, w_neg, w_pos, w_zero);
    }

    /* Interpretation B: [Co, K, Ci] = [1024, 7, 1536] */
    fprintf(stderr, "\n=== Layout B: [Co=%d, K=%d, Ci=%d] ===\n", d0, d1, d2);
    {
        int Co=d0, K=d1, Ci=d2;
        float w_min=1e30, w_max=-1e30, w_sum=0;
        int w_neg=0, w_pos=0, w_zero=0;
        for (int co = 0; co < Co; co++) {
            for (int k = 0; k < K; k++) {
                for (int ci = 0; ci < Ci; ci++) {
                    int src_idx = co * K * Ci + k * Ci + ci;
                    uint8_t v_val = v[src_idx];
                    float g = g_scales[co];
                    float w = g * ((float)v_val - 128.0f) / 127.0f;
                    if (w < w_min) w_min = w;
                    if (w > w_max) w_max = w;
                    w_sum += w;
                    if (w < -0.001f) w_neg++;
                    else if (w > 0.001f) w_pos++;
                    else w_zero++;
                }
            }
        }
        int total_w = Co*K*Ci;
        fprintf(stderr, "  range: [%.4f, %.4f] avg=%.4f neg=%d pos=%d zero=%d\n",
                w_min, w_max, w_sum/total_w, w_neg, w_pos, w_zero);
    }

    /* Also check decoder.model.3 (first conv1d layer that's NOT conv_transpose) */
    fprintf(stderr, "\n=== Checking decoder.model.3 block weights ===\n");
    for (int inner = 2; inner <= 4; inner++) {
        char wv_name[128], wg_name[128];
        snprintf(wv_name, sizeof(wv_name), "decoder.model.3.block.%d.block.1.weight_v", inner);
        snprintf(wg_name, sizeof(wg_name), "decoder.model.3.block.%d.block.1.weight_g", inner);
        DACTensor *iwv = tf(model.tensors, model.n_tensors, wv_name);
        DACTensor *iwg = tf(model.tensors, model.n_tensors, wg_name);
        if (!iwv) continue;
        fprintf(stderr, "  %s: dims=[%d,%d,%d] size=%d\n", wv_name, iwv->dims[0], iwv->dims[1], iwv->dims[2], iwv->data_size);
        if (iwg) fprintf(stderr, "  %s: dims=[%d,%d,%d] size=%d\n", wg_name, iwg->dims[0], iwg->dims[1], iwg->dims[2], iwg->data_size);
    }

    model_loader_free(&model);
    return 0;
}
