#include "src/model_loader.h"
#include "src/dac_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static DACTensor *tf(DACTensor *ts, int nt, const char *name) {
    for (int i = 0; i < nt; i++) if (!strcmp(ts[i].name, name)) return &ts[i];
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_path = "/usr/share/tsac/dac_stereo_q8.bin";
    DACModel model = {0};
    model_loader_load(model_path, &model);
    
    fprintf(stderr, "=== 内部残差块张量信息 ===\n\n");
    
    for (int block = 1; block <= 4; block++) {
        for (int inner = 1; inner <= 4; inner++) {
            if (inner == 1) continue; // block.X.block.1 是 conv_transpose
            
            char wv_name[128], wg_name[128], b_name[128];
            char a_name[128];
            snprintf(wv_name, sizeof(wv_name), "decoder.model.%d.block.%d.block.1.weight_v", block, inner);
            snprintf(wg_name, sizeof(wg_name), "decoder.model.%d.block.%d.block.1.weight_g", block, inner);
            snprintf(b_name, sizeof(b_name), "decoder.model.%d.block.%d.block.1.bias", block, inner);
            snprintf(a_name, sizeof(a_name), "decoder.model.%d.block.%d.block.0.alpha", block, inner);
            
            DACTensor *wv = tf(model.tensors, model.n_tensors, wv_name);
            DACTensor *wg = tf(model.tensors, model.n_tensors, wg_name);
            DACTensor *b = tf(model.tensors, model.n_tensors, b_name);
            DACTensor *a = tf(model.tensors, model.n_tensors, a_name);
            
            if (wv || wg || b || a) {
                fprintf(stderr, "decoder.model.%d.block.%d:\n", block, inner);
                if (wv) fprintf(stderr, "  weight_v: dims=[%d,%d,%d,%d] elem_size=%d\n",
                        wv->dims[0], wv->dims[1], wv->dims[2], wv->dims[3], wv->elem_size);
                if (wg) fprintf(stderr, "  weight_g: dims=[%d,%d,%d,%d] elem_size=%d\n",
                        wg->dims[0], wg->dims[1], wg->dims[2], wg->dims[3], wg->elem_size);
                if (b)  fprintf(stderr, "  bias: dims=[%d,%d,%d,%d] elem_size=%d\n",
                        b->dims[0], b->dims[1], b->dims[2], b->dims[3], b->elem_size);
                if (a)  fprintf(stderr, "  alpha: dims=[%d,%d,%d,%d] elem_size=%d\n",
                        a->dims[0], a->dims[1], a->dims[2], a->dims[3], a->elem_size);
                fprintf(stderr, "\n");
            }
        }
    }
    
    /* 打印所有包含 "inner" 或 "block.2"、"block.3"、"block.4" 的张量 */
    fprintf(stderr, "=== 所有内部块张量列表 ===\n\n");
    for (int i = 0; i < model.n_tensors; i++) {
        if (strstr(model.tensors[i].name, "block.2.") || 
            strstr(model.tensors[i].name, "block.3.") ||
            strstr(model.tensors[i].name, "block.4.")) {
            fprintf(stderr, "  %s: dims=[%d,%d,%d,%d] elem_size=%d\n",
                    model.tensors[i].name,
                    model.tensors[i].dims[0], model.tensors[i].dims[1],
                    model.tensors[i].dims[2], model.tensors[i].dims[3],
                    model.tensors[i].elem_size);
        }
    }
    
    model_loader_free(&model);
    return 0;
}
