#include "model_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAGIC_HDR 0x23f4aefb
#define MAGIC_TNS 0x23f4aefa

int model_loader_load(const char *path, DACModel *model)
{
    if (!path || !model) return TSAC_ERR_PARAM;

    FILE *f = fopen(path, "rb");
    if (!f) return TSAC_ERR_FILE;

    /* Read all file contents for efficient scanning */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) { fclose(f); return TSAC_ERR_MEMORY; }
    fread(buf, 1, fsize, f);
    fclose(f);

    /* Verify header */
    uint32_t magic, type;
    memcpy(&magic, buf, 4);
    memcpy(&type, buf + 4, 4);
    if (magic != MAGIC_HDR) { free(buf); return TSAC_ERR_FORMAT; }

    /* Find JSON end */
    long json_end = 8;
    while (json_end < fsize && buf[json_end] != '}') json_end++;
    if (json_end >= fsize) { free(buf); return TSAC_ERR_FORMAT; }
    json_end++;

    /* Scan for all tensor magic offsets */
    long *tensor_offsets = NULL;
    int n_tensors = 0;
    for (long pos = json_end; pos < fsize - 4; pos++) {
        uint32_t m;
        memcpy(&m, buf + pos, 4);
        if (m == MAGIC_TNS) {
            tensor_offsets = realloc(tensor_offsets, (n_tensors + 1) * sizeof(long));
            tensor_offsets[n_tensors++] = pos;
            pos += 3; /* skip known magic bytes */
        }
    }

    if (n_tensors == 0) { free(buf); return TSAC_ERR_FORMAT; }

    /* Allocate tensors */
    model->n_tensors = n_tensors;
    model->tensors = (DACTensor *)calloc(n_tensors, sizeof(DACTensor));

    /* Read each tensor */
    for (int i = 0; i < n_tensors; i++) {
        DACTensor *t = &model->tensors[i];
        long pos = tensor_offsets[i];

        uint32_t m, f1, nd, nl;
        memcpy(&m, buf + pos, 4); pos += 4;
        memcpy(&f1, buf + pos, 4); pos += 4;
        memcpy(&nd, buf + pos, 4); pos += 4;
        memcpy(&nl, buf + pos, 4); pos += 4;
        t->ndims = nd;

        for (int d = 0; d < (int)nd; d++) {
            uint32_t v; memcpy(&v, buf + pos, 4); pos += 4;
            t->dims[d] = v;
        }

        int name_bytes = nl < 128 ? nl : 127;
        memcpy(t->name, buf + pos, name_bytes);
        t->name[name_bytes] = '\0';
        pos += nl;

        /* Data size = next tensor start - current data position */
        long next_start = (i + 1 < n_tensors) ? tensor_offsets[i + 1] : fsize;
        t->data_size = (int)(next_start - pos);
        if (t->data_size > 0) {
            t->data = (uint8_t *)malloc(t->data_size);
            memcpy(t->data, buf + pos, t->data_size);
        }
        if (strstr(t->name, "weight_v") != NULL) {
            int dims_product = 1;
            for (int d = 0; d < (int)nd; d++) dims_product *= t->dims[d];
            int as_uint8  = dims_product;
            int as_float32 = dims_product * 4;
            if (t->data_size == as_float32) t->elem_size = 4;
            else if (t->data_size == as_uint8) t->elem_size = 1;
            else t->elem_size = 1;
        } else {
            t->elem_size = 4;
        }
    }

    free(tensor_offsets);
    free(buf);
    return TSAC_OK;
}

void model_loader_free(DACModel *model)
{
    if (!model || !model->tensors) return;
    for (int i = 0; i < model->n_tensors; i++)
        free(model->tensors[i].data);
    free(model->tensors);
    model->tensors = NULL;
    model->n_tensors = 0;
}
