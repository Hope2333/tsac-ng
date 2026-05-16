/*
 * txc_format.c — .txc container parser (original TSAC format)
 *
 * Real format reverse-engineered from binary analysis:
 *   Offset  Description
 *   ------  -----------
 *   0-3     "FBAZ" magic (ASCII)
 *   4-5     version (BE u16)
 *   6       flags (u8): bit0=stereo, other bits=encoding mode
 *   7       n_codebooks (u8): 1-12
 *   8-11    parameter u32 BE (sample count or frame reference)
 *   12-15   parameter u32 BE (additional metadata)
 *   16+     optional extended header fields
 *   ?+      uint8 codebook_indices[n_frames * n_codebooks]
 *
 * Header end is auto-detected: the first offset after byte 7 where
 * (file_size - offset) is evenly divisible by n_codebooks.
 * Each codebook index occupies 1 byte (uint8, codebook size 256).
 *
 * n_frames = (file_size - header_end) / n_codebooks
 */
#include "txc_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define TXC_MAGIC_BYTES { 'F', 'B', 'A', 'Z' }

static int find_header_end(const uint8_t *data, size_t data_size, int n_codebooks)
{
    /* Minimum header: magic(4) + version(2) + flags(1) + n_codebooks(1) = 8 */
    int min_header = 8;

    /* Scan forward until the remaining data is evenly divisible by n_codebooks.
     * This handles both 17-byte and 24-byte header variants. */
    for (int h = min_header; h < (int)data_size && h < 256; h++) {
        if ((data_size - (size_t)h) % (size_t)n_codebooks == 0)
            return h;
    }
    /* Fallback: use min_header */
    return min_header;
}

void txc_header_init(TSCHeader *hdr, int stereo, int n_codebooks, int sample_rate)
{
    if (!hdr) return;
    hdr->magic[0] = 'F';
    hdr->magic[1] = 'B';
    hdr->magic[2] = 'A';
    hdr->magic[3] = 'Z';
    hdr->version     = 1;
    hdr->n_codebooks = (uint16_t)(n_codebooks & 0xFFFF);
    hdr->block_len   = 320;
    hdr->n_blocks    = 0;
    hdr->sample_rate = (uint32_t)sample_rate;
    hdr->flags       = stereo ? 1U : 0U;
    hdr->data_offset = sizeof(TSCHeader);
}

int txc_write(const TSCHeader *hdr,
              const int *codebook_indices, int n_frames,
              uint8_t **out_data, size_t *out_size)
{
    if (!hdr || !codebook_indices || !out_data || !out_size)
        return TSAC_ERR_PARAM;

    /* Use uint8 encoding: 1 byte per codebook index */
    size_t hdr_size = sizeof(TSCHeader);
    size_t idx_count = (size_t)n_frames * hdr->n_codebooks;
    size_t idx_bytes = idx_count; /* 1 byte per index */
    size_t total_size = hdr_size + idx_bytes;

    uint8_t *buf = (uint8_t *)calloc(total_size, 1);
    if (!buf) return TSAC_ERR_MEMORY;

    TSCHeader *out_hdr = (TSCHeader *)buf;
    memcpy(out_hdr, hdr, hdr_size);
    out_hdr->n_blocks   = (uint32_t)n_frames;
    out_hdr->data_offset = (uint32_t)hdr_size;
    out_hdr->n_codebooks = hdr->n_codebooks;

    uint8_t *idx_data = buf + hdr_size;
    for (size_t i = 0; i < idx_count; i++)
        idx_data[i] = (uint8_t)(codebook_indices[i] & 0xFF);

    *out_data = buf;
    *out_size = total_size;
    return TSAC_OK;
}

int txc_read(const uint8_t *data, size_t data_size,
             TSCHeader *hdr,
             int **codebook_indices, int *n_frames)
{
    if (!data || !hdr || !codebook_indices || !n_frames)
        return TSAC_ERR_PARAM;

    if (data_size < 8)
        return TSAC_ERR_FORMAT;

    /* Validate magic */
    if (data[0] != 'F' || data[1] != 'B' ||
        data[2] != 'A' || data[3] != 'Z')
        return TSAC_ERR_FORMAT;

    /* Parse fixed header fields */
    memset(hdr, 0, sizeof(TSCHeader));
    hdr->magic[0] = 'F'; hdr->magic[1] = 'B';
    hdr->magic[2] = 'A'; hdr->magic[3] = 'Z';

    hdr->version = (uint16_t)((data[4] << 8) | data[5]);

    if (hdr->version < 1 || hdr->version > 255)
        return TSAC_ERR_FORMAT;

    hdr->flags       = data[6];
    hdr->n_codebooks = data[7];
    hdr->sample_rate = 48000; /* default, overridden if available */

    if (hdr->n_codebooks < 1 || hdr->n_codebooks > 12)
        return TSAC_ERR_FORMAT;

    /* Parse optional fields at offset 8-15 if present */
    if (data_size >= 12) {
        uint32_t p1, p2;
        p1 = ((uint32_t)data[ 8] << 24) | ((uint32_t)data[ 9] << 16)
           | ((uint32_t)data[10] <<  8) | ((uint32_t)data[11]);
        p2 = ((uint32_t)data[12] << 24) | ((uint32_t)data[13] << 16)
           | ((uint32_t)data[14] <<  8) | ((uint32_t)data[15]);
        hdr->n_blocks   = p1; /* likely total sample count or frame reference */
        hdr->block_len  = p2; /* stored in block_len slot as metadata */
    }

    /* Auto-detect header end: find offset where remaining data is
     * cleanly divisible by n_codebooks (uint8 index encoding) */
    int codebooks = (int)hdr->n_codebooks;
    int header_end = find_header_end(data, data_size, codebooks);

    if (header_end < 8) {
        return TSAC_ERR_FORMAT;
    }

    hdr->data_offset = (uint32_t)header_end;

    size_t payload_size = data_size - (size_t)header_end;
    size_t idx_count = payload_size; /* n_frames * n_codebooks, uint8 each */

    int total_frames = (int)(idx_count / (size_t)codebooks);

    if (total_frames < 1 || (size_t)total_frames * (size_t)codebooks != idx_count)
        return TSAC_ERR_FORMAT;

    /* Allocate and convert: uint8 → int (0..255 codebook indices) */
    int *indices = (int *)calloc(idx_count, sizeof(int));
    if (!indices) return TSAC_ERR_MEMORY;

    const uint8_t *src = data + header_end;
    for (size_t i = 0; i < idx_count; i++)
        indices[i] = (int)src[i];

    *codebook_indices = indices;
    *n_frames = total_frames;

    /* Store actual frame count in header */
    hdr->n_blocks    = (uint32_t)total_frames;
    hdr->block_len   = 600; /* default frame length (48kHz / 80fps) */
    hdr->sample_rate = 48000;

    return TSAC_OK;
}
