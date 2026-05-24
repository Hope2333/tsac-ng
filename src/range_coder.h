#ifndef RANGE_CODER_H
#define RANGE_CODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RC_MIN_VALUE      0x0000FF00U
#define RC_INIT_RANGE     0xFFFFFFFFU
#define RC_INIT_CODE_BYTES 4

typedef struct {
    uint32_t        low;
    uint32_t        range;
    const uint8_t  *buf;
    size_t          buf_pos;
    size_t          buf_len;
} RangeCoder;

int  rc_decoder_init(RangeCoder *rc, const uint8_t *buf, size_t len);
int  rc_decoder_direct_bit(RangeCoder *rc);
int  rc_decoder_get_freq(RangeCoder *rc, uint32_t freq);

#ifdef __cplusplus
}
#endif
#endif
