#ifndef RANGE_CODER_H
#define RANGE_CODER_H

/* Range coder: get_freq (15-bit adaptive) + direct bit. RE from arith.c. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Range coder state.
 *  Stores the current decoding window (low, range) and buffer position. */

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

/** Initialize a range coder from a byte buffer.
 *  Reads RC_INIT_CODE_BYTES (4) bytes to seed the low value.
 *  Returns 0 on success, negative on error. */
int  rc_decoder_init(RangeCoder *rc, const uint8_t *buf, size_t len);

/** Decode a single bit with 50/50 equal probability.
 *  Used in fast TXC mode as a fallback (confirmed dead code in original). */
int  rc_decoder_direct_bit(RangeCoder *rc);

/** Decode a single bit with adaptive frequency.
 *  freq is a 15-bit probability value (1..32767).
 *  Used in normal TXC mode with binary search decoder. */
int  rc_decoder_get_freq(RangeCoder *rc, uint32_t freq);

#ifdef __cplusplus
}
#endif
#endif
