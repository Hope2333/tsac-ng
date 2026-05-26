#ifndef RANGE_CODER_H
#define RANGE_CODER_H

/*
 * range_coder.h — Arithmetic range coder for codebook index decoding.
 *
 * Reverse-engineered from original tsac binary's arith.c.
 * Supports two decoding modes:
 *   - rc_decoder_get_freq(): Adaptive 15-bit probability (normal TXC mode)
 *   - rc_decoder_direct_bit(): Fixed 50/50 probability (fast TXC fallback)
 *
 * Algorithm (from GDB RE @ 0x42bbe0):
 *   range0 = (range * freq) >> 15
 *   Normalization threshold: RC_MIN_VALUE (0xFF00)
 *
 * The original tsac uses get_freq with a binary search decoder
 * to extract codebook indices from the range-coded bitstream.
 * get_bit (at 0x42bd30) is confirmed dead code — never called.
 */

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
