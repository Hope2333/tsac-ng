/*
 * range_coder.c — Arithmetic range coder implementation.
 *
 * Reverse-engineered from original tsac binary's arith.c.
 * The original uses get_freq (adaptive 15-bit probability) for
 * codebook index decoding, NOT equal-probability direct bits.
 */

#include "range_coder.h"

/* Ensure range is above minimum threshold after bit consumption. */
static void rc_normalize(RangeCoder *rc)
{
    while (rc->range <= RC_MIN_VALUE) {
        rc->range <<= 8;
        rc->low   <<= 8;
        if (rc->buf_pos < rc->buf_len)
            rc->low |= rc->buf[rc->buf_pos++];
    }
}

int rc_decoder_init(RangeCoder *rc, const uint8_t *buf, size_t len)
{
    if (!rc || !buf) return -1;
    if (len < RC_INIT_CODE_BYTES) return -2;

    rc->low     = 0;
    rc->range   = RC_INIT_RANGE;
    rc->buf     = buf;
    rc->buf_pos = 0;
    rc->buf_len = len;

    for (int i = 0; i < RC_INIT_CODE_BYTES; i++)
        rc->low = (rc->low << 8) | rc->buf[rc->buf_pos++];

    return 0;
}

int rc_decoder_direct_bit(RangeCoder *rc)
{
    rc_normalize(rc);
    uint32_t r0 = rc->range >> 1;
    if (rc->low >= r0) { rc->low -= r0; rc->range -= r0; return 1; }
    rc->range = r0;
    return 0;
}

int rc_decoder_get_freq(RangeCoder *rc, uint32_t freq)
{
    rc_normalize(rc);
    uint32_t r0 = ((uint64_t)rc->range * freq) >> 15;
    if (r0 < 1) r0 = 1;
    if (r0 >= rc->range) r0 = rc->range - 1;

    if (rc->low >= r0) {
        rc->low   -= r0;
        rc->range -= r0;
        return 1;
    }
    rc->range = r0;
    return 0;
}
