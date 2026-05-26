# Round 046 — ConvTranspose Weight Dump & Comparison

**Date**: 2026-05-26
**Status**: Complete

## Achievement
Intercepted nc_conv_transpose_1d (PLT 0x403f00) and dumped convtr weight for model.1.block.1.

## Data
- libnc convtr weight: /tmp/libnc_cvt1.bin — [768, 16, 1536] = 18,874,368 floats
- Values: F0=0.001577, range=[-0.078, 0.074]
- LD_PRELOAD: /tmp/preload_convt2.so with nc_conv_transpose_1d intercept

## Our Dequant
- Model: decoder.model.1.block.1.weight_v [768, 16, 1536] BF8 grouped
- is_ct=1 (bias dims[0]=768 == weight_v dims[0]=768)
- Ci=1536, Co=768, K=16
- Norm per input channel (1536 channels)
- Weight_g: [1, 1, 1536] — 1536 gain values

## Comparison
- libnc layout: [Co, K, Ci] = [768, 16, 1536]
- Our output layout: [Co, Ci, K] = [768, 1536, 16]
- Transposition needed for direct comparison
- Python comparison timed out (18.8M elements × nested loops)

## Key Finding
nc_conv_transpose_1d uses same calling convention as nc_conv_1d (weight in rsi). Convtr weight can be reliably dumped for comparison.
