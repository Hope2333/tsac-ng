# Round 041 — Weight Comparison: Our vs libnc

**Date**: 2026-05-26
**Status**: Complete

## Comparison Results

### Raw .bin data vs libnc converted weight
- out_proj.weight_v: float32, [8, 1, 1024]
- Raw vs libnc: avg_diff=0.017, max_diff=0.160, 0/8192 exact
- Ratio raw/lib: mean=1.128 ± 0.142 — raw data ~13% larger

### Our dequantized vs libnc
- Our op_f32 vs libnc: avg_diff=0.321, max_diff=1.054, 0/8192 exact
- Ratio our/lib: mean=0.953 ± 346.5 — highly variable

### Key Finding
libnc's nc_convert_from_old_bf transforms the raw float32 data via L2 normalization + weight_g scaling. Our dequant_weights also applies L2 norm + weight_g but produces different output. The difference is NOT in the raw data (same .bin file) but in how the norm/gain is computed.

## Layout Issue
- Our output: [Co=1024, Ci=8, K=1] — transposed for our conv1d kernels
- libnc output: [Ci=8, K=1, Co=1024] — original stored layout
- nc_conv_1d reads dims as [Ci, K, Co] — expects non-transposed layout
- Our kernels expect [Co, Ci, K] — different access pattern
- Both are internally consistent but lay out values differently in memory
