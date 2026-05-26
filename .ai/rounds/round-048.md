# Round 048 — Convtr Investigation Synthesis

**Date**: 2026-05-26
**Status**: Complete

## Key Finding
Successfully intercepted nc_conv_transpose_1d — same calling convention as nc_conv_1d (weight=rsi). Convtr weight can be dumped for any layer in the DAC graph.

## Divergence Reversal Root Cause Narrowed
- model.0 (regular conv1d): libnc 2.901 vs our 0.609 — libnc 4.8× LARGER
- model.6 (regular conv1d): libnc 0.203 vs our 0.641 — our 3.2× LARGER
- The reversal happens in models 1-4 (convtr + residual blocks)
- Convtr weight dequant (is_ct=1) and residual block conv1d weights are the prime suspects

## Next Steps
- Round 049: Implement efficient C comparison of convtr weights
- Round 050: If weights match, compare convtr INPUT/OUTPUT tensors
- Round 051: If weights differ, fix convtr dequant and verify RMS
