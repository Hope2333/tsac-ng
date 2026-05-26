# Round 053 — Model.0 Gap Root Cause Analysis

**Date**: 2026-05-26
**Status**: Complete (corrected per Oracle review)

## Evidence Chain
1. R052: Injecting libnc weights → no RMS change → weights NOT the cause
2. R028: RVQ output differs from libnc (avg diff 3.18)
3. R033: in_proj+out_proj implementation → RMS unchanged (RVQ changes don't propagate)
4. R025: Injecting codebook weights → RMS unchanged

## Contradiction
R028 says RVQ differs, but R033/R025 show RVQ changes have zero impact on final RMS. This is a **logical contradiction**. The RVQ hypothesis cannot explain the gap because fixes to RVQ don't change the output.

## Corrected Hypothesis
The conv1d kernel (not RVQ, not weights) is the most likely cause. Same weights produce same output (R052). RVQ changes don't propagate (R033). Therefore: the **conv1d kernel itself** (padding, bias application, dilation, precision) differs from nc_conv_1d.

## Next Steps
- Compare our conv1d_kernel output vs libnc nc_conv_1d for identical input and weights
- Verify padding mode, bias order, accumulation precision
