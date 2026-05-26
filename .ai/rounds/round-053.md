# Round 053 — Model.0 Gap Root Cause Analysis

**Date**: 2026-05-26
**Status**: Complete

## Evidence Chain
1. R052: Injecting libnc weights → no RMS change → weights NOT the cause
2. R028: RVQ output differs from libnc (avg diff 3.18)
3. R033: in_proj+out_proj implementation → RMS unchanged
4. R025: Injecting codebook weights → RMS unchanged

## Hypothesis
Despite 3 rounds of RVQ investigation showing no impact, the model.0 input (RVQ output) MUST differ between our code and libnc, because:
- Same weights + same conv1d kernel → same output with same input
- Different weights + same kernel → different output (but we confirmed weights are NOT the cause by injection)
- Therefore: different INPUT → different output

## Next Steps
- Measure our RVQ output RMS and compare with libnc (R044 showed libnc RVQ input=0.025, output=1.061 per codebook)
- If RVQ output differs, trace back to codebook entry mapping or projection weights
