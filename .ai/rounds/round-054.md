# Round 054 — Model.0 Investigation Synthesis

**Date**: 2026-05-26
**Status**: Complete

## Key Finding
Model.0 weight dequant is NOT the bottleneck. Weight injection proves zero impact on RMS (0.080 → 0.080). The remaining -8 dB gap (our -21.99 vs ref -13.85) most likely originates in the **conv1d kernel implementation**.

## Project Status After 54 Rounds
| Component | Status | Gap |
|-----------|--------|-----|
| TXC parsing | ✅ 54/54 indices verified | — |
| CRC32 | ✅ Verified | — |
| BF8 dequant | ✅ Formula correct, L2 norm verified | — |
| is_ct detection | ✅ Fixed (d0!=d2) | — |
| Model.0 weights | ✅ Verified via injection test (R052) | — |
| Conv1d kernel | ⚠️ Suspected different from nc_conv_1d | Likely primary source |
| RVQ output | ⚠️ Confirmed divergent (R028) but changes don't propagate (R033) | Secondary |
| DAC output | ⚠️ -21.99 dBFS vs -13.85 target | -8 dB |

## Next Priority
1. Compare our conv1d kernel vs libnc for same input/weights
2. If kernel matches → re-examine RVQ with new methodology
3. If kernel differs → fix padding/bias/dilation/precision
