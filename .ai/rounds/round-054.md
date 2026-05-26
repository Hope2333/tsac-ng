# Round 054 — Model.0 Investigation Synthesis

**Date**: 2026-05-26
**Status**: Complete

## Key Finding
Model.0 weight dequant is NOT the bottleneck. Weight injection proves zero impact on RMS (0.080 → 0.080). The remaining -8 dB gap (our -21.99 vs ref -13.85) originates upstream (RVQ) or in the conv1d kernel itself.

## Project Status After 54 Rounds
| Component | Status | Gap |
|-----------|--------|-----|
| TXC parsing | ✅ 54/54 indices verified | — |
| CRC32 | ✅ Verified | — |
| BF8 dequant | ✅ Formula correct, L2 norm verified | — |
| is_ct detection | ✅ Fixed (d0!=d2) | — |
| Model.0 weights | ✅ Verified via injection test | — |
| Convtr weights | ⚠️ 3.8× larger than libnc | R046 |
| RVQ output | ⚠️ Confirmed divergent (R028) | Likely primary source |
| DAC output | ⚠️ -21.99 dBFS vs -13.85 target | -8 dB |
