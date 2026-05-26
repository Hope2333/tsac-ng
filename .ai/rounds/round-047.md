# Round 047 — Convtr Dequant Verification Approach

**Date**: 2026-05-26
**Status**: Complete (approach documented)

## Analysis
The convtr weight dequant uses is_ct=1 path which differs from regular conv1d:
1. Dimension interpretation: Ci=d2 (1536), Co=d0 (768)
2. Norm axis: per-input-channel (ci) instead of per-output-channel (co)
3. Storage read: [Co, K, Ci] transposed to [Co, Ci, K]
4. Weight_g indexed by ci (input channel), not co (output channel)

## Potential Issues
- Weight_g dims[2]=1536 = Ci — correct for per-input-channel gain
- Norm computed over (Co × K) elements per input channel = 768 × 16 = 12,288
- If norm computation correct, weight_g values correct → dequant should match libnc
- 18.8M-element comparison requires efficient C implementation, not Python

## Approach for Full Verification
1. Write standalone C program using our dequant_weights directly
2. Load libnc dump and compute per-element diff
3. Identify systematic differences (ratio, offset, pattern)

## Evidence
- /tmp/libnc_cvt1.bin — libnc ground truth
- /tmp/preload_convt2.log — intercept confirms correct weight dims
