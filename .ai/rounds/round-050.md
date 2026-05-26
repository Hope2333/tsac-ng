# Round 050 — Convtr Norm Loop Fix Attempted

**Date**: 2026-05-26
**Status**: Investigated (fix attempted, needs debugging)

## Goal
Fix the norm accumulation loop for is_ct=1 tensors to use the correct [Co,K,Ci] indexing instead of [Ci,K,Co].

## Attempted Fix
```c
if (is_ct) {
    // ConvTranspose: [Co,K,Ci] layout, norms per Ci
    src_idx = i2 * K * d0 + k * d0 + i0;
    norms[i0] += v * v;
} else {
    // Regular: [Ci,K,Co] layout, norms per Co
    src_idx = i0 * K * d2 + k * d2 + i2;
    norms[i2] += v * v;
}
```

## Result
Output saturated (RMS=1.000). The fix produces norm values that are too small (per-Ci with 12288 elements vs per-Co with 24576 elements), making weights too large. 

## Analysis
Per-Ci norms (avg=12,405) are ~1.4× smaller than per-Co norms (avg=17,467), making weights ~1.4× larger. This alone doesn't explain saturation. The issue may be in how src_f32 is populated or index arithmetic.

## Next
- Debug with smaller tensor to verify indexing
- Check if d0/d2 swap affects norm loop iteration ranges
- Consider alternative: apply weight_g BEFORE norm rather than after
