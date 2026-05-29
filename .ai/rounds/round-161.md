# Round 161 — Fix AVX-512 Conv1d Kernel (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
R160 confirmed the AVX-512 conv1d kernel (`conv1d_avx512` in `src/cpu_simd.inc:342-385`) produces incorrect output compared to scalar. Two known issues:

1. **gather16 scalar temp array** (line 141): `gather16()` loads 16 floats one-by-one into a temp array, then loads with `_mm512_loadu_ps`. This defeats hardware gather and may cause 27-70× activation amplification. Replace with `_mm512_mask_i32gather_ps`.

2. **FMA reassociation under -ffast-math** (line 364): `_mm512_fmadd_ps(x16, w16, sum16)` may be reassociated by the compiler when `-ffast-math` is active. The compiler can fuse or split FMAs differently than libnc's sequential multiply-add loop.

**Additional concern**: The horizontal reduction `hsum512_ps(sum16)` at line 370 adds 16 partial sums into a single float after the channel loop. This is correct only if the outer `oc` loop produces one output per channel, but the structure has `sum16` accumulating across multiple K iterations per channel block, making the reduction a partial sum over (Ci_block/16 * K) products — NOT a single channel's complete sum. This is a **structural bug in the kernel**: sum16 should accumulate across `icb` + `j` loops and the reduction should produce the final output value without further channel loop.

Wait — let me trace the exact logic:
- Outer `oc`: loop over output channels
- Inner `oi`: loop over output time steps
- `sum16 = bias_vec`: initialize
- Inner `icb`: loop over input channel blocks of 16
  - Inner `j`: loop over kernel positions (K)
    - `sum16 = _mm512_fmadd_ps(x16, w16, sum16)` — accumulates ALL input channels and ALL K positions into sum16

This is WRONG. sum16 accumulates across the ENTIRE channel dimension (Ci_block/16 blocks * K kernel positions), but `hsum512_ps(sum16)` then reduces the 16 lanes into 1 float. The 16 lanes were accumulating different output channels? No — sum16 has 16 lanes that all accumulate the SAME output channel oc across different input channel groups. Then `hsum512_ps` adds them together, which double-counts.

**Actually wait**: Let me re-read more carefully. The 16 lanes of sum16 each hold a partial sum for DIFFERENT output channels? No — the v16_x and v16_w both span 16 INPUT channels, and they're multiplied element-wise. The output channel is fixed (oc). So the 16 lanes are partial sums for the SAME output channel oc across 16 input channels. Then hsum512 adds them together. That IS correct for a single output channel.

But then after the loop, the remaining channels are handled with scalar. This seems structurally OK.

**The real issue is hsum512_ps being called inside the oi loop, after the icb loop, but the reduction should happen AFTER the full icb + j loop completes for each (oc, oi) pair**. Looking at the code flow:

```
for oc:
  for oi:
    sum16 = bias_vec
    for icb:
      for j:
        sum16 = fmadd(x16, w16, sum16)
    float sum = hsum512_ps(sum16)
    for ic in remaining:
      sum += ...
    o[oc*T+oi] = sum
```

This IS correct — sum16 accumulates all (Ci_block * K) products into 16 lanes (one per input channel within a block), then hsum reduces to a single value, then remaining channels are added. The correctness of this structure depends on whether the 16 lanes of x16 * w16 are all for the same output channel — and they ARE.

So the actual bug must be in gather16 (scalar temp array) or FMA reassociation.

## Round Strategy
1. Fix `gather16` to use `_mm512_mask_i32gather_ps` instead of scalar temp array
2. Add `-fno-associative-math` or `-ffp-contract=on` to prevent FMA reassociation
3. Verify AVX-512 conv1d output matches scalar conv1d on model.0 data
4. Re-enable AVX-512 dispatch in `get_ops()` (`src/cpu_decoder.c`)
5. Full decoder run with AVX-512 active, compare activations against libnc

## Tasks

### T1: Fix gather16 — replace scalar temp array with hardware gather
**File**: `src/cpu_simd.inc` lines 141-150

**Action**: Replace `gather16` function:
```c
static inline __m512 gather16(const float *x, int ic_base, int ii, int T, int Ci) {
    // Build byte-offset indices for gather
    // x[(ic_base+k)*T + ii] → byte_offset = ((ic_base+k)*T + ii) * sizeof(float)
    __m512i indices = _mm512_set_epi32(
        ((ic_base + 15) * T + ii) * 4,
        ((ic_base + 14) * T + ii) * 4,
        ((ic_base + 13) * T + ii) * 4,
        ((ic_base + 12) * T + ii) * 4,
        ((ic_base + 11) * T + ii) * 4,
        ((ic_base + 10) * T + ii) * 4,
        ((ic_base +  9) * T + ii) * 4,
        ((ic_base +  8) * T + ii) * 4,
        ((ic_base +  7) * T + ii) * 4,
        ((ic_base +  6) * T + ii) * 4,
        ((ic_base +  5) * T + ii) * 4,
        ((ic_base +  4) * T + ii) * 4,
        ((ic_base +  3) * T + ii) * 4,
        ((ic_base +  2) * T + ii) * 4,
        ((ic_base +  1) * T + ii) * 4,
        ((ic_base +  0) * T + ii) * 4
    );
    // Mask: only valid channels within Ci
    int remaining = Ci - ic_base;
    __mmask16 mask = remaining >= 16 ? 0xFFFF : (__mmask16)((1 << remaining) - 1);
    // Gather with zero-masking for out-of-bounds
    return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, indices, x, 4);
}
```

**Acceptance**: `gather16` produces identical output to scalar loop for all valid inputs.

### T2: Fix FMA reassociation — control compiler fusion
**File**: `src/cpu_simd.inc` and `CMakeLists.txt`

**Action**: Two approaches (try both):
1. **Per-function**: Add `__attribute__((target("avx512f"), optimize("no-associative-math")))` to `conv1d_avx512`
2. **Per-compilation-unit**: Add `-ffp-contract=on` (not `=fast`) to the compilation flags for `cpu_simd.inc`

In `CMakeLists.txt`:
```cmake
# AVX-512 source needs precise FMA semantics
set_source_files_properties(src/cpu_simd.inc PROPERTIES COMPILE_FLAGS "-ffp-contract=on")
```

Or better: compile `cpu_simd.inc` as a separate translation unit with controlled flags.

**Acceptance**: `_mm512_fmadd_ps` produces bit-identical results to sequential `a*b + c` on all test vectors.

### T3: Verify AVX-512 conv1d matches scalar on model.0
**File**: `experimental/test_conv1d_kernels.py` (from R160)

**Action**: Re-run comparison after fix:
```bash
# Rebuild
cmake --build build

# Run test harness
./build/test_conv1d_kernels

# Compare
python3 -c "
import numpy as np
s = np.fromfile('/tmp/conv1d_scalar.bin', dtype=np.float32)
a = np.fromfile('/tmp/conv1d_avx512.bin', dtype=np.float32)
corr = np.corrcoef(s, a)[0,1]
rmse = np.sqrt(np.mean((s-a)**2))
max_diff = np.max(np.abs(s-a))
print(f'AVX-512 vs scalar: corr={corr:.10f} rmse={rmse:.10f} max_diff={max_diff:.10f}')
assert corr > 0.999999, f'AVX-512 still diverges from scalar! corr={corr}'
assert max_diff < 1e-5, f'AVX-512 max_diff too large: {max_diff}'
print('PASS: AVX-512 matches scalar')
"
```

**Acceptance**: AVX-512 conv1d output matches scalar with corr > 0.999999.

### T4: Re-enable AVX-512 dispatch in get_ops()
**File**: `src/cpu_decoder.c` lines 204-208

**Action**: Remove the workaround that forces scalar:
```c
// BEFORE (lines 204-208):
    /* NOTE: convt1d_avx512 and conv1d_avx512 both have suspected FMA bugs
     * producing 27× and 70× larger activations respectively. Using scalar
     * kernels for correctness until SIMD bugs are fixed. */
    ops.conv1d = conv1d_s;
    ops.conv_transpose1d = convt1d_s;
    return ops;

// AFTER:
// (Let the earlier dispatch assignments stand — they set ops.conv1d = conv1d_avx512
//  at line 146 if AVX-512 is detected)
    return ops;
```

**Acceptance**: AVX-512 kernels are auto-selected on AVX-512-capable hardware. Scalar used only as fallback.

### T5: Full decoder run — compare activations against libnc
**File**: `src/cpu_decoder.c`, `experimental/compare_activations.py`

**Action**: Run full decoder with AVX-512 active:
```bash
cmake --build build && ./build/tsac-ng -v -f d /tmp/short_fast.txc /tmp/out.wav
python3 experimental/compare_activations.py
```

**Expected improvement**:
- m0_conv1d correlation should now match libnc (or at least improve)
- Downstream layers should see compounding improvement

**Acceptance**: Per-layer correlation improves for model.0 conv1d. If not, further investigation needed.

## Acceptance Criteria
- [ ] `gather16` uses hardware gather (`_mm512_mask_i32gather_ps`) — no scalar temp array
- [ ] FMA reassociation disabled for AVX-512 conv kernels
- [ ] AVX-512 conv1d matches scalar with corr > 0.999999 on model.0 data
- [ ] AVX-512 dispatch re-enabled (no scalar workaround)
- [ ] Full decoder run shows improved model.0 conv1d correlation vs libnc
- [ ] No new NaN or numerical stability issues introduced
