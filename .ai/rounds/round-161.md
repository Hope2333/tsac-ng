# Round 161 — Fix AVX-512 Conv1d Kernel Bug (Header Dispatch)
**Signed**: Worker | **Date**: 2026-05-28 | **Status**: PENDING

## Summary
The AVX-512 conv1d and convt kernels in `src/cpu_simd.inc` are known-broken: they produce 10-70× larger activations compared to scalar kernels. Both `conv1d_avx512` (line 342) and `convt1d_avx512` (line 388) are disabled at runtime in `src/cpu_decoder.c` line 207-208, which forces scalar fallback for ALL architectures. This creates a 16× performance regression on AVX-512-capable hardware (Skylake-SP, Zen 4). The root cause is suspected in the `gather16` helper (line 141) which uses a scalar temp-array fallback instead of proper `_mm512_mask_i32gather_ps`, combined with potential `-ffast-math` interaction with `_mm512_fmadd_ps` accumulation order. Fix and re-enable AVX-512 dispatch.

**Strategy**: Build a dedicated test harness in `/tmp/test_avx512.c` that compares AVX-512 vs scalar conv1d kernel outputs on controlled test vectors. Fix identified bugs in `gather16` and accumulation paths. Verify bit-exact (or near-exact) output. Remove the scalar-force override in `get_ops()`. Run full decoder verify no regression.

## Tasks

### T1: Build AVX-512 vs Scalar Comparison Harness
- Create `/tmp/test_avx512.c` that:
  1. Allocates random input tensor `x[Ci*T]` with known seed (e.g., srand(42))
  2. Allocates weight tensor `w[Co*Ci*K]` with small uniform values [-0.1, 0.1]
  3. Calls `conv1d_s()` (scalar, from cpu_simd.inc) → reference output `o_ref`
  4. Calls `conv1d_avx512()` (from cpu_simd.inc) → test output `o_test`
  5. Computes per-element `max_abs_diff` and `RMS_diff` between o_ref and o_test
- Test matrix: `T={16,32,64}, K={7}, Ci={16,32,64}, Co={16,32}`
- Test specific edge: Ci not multiple of 16 (e.g., Ci=20, Ci=24, Ci=36) to test remainder path
- Test `-ffast-math` vs `-fno-fast-math` by compiling with and without (verify __AVX512F__ defined)
- **Look for**: The `gather16()` function at line 141-151 uses scalar load into temp array `tmp[16]` then `_mm512_loadu_ps`. Replace with proper `_mm512_i32gather_ps()` or `_mm512_mask_i32gather_ps()`:
  ```c
  // Current broken implementation:
  static inline __m512 gather16(const float *x, int ic_base, int ii, int T, int Ci) {
      float tmp[16];
      for (int k = 0; k < 16 && (ic_base + k) < Ci; k++)
          tmp[k] = x[(ic_base + k) * T + ii];
      for (int k = (Ci - ic_base); k < 16; k++)
          tmp[k] = 0.0f;
      return _mm512_loadu_ps(tmp);
  }
  ```
  Replace with:
  ```c
  static inline __m512 gather16(const float *x, int ic_base, int ii, int T, int Ci) {
      // Use AVX-512 masked gather
      __m512i idx = _mm512_set_ps(15*T, 14*T, 13*T, 12*T, 11*T, 10*T, 9*T, 8*T,
                                    7*T,  6*T,  5*T,  4*T,  3*T,  2*T,  1*T,  0*T);
      idx = _mm512_add_epi32(idx, _mm512_set1_epi32(ii));
      __mmask16 mask = (ic_base + 16 <= Ci) ? 0xFFFF : ((1 << (Ci - ic_base)) - 1);
      return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, idx, x + ic_base * T, 4);
  }
  ```

### T2: Fix the FMA Accumulation Accuracy Issue
- The `conv1d_avx512` kernel at line 364 uses `_mm512_fmadd_ps(x16, w16, sum16)` inside a triple-nested loop. Under `-ffast-math`, the compiler may reassociate FMA operations differently than the scalar kernel's sequential accumulation.
- **Fix approach**: Use `_mm512_fmadd_ps` with `#pragma STDC FP_CONTRACT ON` to ensure fused multiply-add is used consistently. Alternatively, accumulate in a `double`-precision reduction:
  ```c
  // After inner loops, convert sum16 to double for reduction:
  double sum_d = (double)_mm512_reduce_add_ps(sum16);
  float sum = (float)sum_d;
  ```
- Test with known input: create a case where `x[ic*T+ii] * w[oc*Ci*K + ic*K + j]` has alternating signs → FMA reassociation could produce different results.
- **Critical check**: The `hsum512_ps` at line 118 uses `_mm512_reduce_add_ps` which is a full-width horizontal add. Verify this matches scalar reduction on edge cases (large positive + large negative values).

### T3: Fix convt1d_avx512 (Transpose Convolution)
- `convt1d_avx512` at line 388 also uses `_mm512_fmadd_ps` for accumulation into output buffer.
- The transpose conv has a different access pattern (`o[ocb*To + oi] += v * w[...]`). The FMA is `ov = _mm512_fmadd_ps(vb, w16, ov)` followed by store back.
- **Issue**: With `-ffast-math`, the compiler may merge the load-FMA-store pattern into a single `vfmadd132ps` memory operand variant, potentially reading stale values if threading is used.
- **Fix**: Make the load-FMA-store sequence explicit:
  ```c
  __m512 ov = _mm512_loadu_ps(&o[ocb*To + oi]);
  ov = _mm512_fmadd_ps(vb, w16, ov);
  _mm512_storeu_ps(&o[ocb*To + oi], ov);
  ```
  Add `_mm512__storeu_ps` with `_mm256_zeroupper()` after the loop to prevent AVX-SSE transition penalties.
- Test with the same harness, comparing `convt1d_avx512` vs `convt1d_s`.

### T4: Remove Scalar Fallback Override + Enable AVX-512 Dispatch
- In `src/cpu_decoder.c`, `get_ops()` function (line 138):
  ```c
  /* NOTE: convt1d_avx512 and conv1d_avx512 both have suspected FMA bugs */
  ops.conv1d = conv1d_s;
  ops.conv_transpose1d = convt1d_s;
  return ops;
  ```
- **Change**: Remove lines 206-208 (the override). The dispatch logic at lines 143-156 already correctly selects AVX-512 kernels when `HAS(7, 1, 16)` is true and `__AVX512F__` is defined.
- Verify that after fix, `ops.conv1d == conv1d_avx512` and `ops.conv_transpose1d == convt1d_avx512` on AVX-512 hardware.
- Add a compile-time check:
  ```c
  #if defined(__AVX512F__) && !defined(ALLOW_AVX512)
  #error "AVX-512 kernels enabled but not verified — define ALLOW_AVX512 to proceed"
  #endif
  ```
  Remove this after verification passes.

### T5: Full Regression Test
- Build with `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
- Run `./build/tsac-ng -v d test-simples/P丸様。-自分後回し@A.txc /tmp/test_avx512_output.wav`
- Compare with reference output from scalar-only build (baseline before this round)
- Verify: output RMS within 0.001 of baseline, no NaN, no clipping > 1%
- Test with `-fno-fast-math` build to confirm numerical stability independent of compiler flags

## Acceptance Criteria
1. `gather16` uses proper AVX-512 masked gather (`_mm512_mask_i32gather_ps`), not scalar fallback
2. `conv1d_avx512` produces output within 1e-5 relative error of `conv1d_s` on all test vectors
3. `convt1d_avx512` produces output within 1e-5 relative error of `convt1d_s` on all test vectors
4. The scalar-force override at `cpu_decoder.c:207-208` is removed
5. `get_ops()` returns AVX-512 kernels when CPU supports AVX-512F
6. Full decoder WAV output matches scalar-only build (RMS diff < 0.001)
7. All builds (Release, Debug) compile without warnings on `-mavx512f` enabled toolchain
