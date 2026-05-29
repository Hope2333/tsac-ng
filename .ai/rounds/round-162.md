# Round 162 — Fix Convt Kernel Layout & convt1d_parallel (Header Dispatch)
**Signed**: Worker | **Date**: 2026-05-28 | **Status**: PENDING

## Summary
The 4 convtranspose layers (decoder blocks 1-4, model.{1,2,3,4}.block.1) have a GDB-confirmed weight access pattern of `[Co][K][Ci]` — meaning the native flat order from `dequant_weights` IS `[Co][K][Ci]` for convtranspose layers. The scalar `convt1d_s` kernel at `cpu_simd.inc:58-72` correctly accesses weights as `w[oc * K * Ci + j * Ci + ic]`. However, `convt1d_parallel` at `cpu_threads.inc:69-93` may have a threading issue: each thread writes to `o[oc*To+oi]` and the `memset(o, 0, ...)` in `convt1d_parallel` zeroes the ENTIRE output before threading starts, but each thread's `convt1d_thread` passes a sub-view `o + oc_start * To` which is correct. However, the SIMD kernels (`convt1d_avx`, `convt1d_avx512`) also zero output internally via `memset`. This double-zeroing is harmless but wasteful. The real issue is verifying that ALL 4 convtr layers use the `[Co][K][Ci]` layout and that our output matches libnc per-layer.

**Strategy**: Verify the convtranspose weight layout for all 4 layers by loading libnc override weights and comparing outputs. Audit the convtr access pattern in all kernel variants. Build per-layer comparison against libnc captured activations (from R103 LD_PRELOAD captures at `/tmp/lcc_*.bin`). Compare with reference convtr output from original tsac GDB trace.

## Tasks

### T1: Verify [Co][K][Ci] Weight Layout for ALL 4 Convtr Layers
- The 4 convtr layers and their tensor dims (from `dac_stereo_q8.bin`):
  - decoder.model.1.block.1: dims=[768, 11, 1536] → Co=768, K=11, Ci=1536, stride=5
  - decoder.model.2.block.1: dims=[384, 11, 768]  → Co=384, K=11, Ci=768,  stride=5
  - decoder.model.3.block.1: dims=[192, 11, 384]  → Co=192, K=11, Ci=384,  stride=5
  - decoder.model.4.block.1: dims=[96,  7,  192]  → Co=96,  K=7,  Ci=192,  stride=3
- **Validation**: For each layer, load the libnc override weight (float32 reference from R103, available at `/tmp/libnc_OVR_decoder_model_X_block_1_weight_v.bin`). Compare our dequantized weights with libnc's float32 reference. The `dequant_weights` function at `cpu_decoder.c:220-350` should output `[Co][K][Ci]` layout when `is_ct=1`.
- **Check the specific code path**:
  ```c
  // cpu_decoder.c:330-345 — output rearrangement
  for (int ci = 0; ci < Ci; ci++) {
      for (int k = 0; k < K; k++) {
          for (int co = 0; co < Co; co++) {
              if (is_ct) {
                  // convt: keep flat [Co][K][Ci] order (no rearrangement)
                  int src_idx = co * K * Ci + k * Ci + ci;
                  w_f32[src_idx] = src_f32[src_idx];
              } else { ... }
          }
      }
  }
  ```
- **Verify**: `src_f32` is already in the stored flat order from the model file. For convtr, the model stores in `[Co][K][Ci]` order. The `is_ct` branch copies without rearrangement — this is ONLY correct if `src_f32` is already `[Co][K][Ci]`. Confirm by checking: does the model file store `.weight_v` tensor data as `d0*d1*d2` values in row-major order where `d0=Co, d1=K, d2=Ci`? The libnc override data (elem_size=0 sentinel) stores in `[Co][Ci][K]` order per model_loader.c:130-133 — for convtr this must be converted to `[Co][K][Ci]`. The current code skips rearrangement for `is_ct` which assumes the source is already `[Co][K][Ci]`. **THIS IS A BUG FOR LIBNC OVERRIDE**: When `elem_size=0` (libnc override), the data is stored as `[Co][Ci][K]` but the `is_ct` branch does NO rearrangement, leaving it as `[Co][Ci][K]` instead of the required `[Co][K][Ci]`.
- **Fix the libnc override case for convtr**: When `is_ct=1` and `elem_size==0` (libnc override), we MUST rearrange from `[Co][Ci][K]` → `[Co][K][Ci]`:
  ```c
  if (is_ct && weight_v->elem_size == 0) {
      // LibNC override stores [Co][Ci][K], convert to [Co][K][Ci]
      // src_f32 was loaded directly from override, need to rearrange
      float *tmp = malloc(total_size * sizeof(float));
      memcpy(tmp, w_f32, total_size * sizeof(float)); // backup [Co][Ci][K]
      for (int co = 0; co < Co; co++)
          for (int ci = 0; ci < Ci; ci++)
              for (int k = 0; k < K; k++)
                  w_f32[co * K * Ci + k * Ci + ci] = tmp[co * Ci * K + ci * K + k];
      free(tmp);
  }
  ```

### T2: Fix convt1d_parallel Threading + memset Issue
- Current `convt1d_parallel` at `cpu_threads.inc:69-93`:
  ```c
  static void convt1d_parallel(...) {
      memset(o, 0, (size_t)Co * To * sizeof(float));  // FULL zero before threading
      if (n_threads <= 1 || Co < n_threads * 2) {
          kern(o, x, w, Ti, To, K, Ci, Co, stride);
          return;
      }
      // ... thread dispatch with o + oc_start*To view ...
  }
  ```
- **Problem 1**: The `convt1d_s` kernel ALSO does `memset(o, 0, ...)` internally (line 63). When called from `convt1d_parallel`, the thread worker calls `kern(o + oc_start * To, ...)` which zeroes only its slice. This is fine.
- **Problem 2**: SIMD kernels `convt1d_avx`/`convt1d_avx512` also zero internally with `memset(o, 0, Co*To*sizeof(float))`. When called from `convt1d_parallel` with a sub-view, this zeroes only the sub-view — but the SIMD kernel uses `Co_block = Co & ~7` which is the FULL Co of the sub-view, not the original Co. This is correct as long as the sub-view's Co matches the kernel's Co parameter.
- **Fix**: Remove the `memset` from `convt1d_parallel` and let each kernel handle its own zeroing. This avoids double-zeroing. Update the comment to document this contract.
- **Thread Safety**: Verify that `convt1d_avx` FMA load-add-store is atomic within each thread's Co slice. Each thread owns exclusive `[oc_start*To : oc_end*To)` output range — no overlap, so thread-safe.

### T3: Compare Convtr Output with libnc GDB Captures
- Use the libnc override mechanism to inject exact float32 weights for ALL 4 convtr layers.
- Place override files at `/tmp/libnc_OVR_decoder_model_{1,2,3,4}_block_1_weight_v.bin`
- Run our decoder (with scalar kernel) on a short TXC file and capture convtr outputs via `DUMP_ACT`.
- Create a Python comparison script that:
  1. Loads our dumped activation (binary float32)
  2. Loads libnc reference activation from GDB trace (if available) or from a reference run
  3. Computes per-output-channel correlation
  4. Reports per-layer: `max_abs_diff`, `mean_abs_diff`, `correlation`
- **Expected**: After fixing the libnc override layout (T1), convtr layer outputs should match libnc reference with correlation > 0.99.

### T4: Verify Convtr Stride = K/2 for All Layers
- The stride computation at `cpu_blocks.inc:39` uses `conv_stride = conv_K / 2`. This was GDB-confirmed for the general case.
- Verify for each of the 4 layers:
  - model.1.block.1: K=11 → stride=5 (correct, GDB trace at docs/evidence/)
  - model.2.block.1: K=11 → stride=5
  - model.3.block.1: K=11 → stride=5
  - model.4.block.1: K=7 → stride=3
- **Check**: `n_frames_out = cur_frames * conv_stride` at `cpu_blocks.inc:40`. The DAC convtranspose expects output length = input_length * stride. This differs from typical convtranspose which uses `output = (input-1)*stride + K - 2*P`. Verify with GDB trace that the DAC's convtranspose is indeed a simple nearest-neighbor upsampling conv (not standard transposed conv).
- **Edge case**: When `P = K/2`, the formula `output = (input-1)*stride + K - 2*P = (input-1)*stride + K - K = (input-1)*stride`. With stride=K/2, this gives `(input-1)*K/2`. But our code uses `input * K/2`. The difference is K/2 frames at the end. **This could cause the decoder to read OOB or produce shifted output**. Verify which formula libnc uses by checking GDB trace for convtr output sizes.

## Acceptance Criteria
1. All 4 convtr layers verified with `[Co][K][Ci]` weight layout — libnc override weight comparison
2. LibNC override path fixed: `elem_size=0` convtr data properly rearranged `[Co][Ci][K]`→`[Co][K][Ci]`
3. Convtr output sizes verified against GDB trace (no OOB, correct temporal dimension)
4. No double-memset in convt1d_parallel path
5. Per-layer convtr output correlation with libnc > 0.99
6. Build clean, no regressions in decoder output (compare with scalar-only baseline)
