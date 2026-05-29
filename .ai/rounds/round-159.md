# Round 159 — Deep Dive: RVQ Output Comparison (5 Formula Variants)
**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-158

## Strategy — WHY this round exists

The RVQ output (1024-dim feature vector, `rvq_out` at `cpu_decoder.c:396`) is the **very first step** of the DAC decoder. If the RVQ formula diverges from the original tsac, every downstream layer will amplify the error. The current implementation uses a two-step process — `in_proj` lookup (codebook index → 8-dim vector) → `out_proj` matrix multiply (8-dim → 1024-dim) — accumulated across 12 codebooks. However, the original tsac's `libnc` may use a different formulation: fused in_proj+out_proj, different matmul order, per-codebook scaling, or a different codebook layout (shape [1024,8] vs [8,1024] orientation). This round tests **5 RVQ formula variants** against the GDB-captured `gdb_model0_input_9216f32.bin` (the true RVQ output that feeds model.0 conv1d), and against the 6 captured `libnc_inproj_cb*_f32.bin` in_proj weights. The variant with the highest correlation against GDB ground truth is the correct RVQ formulation.

## Tasks

### T1: Create standalone RVQ test harness (⬜)
- Create file: `experimental/test_rvq_variants.py`

```python
#!/usr/bin/env python3
"""
Test 5 RVQ formula variants against GDB-captured model.0 input.
Compare: implementation from cpu_decoder.c lines 396-444 vs alternatives.

GDB ground truth: docs/evidence/gdb_model0_input_9216f32.bin
  - 9216 float32 values = 1024 dims × 9 frames
  - This is the ACTUAL RVQ output the original tsac feeds into model.0 conv1d

LibNC captured in_proj weights: docs/evidence/libnc_inproj_cb{0-5}_f32.bin
  - 8192 float32 values each = 1024 entries × 8 dims
  - Native format from LD_PRELOAD capture, may have different layout than our model loader

Model tensors from TSAC model file (loaded via model_loader):
  - quantizer.quantizers.{0-11}.in_proj.weight_v [1024, 1, 8]
  - quantizer.quantizers.{0-11}.out_proj.weight_v [8, 1, 1024]
  - quantizer.quantizers.{0-11}.codebook.weight [1024, 1024] or [n_entries, dim]
"""
import numpy as np
import json, os, sys, struct

# Paths
GDB_PATH = "docs/evidence/gdb_model0_input_9216f32.bin"
LIBNC_IP_PATTERN = "docs/evidence/libnc_inproj_cb{}_f32.bin"
MODEL_PATH = "/usr/share/tsac/dac_stereo_q8.bin"
TXC_PATH = "test-simples/short_fast.txc"

def load_gdb_ground_truth():
    """Load GDB-captured model.0 input (9216 float32)."""
    data = np.fromfile(GDB_PATH, dtype=np.float32)
    assert len(data) == 9216, f"Expected 9216 floats, got {len(data)}"
    return data.reshape(1024, -1)  # (1024, 9)

def load_libnc_inproj(cb_idx):
    """Load LD_PRELOAD-captured in_proj weight for codebook idx (0-5)."""
    path = LIBNC_IP_PATTERN.format(cb_idx)
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.float32)
    # 8192 = 1024 entries × 8 dims
    return data.reshape(1024, 8)  # (entries, dim)

# ... rest in T2-T6
```

- Define main function that loads:
  1. GDB ground truth (1024, 9)
  2. TXC codebook indices for the same 9 frames (from TXC file)
  3. Model codebook weights (from .bin model file)
- **Expected**: Script runs and loads all data without errors.
- **Verification**: `python3 experimental/test_rvq_variants.py --load-only` prints shapes of all loaded data.

### T2: Implement Variant A — Current Implementation (⬜)
- **Variant A (current code)**: Exactly as `cpu_decoder.c` lines 396-444:
  ```python
  def variant_A(codes, in_proj_list, out_proj_list, n_frames, n_cb=12):
      """
      Current implementation:
      For each codebook cb:
        - in_proj: ip_f32[raw * Co + o] — lookup row `raw`, get 8 values
        - out_proj: sum over o of ip_vec[o] * op_f32[o * op_Co + d]
        - Accumulate: rvq_out[d, f] += sum
      Weight layouts:
        in_proj: shape (1024, 8) = [entries, dim]
        out_proj: shape (8, 1024) = [dim, features]  
      """
      rvq_dim = 1024
      rvq_out = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          ip = in_proj_list[cb]  # (1024, 8) or None
          op = out_proj_list[cb] # (8, 1024) or None
          if ip is None or op is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, ip.shape[0] - 1))
              ip_vec = ip[raw, :]  # (8,)
              rvq_out[:, f] += ip_vec @ op  # (8,) @ (8, 1024) = (1024,)
      return rvq_out
  ```

- **Test against GDB**:
  ```python
  gdb = load_gdb_ground_truth()  # (1024, 9)
  our_vA = variant_A(codes, ips, ops, n_frames=9)
  corr_vA = np.corrcoef(gdb.flatten(), our_vA.flatten())[0,1]
  print(f"Variant A corr: {corr_vA:.6f}")
  ```

- **Expected**: Variant A correlation with GDB data — measure and record.
- **Verification**: If corr < 0.5, the current RVQ formula is wrong.

### T3: Implement Variant B — Fused in_proj + out_proj (⬜)
- **Variant B (fused weights)**: Pre-multiply in_proj × out_proj per codebook:
  ```python
  def variant_B(codes, in_proj_list, out_proj_list, n_frames, n_cb=12):
      """
      Pre-fused: W_fused[c] = in_proj[c] @ out_proj[c] for codebook c
      Shape: W_fused[c] = (1024, 8) @ (8, 1024) = (1024, 1024)
      Then: rvq_out[:, f] += W_fused[raw, :] for each codebook
      This is mathematically equivalent to Variant A but tests if
      libnc pre-computes fused weight matrices.
      """
      rvq_dim = 1024
      fused_mats = []
      for cb in range(min(n_cb, 12)):
          ip = in_proj_list[cb]
          op = out_proj_list[cb]
          if ip is None or op is None:
              fused_mats.append(None)
              continue
          # (1024, 8) @ (8, 1024) = (1024, 1024)
          fused = ip @ op
          fused_mats.append(fused)
      
      rvq_out = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          if fused_mats[cb] is None: continue
          W = fused_mats[cb]
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, W.shape[0] - 1))
              rvq_out[:, f] += W[raw, :]
      return rvq_out
  ```

- **Test against GDB**: Compute correlation. Compare with Variant A.
- **Expected**: Should be identical to Variant A (mathematically equivalent), corr_diff < 1e-6.
- **Verification**: `np.allclose(variant_A_result, variant_B_result)` should be True.

### T4: Implement Variant C — Direct Codebook Lookup (no in_proj/out_proj split) (⬜)
- **Variant C (direct codebook lookup)**: The model also has `codebook.weight` for each quantizer:
  ```python
  def variant_C(codes, codebook_list, n_frames, n_cb=12):
      """
      Direct codebook lookup: each quantizer has a codebook.weight tensor
      of shape (1024, 1024) — 1024 entries, each 1024-dim.
      This completely bypasses in_proj/out_proj.
      If this variant matches GDB, then libnc uses direct codebook lookup,
      not the in_proj→out_proj decomposition.
      """
      rvq_dim = 1024
      rvq_out = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          cb_data = codebook_list[cb]  # (1024, 1024) or None
          if cb_data is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, cb_data.shape[0] - 1))
              rvq_out[:, f] += cb_data[raw, :]
      return rvq_out
  ```
  
- **Load codebook weights**: The model stores codebook weights under:
  ```python
  # From model_loader, tensor name: "quantizer.quantizers.{cb}.codebook.weight"
  # Typical shape: [1024, 1024] — but may vary. Check actual dims.
  cb_dims = model_get_tensor_dims(f"quantizer.quantizers.{cb}.codebook.weight")
  ```
  
- **Sanity check**: If codebook dims are [1024, 1024], this is 12×1024×1024 = 12M floats = 48MB — feasible.
  If they're [1024, 8], then it's the same as in_proj.

- **Expected**: Compare corr with GDB. If corr > 0.9, the direct codebook approach is correct and the in_proj/out_proj decomposition is unnecessary.
- **Verification**: Corr_V_C recorded alongside V_A and V_B.

### T5: Implement Variant D — LibNC Native In_Proj Weights (⬜)
- **Variant D (libnc native weights)**: Use the LD_PRELOAD-captured `libnc_inproj_cb{0-5}_f32.bin` files instead of our model-loaded weights:
  ```python
  def variant_D(codes, libnc_inproj_list, out_proj_list, n_frames, n_cb=6):
      """
      Use libnc's native in_proj weights (captured via LD_PRELOAD) instead of
      our BF8-dequantized weights. Only 6 codebooks available (cb0-cb5).
      The layout may differ from our model loader's output format.
      """
      rvq_dim = 1024
      rvq_out = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, len(libnc_inproj_list))):
          libnc_ip = libnc_inproj_list[cb]  # (1024, 8) — libnc native format
          op = out_proj_list[cb]             # (8, 1024) — from our model loader
          if libnc_ip is None or op is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, libnc_ip.shape[0] - 1))
              ip_vec = libnc_ip[raw, :]
              rvq_out[:, f] += ip_vec @ op
      return rvq_out
  ```
  
- **Analyze libnc in_proj layout**:
  ```bash
  python3 -c "
  import numpy as np
  for cb in range(6):
      path = f'docs/evidence/libnc_inproj_cb{cb}_f32.bin'
      data = np.fromfile(path, dtype=np.float32)
      print(f'cb{cb}: {data.shape} min={data.min():.4f} max={data.max():.4f} mean={data.mean():.6f} std={data.std():.6f}')
      # Check if values are similar to our dequantized weights
  "
  ```
  
- **Cross-validate with our in_proj**: Load our dequantized in_proj for cb0 and compare:
  ```python
  # Requires compiling and running test_minimal or similar to dump weights:
  # ./build/test_minimal_dump_weights
  our_ip_cb0 = np.fromfile('/tmp/inproj_cb0_ours.bin', dtype=np.float32).reshape(1024, 8)
  libnc_ip_cb0 = np.load('docs/evidence/libnc_inproj_cb0_f32.bin').reshape(1024, 8)
  corr_weights = np.corrcoef(our_ip_cb0.flatten(), libnc_ip_cb0.flatten())[0,1]
  print(f'Weight correlation (ours vs libnc): {corr_weights:.6f}')
  ```

- **Expected**: Variant D tests whether the weight format difference (our BF8 dequant vs libnc native) causes the RVQ divergence.
- **Verification**: If corr_V_D > corr_V_A significantly, the BF8 dequant format is the root cause.

### T6: Implement Variant E — Per-Codebook Scaling / Alternate Matmul (⬜)
- **Variant E (scaled+fused)**: Test hypotheses about per-codebook scaling, signed accumulation, or alternate matmul order:
  ```python
  def variant_E(codes, in_proj_list, out_proj_list, codebook_list, n_frames, n_cb=12):
      """
      Variant E1: in_proj lookup uses transposed layout
        ip_f32[o * Ci + raw] instead of ip_f32[raw * Co + o]
        (See round-145: 'transpose_fix' finding — old access was transposed)
      
      Variant E2: out_proj transpose: op @ ip_vec instead of ip_vec @ op
        rvq_out[:, f] += op @ ip_vec  (uses (1024, 8) @ (8,) instead of (8,) @ (8, 1024))
      
      Variant E3: Per-codebook learned scale
        scale[cb] = some_value; rvq_out += scale * result
        (Test scale = 1/12, 1/sqrt(12), or learned from data)
      
      Variant E4: Fused codebook matmul
        Use codebook.weight directly but with sign flip or offset
      
      Returns dict of all sub-variant results.
      """
      results = {}
      rvq_dim = 1024
      
      # E1: Transposed in_proj access
      rvq_e1 = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          ip = in_proj_list[cb]
          op = out_proj_list[cb]
          if ip is None or op is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, ip.shape[0] - 1))
              # Transposed: was ip[raw, :], now try ip[:, raw]
              # But ip is (1024, 8), so ip[:, raw] = (1024,) — wrong dims.
              # Actually the old code (before fix) was: ip_f32[raw * Co + o]
              # which reads 8 consecutive values starting at raw*8.
              # This is the same as ip[raw, :] in our layout.
              # But if the ACTUAL libnc layout is [8, 1024], then ip_f32[raw * Co + o]
              # reads ip_f32[raw*8 + 0..7] = one element from each of 8 rows at column raw.
              # So test: ip = ip.T  (transpose to (8, 1024))
              ip_vec = ip.T[:, raw]  # (8, 1024)[:, raw] = (8,)
              rvq_e1[:, f] += ip_vec @ op  # (8,) @ (8, 1024) = (1024,)
      results['E1_transposed_ip'] = rvq_e1
      
      # E2: Out-proj transposed
      rvq_e2 = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          ip = in_proj_list[cb]
          op = out_proj_list[cb]
          if ip is None or op is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, ip.shape[0] - 1))
              ip_vec = ip[raw, :]  # (8,)
              # Transposed out: op is (8, 1024), op.T is (1024, 8)
              # ip_vec (8,) @ op.T (1024, 8) doesn't work.
              # Try: op (1024, 8) @ ip_vec (8,) = (1024,)
              op_T = op.T  # (1024, 8)
              rvq_e2[:, f] += op_T @ ip_vec  # (1024, 8) @ (8,) = (1024,)
      results['E2_transposed_op'] = rvq_e2
      
      # E3: Uniform scale 1/n_cb
      if 'variant_A_result' in dir():  # if we have V_A
          rvq_e3 = variant_A_result / n_cb
          results['E3_scaled_1_over_ncb'] = rvq_e3
      
      # E4: Both transposed (IP + OP)
      rvq_e4 = np.zeros((rvq_dim, n_frames), dtype=np.float32)
      for cb in range(min(n_cb, 12)):
          ip = in_proj_list[cb]
          op = out_proj_list[cb]
          if ip is None or op is None: continue
          for f in range(n_frames):
              raw = int(codes[f * n_cb + cb])
              raw = max(0, min(raw, ip.shape[0] - 1))
              ip_vec = ip.T[:, raw]
              op_T = op.T
              rvq_e4[:, f] += op_T @ ip_vec
      results['E4_both_transposed'] = rvq_e4
      
      return results
  ```

- **Test all E sub-variants** and report best correlation:
  ```python
  gdb = load_gdb_ground_truth()
  best_corr = -1
  best_name = None
  for name, result in variant_E_results.items():
      corr = np.corrcoef(gdb.flatten(), result.flatten())[0,1]
      print(f"  {name}: corr={corr:.6f}")
      if corr > best_corr:
          best_corr = corr
          best_name = name
  print(f"Best variant E: {best_name} with corr={best_corr:.6f}")
  ```

- **Expected**: One sub-variant may show significantly higher correlation (e.g., E1 with transposed IP may match the old "buggy" code that had higher RMS).
- **Verification**: Record best E variant in JSON report.

### T7: Run all variants and produce summary report (⬜)
- Run full comparison:
  ```bash
  python3 experimental/test_rvq_variants.py --all --output /tmp/r159_rvq_report.json
  ```

- The script should produce a summary table:
  ```
  ========================================
  RVQ Variant Comparison Summary
  ========================================
  Variant     Corr     RMSE     MaxDiff   Notes
  A (current) 0.1234   0.5678   2.3456    Current implementation
  B (fused)   0.1234   0.5678   2.3456    Should match A
  C (direct)  0.0102   1.2345   5.6789    Direct codebook
  D (libnc)   0.2345   0.4567   1.8901    LibNC native weights
  E1          0.3456   0.3456   1.2345    Transposed IP
  E2          0.0101   1.1111   4.4444    Transposed OP
  E3          0.1234   0.5678   2.3456    Scaled
  E4          0.0056   1.5678   6.7890    Both transposed
  ========================================
  Best: E1 (corr=0.3456)
  ```

- **Decision gate**:
  ```python
  BEST = best_corr
  if BEST > 0.95:
      decision = "RVQ formula solved. Proceed to R160 for conv kernel verification."
  elif BEST > 0.5:
      decision = f"Partial RVQ match ({BEST:.4f}). Investigate remaining gap in R160."
  else:
      decision = f"RVQ still divergent (best corr={BEST:.4f}). Root cause is NOT RVQ formula. Proceed to R160 (conv kernel)."
  print(f"Decision: {decision}")
  ```

- **Expected**: Clear decision on whether RVQ is the root cause of WAV divergence.
- **Verification**: Report JSON written with all variant results and decision.

## Acceptance
- [ ] `experimental/test_rvq_variants.py` implemented with all 5 variants (+4 sub-variants)
- [ ] All variants execute without errors on test data
- [ ] GDB ground truth (9216 floats) loaded and used as reference
- [ ] Six libnc in_proj weights loaded and compared with our dequantized weights
- [ ] Summary table printed with correlations for all variants
- [ ] Best variant correlation against GDB recorded
- [ ] Decision rendered: "RVQ solved" (>0.95), "partial" (>0.5), or "not RVQ" (<0.5)
- [ ] Report JSON `/tmp/r159_rvq_report.json` written with all results
