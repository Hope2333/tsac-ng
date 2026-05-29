# Round 162 — Fix Convt Kernel [Co][K][Ci] Access Pattern (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
The convtranspose1d kernels (`convt1d_avx512`, `convt1d_avx2`, `convt1d_s` in `src/cpu_simd.inc` and `src/cpu_threads.inc`) access weights in `[Co][K][Ci]` layout. GDB confirmed this is correct for libnc (R134-R145). However, the dequant_weights function outputs `[Co][K][Ci]` for is_ct=1 (convtranspose) layers by default — so the access pattern matches.

**The actual issue**: R159 may show convt layers (block.1 through block.4) as the first divergence point. If so, the problem is NOT the access pattern (which is verified) but one of:
1. **Weight order within `[Co][K][Ci]`**: The BF8 dequant may produce weights in a different element order than libnc expects
2. **Stride handling**: convt stride = K/2 may be computed differently
3. **Bias application**: Bias is added AFTER convt, may use wrong channel ordering
4. **Output accumulation order**: The `memset(o, 0, ...)` + `o[...] += v * w` pattern may produce different FP rounding than libnc's single-pass approach

**Number of convt layers in DAC**: 4 (block.1 through block.4), each with K=16, stride=8, upsampling by 8× per block.

## Round Strategy
1. Verify all 4 convt layers individually against GDB reference
2. Test weight rearrangement: compare `[Co][K][Ci]` vs `[Ci][K][Co]` layouts
3. Test stride formula: verify K/2 matches libnc's stride
4. Test bias application order (pre-FMA vs post-FMA)
5. Compare convolution output with libnc at identical input+weights

## Tasks

### T1: GDB capture convt weights for all 4 blocks
**File**: `docs/evidence/gdb_convt_weights_block*.bin` (new captures)

**Action**: Extend GDB capture (R158) to dump weight tensors for all 4 convt layers:
- `decoder.model.{1,2,3,4}.block.1.weight_v` (BF8 format)
- `decoder.model.{1,2,3,4}.block.1.weight_g` (scale bytes)
- `decoder.model.{1,2,3,4}.block.1.bias` (float32)

GDB capture approach:
```gdb
# Break on nc_load to capture weight tensors by name
break nc_load
commands
  silent
  set $name_ptr = *(long*)($rdi + 0x20)
  if $name_ptr != 0
    set $name = $name_ptr
    printf "nc_load: %s\n", $name
  end
  continue
end
```

Or use LD_PRELOAD intercept:
Capture via modified `docs/evidence/libnc_preload.so` that logs tensor name + data to file.

**Acceptance**: Weight tensors for all 4 convt layers captured as float32 dumps.

### T2: Compare our dequant_weights output with libnc raw weights
**File**: `src/cpu_decoder.c` function `dequant_weights()` lines 220-369

**Action**: For each convt layer:
1. Load GDB-captured raw BF8 weights (`docs/evidence/gdb_convt_wv_block1.bin`)
2. Run our `dequant_weights()` on them
3. Load GDB-captured float32 weights (libnc's own dequant output)
4. Compute correlation between our dequant output and libnc's

```python
for block in [1, 2, 3, 4]:
    our_w = np.fromfile(f'/tmp/our_convt_block{block}_weights.bin', dtype=np.float32)
    libnc_w = np.fromfile(f'docs/evidence/gdb_convt_block{block}_weights_f32.bin', dtype=np.float32)
    n = min(len(our_w), len(libnc_w))
    corr = np.corrcoef(our_w[:n], libnc_w[:n])[0, 1]
    rmse = np.sqrt(np.mean((our_w[:n] - libnc_w[:n])**2))
    print(f'Block {block} weights: corr={corr:.6f} rmse={rmse:.6f}')
    if corr < 0.95:
        # Reshape and check pattern
        our_r = our_w.reshape(Co, K, Ci)
        libnc_r = libnc_w.reshape(Co, K, Ci)
        # Check if the issue is element order (transpose needed?)
        our_T = our_w.reshape(Ci, K, Co).transpose(2, 1, 0).flatten()
        corr_T = np.corrcoef(our_T[:n], libnc_r.flatten()[:n])[0, 1]
        print(f'  After transpose [Ci,K,Co]→[Co,K,Ci]: corr={corr_T:.6f}')
```

**Acceptance**: Weight correlation > 0.99 for all blocks, OR precise identification of which rearrangement is needed.

### T3: Compare convt output for all 4 blocks
**File**: `experimental/test_convt_kernels.py` (new)

**Action**: Create test harness that:
1. Loads GDB reference input + weights + output for each block
2. Runs our convt kernel (scalar) with identical input
3. Computes output correlation vs libnc output

```python
def test_convt_block(block, Ti=9, K=16, Ci=768, Co=384, stride=8):
    """Test block convt against GDB reference."""
    # Load reference
    x = np.fromfile(f'docs/evidence/gdb_block{block}_input.bin', dtype=np.float32).reshape(Ci, Ti)
    w = np.fromfile(f'docs/evidence/gdb_convt_block{block}_weights.bin', dtype=np.float32)
    ref = np.fromfile(f'docs/evidence/gdb_block{block}_convt.bin', dtype=np.float32)
    b = np.fromfile(f'docs/evidence/gdb_convt_block{block}_bias.bin', dtype=np.float32)

    # Run our kernel via C test harness
    # ... (compile C code, run, load output)

    our = np.fromfile(f'/tmp/convt_block{block}_our.bin', dtype=np.float32)
    corr = np.corrcoef(our.flatten(), ref.flatten())[0, 1]
    return corr
```

**Acceptance**: Each block's convt output correlation measured.

### T4: Test alternative weight layouts and bias application
**File**: `src/cpu_simd.inc` convt kernels (lines 320-429), `src/cpu_blocks.inc` lines 46-56

**Action**: Test hypotheses in order:

**Hypothesis A** — Weight layout needs transposition:
```python
# Current: w_f32 layout [Co][K][Ci] 
# Alternative: w_f32 layout [Co][Ci][K] (conv1d-like but with convt stride)
```

**Hypothesis B** — Bias should be applied within the FMA loop, not after:
```c
// Current (cpu_blocks.inc:49-55):
if (b_data) {
    for (int c = 0; c < conv_Co; c++)
        for (int t = 0; t < n_frames_out; t++)
            next_buf[c * n_frames_out + t] += b_data[c];
}

// Alternative: bias integrated in kernel
o[oc*To+oi] = b[oc] + sum;  // inside convt kernel
```

**Hypothesis C** — Stride should be fixed K/2, not computed per-dim:
```c
// Current: int stride = K/2;
// libnc might use different stride for some layers
```

**Acceptance**: Which hypothesis (A, B, or C) produces corr > 0.99.

### T5: Fix and verify all 4 convt layers
**File**: `src/cpu_simd.inc` and/or `src/cpu_blocks.inc`

**Action**: Based on T4 findings, apply fix:
- If access pattern wrong: fix `dequant_weights()` output layout for convt layers
- If bias wrong: move bias into kernel
- If stride wrong: fix stride computation

Then re-verify:
```bash
cmake --build build
./build/tsac-ng -v -f d /tmp/short_fast.txc /tmp/out.wav
python3 experimental/compare_activations.py
```

**Expected**: Block{1,2,3,4}_convt correlations improve from current to > 0.95.

**Acceptance**: All 4 convt layer outputs correlate > 0.95 with libnc reference.

## Acceptance Criteria
- [ ] GDB weight captures for all 4 convt layers
- [ ] Weight dequant correlation > 0.99 for all blocks
- [ ] Convt output correlation measured for each block
- [ ] Fix applied based on hypothesis testing (A/B/C)
- [ ] All 4 convt outputs correlate > 0.95 with libnc
- [ ] Downstream layers (snake, m6) show corresponding improvement
