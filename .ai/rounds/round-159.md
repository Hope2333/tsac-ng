# Round 159 — Complete Per-Layer Correlation Heatmap (Phase 4A)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
With R157 activation dumps and R158 GDB reference captures, compute correlation for ALL 32+4 DAC decoder layers (not just the 7 key points). Generate a comprehensive correlation heatmap to identify every divergence hotspot quantitatively.

**Context**: R156 fixed the RVQ 50× scale error, but residual errors remain:
- RVQ RMS improved to 0.0644 (target ~0.03-0.06) — close
- m6_pre_tanh RMS improved to 1.81 (target <3) — within spec
- WAV corr remains ~0 — indicating non-scale errors accumulate in later layers

**Hypothesis**: Layer-by-layer correlation reveals:
1. Some layers have corr > 0.95 (conv1d with small K, snake)
2. Some layers have corr < 0.5 (convt layers, dilated convs)
3. Divergence accelerates after a specific block (error compounding)

## Round Strategy
1. Extend `experimental/compare_activations.py` to compare all 32+4 layers
2. For inner layers without GDB reference, compare our own conv1d vs convt layers at same dims (self-consistency check)
3. Generate comprehensive heatmap of ALL layers (not just 7)
4. Generate per-layer error statistics (mean, std, skew, kurtosis of error distribution)
5. Identify exact function and line of first significant divergence

## Tasks

### T1: Extend compare_activations.py for full 32+4 layer list
**File**: `experimental/compare_activations.py` function `main()` lines 114-123

**Action**: Replace the 7-layer LAYERS list with complete 36-layer list:

```python
LAYERS = [
    # Core path (7 layers)
    ("rvq_out", (1024, 16), "gdb_model0_input.bin"),
    ("m0_conv1d", (1536, 16), "gdb_m0_conv1d.bin"),
    ("block1_convt", (768, 128), "gdb_block1_convt.bin"),
    ("block2_convt", (384, 512), "gdb_block2_convt.bin"),
    ("block3_convt", (192, 1024), "gdb_block3_convt.bin"),
    ("block4_convt", (96, 2048), "gdb_block4_convt.bin"),
    ("m6_pre_tanh", (2, 4096), "gdb_m6_pre_tanh.bin"),
    # Inner layers block 1
    ("b1_snake2", None, None),        # block.1.block.2.block.0.alpha
    ("b1_conv_dil1", None, None),     # block.1.block.2.block.1
    ("b1_snake3", None, None),        # block.1.block.3.block.0.alpha
    ("b1_conv_dil3", None, None),     # block.1.block.3.block.1 (dilation=3)
    ("b1_snake4", None, None),        # block.1.block.4.block.0.alpha
    ("b1_conv_dil9", None, None),     # block.1.block.4.block.1 (dilation=9)
    ("b1_add4", None, None),          # block.1.block.4 output (skip-add)
    # Inner layers block 2 (repeat 11 layers)
    ("b2_snake2", None, None),
    ...
    # Inner layers block 3
    ...
    # Inner layers block 4
    ...
    ("b4_add4", None, None),
    # Tail
    ("m5_snake", None, None),         # model.5 snake output
]
```

**Acceptance**: Script handles layers with and without GDB references gracefully.

### T2: Add per-layer error distribution statistics
**File**: `experimental/compare_activations.py` — new function `error_statistics()`

**Action**: For each compared layer, compute:
```python
error = ours - reference
stats = {
    "mean_error": float(np.mean(error)),
    "std_error": float(np.std(error)),
    "skewness": float(scipy.stats.skew(error.flatten())),  # if scipy available
    "kurtosis": float(scipy.stats.kurtosis(error.flatten())),
    "pct_outliers": float(np.mean(np.abs(error) > 3*np.std(error))),
    "error_rms": float(np.sqrt(np.mean(error**2))),
    "error_vs_signal_db": float(20*np.log10(np.sqrt(np.mean(reference**2)) / (np.sqrt(np.mean(error**2)) + 1e-30))),
}
```

**Interpretation rules**:
- `skewness` near 0 = symmetric error (scale/bias issue)
- `skewness` far from 0 = structural error (wrong access pattern)
- `outliers > 5%` = numerical stability problem (NaN precursors)

**Acceptance**: JSON report includes per-layer error distribution statistics.

### T3: Generate comprehensive heatmap (36-layer version)
**File**: `experimental/compare_activations.py` function `generate_heatmap()` line 55

**Action**: Create two heatmaps:
1. **7-layer core heatmap** — `experimental/correlation_heatmap_core.png` (same as R157)
2. **36-layer full heatmap** — `experimental/correlation_heatmap_full.png` with:
   - Rows: correlation, MAE, RMSE, SNR, outliers%, error_std
   - Columns: all 36 layers grouped by block
   - Color-scaled: RdYlGn for correlation, RdBu_r for error metrics
   - Annotations on each cell

**Acceptance**: Two heatmap PNGs generated and visually interpretable.

### T4: Self-consistency test — compare same-shape layers
**File**: `experimental/compare_activations.py` — new test

**Action**: For layers with identical dims (e.g., block.1.snake2 and block.2.snake2 both have shape (768, 128)), cross-compare to verify:
1. If our block.1 and block.2 snakes produce statistically similar distributions
2. If our dilated convs at same dims produce similar scaling to libnc

This catches cases where the layer IS correct but the GDB reference is mis-identified.

```python
def self_consistency(metrics_dict):
    """Verify layers with identical dims produce similar statistics."""
    by_dims = {}
    for name, metrics in metrics_dict.items():
        shape = tuple(metrics.get("shape", []))
        if shape:
            by_dims.setdefault(shape, []).append((name, metrics))
    for shape, layers in by_dims.items():
        if len(layers) >= 2:
            corrs = [m.get("correlation", 0) for n, m in layers]
            if max(corrs) - min(corrs) > 0.1:
                print(f"  [WARN] {shape}: corr range {min(corrs):.3f}-{max(corrs):.3f} "
                      f"between {[n for n,m in layers]}")
```

**Acceptance**: Self-consistency check passes or identifies new anomalies.

### T5: Divergence root cause analysis
**File**: `experimental/activation_comparison.json` → updated with analysis

**Action**: With full correlation matrix, determine:

1. **First divergence layer**: The earliest layer where corr < 0.95
2. **Divergence acceleration point**: Layer where corr drops below 0.5
3. **Suspect function(s)**: Map divergence layer to code:
   - If corr drops at model.0 → `conv1d_*` in `src/cpu_simd.inc`
   - If corr drops at block.N convt → `convt1d_*` in `src/cpu_simd.inc`
   - If corr drops at snake layer → `snake_*` in `src/cpu_simd.inc`
   - If corr drops at inner conv → `conv1d_dilated_s` in `src/cpu_decoder.c`

**Acceptance**: JSON report includes `recommended_action` field with exact function name and file to fix.

## Acceptance Criteria
- [ ] 36-layer activation comparison computed
- [ ] Per-layer error distribution statistics in JSON report
- [ ] Two heatmap PNGs generated (core + full)
- [ ] Self-consistency check run and documented
- [ ] Divergence root cause analysis with exact function/file recommendation
- [ ] Report guides R160-R164 priority (which kernel to fix first)
