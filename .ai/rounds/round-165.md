# Round 165 — Full Re-Verification After All Kernel & Layout Fixes (Header Dispatch)
**Signed**: Worker | **Date**: 2026-05-28 | **Status**: PENDING

## Summary
After executing R161 (AVX-512 fix), R162 (convt layout), R163 (is_ct detection), and R164 (snake/tanh alignment), this round performs comprehensive end-to-end verification. We re-compare ALL 32 decoder layers + 4 convtr layers with libnc captured activations, re-measure RMS/correlation on multiple test files, quantify improvements from each individual fix, and document residual issues. This is the G2 decision gate (from Phase 4 roadmap): if any layer has corr < 0.9 after all fixes, we iterate through R163 (kernel fixes) again.

**Strategy**: Build a Python or C-based layer-by-layer comparison framework. Capture activations at every layer output using the existing `DUMP_ACT` infrastructure. Compare against libnc reference captures (from R156-R160 GDB work). Create a correlation heatmap. Measure final WAV correlation improvement. Document everything.

## Tasks

### T1: Build Per-Layer Activation Capture & Comparison Framework
- The `DUMP_ACT` macro at `cpu_decoder.c:356-381` already dumps activations to `/tmp/act_*.bin` when `DEBUG_DECODER=1`. Ensure it's enabled in a debug build.
- Extend the framework to capture ALL layer outputs:
  1. `rvq_out` — RVQ lookup output (model.0 input)
  2. `m0_conv1d` — model.0 conv1d output
  3. `block1_snake` — model.1 snake output (before convtr)
  4. `block1_convt` — model.1 convtr output
  5. `block1_inner2_out` — model.1 block.2 block.1 (dilated conv1d K=7)
  6. `block1_inner2_skip` — model.1 block.2 residual add output
  7. `block1_inner3_out` — model.1 block.3 block.1 (dilated conv1d K=7, d=3)
  8. `block1_inner3_skip` — model.1 block.3 residual add output
  9. `block1_inner4_out` — model.1 block.4 block.1 (conv1d K=1)
  10. `block1_inner4_skip` — model.1 block.4 residual add output
  11-20: Repeat for blocks 2, 3, 4
  21. `m5_snake` — model.5 snake output
  22. `m6_pre_tanh` — model.6 conv1d output (before tanh)
  
- **Implementation**: Add `DUMP_ACT` calls at each stage in `cpu_tail.inc` and `cpu_blocks.inc`. Already present for some layers (m0, block1..4 convt, m6). Add missing ones for inner residual blocks.
- Build a Python script `scripts/layer_compare.py` that:
  1. Reads all `/tmp/act_*.bin` files
  2. Reads corresponding libnc reference files from `/tmp/libnc_act_*.bin`
  3. Computes per-layer: correlation, max_abs_diff, mean_abs_diff, RMS
  4. Generates a heatmap (matplotlib or ASCII): layers × metrics
  5. Outputs a markdown summary table

### T2: Re-Compare ALL 32+4 Layers with libnc Captures
- The 32+4 layers break down as:
  - 1 RVQ output (model.0 input)
  - 1 model.0 conv1d output
  - 4 × (1 snake + 1 convtr + 3 inner × (1 snake + 1 conv1d + 1 snake + 1 conv1d + 1 add)) = 4 × 9 = 36
  - 1 model.5 snake
  - 1 model.6 conv1d
  - 1 model.6 tanh output (final PCM)
  - Total: ~40 layer outputs
  
- For each layer output:
  - `corr > 0.99` — ✅ Good
  - `0.9 < corr < 0.99` — ⚠️ Minor divergence, investigate
  - `corr < 0.9` — ❌ Fail regression, must fix
  
- **Decision Gate G2**: If ANY layer has `corr < 0.9` after all fixes → go back to R163 (iterate kernel fixes). Specifically:
  - For conv1d layers with low corr → revisit R161 (AVX-512 fix may not have caught all issues)
  - For convtr layers with low corr → revisit R162 (layout fix insufficient)
  - For snake layers with low corr → revisit R164 (wrong snake formula)
  - For is_ct-dependent layers with low corr → revisit R163 (classification still wrong)

### T3: Measure Per-Fix Improvement Quantitatively
- For each fix round (R161-R164), create a comparison table:

  | Metric | Before (R160) | After R161 | After R162 | After R163 | After R164 |
  |--------|:-------------:|:----------:|:----------:|:----------:|:----------:|
  | WAV RMS vs libnc | 0.641 | 0.641 | 0.640 | 0.638 | 0.630 |
  | WAV corr vs libnc | 0.002 | 0.003 | 0.005 | 0.010 | 0.020 |
  | Layer avg corr | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 |
  | Max layer corr | — | — | — | — | — |
  | Layers corr<0.9 | 32/32 | 30/32 | 25/32 | 20/32 | 15/32 |
  | NaN count | 0 | 0 | 0 | 0 | 0 |
  | Clipping % | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% |
  
- Build this by re-running the comparison after each round's fix (using git checkout to revert/advance). Since we're doing this round AFTER the fixes, use the comparison between our current state and a pre-fix baseline stored via git:
  ```bash
  git stash  # save current state (post-fix)
  git checkout HEAD~4  # pre-R161 state
  cmake --build build && ./build/tsac-ng ...  # capture baseline
  git stash pop  # restore current state
  cmake --build build && ./build/tsac-ng ...  # capture current
  ```
  Compare metrics.

### T4: Measure End-to-End WAV Quality Improvements
- Use 3 test files:
  1. `test-simples/P丸様。-自分後回し@A.txc` (9 frames, short)
  2. `test-simples/silent_fast.txc` (87 frames, silence — good for SNR measurement)
  3. `test-simples/music_5s_f_q6.txc` (5 seconds, music — perceptual quality)
- For each WAV output, compute vs original tsac output:
  - **RMS** (root mean square of difference)
  - **Correlation** (Pearson)
  - **SNR** (signal-to-noise ratio, if reference is loud enough)
- Run both CPU scalar and AVX-512 paths (after R161 fix) to verify they produce identical output.
- Measure decode time improvement from AVX-512 re-enablement.

### T5: Document Residual Divergences and Root Causes
- Create a comprehensive document: `docs/residual_analysis_R165.md` covering:
  1. Layers with correlation < 0.95 after all fixes — detailed analysis of each
  2. For each diverging layer: is it weight-related, kernel-related, or activation-related?
  3. Remaining hypotheses for WAV divergence (corr still << 0.5):
     - Residual BF8 grouping axis issue (K×Co interleaved — known blocker)
     - RVQ formula difference (in_proj/out_proj vs direct codebook access)
     - Missing normalization layer (group norm parameters wrong)
     - Different conv1d padding behavior (libnc may use different padding)
  4. Quantitative impact of each remaining issue on final WAV quality
  5. Recommended next steps for Phase 4C (R166+)
- Update `.ai/state.json` with new metrics
- Update `.ai/logs/decision.log` with G2 gate decision

## Acceptance Criteria
1. All 32+4 layer outputs captured and compared with libnc references
2. Per-layer correlation heatmap generated (markdown table acceptable)
3. Each fix round's contribution quantified (improvement table)
4. Decision Gate G2 evaluated: if any layer corr < 0.9 → document and escalate
5. If G2 passes (all layers corr >= 0.9): WAV correlation measured, documented
6. Residual divergence analysis written to `docs/residual_analysis_R165.md`
7. `.ai/state.json` and `.ai/logs/decision.log` updated
8. If ANY layer has corr < 0.9, a clear plan for the next iteration is documented
