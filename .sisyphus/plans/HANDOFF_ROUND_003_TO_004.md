# HANDOFF: tsac-ng Round 003 → Round 004

## Session Info
- **Date**: 2026-05-18
- **Round**: 003 (COMPLETE) → 004 (PENDING)
- **Author**: Sisyphus
- **Status**: HANDOFF READY

---

## What Was Achieved

### Fixed (ISS-006 through ISS-010)
1. **ISS-006**: PCM buffer heap overflow — fixed `discard_samples` formula
2. **ISS-007**: AVX-512 not detected — switched to `__get_cpuid_count(7, 0, ...)`
3. **ISS-008**: m6 output buffer hardcoded — now uses `m6_Co*cur_frames`
4. **ISS-009**: NaN output (~68-75% NaNs) — 8 bugs fixed, NaN=0 everywhere
5. **ISS-010**: conv_transpose value explosion (4.5e15) — `1.0f/K` normalization via `convt1d_s` kernel

### Key Breakthrough
- **conv_transpose 1.0f/K normalization** tames value explosion from `4.5e15` to `max ~25`
- Root cause: `test_minimal.c`'s hand-written im2col was missing normalization factor
- Fix: Replaced im2col with scalar `convt1d_s` kernel which already has `norm = 1.0f / K`
- `cpu_decoder.c` scalar kernel was already correct (no im2col used there)

---

## What Is BROKEN (Must Fix in Round 004)

### ISS-011: Output Audio Structure Mismatch
- **Symptom**: WAV output has wrong sample values
  - Our output: `(160, 867, 183, -1053, -2785, -10588...)` — large oscillating values
  - Reference: `(-1, -16, 5, -24, 3, -23...)` — near-zero quiet audio start
- **Impact**: Audio is completely wrong, not usable

### ISS-012: Inner Residual Blocks Cause 10^15+ Explosion
- **Symptom**: Enabling inner blocks (decoder.model.N.block.2-4) causes MASSIVE value explosion
  - block1 inner: `[-13M, 8.9M]` (was `[-289, 282]`)
  - block2 inner: `[-1.4e11, 2.7e11]`
  - block3 inner: `[-6.9e14, 7.0e14]`
  - block4 inner: `[-3.9e14, 3.3e14]`
- **Status**: Inner blocks **DISABLED** in `test_minimal.c`
- **Attempts**:
  1. Group norm indexing: `idx % ic_Co` → `idx / cur_frames` — still explodes
  2. BF8 dequantization with weight_g — still explodes
- **Hypothesis**: Inner block BF8 dequantization wrong (weight_g per-output-channel vs per-input), or group norm weight layout mismatch

---

## Pipeline Value Trajectory (Inner Blocks DISABLED)

| Stage | Min | Max | Notes |
|-------|-----|-----|-------|
| RVQ | -19.37 | 21.06 | Looks reasonable |
| m0 conv1d | - | - | NaN=0 |
| block1 snake | - | - | NaN=0 |
| block1 convt | -289 | 282 | After 1.0f/K norm |
| block1 inner | -289 | 282 | **SKIPPED** |
| block2 convt | -208 | 208 | After 1.0f/K norm |
| block2 inner | -208 | 208 | **SKIPPED** |
| block3 convt | -211 | 190 | After 1.0f/K norm |
| block3 inner | -211 | 190 | **SKIPPED** |
| block4 convt | -33 | 36 | After 1.0f/K norm |
| block4 inner | -33 | 36 | **SKIPPED** |
| m5 snake | -32.6 | 35.9 | NaN=0 |
| m6 conv1d | -20.0 | 25.0 | NaN=0, scale=0.04 |

---

## Critical Files

| File | Purpose | Status |
|------|---------|--------|
| `test_minimal.c` | Debug harness — load model, decode 20 frames, write WAV | WORKING (inner blocks disabled) |
| `src/cpu_decoder.c` | Production CPU decoder with SIMD dispatch | NaN=0, but inner blocks not fully tested |
| `src/model_loader.c` | Model loading, BF8/fp32 detection | WORKING |
| `test-simples/P丸様。-自分後回し@A.txc` | Test input file | EXISTS |
| `test-simples/tsac_orig_decode.wav` | Reference output from original tsac | EXISTS (28MB, full decode) |
| `/usr/share/tsac/dac_stereo_q8.bin` | Decoder model (322 tensors, BF8) | EXISTS |

---

## Next Session Action Items (Priority Order)

### 1. Fix Inner Residual Block Dequantization
- Inner blocks use BF8 weights with `weight_g` — the dequantization logic in `dequant_weights()` may be wrong for these layers
- Check: `weight_g` dims for inner blocks vs outer blocks
- Check: `per_input` detection logic (`Ci == weight_g->dims[2]`)
- **Action**: Add debug output for inner block weight_g dimensions and dequant values

### 2. Fix Group Norm Weight Indexing
- Group norm weights have `ic_Co` elements (per-channel), not `ic_Co * cur_frames`
- Current: `ch_idx = idx / cur_frames` — verify this is correct
- **Action**: Compare with PyTorch `nn.GroupNorm` implementation

### 3. Verify RVQ Lookup
- RVQ accumulation: `rvq_out[d * ctx_frames + f] += cb_data[entry * dim + d]`
- Check: Is this the correct layout? Should it be `rvq_out[f * rvq_dim + d]`?
- **Action**: Print first 10 RVQ values and compare with expected

### 4. Compare with Python/PyTorch Descript DAC Reference
- The model is based on Descript DAC (https://github.com/descriptinc/descript-audio-codec)
- **Action**: Run Python DAC decode on same input, compare intermediate tensor stats

### 5. Re-enable Inner Blocks in test_minimal.c
- Only after #1 and #2 are fixed
- **Action**: Uncomment inner block code, verify no explosion

---

## Build Commands

```bash
cd /home/miao/Projects/tsac-ng
# Build library
cmake -B build && cmake --build build

# Build test harness
gcc -O2 -I. -Iinclude -o test_minimal test_minimal.c build/libtsac-ng.so -lm -lpthread

# Run test
LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./test_minimal 2>&1
```

---

## Lessons Learned

1. **conv_transpose normalization**: PyTorch ConvTranspose1d does NOT apply `1/K` normalization by default. The `1.0f/K` factor in `convt1d_s` was added to prevent explosion. Verify if this is correct or if the model weights already account for this.

2. **Group norm indexing**: Group norm weights are per-channel, not per-element. The indexing `w[idx]` where `idx` spans `[0, Co*frames)` is WRONG — should be `w[idx % Co]` or `w[idx / frames]`.

3. **Inner blocks are essential**: The decoder has 3 inner residual blocks per stage (block.2, block.3, block.4). Skipping them produces stable but WRONG output. They must be fixed, not skipped.

4. **test_minimal.c is the debug harness**: Use it for rapid iteration. Don't test in the full binary.

---

## End of Handoff
