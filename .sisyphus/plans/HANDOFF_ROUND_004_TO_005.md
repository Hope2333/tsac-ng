# HANDOFF: Round 004 → Round 005

**Created**: 2026-05-21T15:45:00+08:00
**By**: Hephaestus (Sisyphus session)
**Status**: Round 004 active, handoff ready for Round 005 continuation

---

## What Changed (Round 004)

### CPU Decoder (`src/cpu_decoder.c`)
- **DAC realignment**: ResidualUnit now `Snake→dilated Conv7→Snake→Conv1→residual` (was: 1 Snake+1 Conv+GroupNorm)
- **weight_norm**: L2 reconstruction `W = g * v / ||v||₂` added
- **BF8 grouped scale**: `scale_byte / 127.0f` applied per group (was: skipped)
- **ConvTranspose stride**: `K/2` with stride parameter (was: hardcoded 2 in GPU)
- **ConvTranspose output**: `Ti * stride` (was: `Ti*stride - stride`)
- **Removed**: 1/K convt normalization, max_abs scaling
- **Added**: tanhf() final activation, `conv1d_dilated_s`, threaded batch decode
- **BATCH_FRAMES**: 5000 → 16

### TXC Parser (`src/txc_format.c`)
- **Fast-mode**: version=0 + flags&0x80 → uint8 RVQ indices (new path)
- **Normal reject**: version=1 + flags&0x80 → fail-fast with transformer message
- **Block count**: read from header bytes 8-11 (was: computed from payload size)

### CUDA/HIP Backends
- **Stride fix**: ConvTranspose stride now K/2 parameter (was: hardcoded 2)
- **Output length**: `next_T = cur_T * conv_stride` (was: `cur_T * 2`)
- **GN removal**: Removed incorrect GroupNorm upload/call from inner blocks
- **Residual add**: Added to inner block loop

### GroupNorm (ARM/CPU/GPU)
- **Signature changed**: `gn_s/gn_neon/gn_k` now takes `(C, T, G)` instead of `(N, G)`
- **Affine indexing**: Now per-channel `w[channel]` instead of per-element `w[idx]`
- **Note**: GroupNorm is NOT part of DAC decoder → these functions are unused in decode path

### AI Framework
- `.ai/state.json`: Round 004, ISS-011/012/013 updated, 19 tasks
- `.ai/memories/technical.md`: COT-015 to COT-019 added
- `.ai/logs/decision.log`: 9 new Round 004 decisions
- `.ai/logs/error.log`: Deduplicated, 8 new Round 004 errors (BF8, weight_norm, ResidualUnit, stride, etc.)
- `.ai/rounds/round-004.md`: Created

---

## Current State

| Layer | NaN | Status |
|-------|-----|--------|
| RVQ | 0 | Stable |
| m0 conv1d | 0 | Stable |
| block1-4 snake/convt | 0 | Stable |
| **Inner residual blocks** | 0 | **Stable — fully enabled** |
| m5 snake | 0 | Stable |
| m6 conv1d | 0 | Stable |
| tanh | 0 | Stable |

**Fast-mode WAV**: Decodes but SNR=-1.44dB vs reference (remaining bit-match issue).

---

## Open Blockers

### ISS-011 (PARTIAL): DAC output WAV mismatch
- Fast-mode decodes to stable WAV but samples differ significantly from original
- Max diff 20677@sample2475, SNR -1.44dB, 49.6% sign mismatch
- Our output oscillates; reference builds up smoothly
- **Hypothesis**: RVQ index ordering, interleaving, or delay/padding off by one

### ISS-013: Normal compressed .txc not supported
- Requires Transformer inference engine (tsac_stereo_q8.bin, 190 tensors)
- Requires arithmetic/range decoder (arith.c)
- Model: GPT-NeoX style, 12 layers, d_model=512, 3-layer codebook_decoder
- LibNC API exists on system: `nc_load_param`, `nc_convert_from_old_bf`, etc.

---

## Round 005 Priority Actions

1. **Debug fast-mode WAV bit-match**
   - Dump RVQ indices from fast file, compare with expected
   - Trace per-layer output vs original (use Python DAC reference if possible)
   - Check channel interleaving, delay offset, padding
   - Try shorter input (4 frames) to compare per-sample

2. **Transformer/Entropy subsystem design**
   - Map tsac_stereo_q8.bin tensor architecture
   - Implement GPT-NeoX forward pass (attention + MLP)
   - Implement arithmetic/range decoder
   - Wire into txc_read compressed path

3. **Port ResidualUnit fixes to CUDA/HIP**
   - Add second Snake + second Conv1d(K=1) to inner blocks
   - Sync weight_norm L2 reconstruction in GPU dequant paths
   - Test with fast-mode input

4. **Optimize dilated conv1d**
   - AVX/AVX2/AVX-512 vectorization for dilation>1
   - Threaded parallel over output channels

---

## File Map (Key Changed Files)

| File | What Changed |
|------|-------------|
| `src/cpu_decoder.c` | DAC realignment, BF8 scale, weight_norm, ResidualUnit, convt stride, tanh, BATCH_FRAMES |
| `src/txc_format.c` | Fast-mode parsing, header reads, version branching, fail-fast |
| `src/cuda/cuda_backend.cu` | Stride K/2, output length, GN removal, residual add, GN upload removal |
| `src/cuda/cuda_kernels.cu` | conv_transpose stride parameter, group_norm (C,T,G) signature |
| `hip/dac_decoder.hip.cpp` | Stride K/2, output length, GN removal, residual add, gn_k (C,T,G) |
| `src/arch/arm/cpu_arm.c/h` | group_norm_neon (C,T,G) signature |
| `.ai/state.json` | Round 004, ISS updates, 19 tasks, 3 backends |
| `.ai/memories/technical.md` | COT-015 to COT-019 |
| `.ai/logs/decision.log` | 9 new decisions |
| `.ai/logs/error.log` | Deduplicated + 8 new error records |
| `.ai/rounds/round-004.md` | Created |

---

## Quick Start (Next Session)

```bash
cd /home/miao/Projects/tsac-ng
cmake --build build
# Fast-mode decode test
./build/tsac-ng -T 8 -v d /tmp/short_fast.txc /tmp/test.wav
# Normal .txc should fail with clear message
./build/tsac-ng -v d test-simples/P丸様。-自分後回し@A.txc /tmp/should_fail.wav
```
