# Round 039 — RMS Root Cause Synthesis

**Date**: 2026-05-26
**Status**: Complete

## Evidence Summary

### Verified Correct
- ✅ 10-bit codebook indices: 54/54 GDB ground truth (R016)
- ✅ BF8 dequant formula: all variations produce identical RMS (R038)
- ✅ in_proj+out_proj RVQ lookup: implemented, RMS unchanged (R033)
- ✅ Codebook weights: injected originals, RMS unchanged (R025)
- ✅ CRC32, range coder (get_freq), TXC format parsing

### Known Divergences
- ⚠️ RVQ output: avg diff 3.18 vs original (R028)
- ⚠️ Audio output: 10 dB too loud (RMS=0.641 vs ref 0.203)
- ⚠️ libnc weight comparison: blocked by closed binary

### Hypothesis Ranking

| Hypothesis | Evidence | Likelihood |
|-----------|----------|:----------:|
| Conv1d kernel differs from nc_conv_1d | 32 calls match, but kernel implementation unknown | High |
| tanh implementation differs | nc_tanh absent from PLT imports | Medium |
| RVQ codebook summation scaling | Residual vectors vs simple sum | Medium |
| Missing post-processing gain | Audio scaling factor | Low |
| Group structure differs | L2 norm cancels within-channel differences | Low |

## Next Steps
1. Compare conv1d outputs layer-by-layer using GDB breakpoints
2. Verify tanh implementation matches original
3. Test with a single-frame audio (eliminate context frame issues)

## Blockers
- libnc nc_conv_1d calling convention unknown
- GDB inferior calls to libnc functions hang/crash
- LD_PRELOAD interception disturbs multi-threaded state
