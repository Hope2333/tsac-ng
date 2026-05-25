# HANDOFF: Round 039 → Round 040

**Date**: 2026-05-26
**State**: ROUND_039_COMPLETE

## Key Finding
BF8 dequant formula variations under the **same group structure** produce identical RMS (~0.641, -3.86 dBFS). The L2 norm cancels within-channel scale differences. However, the **group structure** (normalization axis, group boundaries) was NOT tested and remains an open variable.

## RMS Status
- Reference: RMS=0.203, -13.85 dBFS
- Our: RMS=0.641, -3.86 dBFS (10 dB too loud)
- 9 formula variants tested, none converged toward reference

## Top Hypotheses for Round 040+
1. **Group structure**: libnc may group BF8 bytes differently (across Ci vs linear), not verified
2. **Conv1d kernel**: nc_conv_1d implementation may differ from our conv1d_s/avx/avx512
3. **tanh**: Original may use different activation
4. **RVQ summation**: Accumulation across codebooks may differ

## Next Steps
- Round 040: Compare conv1d outputs layer-by-layer (GDB breakpoints at each conv1d call, capture input/output tensors)
- Round 041: Test alternative BF8 group structures (group across Ci instead of linear)
- Round 042: Verify tanh matches original

## Blockers
- libnc nc_convert_from_old_bf not interceptable (internal call, not through PLT)
- nc_conv_1d calling convention unclear (weight not in expected register)
- GDB inferior calls to libnc functions hang/crash in multi-threaded state

## Files
- src/cpu_decoder.c: dequant_weights (line 843), decode_batch (line 952)
- .ai/state.json: phase=ROUND_039_COMPLETE
