# Handoff Plan — R117 Final: All Investigation Rounds Complete

## Project State

**Status**: FINAL — 38 investigation rounds (079-119) complete
**Quality**: fuck-u-code 86.67 (stable), time-complexity clean, CodeWrench 52 FP

## What Was Achieved

1. **TXC format fully reversed**: 10-bit bitpacking, CRC32 polynomial, fast/normal modes
2. **DAC architecture confirmed**: 32 conv1d/29 snake/4 convtr verified via GDB
3. **BF8 formula isolated**: gs=32, int8_t signed, uint16→shl16→float32 (corr 0.799 in isolation)
4. **Root cause identified**: K×Co interleaved grouping axis in nc_reduce_sum_sqr cannot be replicated without libnc source
5. **is_ct regression fixed**: bias->dims[0]==d0 guard
6. **All GDB sessions complete**: gdb_tsac + gdb_sqr → CLOSED

## What Remains (Requires libnc source)

1. **BF8 grouping axis replication**: The uint16→shl16 scale pre-computation happens during model loading (libnc internal). K×Co interleaved grouping requires libnc model loader source.
2. **Normal TXC decode**: Transformer + range coder not implemented
3. **Encoder strided convs**: Temporal encoding not yet correct

## Investigation Artifacts

- Model weights: `/tmp/lcc_*.bin` (62 files, 470MB)
- GDB evidence: `docs/evidence/`
- Test WAVs: `/tmp/_r117*.wav`, `/tmp/_int8.wav`, `/tmp/_signed.wav`, `/tmp/_gs32.wav`
- LD_PRELOAD shim: `/tmp/libnc_preload_v3.so`

## Resume Path

When libnc source becomes available:
1. Study `nc_reduce_sum_sqr` (0x8310) SIMD kernel for BF8 grouping axis
2. Replicate K×Co interleaved pattern in `dequant_weights()` in `src/cpu_decoder.c`
3. Test with existing WAV corpus for RMS/correlation improvement
4. Re-run quality tools before commit

## Key Contacts/Files

- Main decoder: `src/cpu_decoder.c` → `dequant_weights()`, `decode_batch()`
- Model loader: `src/model_loader.c` → `model_loader_load()`
- TXC format: `src/txc_format.c`
- State: `.ai/state.json`, `.ai/active-lane/current-status.md`
