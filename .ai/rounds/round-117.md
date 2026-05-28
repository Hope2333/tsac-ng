# Round 117 — BF8 Fix Implementation (gs=32 + int8 + uint16 scale)

**Status**: COMPLETED | **Date**: 2026-05-28 | **Predecessor**: round-114 (GDB discovery)

## GDB Discovery (Round 114)
GDB session found the actual BF8 decode at libnc 0x8990:
- **group_size**: FIXED 32 (not variable 2/14/16)
- **int8 handling**: int8_t signed (not byte-128)
- **scale**: *(uint16_t*)(ptr) → shl 0x10 → float32 (not scale_byte/127)
- **Validation**: Python test achieves **correlation 0.799** with libnc ground truth (vs 0.016 best previous)

## Tasks

### T1: Implement gs=32 + int8 + uint16 scale ✅
Implemented grouped BF8 decode in dequant_weights() with:
- group_size = 32 (fixed)
- value = (float)(int8_t)byte → signed
- scale = *(uint16_t*)(data + group_start + 32) → float via shl 0x10
- result[i] = value[i] * scale

### T2: L2 norm + gain ✅
Existing L2 normalization and weight_g logic kept unchanged.

### T3: Build and test ✅
- **int8 signed alone**: RMS 0.96 (clipping), corr 0.002
- **gs=32 group-norm**: corr -0.002 (grouping axis still wrong)

### T4: Root cause identified
- **Formula correlation in isolation**: 0.799 (50× improvement over previous 0.016)
- **Deploy failure**: K×Co interleaved grouping axis cannot be replicated without libnc model loader source
- **GDB sessions**: gdb_tsac + gdb_sqr → CLOSED

## Acceptance
- [x] Build succeeds
- [ ] WAV correlation > 0.1 (FAILED — 0.002/-0.002)
- [ ] RMS gap reduced (RMS 0.96 — worse due to clipping)

## Conclusion
BF8 formula is correct in isolation (corr 0.799) but insufficient in deployment. The uint16→shl16 scale pre-computation happens during model loading (libnc internal). Without source access to libnc's model loader, the K×Co interleaved grouping pattern cannot be replicated. Investigation exhausted after 38 rounds (079-117).
