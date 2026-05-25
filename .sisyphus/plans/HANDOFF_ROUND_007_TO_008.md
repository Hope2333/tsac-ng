# HANDOFF Round 007 → Round 008

**Date**: 2026-05-25
**From**: Sisyphus (Round 007)
**To**: Round 008

## Primary Deliverables
- `src/range_coder.c/.h`: get_freq adaptive probability range coder (0xFF00 threshold, 15-bit)
- `src/txc_format.c`: range-coded detection + get_freq integration
- 23 TXC test files in `/tmp/mogra_slices/` (fast/normal × q6/q12)
- Normal mode TXC header: n_blocks @ bytes 8-11 BE uint32

## Critical Context
- get_freq IS the adaptive probability decoder used by original tsac (arith.c)
- get_bit at 0x42bd30 is DEAD CODE — never called
- Fast mode does NOT call vec_sum_f32 or get_freq (GDB confirmed)
- get_freq uses hardcoded 0x4000 (50/50) in txc_read — needs probability model init
- Audio output still wrong for original fast TXC files

## Next Steps (Round 008)
1. Find fast-mode range coder path (the one actually called during fast decode)
2. Implement probability model initialization from TXC header
3. Fix audio output correctness
4. Explore normal mode decode (Transformer inference needed)

## Files Modified
- src/range_coder.c (new)
- src/range_coder.h (new)
- src/txc_format.c
- src/tsac_codec.c
- src/cpu_decoder.c
- CMakeLists.txt
- .ai/state.json
- .ai/rounds/round-007.md
- .ai/active-lane/current-status.md

## Test Data
- /tmp/mogra_slices/ — 23 TXC files from MOGRA DJ audio
- Various durations: 0.1s, 0.5s, 1s, 5s (music + silence)
- Modes: fast + normal, codebooks: q6 + q12
