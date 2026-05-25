# HANDOFF Round 009 → Round 010

**Date**: 2026-05-25
**From**: Sisyphus (Round 009)
**To**: Round 010

## Primary Deliverable
**Fast-mode range coder is INLINE code, not a function call.**

### Evidence
- `/tmp/gdb_arith_prologues.log` — 10 arith.c prologues tested, ALL 0 hits during fast decode
- libnc.so: nm -D shows NO range coder symbols
- Combined with fgetc=4/fread=2 I/O trace: all processing is in-memory, inline

## Critical Context for Round 010
- Entire arith.c unit NOT used in fast mode (get_freq, get_bit, put_bit, init, carry — all 0 GDB hits)
- Range coder operations are inline in TXC read function
- tsac-ng should use inline bit reader (shift/mask on byte buffer), not get_freq function call
- The "function address" premise was wrong — there IS no address to find

## Next Steps (Round 010)
1. Implement inline bit reader for fast-mode TXC decode
2. Determine correct bitstream format (byte order, bit alignment)
3. Fix audio output correctness
