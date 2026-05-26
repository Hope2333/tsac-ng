# HANDOFF: Round 042 → 043

**Date**: 2026-05-26
**State**: ROUND_042_COMPLETE

## BREAKTHROUGH
nc_conv_1d calling convention cracked. LD_PRELOAD intercept now works correctly.
Weight dump confirmed: raw .bin data differs from libnc converted data by avg 0.017.

## Key Files
- /tmp/preload_correct.so — Correct nc_conv_1d intercept
- /tmp/libnc_conv1.bin — First conv1d weight dump (8192 floats, [8,1,1024])
- /tmp/preload_correct.log — 32-call trace with int args

## Next
- Round 043: Dump ALL 32 conv1d weights, compare with our dequant
- Round 044: Dump conv1d inputs/outputs for layer-by-layer comparison
- Round 045: Fix the specific layer that first diverges
