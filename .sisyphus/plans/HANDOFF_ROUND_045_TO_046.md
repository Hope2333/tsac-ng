# HANDOFF: Round 045 → 046

**Date**: 2026-05-26
**State**: ROUND_045_COMPLETE

## BREAKTHROUGH
Divergence REVERSES through DAC graph:
- model.0: libnc 2.901 vs our 0.609 (libnc 4.8× LARGER)
- model.6: libnc 0.203 vs our 0.641 (our 3.2× LARGER)

Root cause in residual blocks + convtr upsampling (models 1-4).

## Key Data
- 32 libnc weight dumps: /tmp/libnc_w01.bin through /tmp/libnc_w32.bin
- 6 libnc output dumps: /tmp/libnc_out{01,07,14,20,26,32}.bin
- Our model.0 output: RMS=0.609 (added full_rms() debug to cpu_decoder.c)
- Full comparison table in round-044.md

## Next
- Round 046: Dump model.1 block1 convtr I/O from libnc
- Round 047: Verify convtr weight dequant (is_ct=1 path)
- Round 048: Test with fixed convtr weights
