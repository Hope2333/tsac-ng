# Current Status

**Phase**: ROUND_054_COMPLETE → ROUND_055_PENDING
**Date**: 2026-05-26

## Active Issues
- **ISS-011**: Decoder WAV mismatch — model.0 dequant verified correct (injection test); gap is in conv1d kernel or RVQ
- **ISS-013**: Normal TXC not supported — format documented, implementation deferred

## RMS Status
- Reference: 0.203 (-13.85 dBFS)
- Our (after is_ct fix): 0.080 (-21.99 dBFS)
- Gap: libnc output is 2.5× larger

## Next Round: 055
Compare our conv1d kernel output vs libnc nc_conv_1d for same input and weights.

## Key Evidence
- /tmp/libnc_w07.bin — libnc model.0 weight (R043)
- Model.0 weight injection test: RMS unchanged (R052)
- is_ct fix committed: 6119c3c (R049)
