# Current Status

**Phase**: ROUND_048_COMPLETE → ROUND_049_PENDING (convtr norm comparison done, fix pending)
**Date**: 2026-05-26

## Active Issues
- **ISS-011**: Decoder WAV mismatch — BF8 dequant formula verified, group structure/conv kernel remain open
- **ISS-013**: Normal TXC not supported — format documented, implementation deferred

## Next Round: 049
Conv1d layer-by-layer output comparison via GDB to isolate which layer first diverges.

## Key Evidence
- 9 BF8 formula variants tested (R038): all equivalent under same group structure
- libnc weight comparison blocked (R037): nc_convert_from_old_bf is internal call
- Reference RMS: -13.85 dBFS; Our RMS: -3.86 dBFS (10 dB gap)
