# Round 142 — BF8 Full Pipeline Reverse Engineering

**Status**: COMPLETED | **Date**: 2026-05-28

## Summary
Complete reverse engineering of libnc BF8 decode pipeline:
- **0x8990**: uint16→shl16→float32, gs=32 (confirmed)
- **elem_size=0 sentinel**: enables direct [Co][Ci][K] weight override
- **bfloat16 encoding**: confirmed for scale storage
- **gs=32 re-grouping**: weight correlation improved 0.71→0.82

## Key Finding
BF8 pipeline fully understood. Remaining gap: sample-level correlation (0.002) despite weight correlation (0.82) — suggests conv kernel or RVQ formula divergence.
