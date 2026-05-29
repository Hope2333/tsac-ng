# Round 147 — Encoder Full Pipeline + CLI Integration

**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-146 | **Priority**: HIGH

## Strategy
Integrate strided convs into encoder and add encode path to CLI.

## Tasks

### T1: Add encode path to CLI
Add `tsac-ng c input.wav output.txc` command using encoder framework.

### T2: Test fast TXC encode → our decode
Round-trip: encode WAV → TXC → decode → WAV. Compare RMS.

### T3: Test fast TXC encode → original tsac decode
Cross-validate: our TXC decoded by original tsac.

### T4: Measure encode quality
PESQ/STOI comparison vs original tsac encoder output.

### T5: Document encoder status in README
Update compatibility table with encoder completion percentage.
