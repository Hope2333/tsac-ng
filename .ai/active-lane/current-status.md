# Current Status

**Phase**: ROUND_036_COMPLETE → ROUND_037_PENDING
**Date**: 2026-05-26

## Active Issues
- **ISS-011**: BF8 dequant RMS error −3.44 dB FS — root cause confirmed, fix blocked by libnc
- **ISS-013**: Normal TXC not supported — format documented, Transformer deferred

## Next Round: 037
BF8 dequant fix via mathematical formula reverse-engineering or alternative GDB memory dump technique.

## Key Evidence
- `docs/evidence/gdb_indices_round016.txt` — 54/54 GDB ground truth
- `/tmp/gdb_q0_weights.bin` — original codebook 0 weights (8192 float32)
- `/tmp/gdb_rvq_out.bin` — original RVQ output (1024 float32)
- `docs/evidence/nc_convert_from_old_bf_disasm.txt` — libnc disassembly
