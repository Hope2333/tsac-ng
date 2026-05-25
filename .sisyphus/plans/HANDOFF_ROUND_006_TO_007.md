# HANDOFF: Round 006 → Round 007

**Created**: 2026-05-24T12:00:00+08:00
**By**: Hephaestus

## Round 006 Context
CRC32 fully reversed (Oracle-verified). Architecture confirmed (RWKV + range coder + DAC). GDB evidence: 32 conv1d/batch. Fast TXC has 2.5x WAV mismatch — frame count discrepancy.

## Key Findings (Round 005→006)
- **CRC32**: polynomial 0x04C11DB7, table@0x43dda0, function@0x42a610, patch@0x5D6A
- **batch_size**: normal=16, fast=unused
- **Decoder batch**: 32 nc_conv_1d/batch = 30 DAC + 2 RVQ
- **Architecture**: 9 source files (arith.c, transformer.c, nc_block.c, ...)

## Round 007 Quick Start
```bash
cd /home/miao/Projects/tsac-ng
cat .ai/AGENTS.md          # Workflow
cat .ai/state.json         # Current state
cat .ai/memories/RE_NOTES.md  # RE reference
cmake --build build        # Verify build
```
