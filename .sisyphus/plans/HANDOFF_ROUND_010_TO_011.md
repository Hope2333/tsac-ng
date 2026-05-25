# HANDOFF Round 010 → Round 011

## Finding
All validity tests since Round 001 are tautological: `sym >= cb` can NEVER fail because `bits=log2(cb)` → max sym = cb-1 < cb. The test proves nothing about the bitstream.

## Corrected Approach
Use GDB to capture actual decoded indices from original tsac, then byte-compare. Or compare WAV output directly.

## Files
- /tmp/test_inline_fixed — inline bit reader test
- .ai/rounds/round-010.md
