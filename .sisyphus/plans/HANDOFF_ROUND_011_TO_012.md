# HANDOFF Round 011 → Round 012

**Date**: 2026-05-25
**From**: Sisyphus (Round 011)
**To**: Round 012

## Primary Finding
0x4044d0 is NOT the fast-mode decoder dispatch. Fast mode uses an unknown path.

## Round 012 Priorities
1. Locate actual fast-mode dispatch (libnc.so internal or inline)
2. WAV comparison: try different offset/init combos → compare with original tsac output
3. Determine probability model / bitstream format
