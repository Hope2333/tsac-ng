# Round 143 — Convtr Stride + Weight Access Verification

**Status**: COMPLETED | **Date**: 2026-05-28

## Summary
GDB trace confirmed:
- Convtr stride = K/2
- Convtr weight access pattern: [Co][K][Ci]
- Scalar kernels forced for correct output
- All 4 convtr layers verified

## Key Finding
Convt weight layout [Co][K][Ci] confirmed via GDB. Matches our is_ct=1 path.
