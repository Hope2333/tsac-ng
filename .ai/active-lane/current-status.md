# Current Status — TSAC Reverse Engineering

## Active Lane: ROUND_084_COMPLETE

### Last Commit
Pending — is_ct fix applied to src/cpu_decoder.c

### Status Summary

| Metric | Value | Target |
|--------|-------|--------|
| RMS (our) | 0.641 (-3.86 dBFS) | 0.203 (-13.85 dBFS) |
| RMS gap | +9.99 dB | 0 dB |
| WAV correlation | 0.002 | 1.000 |
| Codebook index accuracy | 54/54 (100%) | 54/54 ✓ |
| fuck-u-code score | 86.85 | 87+ (stalled) |
| GDB capture | Working | - |

### Key Finding from Rounds 079-084

The is_ct regression (commit 6119c3c) was identified and fixed. The `d0 != d2` guard incorrectly excluded square inner residual blocks (Ci==Co) from convtr dequant layout. Removing this guard restored RMS from 0.080 (-21.99 dBFS) to 0.641 (-3.86 dBFS).

However, RMS restoration is superficial — waveform correlation with reference is 0.002, meaning the audio output is essentially unrecognizable. The root cause is in the BF8 dequant formula for in_proj codebook weights, which produces codebook entry vectors that diverge from libnc by factors up to 545×.

GDB capture of libnc's model.0 input confirms: the first RVQ dimension matches (0.038≈0.038), but most dimensions diverge wildly.

### Blockers
1. **BF8 dequant formula**: our `(byte-128)*group_scale` doesn't match libnc's `nc_convert` BF8 decode
2. **nc_convert BF8 decode**: disassembled but formula not extracted from SIMD-optimized assembly

### Next Round (085)
Disassemble the BF8-specific decode path in `nc_convert` (type == 0xe or 0xf) to extract the exact formula for grouped block floating point decode.

### Round History
- Round 084: GDB capture + nc_convert disassembly (COMPLETE)
- Round 083: MOGRA audio experiments (COMPLETE)
- Round 082: Quality tools re-evaluation (COMPLETE)
- Round 081: Out-proj dequant + WAV correlation (COMPLETE)
- Round 080: is_ct fix + HIP analysis (COMPLETE)
- Round 079: in_proj dequant + bisection (COMPLETE)
- Rounds 001-078: see .ai/rounds/ for complete history
