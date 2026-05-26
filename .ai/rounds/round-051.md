# Round 051 — is_ct Fix Committed, RMS Progress

**Date**: 2026-05-26
**Status**: Complete (partial fix committed)

## Achievement
is_ct detection fix committed. The bug affected ALL inner residual blocks (model.1-4.block.2/3/4.block.1/3) where Ci=Co, falsely treating them as convtr.

## RMS Trajectory
| Stage | RMS | dBFS | Delta |
|-------|-----|------|-------|
| Reference | 0.203 | -13.85 | — |
| Before fix | 0.641 | -3.86 | +10 dB (too loud) |
| After fix | 0.080 | -21.99 | -8 dB (too quiet) |

The fix reversed the error direction (was too loud, now too quiet), confirming that is_ct false positives were a significant factor. Full RMS convergence requires also fixing the convtr norm loop (deferred to R052).

## Remaining Gap
The 8 dB gap after the is_ct fix is suspected to come from:
1. Convtr norm axis mismatch (per-Co vs per-Ci)
2. Residual connection feed_forward differences
3. Snake activation alpha mismatch
