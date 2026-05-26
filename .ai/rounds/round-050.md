# Round 050 — Convtr Norm Loop Verified Correct (Oracle Finding)

**Date**: 2026-05-26
**Status**: Verified correct — no fix needed

## Oracle Review
The norm loop "fix" attempted in R050 was WRONG. The current norm loop:
```
for (i0, k, i2) → src_idx = i0*K*d2 + k*d2 + i2 → norms[i2]
```
is axis-correct WITHOUT branching:
- Conv1d (stored [Ci,K,Co], is_ct=0): d0=Ci, d2=Co → norms per output channel ✓
- Convtr (stored [Co,K,Ci], is_ct=1): d0=Co, d2=Ci → norms per input channel ✓

Both match PyTorch weight_norm(dim=0) semantics. The R050 transposition attempt broke this.

## weight_g Verification (BLOCKER-2)
All convtr weight_g→dims[2] confirmed to equal Ci:
- model.1: 1536 ✓, model.2: 768 ✓, model.3: 384 ✓, model.4: 192 ✓
- No OOB risk. norm_channels always ≥ Ci.

## Layer-by-layer after fix (BLOCKER-1)
| Layer | Our RMS | libnc RMS | Ratio |
|-------|---------|-----------|-------|
| model.0 | 0.609 | 2.901 | libnc 4.8× larger |
| final WAV | 0.080 | 0.203 | libnc 2.5× larger |

Direction reversed from pre-fix (was our 3.2× larger). Remaining gap: need 2.5× more gain.
