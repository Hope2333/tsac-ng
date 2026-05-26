# Round 051 — Post-is_ct Fix Analysis

**Date**: 2026-05-26
**Status**: Complete

## RMS Comparison After Fix
| Metric | Before Fix | After Fix | libnc |
|--------|-----------|-----------|-------|
| model.0 RMS | 0.609 | 0.609 | 2.901 |
| Final RMS | 0.641 | 0.080 | 0.203 |
| Final dBFS | -3.86 | -21.99 | -13.85 |

## Analysis
- model.0 unchanged (is_ct fix doesn't affect regular conv1d)
- Final RMS dropped 8×: false is_ct=1 was inflating inner residual block weights
- Direction reversed: was our 3.2× too loud, now libnc 2.5× larger

## Remaining Gap
Target: increase our output 2.5× (from 0.080 to 0.203)
Suspected contributors:
- model.0 output: our 0.609 vs libnc 2.901 (4.8× gap at first layer!)
- Convtr layers pass through the 4.8× gap into residual blocks
- Residual blocks amplify further through feed_forward

## Next
- Investigate model.0 4.8× gap as primary error source
- Re-run libnc weight comparison for model.0 with corrected norm understanding
