# Round 049 — is_ct Detection Bug Found and Fixed

**Date**: 2026-05-26
**Status**: Fixed (committed)

## Bug
The is_ct detection `bias->dims[0] == d0` falsely triggered for inner residual conv1d tensors where Ci=Co. These square tensors (e.g., [768,7,768]) have d0 == bias->dims[0] but are REGULAR conv1d, not convtr.

## Fix
Added `&& d0 != d2` to the check. Convtr tensors have asymmetric dims (Co≠Ci), so d0≠d2. Regular conv1d with Ci=Co have d0==d2.

```c
- int is_ct = (bias && bias->dims[0] == d0) ? 1 : 0;
+ int is_ct = (bias && bias->dims[0] == d0 && d0 != d2) ? 1 : 0;
```

## Impact
RMS changed from 0.641 (-3.86 dBFS) to 0.080 (-21.99 dBFS). The false is_ct=1 was inflating residual block weights, making the output 10 dB too loud.

## Evidence
- Before: /tmp/_rms2.wav (RMS=0.641)
- After: /tmp/_fix8.wav (RMS=0.080)
- Reference: /tmp/ref_now.wav (RMS=0.203, -13.85 dBFS)
