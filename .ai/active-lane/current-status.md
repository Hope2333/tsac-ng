# Current Status — TSAC Reverse Engineering
## Active Lane: ROUND_113_COMPLETE → ROUND_114_PLANNED

| Metric | Value | Target |
|--------|-------|--------|
| RMS | 0.380 (current) | 0.203 |
| Quality | 86.85 | 87+ |
| Rounds | 34 (079-113) | — |

### R113 Breakthrough
libnc BF8 grouping: K×Co interleaved. gs = f(K). K=7→14 values = 2Co×7K.

### R114: GDB single-step nc_reduce_sum_sqr to extract exact stride formula.
