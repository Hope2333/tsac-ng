# Current Status — TSAC Reverse Engineering
## Active Lane: ROUND_109_COMPLETE → ROUND_110_PLANNED

| Metric | Value | Target |
|--------|-------|--------|
| RMS | 0.641 (std) / 0.380~0.810 (bypass) | 0.203 |
| Correlation | 0.002 | 1.000 |
| Quality | 86.85 | 87+ |
| Rounds | 31 (079-109) | — |

### All Empirical Approaches Exhausted
13+ formulas, 3 injection methods, layout fixes — all fail.
Root cause: BF8 grouping axis in nc_reduce_sum_sqr (500+ instr SIMD kernel).

### R110: Close-out — accept or plan GDB single-step
