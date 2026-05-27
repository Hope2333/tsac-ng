# Current Status — TSAC Reverse Engineering
## Active Lane: ROUND_104_COMPLETE → ROUND_105_PLANNED
| Metric | Value | Target |
|--------|-------|--------|
| RMS | 0.380 (bypass) / 0.641 (std) | 0.203 |
| Correlation | 0.002 | 1.000 |
| Quality | 86.85 | 87+ |
| Rounds | 26 (079-104) | 3+ |
### Blocker
BF8 grouping axis mismatch in nc_reduce_sum_sqr. Fix requires SIMD kernel analysis.
