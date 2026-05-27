# Current Status
## Active Lane: ROUND_105_COMPLETE
| Metric | Value | Target |
|--------|-------|--------|
| RMS | 0.380 (bypass) / 0.641 (std) | 0.203 |
| Correlation | 0.002 | 1.000 |
| Quality | 86.85 | 87+ |
| Rounds | 27 (079-105) | 3+ |
### Blocker
BF8 grouping axis mismatch in libnc nc_reduce_sum_sqr — 500+ instruction SIMD kernel.
### Evidence
docs/libnc_weights/ (14 layer files), docs/evidence/ (GDB + disassembly + LD_PRELOAD)
