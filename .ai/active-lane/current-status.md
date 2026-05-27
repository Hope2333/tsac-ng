# Current Status — TSAC Reverse Engineering
## Active Lane: ROUND_104_COMPLETE → ROUND_105_PLANNED

| Metric | Value | Target |
|--------|-------|--------|
| RMS | 0.641 (std) | 0.203 |
| Correlation | 0.002 | 1.000 |
| Quality | 86.85 | 87+ |
| Rounds | 26 (079-104) | — |

### R104 Findings
- model.0 injection: no effect → not bottleneck
- 22-layer injection: RMS 0.9999 → transpose bug in injection code
- Root cause: BF8 grouping axis in nc_reduce_sum_sqr

### R105 Plan
Fix injection transpose → single-layer bottleneck isolation → RVQ vs DAC identification
