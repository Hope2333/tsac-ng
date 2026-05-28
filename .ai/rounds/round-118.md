# Round 118 — WAV Validation and Tuning

**Status**: CANCELLED — BLOCKED BY ROOT CAUSE | **Predecessor**: round-117

## Reason for Cancellation
R117 confirmed the BF8 formula (gs=32, int8 signed, uint16→shl16 scale) achieves corr 0.799 in isolation but fails in deployment (corr 0.002/-0.002). Root cause is the K×Co interleaved grouping axis in nc_reduce_sum_sqr which cannot be replicated without libnc model loader source.

### Planned Tasks (not executed)
- T1: Full WAV comparison (short_fast, silent_fast, MOGRA 5s) — SUPERSEEDED by R117 test results
- T2: Tune scale interpretation — SUPERSEEDED: formula correct in isolation, grouping axis is the blocker
- T3: If corr < 0.5, analyze per-layer divergence — SUPERSEEDED by R117 layer injection findings
- T4: Iterate on formula — NOT VIABLE: no empirical approach can fix K×Co interleaved grouping

## Header Decision Required
R118 requires either:
1. libnc source access (external)
2. Complete disassembly of nc_reduce_sum_sqr (human-led GDB task)
3. Re-scope to a different objective
