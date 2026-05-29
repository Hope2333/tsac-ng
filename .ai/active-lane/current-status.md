# Current Status — TSAC Reverse Engineering

## Active Lane: M2 COMPLETE → M3 PLANNED

| Metric | Value | Target |
|--------|-------|--------|
| Rounds | 67 (079-145) | — |
| BF8 weight corr | 0.82 (was 0.71) | >0.95 |
| Spectrogram corr | 0.27 | >0.9 |
| Sample corr | 0.002 | >0.9 |
| Quality | 86.67 | 87+ |

### M2 Complete
BF8 pipeline fully RE'd. gs=32 re-grouping improved weight corr.
Sample-level gap remains — conv kernel or RVQ divergence.

### Next: M3 (R146-R155) — Production Readiness
Encoder, GPU backends, quality 87+, cross-platform.
