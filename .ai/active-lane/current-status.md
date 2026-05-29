# Current Status — TSAC Reverse Engineering

## Active Lane: Phase 4 Planned — WAV Divergence Resolution

| Metric | Value | Target |
|--------|-------|--------|
| Rounds | 77 (079-155) + 10 planned | — |
| BF8 weight corr | 0.82 | >0.95 |
| WAV corr | ~0 | >0.5 (Ph4) |
| Quality | 85.53 | 87+ |

### Phase 4 Strategy
Layer-by-layer GDB activation capture → compare with ours → identify first divergence layer → fix kernel/RVQ/activation → iterate.

| Round | Focus |
|:-----:|-------|
| 156 | GDB activation capture infra |
| 157 | Layer 0-1 comparison (model.0→convtr) |
| 158 | Conv1d kernel investigation |
| 159 | RVQ formula fix |
| 160 | Snake + activation fix |
| 161 | Convt kernel fix |
| 162 | Full re-measurement |
| 163 | Iterative tuning |
| 164 | Quality + docs |
| 165 | Final assessment + Oracle |

### Version Plan
- corr > 0.5 → v0.2.0 (first meaningful audio)
- corr < 0.5 → v0.1.4 (accept limitation, document residual)
