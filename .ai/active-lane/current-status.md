# Current Status — TSAC Reverse Engineering

## Active Lane: Phase 4 Planned (R156-R180) — WAV Divergence Resolution

| Metric | Value | Phase 4 Target |
|--------|-------|:--------------:|
| Rounds | 77 (079-155) + 25 planned | 102 total |
| WAV corr | ~0 | > 0.5 |
| Quality | 85.53 | 87+ |

### Phase 4 Sub-Phases
| Sub | Rounds | Focus |
|-----|:------:|-------|
| 4A | R156-R160 | GDB activation capture (25 tasks) |
| 4B | R161-R165 | Kernel + layout fixes (21 tasks) |
| 4C | R166-R170 | Multi-file + multi-backend (21 tasks) |
| 4D | R171-R175 | Normal TXC integration (21 tasks) |
| 4E | R176-R180 | Quality + release (22 tasks) |

### Version Plan
corr > 0.5 → v0.2.0 | corr > 0.9 → v0.3.0 | corr < 0.5 → v0.1.4
