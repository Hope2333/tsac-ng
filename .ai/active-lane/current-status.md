# Current Status — TSAC Reverse Engineering

## Active Lane: Phase 4 — R156 COMPLETE, R157-R180 PENDING

| Metric | Value | Target |
|--------|-------|--------|
| R156 (RVQ fix) | ✅ COMPLETED | — |
| RVQ RMS | 0.0012→0.0644 | ~0.03-0.06 |
| WAV corr | ~0 | >0.5 |
| R157-R180 | 24 rounds PENDING | — |
| Quality | 85.53 | 87+ |

### R156 Findings
- ROOT CAUSE: RVQ weight access layout mismatch + missing L2 norm for K=1
- Fix applied: correct [Co][Ci][K] access + conditional L2 norm
- RVQ RMS improved 50×, decoder activations now stable

### Next: R157-R180 (24 rounds pending)
GDB capture infrastructure needs refinement before continuing.
